"""The released dipeptide graph transformer, implemented in PyTorch.

Scores follow the source convention: a potential expert returns minus its
coordinate gradient, while a direct expert returns minus its decoder output.
Centering happens before differentiating each expert, as in MixtureOfModels.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def _initialize(module):
    if isinstance(module, nn.Linear):
        # Flax Dense: variance_scaling(1, "fan_in", "truncated_normal").
        std = math.sqrt(1 / module.in_features) / 0.87962566103423978
        nn.init.trunc_normal_(module.weight, std=std, a=-2 * std, b=2 * std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        # Flax Embed initializes with its embedding dimension as fan-in.
        nn.init.normal_(module.weight, std=1 / math.sqrt(module.embedding_dim))


class GatedResidual(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.gate = nn.Linear(3 * width, 1, bias=False)

    def forward(self, x, residual):
        gate = torch.sigmoid(self.gate(torch.cat((x, residual, x - residual), dim=-1)))
        return x * gate + residual * (1 - gate)


class Attention(nn.Module):
    def __init__(self, width, heads=8, head_dim=64):
        super().__init__()
        self.heads, self.head_dim = heads, head_dim
        inner = heads * head_dim
        self.q = nn.Linear(width, inner)
        self.k = nn.Linear(width, inner)
        self.v = nn.Linear(width, inner)
        self.edge_kv = nn.Linear(width, inner)
        self.out = nn.Linear(inner, width)
        self.register_buffer("_edge_weight", None, persistent=False)
        self.register_buffer("_edge_bias", None, persistent=False)

    def forward(self, nodes, edges, factored_positions=None, factored_bias=None):
        b, n, _ = nodes.shape

        def heads(x):
            return x.reshape(b, *x.shape[1:-1], self.heads, self.head_dim).movedim(-2, 1)

        q, k, v = (heads(layer(nodes)) for layer in (self.q, self.k, self.v))
        if factored_positions is not None:
            # edge[i,j] = A*x[j] - A*x[i] + b. The query-row
            # constant -A*x[i] + b cancels exactly inside softmax.
            positions = heads(factored_positions)
            similarity = torch.matmul(q, (k + positions).transpose(-1, -2)) * self.head_dim**-0.5
            attention = similarity.softmax(dim=-1)
            out = torch.matmul(attention, v + positions) - positions
            out = out + factored_bias.reshape(1, self.heads, 1, self.head_dim)
            return self.out(out.transpose(1, 2).reshape(b, n, -1))
        edge = heads(self.edge_kv(edges) if self._edge_weight is None else
                     F.linear(edges, self._edge_weight, self._edge_bias))
        # The same edge embedding is added to both keys and values.
        k = k[:, :, None, :, :] + edge
        v = v[:, :, None, :, :] + edge
        similarity = torch.einsum("bhid,bhijd->bhij", q, k) * self.head_dim**-0.5
        attention = similarity.softmax(dim=-1)
        out = torch.einsum("bhij,bhijd->bhid", attention, v)
        return self.out(out.transpose(1, 2).reshape(b, n, -1))


class TransformerLayer(nn.Module):
    def __init__(self, width, heads, head_dim):
        super().__init__()
        self.attention_norm = nn.LayerNorm(width, eps=1e-6)
        self.attention = Attention(width, heads, head_dim)
        self.attention_residual = GatedResidual(width)
        self.feedforward_norm = nn.LayerNorm(width, eps=1e-6)
        self.feedforward_in = nn.Linear(width, 4 * width)
        self.feedforward_out = nn.Linear(4 * width, width)
        self.feedforward_residual = GatedResidual(width)

    def forward(self, nodes, edges, factored_positions=None, factored_bias=None):
        nodes = self.attention_residual(self.attention(self.attention_norm(nodes), edges,
                                                     factored_positions, factored_bias), nodes)
        h = self.feedforward_norm(nodes)
        h = self.feedforward_out(F.gelu(self.feedforward_in(h), approximate="tanh"))
        return self.feedforward_residual(h, nodes)


class GraphTransformer(nn.Module):
    def __init__(self, width, layers, embedding_dim, heads, head_dim, time_range, potential):
        super().__init__()
        self.t1, self.t0 = time_range
        self.potential = potential
        # Preserve the source's asymmetric test, including its missing abs().
        self.relative_time = self.t0 >= 1e-6 or self.t1 - 1.0 >= 1e-6
        self.scalar_embeddings = nn.ModuleList(nn.Embedding(size, embedding_dim) for size in (20, 5, 3, 2))
        self.edge_embedding = nn.Linear(3, width)
        self.node_embedding = nn.Linear(4 * embedding_dim + 4 + 3 * self.relative_time, width)
        self.layers = nn.ModuleList(TransformerLayer(width, heads, head_dim) for _ in range(layers))
        self.node_decoder = nn.Linear(width if potential else 10 * width, 1 if potential else 30)
        self._compiled_forward = None
        self._compiled_potential = None
        self._inference_cache = False
        self._factored_attention = False

    def time_features(self, t):
        t = t.clamp(self.t0, self.t1)
        features = [t - (self.t1 - self.t0) / 2, (2 * math.pi * t).cos(),
                    (2 * math.pi * t).sin(), -(4 * math.pi * t).cos()]
        if self.relative_time:
            relative = (t - self.t0) / (self.t1 - self.t0)
            features += [(2 * math.pi * relative).cos(), (2 * math.pi * relative).sin(),
                         -(4 * math.pi * relative).cos()]
        return torch.cat(features, dim=-1)

    def _differentiable_forward(self, x, features, t):
        x = x.reshape(-1, 10, 3)
        embedded = torch.cat([embedding(features[..., i].long())
                              for i, embedding in enumerate(self.scalar_embeddings)], dim=-1)
        time = self.time_features(t).unsqueeze(1).expand(-1, x.shape[1], -1)
        nodes = self.node_embedding(torch.cat((embedded, time), dim=-1))
        if self._factored_attention:
            for layer in self.layers:
                attention = layer.attention
                if attention._edge_weight is None:
                    projection = attention.edge_kv
                    weight = projection.weight @ self.edge_embedding.weight
                    bias = projection.bias + projection.weight @ self.edge_embedding.bias
                else:
                    weight, bias = attention._edge_weight, attention._edge_bias
                positions = F.linear(x, weight, bias=None)
                nodes = layer(nodes, None, positions, bias)
            if self.potential:
                return self.node_decoder(nodes)
            return self.node_decoder(nodes.flatten(1)).reshape_as(x)
        # Source xa=x[:, None], xb=x[:, :, None]: edge[i,j]=x[j]-x[i].
        edges = x[:, None, :, :] - x[:, :, None, :]
        if not self._inference_cache:
            edges = self.edge_embedding(edges)
        for layer in self.layers:
            nodes = layer(nodes, edges)
        if self.potential:
            return self.node_decoder(nodes)
        return self.node_decoder(nodes.flatten(1)).reshape_as(x)

    def log_q_and_score(self, x, features, t, create_graph=False):
        if not self.potential:
            raise ValueError("A direct-score expert has no energy.")
        if self._compiled_potential is not None:
            score, (_, log_q) = self._compiled_potential(x, features, t)
            return log_q, score if create_graph else score.detach()
        with torch.enable_grad():
            if not x.requires_grad:
                x = x.requires_grad_(True)
            energy = self._differentiable_forward(x, features, t).sum(dim=(1, 2)).unsqueeze(1)
            score = -torch.autograd.grad(energy.sum(), x, create_graph=create_graph,
                                         retain_graph=create_graph)[0]
        return -energy, score

    def _functional_potential(self, x, features, t):
        def value_and_aux(position):
            log_q = -self._differentiable_forward(position, features, t).sum(dim=(1, 2)).unsqueeze(1)
            return log_q.sum(), log_q

        return torch.func.grad_and_value(value_and_aux, has_aux=True)(x)

    def log_q(self, x, features, t):
        forward = self._compiled_forward or self._differentiable_forward
        return -forward(x, features, t).sum(dim=(1, 2)).unsqueeze(1)

    def forward(self, x, features, t, create_graph=False):
        if self.potential:
            return self.log_q_and_score(x, features, t, create_graph)[1]
        forward = self._compiled_forward or self._differentiable_forward
        return -forward(x, features, t).reshape_as(x)


class DipeptideModel(nn.Module):
    """Single potential, or two direct-score experts plus a low-time potential.

    ``active_experts`` restricts evaluated experts without changing the source
    range weights. ``freeze_for_stage(i)`` enables training only expert ``i``.
    Use ``create_graph=True`` when differentiating scores in DSM/FP training.
    """

    def __init__(self, method, config):
        super().__init__()
        self.method = method
        settings = config.get("model", {})
        if method not in ("diffusion", "fokker_planck", "two_for_one", "mixture"):
            raise ValueError(f"Unknown method: {method}")
        self.ranges = [(1.0, 0.6), (0.6, 0.1), (0.1, 0.0)] if method == "mixture" else [(1.0, 0.0)]
        self.experts = nn.ModuleList()
        for index, time_range in enumerate(self.ranges):
            potential = method != "mixture" or index == 2
            width = settings.get("hidden_dim", 128) if potential else settings.get("score_hidden_dim", 96)
            layers = settings.get("layers", 3) if potential else settings.get("score_layers", 2)
            self.experts.append(GraphTransformer(width, layers, settings.get("embedding_dim", 16),
                                                settings.get("heads", 8), settings.get("head_dim", 64),
                                                time_range, potential))
        self.apply(_initialize)

    def freeze_for_stage(self, stage):
        for index, expert in enumerate(self.experts):
            expert.requires_grad_(index == stage)

    def enable_compile(self):
        """Compile pure expert calculations without altering checkpoint keys.

        Functional differentiation lets AOTAutograd compile the parameter
        gradients of conservative scores, including the residual FP loss.
        """
        for expert in self.experts:
            expert._compiled_forward = torch.compile(expert._differentiable_forward, fullgraph=True)
            if expert.potential:
                expert._compiled_potential = torch.compile(expert._functional_potential, fullgraph=True)
        return self

    def enable_factorized_attention(self, enabled=True):
        """Compute exact affine relative-edge attention without pairwise edges.

        Trainable forwards recompute affine composition with ordinary autograd.
        Frozen inference can reuse the validated affine projection cache.
        Plain matmul/softmax retains all coordinate and parameter derivatives.
        The disabled path and all checkpoint keys stay unchanged.
        """
        if (enabled and any(expert._inference_cache for expert in self.experts)
                and any(parameter.requires_grad for parameter in self.parameters())):
            raise ValueError("Freeze model parameters before combining factored attention and inference caching.")
        for expert in self.experts:
            expert._factored_attention = bool(enabled)
        return self

    def enable_inference_cache(self):
        """Compose adjacent edge projections after freezing/loading weights.

        There is no nonlinearity between these projections. Cached tensors
        preserve coordinate derivatives and leave checkpoint keys unchanged.
        Rebuild the cache after loading different weights.
        """
        if any(parameter.requires_grad for parameter in self.parameters()):
            raise ValueError("Freeze model parameters before caching inference projections.")
        with torch.no_grad():
            for expert in self.experts:
                for layer in expert.layers:
                    attention = layer.attention
                    attention._edge_weight = attention.edge_kv.weight @ expert.edge_embedding.weight
                    attention._edge_bias = attention.edge_kv.bias + attention.edge_kv.weight @ expert.edge_embedding.bias
                expert._inference_cache = True
        return self

    def _weights(self, t, active_experts):
        time = t.clamp(0.0, 1.0).reshape(-1)
        weights = torch.stack([((time >= t0) & (time <= t1)).to(t.dtype)
                               for t1, t0 in self.ranges])
        weights = weights / weights.sum(dim=0, keepdim=True)
        # Preserve the released mixture's duplicate-index condition verbatim.
        if active_experts and len(set(active_experts)) != len(active_experts):
            mask = torch.zeros_like(weights)
            mask[list(active_experts)] = 1
            weights = weights * mask
            weights = weights / weights.sum(dim=0, keepdim=True)
        return weights

    def _evaluate(self, x, features, t, mode, create_graph, active_experts):
        original_shape = x.shape
        x = x.reshape(-1, 30)
        features = features.reshape(-1, 10, 4)
        t = t.reshape(x.shape[0], 1)
        # Center before the experts' gradient, without projecting direct scores.
        x = (x.reshape(-1, 10, 3) - x.reshape(-1, 10, 3).mean(dim=1, keepdim=True)).reshape_as(x)
        if len(self.experts) == 1:
            expert = self.experts[0]
            if active_experts and 0 not in active_experts:
                return torch.zeros_like(t), torch.zeros_like(x).reshape(original_shape)
            if mode == "score":
                return torch.zeros_like(t), expert(x, features, t, create_graph).reshape(original_shape)
            if mode == "log_q":
                value = expert.log_q(x, features, t)
                return value, torch.zeros_like(x).reshape(original_shape)
            value, force = expert.log_q_and_score(x, features, t, create_graph)
            return value, force.reshape(original_shape)
        weights = self._weights(t, active_experts)
        score, log_q = torch.zeros_like(x), torch.zeros_like(t)
        selected = range(len(self.experts)) if not active_experts else dict.fromkeys(active_experts)
        for index in selected:
            expert = self.experts[index]
            weight = weights[index].unsqueeze(1)
            if not bool(torch.any(weight > 1e-2)):
                continue
            if mode == "score":
                score = score + weight * expert(x, features, t, create_graph)
            elif expert.potential:
                if mode == "log_q":
                    value = expert.log_q(x, features, t)
                else:
                    value, force = expert.log_q_and_score(x, features, t, create_graph)
                    score = score + weight * force
                log_q = log_q + weight * value
        return log_q, score.reshape(original_shape)

    def forward(self, x, features, t, create_graph=False, active_experts=None):
        return self._evaluate(x, features, t, "score", create_graph, active_experts)[1]

    def log_q(self, x, features, t, create_graph=False, active_experts=None):
        return self._evaluate(x, features, t, "log_q", create_graph, active_experts)[0]

    def log_q_and_score(self, x, features, t, create_graph=False, active_experts=None):
        return self._evaluate(x, features, t, "log_q_and_score", create_graph, active_experts)

    def potential_score(self, x, features, t):
        """Graphable force of the potential expert at a fixed simulation time.

        This bypasses mixture routing, so callers must use the low-time range
        for the mixture. Differentiation follows centering, preserving the
        released implementation's order of operations.
        """
        shape = x.shape
        x = x.reshape(-1, 10, 3)
        centered = (x - x.mean(dim=1, keepdim=True)).reshape(-1, 30)
        expert = self.experts[-1]

        def total_log_q(position):
            return -expert._differentiable_forward(position, features, t).sum()

        return torch.func.grad(total_log_q)(centered).reshape(shape)
