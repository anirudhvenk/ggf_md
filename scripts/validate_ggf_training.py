#!/usr/bin/env python3
"""CPU-only synthetic checks for the paired-singleton GGF training contract."""

from argparse import Namespace
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

# This validator never discovers accelerators or loads production datasets.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.nn import functional as F

import ggf_score_matching as ggf
from model import DipeptideModel, GraphTransformer
import train


def tiny_config(config):
    config = copy.deepcopy(config)
    config["model"].update(hidden_dim=8, layers=1, embedding_dim=2, heads=2, head_dim=4)
    config["training"].update(batch_size=8, epochs=2)
    return config


def features(batch):
    values = torch.zeros(batch, 10, 4, dtype=torch.long)
    values[..., 1] = torch.arange(10) % 5
    values[..., 2] = torch.arange(10) % 3
    values[:, 5:, 3] = 1
    return values


def target_checks(config):
    identity = ggf.ggf_identity(config)
    assert identity == {"objective": "ggf_singleton", "noise_conditioned": False,
                        "sigma_min": ggf.SIGMA_MIN, "sigma_max": 20.0,
                        "bandwidth_h": ggf.SIGMA_MIN, "fixed_time": 0.0,
                        "iid_sampler": "ve_euler_maruyama",
                        "iid_score_adapter": "h2_over_sigma2"}
    assert json.loads(json.dumps(identity)) == identity
    for lo, hi in ((0.0, 20.0), (20.0, 20.0), (21.0, 20.0), (float("nan"), 20.0)):
        invalid = copy.deepcopy(config)
        invalid["ggf"].update(sigma_min=lo, sigma_max=hi)
        try:
            ggf.ggf_identity(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid GGF experiment noise bounds were accepted")
    h = identity["sigma_min"]
    data = torch.linspace(-1, 1, 90, dtype=torch.float64).reshape(3, 30)
    noise = torch.linspace(0.2, 1.2, 90, dtype=torch.float64).reshape_as(data)
    scales = torch.tensor([h, 1.0, 20.0], dtype=data.dtype).reshape(-1, 1)
    target = -scales * noise / h**2
    query = data + scales * noise
    captured = []
    mse = F.mse_loss

    def capture(predictions, targets):
        assert not targets.requires_grad
        captured.append(targets.clone())
        return mse(predictions, targets)

    with patch.object(ggf.F, "mse_loss", side_effect=capture):
        loss = ggf.ggf_score_matching(lambda x: 0.125 * x, data, h,
                                      noise=noise, noise_std=scales)
        torch.testing.assert_close(loss, mse(0.125 * query, target), rtol=0, atol=0)
        changed_data, changed_noise, changed_scales = data.clone(), noise.clone(), scales.clone()
        changed_data[1:] += 100
        changed_noise[1:] *= -2
        changed_scales[1:] *= 0.5
        ggf.ggf_score_matching(torch.zeros_like, changed_data, h,
                               noise=changed_noise, noise_std=changed_scales)
    torch.testing.assert_close(captured[0], target, rtol=0, atol=0)
    torch.testing.assert_close(captured[0][0], captured[1][0], rtol=0, atol=0)
    assert captured[0].reshape(3, 10, 3).mean(dim=1).abs().max() > 0  # no COM projection

    # At large coordinates, forming data-query would round the displacement away.
    huge = torch.full((1, 30), 1e10, dtype=torch.float32)
    loss = ggf.ggf_score_matching(torch.zeros_like, huge, h,
                                  noise=torch.ones_like(huge), noise_std=huge.new_tensor([h]))
    torch.testing.assert_close(loss, huge.new_tensor(1 / h**2), rtol=2e-7, atol=0)
    torch.manual_seed(41)
    drawn = ggf.sample_noise_std(torch.zeros(4096, 30), h, 20)
    assert drawn.shape == (4096, 1) and drawn.min() >= h and drawn.max() <= 20
    assert abs(float(drawn.log().mean()) - (np.log(h) + np.log(20)) / 2) < 0.15
    return {"paired_target": True, "no_cross_sample_target_dependence": True,
            "raw_mse_and_no_com_projection": True, "stable_displacement_formula": True}


def energy_checks(config):
    torch.manual_seed(9)
    expert = GraphTransformer(8, 1, 2, 2, 4, (1.0, 0.0), True, fixed_time=0.0).double()
    theta = torch.nn.Parameter(torch.tensor(0.3, dtype=torch.float64))
    expert.register_parameter("test_theta", theta)

    def quadratic(x, atom_features, t):
        return 0.5 * theta * x.reshape(-1, 10, 3).square().sum(dim=-1, keepdim=True)

    expert._differentiable_forward = quadratic
    data = torch.linspace(-0.4, 0.5, 60, dtype=torch.float64).reshape(2, 30)
    noise = torch.linspace(-1, 1, 60, dtype=torch.float64).reshape_as(data)
    scales = data.new_tensor([0.01, 20]).reshape(-1, 1)
    t = data.new_zeros((2, 1))
    atom_features = features(2)
    query = (data + scales * noise).requires_grad_()
    score = expert(query, atom_features, t, create_graph=True)
    torch.testing.assert_close(score, -theta * query, rtol=0, atol=0)
    loss = ggf.ggf_score_matching(
        lambda x: expert(x, atom_features, t, create_graph=True), data,
        noise=noise, noise_std=scales)
    gradient = torch.autograd.grad(loss, theta)[0]
    expected = 2 * ((theta.detach() * query.detach() - scales * noise / ggf.SIGMA_MIN**2)
                    * query.detach()).mean()
    torch.testing.assert_close(gradient, expected, rtol=1e-14, atol=1e-8)

    model = DipeptideModel("ggf", tiny_config(config)).double()
    diffusion = DipeptideModel("diffusion", tiny_config(config)).double()
    assert {key: value.shape for key, value in model.state_dict().items()} == {
        key: value.shape for key, value in diffusion.state_dict().items()}
    x = torch.randn(2, 30, dtype=torch.float64)
    references = []
    direct = []
    for time in (0.0, 0.2, 1.0):
        times = torch.full((2, 1), time, dtype=x.dtype)
        references.append(model(x.clone(), atom_features, times, create_graph=True))
        direct.append(model.experts[0]._differentiable_forward(x, atom_features, times))
    for index in (1, 2):
        torch.testing.assert_close(references[index], references[0], rtol=0, atol=0)
        torch.testing.assert_close(direct[index], direct[0], rtol=0, atol=0)
    x = x.requires_grad_()
    times = x.new_ones((2, 1))
    log_q = model.log_q(x, atom_features, times)
    coordinate_gradient = torch.autograd.grad(log_q.sum(), x)[0]
    torch.testing.assert_close(model(x, atom_features, times), coordinate_gradient,
                               rtol=1e-10, atol=1e-12)
    with torch.no_grad():
        validation_score = model(x.detach().clone(), atom_features, times, create_graph=False)
    assert not validation_score.requires_grad
    return {"energy_score_sign": True, "mixed_parameter_derivative": True,
            "fixed_time_training_and_direct_inference": True,
            "baseline_parameter_shapes": True, "validation_coordinate_autograd": True}


def microbatch_checks(config):
    config = tiny_config(config)
    torch.manual_seed(72)
    initial = DipeptideModel("ggf", config)
    x = torch.randn(8, 30)
    atom_features = features(8)
    before = torch.get_rng_state().clone()

    def run(size):
        model = copy.deepcopy(initial)
        torch.set_rng_state(before)
        scales = ggf.sample_noise_std(x, ggf.SIGMA_MIN, 20)
        noise = torch.randn_like(x)
        after_draw = torch.get_rng_state().clone()
        terms = torch.zeros(3)
        for begin in range(0, len(x), size):
            end = min(begin + size, len(x))
            loss = train.losses(model, x[begin:end], atom_features[begin:end],
                                x.new_zeros((end - begin, 1)), config,
                                noise=noise[begin:end], noise_std=scales[begin:end])
            weighted = loss * ((end - begin) / len(x))
            weighted.sum().backward()
            terms += weighted.detach()
        assert torch.equal(after_draw, torch.get_rng_state())
        return terms, {key: parameter.grad for key, parameter in model.named_parameters()}, after_draw

    full, full_grad, full_rng = run(8)
    micro, micro_grad, micro_rng = run(3)
    torch.testing.assert_close(micro, full, rtol=2e-7, atol=0)
    assert torch.equal(full_rng, micro_rng)
    for key in full_grad:
        if full_grad[key] is None:
            assert micro_grad[key] is None
        else:
            torch.testing.assert_close(micro_grad[key], full_grad[key], rtol=2e-4, atol=0.02)
    return {"microbatch_loss_gradient": True, "microbatch_rng": True}


def synthetic_training_checks(config):
    config = tiny_config(config)
    rng = np.random.default_rng(83)
    atom_features = features(1)[0].numpy()
    fixtures = {split: {"AA": {"x": rng.normal(size=(count, 30)).astype(np.float32),
                               "features": atom_features}}
                for split, count in (("train", 16), ("val", 8))}

    def run(root, microbatch, resume=None, max_steps=None):
        current = copy.deepcopy(config)
        current["data"]["root"] = str(root)
        args = Namespace(method="ggf", device="cpu", microbatch=microbatch, resume=resume,
                         max_steps=max_steps, profile_only=False, factorized_attention=False,
                         compile=False)
        with patch.object(train, "prepare_data", return_value={"norm_factor": 1.0}), \
                patch.object(train, "load_split", side_effect=lambda cfg, split: fixtures[split]):
            train.train(current, args)
        return root / "runs/ggf/seed1" / ("latest.pt" if max_steps else "final.pt")

    with tempfile.TemporaryDirectory(prefix="ggf-training-validator-") as directory:
        root = Path(directory)
        full = torch.load(run(root / "full", 8), map_location="cpu", weights_only=False)
        micro = torch.load(run(root / "micro", 3), map_location="cpu", weights_only=False)
        interrupted_path = run(root / "resume", 3, max_steps=1)
        resumed = torch.load(run(root / "resume", 3, resume=str(interrupted_path)),
                             map_location="cpu", weights_only=False)
        for candidate in (micro, resumed):
            assert candidate["completed"] and candidate["global_step"] == 4
            assert candidate["ggf"] == ggf.ggf_identity(config)
            assert torch.equal(candidate["rng"]["torch"], full["rng"]["torch"])
            assert candidate["rng"]["numpy"] == full["rng"]["numpy"]
            assert candidate["rng"]["python"] == full["rng"]["python"]
        # Gradient accumulation changes floating-point summation. In particular,
        # Adam can amplify roundoff in the softmax-shift-null attention key bias.
        # Resume with the same partition must reproduce the complete state exactly.
        for collection in ("model", "ema"):
            for key in micro[collection]:
                torch.testing.assert_close(resumed[collection][key], micro[collection][key],
                                           rtol=0, atol=0)
        train.check_resume_compatible(full, "ggf", full["config"], 1.0)
        mutations = []
        changed = copy.deepcopy(full["config"])
        changed["ggf"]["sigma_min"] *= 2
        mutations.append((full, changed))
        changed = copy.deepcopy(full["config"])
        changed["ggf"]["sigma_max"] *= 2
        mutations.append((full, changed))
        changed_checkpoint = copy.deepcopy(full)
        changed_checkpoint.pop("ggf")
        mutations.append((changed_checkpoint, full["config"]))
        changed_checkpoint = copy.deepcopy(full)
        changed_checkpoint["ggf"]["iid_score_adapter"] = "none"
        mutations.append((changed_checkpoint, full["config"]))
        for checkpoint, changed_config in mutations:
            try:
                train.check_resume_compatible(checkpoint, "ggf", changed_config, 1.0)
            except ValueError:
                pass
            else:
                raise AssertionError("Incompatible GGF checkpoint was accepted")
    return {"synthetic_full_training": True, "synthetic_resume": True,
            "validation_rng_independent_of_microbatch": True,
            "checkpoint_objective_identity": True, "resume_refuses_ggf_changes": True}


def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    config = train.load_config(Path(__file__).resolve().parents[1] / "configs/dipeptides_ggf.yaml")
    report = {"device": "cpu", "threads": torch.get_num_threads(),
              **target_checks(config), **energy_checks(config),
              **microbatch_checks(config), **synthetic_training_checks(config)}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
