"""The paired-singleton GGF objective and its experiment identity."""

import math

import torch
import torch.nn.functional as F


SIGMA_MIN = 0.0010004971259352385
SIGMA_MAX = 20.0


def _validate_bounds(sigma_min, sigma_max):
    if not (math.isfinite(sigma_min) and math.isfinite(sigma_max)
            and 0 < sigma_min <= sigma_max):
        raise ValueError("GGF noise bounds must be finite and satisfy 0 < sigma_min <= sigma_max")


def ggf_identity(config):
    """Return the JSON-safe identity shared by training and evaluation artifacts."""
    settings = config["ggf"]
    sigma_min, sigma_max = float(settings["sigma_min"]), float(settings["sigma_max"])
    _validate_bounds(sigma_min, sigma_max)
    if sigma_min == sigma_max:
        raise ValueError("GGF experiment noise bounds must satisfy sigma_min < sigma_max")
    return {"objective": "ggf_singleton", "noise_conditioned": False,
            "sigma_min": sigma_min, "sigma_max": sigma_max,
            "bandwidth_h": sigma_min, "fixed_time": 0.0,
            "iid_sampler": "ve_euler_maruyama", "iid_score_adapter": "h2_over_sigma2"}


def sample_noise_std(data, sigma=SIGMA_MIN, sigma_max=SIGMA_MAX):
    """Draw one log-uniform VE scale per sample before any batch partitioning."""
    _validate_bounds(sigma, sigma_max)
    shape = (data.shape[0],) + (1,) * (data.ndim - 1)
    return torch.empty(data.shape[0], device=data.device, dtype=data.dtype).uniform_(
        math.log(sigma), math.log(sigma_max)).exp().view(shape)


def ggf_score_matching(model, data, sigma=SIGMA_MIN, *, sigma_max=SIGMA_MAX,
                       noise=None, noise_std=None):
    """Match each query to its own clean source using the fixed bandwidth ``sigma``.

    ``model`` is a callable score closure, so peptide features and the energy
    gradient graph remain the caller's responsibility. Optional fixed draws
    support batch partitioning, deterministic resume, and numerical checks.
    No other source frame contributes to a query's target.
    """
    _validate_bounds(sigma, sigma_max)
    if data.ndim < 2 or not len(data):
        raise ValueError("GGF data must contain a nonempty batch of coordinate arrays")
    if noise_std is None:
        noise_std = sample_noise_std(data, sigma, sigma_max)
    else:
        shape = (data.shape[0],) + (1,) * (data.ndim - 1)
        if noise_std.numel() != data.shape[0]:
            raise ValueError("GGF noise_std must contain one scale per sample")
        noise_std = noise_std.reshape(shape)
    noise = torch.randn_like(data) if noise is None else noise
    if noise.shape != data.shape:
        raise ValueError("GGF noise must have the same shape as data")
    displacement = noise_std.detach() * noise.detach()
    query = data + displacement
    # Avoid cancellation in (data - query) at the small fixed bandwidth.
    target_score = (-displacement / sigma**2).detach()
    return F.mse_loss(model(query), target_score)
