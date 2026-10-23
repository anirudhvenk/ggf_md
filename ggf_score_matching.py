import math
import torch
import torch.nn.functional as F

def ggf_score_matching(model, data, sigma=1e-3):
    """
    Global Generative Fields (score matching version)
    """
    batch = data.shape[0]
    shape = (batch,) + (1,) * (data.ndim - 1)

    # Broad distribution: add log-uniform variance-exploding noise
    SIGMA_MAX = 20.0
    noise_std = torch.empty(batch, device=data.device).uniform_(
        math.log(sigma), math.log(SIGMA_MAX)
    ).exp().view(shape)
    x = data + noise_std * torch.randn_like(data)

    # Construct the minibatch Gaussian KDE target score
    with torch.no_grad():
        x_flat, data_flat = x.flatten(1), data.flatten(1)
        weights = torch.softmax(
            -torch.cdist(x_flat, data_flat).square() / (2 * sigma**2), dim=1
        )
        target_score = ((weights @ data_flat - x_flat) / sigma**2).view_as(x)

    predictions = model(x)
    loss = F.mse_loss(predictions, target_score)

    return loss