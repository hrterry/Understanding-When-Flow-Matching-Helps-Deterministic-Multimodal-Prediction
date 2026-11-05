"""Linear-path flow matching with x-prediction, and the v-prediction comparison (Figure 5)."""
import math

import torch


def zinb_prior(shape, device, total_count=1.0, logits=0.1, zi_logits=0.0):
    """log1p of a zero-inflated negative binomial; used to start gene-space sampling."""
    mu = total_count * math.exp(logits)
    rate = torch.distributions.Gamma(total_count, total_count / mu).sample(shape).to(device)
    counts = torch.poisson(rate)
    counts[torch.rand(shape, device=device) < torch.sigmoid(torch.tensor(zi_logits))] = 0
    return torch.log1p(counts)


def alpha_cos(t):
    return 0.5 - 0.5 * torch.cos(math.pi * t)


def alpha_cos_derivative(t):
    return 0.5 * math.pi * torch.sin(math.pi * t)


def masked_mse(err, valid, weight=None):
    """Mean of squared errors over real spots; `weight` optionally weights genes."""
    per_gene = (err * valid[..., None]).sum(dim=(0, 1)) / valid.sum().clamp_min(1)
    return per_gene.mean() if weight is None else (per_gene * weight).sum() / weight.sum()


def flow_step(cfg, model, img, x1, weight=None, source=None):
    """One training loss. x1 [B,M,D] is the target, `source` optionally replaces it inside x_t."""
    valid = (img.sum(-1) != 0).to(x1.dtype)  # zero-padded spots do not count
    B = x1.shape[0]
    t = torch.rand(B, device=x1.device, dtype=x1.dtype)
    if cfg["prediction"] == "v":  # vanilla flow on a cosine path, read from the ST tokens
        x0 = torch.randn_like(x1) if cfg["latent"] else zinb_prior(x1.shape, x1.device)
        a = alpha_cos(t)[:, None, None]
        xt = (1 - a) * x0 + a * x1
        pred = model(img, xt, t, from_st_tokens=True)
        target = (x1 - x0) * alpha_cos_derivative(t)[:, None, None]
        return masked_mse((pred - target) ** 2, valid)

    x0 = torch.randn_like(x1)
    src = x1 if source is None else source
    xt = t[:, None, None] * src + (1 - t)[:, None, None] * x0
    tokens = src if cfg["xt_mode"] == "clean" else xt
    t_in = torch.zeros_like(t) if cfg["time"] == "none" else t
    pred = model(img, tokens, t_in)
    denom = (1 - t).clamp_min(1e-4)[:, None, None]
    # velocity loss under x-prediction: ||(x_pred - x_t) - (x_1 - x_t)||^2 / (1 - t)^2
    return masked_mse(((pred - xt) / denom - (x1 - xt) / denom) ** 2, valid, weight)


@torch.no_grad()
def sample(cfg, model, img, steps=50, noise=None, generator=None):
    """Euler integration from the prior to t = 1. Returns the flow-space endpoint."""
    B, M, _ = img.shape
    shape = (B, M, model.dim)
    if noise is not None:
        x = noise
    elif cfg["latent"]:
        x = torch.randn(shape, device=img.device, generator=generator)
    else:
        x = zinb_prior(shape, img.device)
    for s in range(steps):
        t = torch.full((B,), s / steps, device=img.device, dtype=x.dtype)
        t_next = (s + 1) / steps
        t_in = torch.zeros_like(t) if cfg["time"] == "none" else t
        if cfg["prediction"] == "v":
            x = x + model(img, x, t_in, from_st_tokens=True) / steps
        else:
            denom = (1 - t).clamp_min(1e-4)[:, None, None]
            x = x + (t_next - s / steps) * (model(img, x, t_in) - x) / denom
    return x
