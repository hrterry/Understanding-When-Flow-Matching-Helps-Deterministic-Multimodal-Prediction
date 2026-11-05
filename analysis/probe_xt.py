"""What does the model read from x_t? One forward pass with probe states built from test data.

    true      x_t = t x1 + (1-t) e        measured test expression
    shuffled  same values, shuffled across spots of the slide
    mean      slide-mean target only
    deviation per-spot deviations only
    foreign   expression of another (training) slide
    noise     x_t = e

    python analysis/probe_xt.py --run results/IDC_all/genespace_local_seed1/split0 --split 0
"""
import argparse

import numpy as np
import torch
from common import load_run
from flowst.metrics import column_corr

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--run", required=True)
ap.add_argument("--split", type=int, default=0)
ap.add_argument("--device", default="cuda:0")
a = ap.parse_args()
cfg, model, vae, train, slides, _ = load_run(a.run, a.split, a.device)
assert not cfg["latent"], "probes are defined for gene-space models"
g = torch.Generator(device=a.device).manual_seed(0)


@torch.no_grad()
def run(mode, t, foreign=None):
    preds, truth = [], []
    for s in slides:
        img, y = s.features[None].to(a.device), s.expression[None].to(a.device)
        x1 = y
        if mode == "shuffled":
            x1 = y[:, torch.randperm(y.shape[1], device=a.device, generator=g)]
        elif mode == "mean":
            x1 = y.mean(1, keepdim=True).expand_as(y)
        elif mode == "deviation":
            x1 = y - y.mean(1, keepdim=True)
        elif mode == "foreign":
            idx = torch.randint(len(foreign), (y.shape[1],))
            x1 = foreign.expression[idx][None].to(a.device)
        noise = torch.randn(y.shape, device=a.device, generator=g)
        xt = noise if mode == "noise" else t * x1 + (1 - t) * noise
        preds.append(model(img, xt, torch.full((1,), t, device=a.device))[0].cpu().numpy())
        truth.append(s.expression.numpy())
    return np.nanmean(column_corr(np.concatenate(preds), np.concatenate(truth)))


for t in (0.5, 0.8, 0.95):
    row = {m: run(m, t) for m in ("true", "shuffled", "mean", "deviation", "noise")}
    row["foreign"] = [round(float(run("foreign", t, s)), 4) for s in train[:4]]
    print(f"t={t}", {k: (round(float(v), 4) if not isinstance(v, list) else v) for k, v in row.items()})
