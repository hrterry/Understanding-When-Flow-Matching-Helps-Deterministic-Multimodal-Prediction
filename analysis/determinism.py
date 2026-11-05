"""Repeated sampling: per-gene sampling variation vs prediction error and spatial signal.

    python analysis/determinism.py --run results/IDC_all/genespace_local_seed1/split0 --split 0
"""
import argparse

import numpy as np
import torch
from common import load_run
from flowst.flow import sample

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--run", required=True)
ap.add_argument("--split", type=int, default=0)
ap.add_argument("--seeds", type=int, default=8)
ap.add_argument("--device", default="cuda:0")
a = ap.parse_args()
cfg, model, vae, _, slides, _ = load_run(a.run, a.split, a.device)
s = slides[0]
img = s.features[None].to(a.device)
Y = s.expression.numpy()
P = []
for k in range(a.seeds):
    torch.manual_seed(1000 + k)
    out = sample(cfg, model, img, cfg["ode_steps"])
    P.append((vae.decode(out) if vae is not None else out)[0].detach().cpu().numpy())
P = np.stack(P)
ok = Y.std(0) > 0
seed_sd = P.std(0).mean(0)[ok]
print("median gap to prediction error :", round(float(np.median(np.abs(P.mean(0) - Y).mean(0)[ok] / seed_sd)), 1))
print("median gap to spatial signal   :", round(float(np.median(Y.std(0)[ok] / seed_sd)), 1))
print("max |seed - seed|              :", round(float(np.abs(P - P[0]).max()), 5))
