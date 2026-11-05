"""Inference-time routing interventions: mask attention quadrants at every integration step.

    python analysis/routing.py --run results/IDC_all/genespace_local_seed1/split0 --split 0
"""
import argparse

import torch
from common import load_run
from train import test as run_test

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--run", required=True)
ap.add_argument("--split", type=int, default=0)
ap.add_argument("--device", default="cuda:0")
a = ap.parse_args()
cfg, model, vae, _, slides, _ = load_run(a.run, a.split, a.device)
for route in ("none", "no_ig", "no_gi", "no_cross", "no_gg"):
    if cfg["backbone"] == "qwen" and route == "no_gg":
        continue
    model.set_route(route)
    m, _ = run_test(cfg, model, vae, slides, a.device)
    print(f"{route:9s} PCC={m['pcc']:.4f} MSE={m['mse']:.4f} MAE={m['mae']:.4f}")
