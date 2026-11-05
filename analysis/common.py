"""Load a trained run and its test slides for the analyses."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flowst.data import load_split  # noqa: E402
from train import build  # noqa: E402


def load_run(split_dir, split, device):
    ck = torch.load(Path(split_dir) / "best.pt", map_location=device, weights_only=False)
    cfg = ck["config"]
    train, test, genes = load_split(cfg, split)
    model, vae = build(cfg, len(genes), test[0].features.shape[1], device)
    model.load_state_dict(ck["model"])
    model.eval()
    if vae is not None:
        vae.load_state_dict(ck["vae"])
        vae.eval()
    return cfg, model, vae, train, test, genes
