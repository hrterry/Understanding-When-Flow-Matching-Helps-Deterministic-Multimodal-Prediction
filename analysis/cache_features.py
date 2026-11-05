"""Cache one split as an npz for analysis/simulation.py.

    python analysis/cache_features.py --config configs/latent.yaml --dataset LUNG --split 0 --out cache/LUNG_0.npz
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flowst.data import load_split  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--config", required=True)
ap.add_argument("--dataset", required=True)
ap.add_argument("--split", type=int, default=0)
ap.add_argument("--out", required=True)
a = ap.parse_args()
cfg = dict(yaml.safe_load(open(a.config)), dataset=a.dataset)
train, test, genes = load_split(cfg, a.split)
out = {}
for name, slides in (("train", train), ("test", test)):
    out[f"H_{name}"] = np.concatenate([s.features.numpy() for s in slides])
    out[f"Y_{name}"] = np.concatenate([s.expression.numpy() for s in slides])
    out[f"C_{name}"] = np.concatenate([s.coords.numpy() for s in slides])
    out[f"S_{name}"] = np.concatenate([np.full(len(s), i) for i, s in enumerate(slides)])
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
np.savez(a.out, genes=np.array(genes), **out)
