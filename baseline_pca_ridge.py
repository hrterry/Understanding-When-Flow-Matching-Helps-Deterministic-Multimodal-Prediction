"""Original Regression baseline: PCA (256) + ridge on patch features.

    python baseline_pca_ridge.py --config configs/latent.yaml --dataset COAD
"""
import argparse

import numpy as np
import yaml
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from flowst.data import load_split, n_splits
from flowst.metrics import evaluate


def run_split(cfg, split, seed):
    train, test, _ = load_split(cfg, split)
    x = np.concatenate([s.features.numpy() for s in train])
    y = np.concatenate([s.expression.numpy() for s in train])
    xt = np.concatenate([s.features.numpy() for s in test])
    yt = np.concatenate([s.expression.numpy() for s in test])
    scaler = StandardScaler().fit(x)
    pca = PCA(n_components=min(256, x.shape[1], x.shape[0]), random_state=seed).fit(scaler.transform(x))
    z, zt = pca.transform(scaler.transform(x)), pca.transform(scaler.transform(xt))
    # HEST-Bench ridge: no intercept. PCC is unaffected, but MSE/MAE are not comparable to other methods.
    reg = Ridge(alpha=100.0 / (z.shape[1] * y.shape[1]), solver="lsqr", fit_intercept=False,
                random_state=seed, max_iter=1000).fit(z, y)
    coords = np.concatenate([s.coords.numpy() for s in test])
    ids = np.concatenate([np.full(len(s), i) for i, s in enumerate(test)])
    return evaluate(reg.predict(zt), yt, coords, ids)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    cfg = dict(yaml.safe_load(open(a.config)), dataset=a.dataset)
    res = [run_split(cfg, s, a.seed) for s in range(n_splits(cfg))]
    print(a.dataset, {k: round(float(np.mean([r[k] for r in res])), 4) for k in res[0]})
