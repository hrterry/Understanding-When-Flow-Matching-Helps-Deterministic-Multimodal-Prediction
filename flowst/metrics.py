"""PCC-k, Spearman, MSE, MAE and local Moran (LISA) map similarity."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import rankdata


def column_corr(a, b):
    a, b = a - a.mean(0, keepdims=True), b - b.mean(0, keepdims=True)
    den = np.sqrt((a * a).sum(0) * (b * b).sum(0))
    out = np.full(a.shape[1], np.nan)
    ok = den > 1e-12
    out[ok] = (a * b).sum(0)[ok] / den[ok]
    return out


def local_moran(values, coords, k=6):
    nn = cKDTree(coords).query(coords, k=min(k, len(coords) - 1) + 1)[1][:, 1:]
    centered = values - values.mean(0, keepdims=True)
    return centered * centered[nn].mean(1)


def evaluate(pred, target, coords, slides):
    """Gene-wise metrics on pooled test spots; LISA is computed per slide and averaged per gene."""
    pcc = column_corr(pred, target)
    valid = np.sort(pcc[np.isfinite(pcc)])[::-1]
    out = {"pcc": float(valid.mean())}
    for k in (200, 100, 50):
        out[f"pcc{k}"] = float(valid[:k].mean())  # mean over the k best-predicted genes
    out["mse"] = float(((pred - target) ** 2).mean())
    out["mae"] = float(np.abs(pred - target).mean())
    out["spearman"] = float(np.nanmean(column_corr(rankdata(pred, axis=0), rankdata(target, axis=0))))
    lisa = []
    for s in np.unique(slides):
        m = slides == s
        lisa.append(column_corr(rankdata(local_moran(pred[m], coords[m]), axis=0),
                                rankdata(local_moran(target[m], coords[m]), axis=0)))
    out["lisa"] = float(np.nanmean(np.nanmean(np.vstack(lisa), axis=0)))
    return out
