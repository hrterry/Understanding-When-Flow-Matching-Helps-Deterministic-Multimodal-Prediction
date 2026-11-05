"""HEST-style slides: patch embeddings (h5) + log1p expression (h5ad) + predefined splits."""
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy.spatial import KDTree


class Slide:
    def __init__(self, name, features, expression, coords):
        self.name = name
        self.features = torch.from_numpy(features).float()
        self.expression = torch.from_numpy(expression).float()
        self.coords = torch.from_numpy(coords - coords.mean(0)).float()

    def __len__(self):
        return len(self.features)


def load_slide(root, embed_root, dataset, sample_id, genes, encoder):
    with h5py.File(Path(embed_root) / dataset / encoder / "fp32" / f"{sample_id}.h5", "r") as f:
        embeddings, coords = f["embeddings"][:], f["coords"][:].astype(np.float32)
        barcodes = f["barcodes"][:].flatten().astype(str).tolist()
    adata = sc.read_h5ad(Path(root) / dataset / "adata" / f"{sample_id}.h5ad")[barcodes]
    adata = adata[:, [g for g in genes if g in adata.var_names]].copy()
    sc.pp.log1p(adata)
    expression = adata.to_df().reindex(columns=genes, fill_value=0.0).values.astype(np.float32)
    return Slide(sample_id, embeddings, expression, coords)


def load_split(cfg, split):
    root, ds = Path(cfg["source_root"]), cfg["dataset"]
    genes = json.load(open(root / ds / cfg["gene_list"]))["genes"]
    def slides(name):
        ids = pd.read_csv(root / ds / "splits" / f"{name}_{split}.csv")["sample_id"]
        return [load_slide(root, cfg["embed_root"], ds, s, genes, cfg["encoder"]) for s in ids]
    return slides("train"), slides("test"), genes


def n_splits(cfg):
    return len(list((Path(cfg["source_root"]) / cfg["dataset"] / "splits").glob("train_*.csv")))


def gene_weights(slides, eps=1e-6):
    """Inverse detection-rate weights for the gene-space losses, normalized to mean 1."""
    y = torch.cat([s.expression for s in slides])
    w = 1.0 / ((y > 0).float().mean(0).double() + eps)
    return (w / w.mean()).float()


def shuffled_batches(slides, size):
    """One epoch over all training spots, shuffled across slides into groups of `size` spots."""
    F = torch.cat([s.features for s in slides])
    Y = torch.cat([s.expression for s in slides])
    for idx in torch.randperm(len(F)).split(size):
        yield F[idx][None], Y[idx][None]


def local_batches(slides, size, patches_per_slide=50, batch_size=2):
    """Groups of up to `size` spatially nearest spots around random centres."""
    trees = [KDTree(s.coords.numpy()) for s in slides]
    picks = [i for i in range(len(slides)) for _ in range(patches_per_slide)]
    np.random.shuffle(picks)
    for a in range(0, len(picks), batch_size):
        feats, ys = [], []
        for i in picks[a:a + batch_size]:
            s = slides[i]
            k = max(2, int(len(s) * np.random.rand()))
            k = min(k, size, len(s))
            idx = np.arange(len(s)) if k == len(s) else trees[i].query(s.coords[np.random.randint(len(s))].numpy(), k=k)[1]
            feats.append(s.features[idx]); ys.append(s.expression[idx])
        M = max(len(f) for f in feats)
        yield (torch.stack([torch.nn.functional.pad(f, (0, 0, 0, M - len(f))) for f in feats]),
               torch.stack([torch.nn.functional.pad(y, (0, 0, 0, M - len(y))) for y in ys]))
