import json
import os
from typing import Optional, Tuple

import numpy as np
import torch

from hest_utils.file_utils import read_assets_from_h5
from hest_utils.st_dataset import load_adata


def _resolve_gene_list_path(source_dataroot: str, dataset: str, gene_list_path_or_name: str) -> str:
    """
    translated：
      1) translated
      2) translated
      3) translated：translated source_dataroot/dataset translated
    """
    p = str(gene_list_path_or_name)
    if os.path.isabs(p) and os.path.exists(p):
        return p
    if os.path.exists(p):
        return p
    return os.path.join(source_dataroot, dataset, p)


def compute_train_gene_rho(
    train_sample_ids,
    embed_dataroot: str,
    source_dataroot: str,
    dataset: str,
    feature_encoder: str,
    gene_list_rel_path: str,
    normalize_method,
    nonzero_eps: float = 0.0,
) -> Tuple[np.ndarray, list]:
    """
    rho_g = (# spots with expression > nonzero_eps for gene g) / (total spots in train set)
    translated h5 translated barcodes + load_adata translated。
    """
    gene_list_path = _resolve_gene_list_path(source_dataroot, dataset, gene_list_rel_path)
    with open(gene_list_path, "r") as f:
        genes = json.load(f)["genes"]
    n_genes = len(genes)
    nonzero = np.zeros(n_genes, dtype=np.int64)
    total_spots = 0
    for sid in train_sample_ids:
        h5_path = os.path.join(
            embed_dataroot, dataset, feature_encoder, f"fp32/{sid}.h5"
        )
        h5ad_path = os.path.join(source_dataroot, dataset, f"adata/{sid}.h5ad")
        data_dict, _ = read_assets_from_h5(h5_path)
        barcodes = data_dict["barcodes"].flatten().astype(str).tolist()
        df = load_adata(
            h5ad_path,
            genes=genes,
            barcodes=barcodes,
            normalize_method=normalize_method,
        )
        arr = df.values.astype(np.float64)
        n = arr.shape[0]
        total_spots += n
        nonzero += (arr > nonzero_eps).sum(axis=0)
    rho = nonzero.astype(np.float64) / max(int(total_spots), 1)
    return rho, genes


def rho_to_base_weights(
    rho: np.ndarray,
    mode: str,
    eps: float,
    clip_max: Optional[float] = None,
) -> np.ndarray:
    if mode in (None, "none"):
        return np.ones_like(rho, dtype=np.float32)
    rho = np.maximum(rho.astype(np.float64), 0.0)
    if mode == "inv_rho":
        w = 1.0 / (rho + eps)
    elif mode == "inv_sqrt_rho":
        w = 1.0 / np.sqrt(rho + eps)
    else:
        raise ValueError(
            f"Unknown gene_reweight_mode: {mode!r}. Use none | inv_rho | inv_sqrt_rho."
        )
    if clip_max is not None and float(clip_max) > 0:
        w = np.minimum(w, float(clip_max))
    w = w / np.mean(w)
    return w.astype(np.float32)


def sparse_gene_mask_from_quantile(rho: np.ndarray, sparse_quantile: float) -> np.ndarray:
    """rho translated sparse（translated batch translated）。"""
    q = float(sparse_quantile)
    q = min(max(q, 0.0), 1.0)
    thr = np.quantile(rho, q)
    return rho <= thr


def load_or_compute_gene_rho(
    train_sample_ids,
    embed_dataroot: str,
    source_dataroot: str,
    dataset: str,
    feature_encoder: str,
    gene_list_rel_path: str,
    normalize_method,
    nonzero_eps: float,
    cache_path: Optional[str],
) -> Tuple[np.ndarray, list]:
    gene_list_path = _resolve_gene_list_path(source_dataroot, dataset, gene_list_rel_path)
    with open(gene_list_path, "r") as f:
        genes_json = json.load(f)["genes"]
    if cache_path and os.path.isfile(cache_path):
        z = np.load(cache_path, allow_pickle=True)
        genes_ck = z["genes"].tolist() if hasattr(z["genes"], "tolist") else list(z["genes"])
        if genes_ck == genes_json:
            return np.asarray(z["rho"], dtype=np.float64), genes_json
    rho, genes = compute_train_gene_rho(
        train_sample_ids,
        embed_dataroot,
        source_dataroot,
        dataset,
        feature_encoder,
        gene_list_rel_path,
        normalize_method,
        nonzero_eps=nonzero_eps,
    )
    if genes != genes_json:
        raise RuntimeError("Gene list mismatch after rho computation.")
    if cache_path:
        d = os.path.dirname(os.path.abspath(cache_path))
        if d:
            os.makedirs(d, exist_ok=True)
        np.savez(cache_path, rho=rho, genes=np.array(genes_json, dtype=object))
    return rho, genes


def per_gene_mean_sq_error(
    pred: torch.Tensor,
    target: torch.Tensor,
    pad_mask: torch.Tensor,
) -> torch.Tensor:
    """
    pred/target: [B, M, G]
    pad_mask: [B, M]，True translated padding spot
    translated spot translated，translated [G]
    """
    valid = (~pad_mask).unsqueeze(-1).to(pred.dtype)  # [B, M, 1]
    diff2 = (pred - target) ** 2
    num = (diff2 * valid).sum(dim=(0, 1))
    den = valid.sum(dim=(0, 1)).clamp(min=1.0)
    return num / den


def batch_gene_effective_weights(
    base_w: torch.Tensor,
    is_sparse: torch.Tensor,
    resample_mode: str,
    dense_keep_prob: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """base_w: [G]；is_sparse: [G] bool。translated batch translated w_eff（translated dense translated）。"""
    g = int(base_w.shape[0])
    if is_sparse.shape[0] != g:
        raise ValueError("is_sparse length does not match base_w.")
    base_w = base_w.to(device=device, dtype=dtype)
    is_sparse = is_sparse.to(device=device)

    inc = torch.ones(g, device=device, dtype=dtype)
    if resample_mode == "rebalance_sparse_dense":
        dense = ~is_sparse
        if bool(dense.any()) and float(dense_keep_prob) < 1.0:
            r = torch.rand(g, device=device, dtype=dtype)
            inc[dense] = (r[dense] < float(dense_keep_prob)).to(dtype)
        inc[is_sparse] = 1.0

    return base_w * inc


def weighted_per_gene_scalar_loss(mse_per_g: torch.Tensor, w_eff: torch.Tensor) -> torch.Tensor:
    denom = w_eff.sum().clamp(min=1e-8)
    return (mse_per_g * w_eff).sum() / denom
