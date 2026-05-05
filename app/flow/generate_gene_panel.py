import os
import json
import argparse
from typing import Dict

import numpy as np
import scanpy as sc
import scipy.sparse as sps


def _compute_sum_and_sumsq(x):
    if sps.issparse(x):
        gene_sum = np.asarray(x.sum(axis=0)).ravel().astype(np.float64)
        gene_sumsq = np.asarray(x.multiply(x).sum(axis=0)).ravel().astype(np.float64)
    else:
        arr = np.asarray(x, dtype=np.float64)
        gene_sum = arr.sum(axis=0)
        gene_sumsq = np.square(arr).sum(axis=0)
    return gene_sum, gene_sumsq


def _merge_gene_stats(
    stats: Dict[str, np.ndarray],
    var_names,
    n_spots: int,
    gene_sum: np.ndarray,
    gene_sumsq: np.ndarray,
):
    for i, gene in enumerate(var_names):
        record = stats.get(gene)
        if record is None:
            # [count, sum, sumsq]
            stats[gene] = np.array(
                [float(n_spots), float(gene_sum[i]), float(gene_sumsq[i])], dtype=np.float64
            )
        else:
            record[0] += float(n_spots)
            record[1] += float(gene_sum[i])
            record[2] += float(gene_sumsq[i])


def build_var_gene_panel(source_dataroot: str, dataset: str, top_k: int):
    adata_dir = os.path.join(source_dataroot, dataset, "adata")
    if not os.path.isdir(adata_dir):
        raise FileNotFoundError(f"adata directory not found: {adata_dir}")

    h5ad_files = sorted([f for f in os.listdir(adata_dir) if f.endswith(".h5ad")])
    if len(h5ad_files) == 0:
        raise FileNotFoundError(f"No .h5ad files found under: {adata_dir}")

    stats: Dict[str, np.ndarray] = {}
    total_spots = 0
    for idx, fname in enumerate(h5ad_files, start=1):
        fpath = os.path.join(adata_dir, fname)
        print(f"[{idx}/{len(h5ad_files)}] reading {fpath}")
        adata = sc.read_h5ad(fpath)
        x = adata.X
        n_spots = int(x.shape[0])
        var_names = adata.var_names.tolist()
        gene_sum, gene_sumsq = _compute_sum_and_sumsq(x)
        _merge_gene_stats(stats, var_names, n_spots, gene_sum, gene_sumsq)
        total_spots += n_spots

    genes = []
    vars_ = []
    for gene, (count, sum_, sumsq_) in stats.items():
        if count <= 0:
            continue
        mean = sum_ / count
        var = max(0.0, (sumsq_ / count) - (mean * mean))
        genes.append(gene)
        vars_.append(var)

    order = np.argsort(np.asarray(vars_, dtype=np.float64))[::-1]
    k = min(int(top_k), len(order))
    top_genes = [genes[i] for i in order[:k]]
    top_vars = [float(vars_[i]) for i in order[:k]]

    return {
        "genes": top_genes,
        "meta": {
            "method": "top_k_by_global_variance",
            "top_k": int(k),
            "n_h5ad_files": int(len(h5ad_files)),
            "n_total_spots": int(total_spots),
            "n_unique_genes_seen": int(len(genes)),
            "top_gene_variances": top_vars[:10],
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate variance-based gene panel JSON (e.g., var_200genes.json)."
    )
    parser.add_argument("--source_dataroot", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--top-k", dest="top_k", type=int, default=200)
    parser.add_argument(
        "--output-name",
        type=str,
        default="var_200genes.json",
        help="Output filename saved under <source_dataroot>/<dataset>/",
    )
    args = parser.parse_args()

    result = build_var_gene_panel(
        source_dataroot=args.source_dataroot,
        dataset=args.dataset,
        top_k=args.top_k,
    )

    out_path = os.path.join(args.source_dataroot, args.dataset, args.output_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Saved gene panel to: {out_path}")
    print(f"Selected genes: {len(result['genes'])}")


if __name__ == "__main__":
    main()
