import os
import sys
import csv
import json
import argparse
from time import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


# translated Python translated
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data.normalize_utils import get_normalize_method
from hest_utils.file_utils import read_assets_from_h5
from hest_utils.st_dataset import load_adata
from utils.utils import get_current_time, merge_fold_results, set_random_seed


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _resolve_gene_list_path(source_dataroot: str, dataset: str, gene_list_path_or_name: str) -> str:
    path_str = str(gene_list_path_or_name)
    if os.path.isabs(path_str) and os.path.exists(path_str):
        return path_str
    if os.path.exists(path_str):
        return path_str
    return os.path.join(source_dataroot, dataset, path_str)


def _extract_barcodes(assets: Dict[str, np.ndarray]) -> List[str]:
    if "barcodes" in assets:
        barcodes = assets["barcodes"]
    elif "barcode" in assets:
        barcodes = assets["barcode"]
    else:
        raise KeyError("h5 translated barcodes/barcode translated，translated。")
    return barcodes.flatten().astype(str).tolist()


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return float("nan")
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(pearsonr(x, y)[0])


def _top_mean(values: List[float], k: int) -> float:
    if len(values) == 0:
        return float("nan")
    return float(np.mean(values[: min(k, len(values))]))


def calc_metrics(preds_all: np.ndarray, targets_all: np.ndarray, genes: List[str]) -> Dict:
    l2_errors = []
    r2_scores = []
    pearson_items = []
    pearson_values = []

    n_genes = targets_all.shape[1]
    for idx in range(n_genes):
        preds = preds_all[:, idx]
        target = targets_all[:, idx]
        l2_error = float(np.mean((preds - target) ** 2))

        denom = float(np.sum((target - np.mean(target)) ** 2))
        if denom < 1e-12:
            r2_score = float("nan")
        else:
            r2_score = float(1.0 - np.sum((target - preds) ** 2) / denom)

        pcc = _safe_pearson(target, preds)
        if not np.isnan(pcc):
            pearson_values.append(pcc)
        pearson_items.append(
            {
                "name": genes[idx] if idx < len(genes) else f"gene_{idx}",
                "pearson_corr": pcc,
            }
        )
        l2_errors.append(l2_error)
        r2_scores.append(r2_score)

    pearson_values_sorted = sorted(pearson_values, reverse=True)
    mse_overall = float(np.mean((preds_all - targets_all) ** 2))
    mae_overall = float(np.mean(np.abs(preds_all - targets_all)))

    metrics = {
        "l2_errors": l2_errors,
        "r2_scores": r2_scores,
        "pearson_corrs": pearson_items,
        "pearson_mean": float(np.mean(pearson_values)) if len(pearson_values) > 0 else float("nan"),
        "pearson_std": float(np.std(pearson_values)) if len(pearson_values) > 0 else float("nan"),
        "l2_error_q1": float(np.percentile(l2_errors, 25)),
        "l2_error_q2": float(np.median(l2_errors)),
        "l2_error_q3": float(np.percentile(l2_errors, 75)),
        "r2_score_q1": float(np.nanpercentile(r2_scores, 25)),
        "r2_score_q2": float(np.nanmedian(r2_scores)),
        "r2_score_q3": float(np.nanpercentile(r2_scores, 75)),
        "mse_overall": mse_overall,
        "mae_overall": mae_overall,
        "pcc200": _top_mean(pearson_values_sorted, 200),
        "pcc100": _top_mean(pearson_values_sorted, 100),
        "pcc50": _top_mean(pearson_values_sorted, 50),
    }
    return metrics


def load_split_data(
    sample_ids: List[str],
    args,
    genes: List[str],
    normalize_method,
) -> Tuple[np.ndarray, np.ndarray]:
    all_embeddings = []
    all_exprs = []
    missing_samples = []

    for sample_id in sample_ids:
        embed_path = os.path.join(
            args.embed_dataroot,
            args.dataset,
            args.feature_encoder,
            "fp32",
            f"{sample_id}.h5",
        )
        expr_path = os.path.join(args.source_dataroot, args.dataset, "adata", f"{sample_id}.h5ad")

        if not os.path.exists(embed_path) or not os.path.exists(expr_path):
            missing_samples.append(sample_id)
            continue

        assets, _ = read_assets_from_h5(embed_path)
        barcodes = _extract_barcodes(assets)
        embeddings = assets["embeddings"]

        adata_df = load_adata(
            expr_path=expr_path,
            genes=genes,
            barcodes=barcodes,
            normalize_method=normalize_method,
        )
        labels = adata_df.values.astype(np.float32)
        all_embeddings.append(embeddings.astype(np.float32))
        all_exprs.append(labels)

    if len(missing_samples) > 0:
        print(f"[translated] translated embedding translated h5ad，translated: {missing_samples}")
    if len(all_embeddings) == 0:
        raise RuntimeError("translated split translated，translated/translated baseline。")

    x = np.concatenate(all_embeddings, axis=0)
    y = np.concatenate(all_exprs, axis=0)
    return x, y


def fit_predict_pca_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    seed: int,
    pca_dim: int,
    alpha: float,
    max_iter: int,
) -> np.ndarray:
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    if pca_dim > 0:
        n_components = min(int(pca_dim), x_train_scaled.shape[1], x_train_scaled.shape[0])
        pca = PCA(n_components=n_components, random_state=seed)
        x_train_feat = pca.fit_transform(x_train_scaled)
        x_test_feat = pca.transform(x_test_scaled)
    else:
        x_train_feat = x_train_scaled
        x_test_feat = x_test_scaled

    if alpha <= 0:
        alpha = 100.0 / float(x_train_feat.shape[1] * y_train.shape[1])
    print(f"translated Ridge alpha={alpha:.8f} | feature_dim={x_train_feat.shape[1]}")

    reg = Ridge(
        solver="lsqr",
        alpha=float(alpha),
        random_state=seed,
        fit_intercept=False,
        max_iter=max_iter,
    )
    reg.fit(x_train_feat, y_train)
    preds = reg.predict(x_test_feat)
    return preds


def run_one_split(args, split_id: int, genes: List[str], normalize_method, save_dir: str) -> Dict:
    split_dir = os.path.join(args.source_dataroot, args.dataset, "splits")
    train_df = pd.read_csv(os.path.join(split_dir, f"train_{split_id}.csv"))
    test_df = pd.read_csv(os.path.join(split_dir, f"test_{split_id}.csv"))

    train_ids = train_df["sample_id"].tolist()
    test_ids = test_df["sample_id"].tolist()

    x_train, y_train = load_split_data(train_ids, args, genes, normalize_method)
    x_test, y_test = load_split_data(test_ids, args, genes, normalize_method)

    start_time = time()
    preds = fit_predict_pca_ridge(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        seed=args.seed,
        pca_dim=args.pca_dim,
        alpha=args.alpha,
        max_iter=args.max_iter,
    )
    elapsed = time() - start_time

    metrics = calc_metrics(preds, y_test, genes)
    summary = {
        "split_id": split_id,
        "n_train_spots": int(x_train.shape[0]),
        "n_test_spots": int(x_test.shape[0]),
        "n_genes": int(y_train.shape[1]),
        "pca_dim": int(args.pca_dim),
        "alpha": float(args.alpha) if args.alpha > 0 else "auto(100/(d*g))",
        "fit_predict_seconds": float(elapsed),
        "pearson_mean": metrics["pearson_mean"],
        "pcc200": metrics["pcc200"],
        "pcc100": metrics["pcc100"],
        "pcc50": metrics["pcc50"],
        "mse_overall": metrics["mse_overall"],
        "mae_overall": metrics["mae_overall"],
    }

    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(metrics, f, sort_keys=True, indent=4, default=json_default)
    with open(os.path.join(save_dir, "summary.json"), "w") as f:
        json.dump(summary, f, sort_keys=True, indent=4, default=json_default)
    np.savez(
        os.path.join(save_dir, "predictions.npz"),
        predictions=preds,
        targets=y_test,
        gene_list=np.asarray(genes, dtype=object),
    )
    return metrics


def run_dataset(args):
    gene_list_path = _resolve_gene_list_path(args.source_dataroot, args.dataset, args.gene_list)
    with open(gene_list_path, "r") as f:
        genes = json.load(f)["genes"]

    split_dir = os.path.join(args.source_dataroot, args.dataset, "splits")
    split_files = os.listdir(split_dir)
    n_splits = len(split_files) // 2
    if args.split_only >= 0:
        split_ids = [args.split_only]
    else:
        split_ids = list(range(n_splits))

    normalize_method = get_normalize_method(args.normalize_method)
    split_metrics = []
    test_rows = []

    for split_id in split_ids:
        split_save_dir = os.path.join(args.save_dir, args.dataset, f"split{split_id}")
        print(f"translated {args.dataset} split{split_id} baseline ...")
        metrics = run_one_split(args, split_id, genes, normalize_method, split_save_dir)
        split_metrics.append(metrics)
        test_rows.append(
            {
                "split_id": split_id,
                "pearson_mean": metrics["pearson_mean"],
                "pcc200": metrics["pcc200"],
                "pcc100": metrics["pcc100"],
                "pcc50": metrics["pcc50"],
                "mse": metrics["mse_overall"],
                "mae": metrics["mae_overall"],
            }
        )

    kfold_results = merge_fold_results(split_metrics)
    p_corrs = sorted(kfold_results["pearson_corrs"], key=lambda x: x["mean"], reverse=True)
    kfold_results["pearson_corrs"] = p_corrs

    dataset_save_dir = os.path.join(args.save_dir, args.dataset)
    os.makedirs(dataset_save_dir, exist_ok=True)
    with open(os.path.join(dataset_save_dir, "results_kfold.json"), "w") as f:
        json.dump(kfold_results, f, sort_keys=True, indent=4, default=json_default)

    csv_path = os.path.join(dataset_save_dir, "test_results_all_splits.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split_id", "pearson_mean", "pcc200", "pcc100", "pcc50", "mse", "mae"])
        for row in test_rows:
            writer.writerow(
                [
                    row["split_id"],
                    f"{row['pearson_mean']:.6f}",
                    f"{row['pcc200']:.6f}",
                    f"{row['pcc100']:.6f}",
                    f"{row['pcc50']:.6f}",
                    f"{row['mse']:.6f}",
                    f"{row['mae']:.6f}",
                ]
            )
        if len(test_rows) > 0:
            writer.writerow(
                [
                    "mean",
                    f"{np.nanmean([r['pearson_mean'] for r in test_rows]):.6f}",
                    f"{np.nanmean([r['pcc200'] for r in test_rows]):.6f}",
                    f"{np.nanmean([r['pcc100'] for r in test_rows]):.6f}",
                    f"{np.nanmean([r['pcc50'] for r in test_rows]):.6f}",
                    f"{np.nanmean([r['mse'] for r in test_rows]):.6f}",
                    f"{np.nanmean([r['mae'] for r in test_rows]):.6f}",
                ]
            )

    print(f"{args.dataset} baseline translated，translated: {dataset_save_dir}")


def main():
    parser = argparse.ArgumentParser(description="HEST translated PCA256 + Ridge baseline")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="translated cudnn deterministic（translated GPU op）",
    )
    parser.add_argument(
        "--source_dataroot",
        type=str,
        default="dataset",
    )
    parser.add_argument(
        "--embed_dataroot",
        type=str,
        default="dataset/embed_dataroot",
    )
    parser.add_argument("--save_dir", type=str, default="results_dir/baseline_pca_ridge")
    parser.add_argument("--datasets", nargs="+", default=["COAD", "READ","LUNG", "PAAD","LYMPH_IDC", "SKCM","PRAD","CCRCC"])
    parser.add_argument("--split-only", dest="split_only", type=int, default=-1)
    parser.add_argument("--feature_encoder", type=str, default="uni_v1_official")
    parser.add_argument("--gene_list", type=str, default="var_200genes.json")
    parser.add_argument("--normalize_method", type=str, default="log1p")
    parser.add_argument("--pca_dim", type=int, default=256)
    parser.add_argument("--alpha", type=float, default=-1.0, help="<=0 translated alpha=100/(d*g)")
    parser.add_argument("--max_iter", type=int, default=1000)
    args = parser.parse_args()

    set_random_seed(args.seed, deterministic=args.deterministic)

    exp_code = f"pca{args.pca_dim}_ridge_{get_current_time()}"
    args.save_dir = os.path.join(args.save_dir, exp_code)
    os.makedirs(args.save_dir, exist_ok=True)

    with open(os.path.join(args.save_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, sort_keys=True, indent=4, default=json_default)

    for dataset in args.datasets:
        args.dataset = dataset
        run_dataset(args)


if __name__ == "__main__":
    main()

