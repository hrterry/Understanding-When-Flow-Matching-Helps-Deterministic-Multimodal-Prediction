import os
import sys
import json
import hashlib
import wandb
import argparse
import numpy as np
import pandas as pd
from time import time
from tqdm import tqdm
from operator import itemgetter

import torch
import torch.nn.functional as F

# translatedPythontranslated
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.utils import set_random_seed, get_current_time, merge_fold_results
from utils.gene_rho_weighting import (
    batch_gene_effective_weights,
    per_gene_mean_sq_error,
    weighted_per_gene_scalar_loss,
)
from data.dataset import HESTDatasetPath, MultiHESTDataset, padding_batcher, HESTDataset
from data.normalize_utils import get_normalize_method
from model.vpredictor import MMDiTDenoiser
from model.dit import DitFlowDenoiser
from model.latent_ae import GeneLatentAE, GeneLatentVAE
from flow.interpolant import Interpolant
from flow.noise import PriorSampler
from app.flow.test import test
from hest_utils.utils import save_pkl

def build_flow_denoiser(args):
    """translated backbone translated mmDiT（dit.py + mmdit/）translated vpredictor translated。"""
    b = args.backbone.lower()
    if b == "spatial_transformer":
        return DitFlowDenoiser(args)
    if b in ("vpredictor", "mm_dit"):
        return MMDiTDenoiser(args)
    raise ValueError(
        f"translated backbone: {args.backbone!r}，translated spatial_transformer translated vpredictor"
    )


def json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _resolve_gene_list_path(source_dataroot: str, dataset: str, gene_list_path_or_name: str) -> str:
    """translated/translated/translated。"""
    p = str(gene_list_path_or_name)
    if os.path.isabs(p) and os.path.exists(p):
        return p
    if os.path.exists(p):
        return p
    return os.path.join(source_dataroot, dataset, p)


def calculate_gene_expression_stats(preds_all: np.ndarray, targets_all: np.ndarray, gene_list: list):
    """
    translated
    
    Args:
        preds_all: translated (n_samples, n_genes)
        targets_all: translated (n_samples, n_genes)
        gene_list: translated
    
    Returns:
        dict: translated
    """
    gene_stats = {}
    
    # translated
    for i, gene_name in enumerate(gene_list):
        pred_values = preds_all[:, i]
        target_values = targets_all[:, i]
        
        gene_stats[gene_name] = {
            'predicted': {
                'min': float(np.min(pred_values)),
                'max': float(np.max(pred_values)),
                'mean': float(np.mean(pred_values)),
                'std': float(np.std(pred_values))
            },
            'ground_truth': {
                'min': float(np.min(target_values)),
                'max': float(np.max(target_values)),
                'mean': float(np.mean(target_values)),
                'std': float(np.std(target_values))
            }
        }
    
    # translated
    all_pred_values = preds_all.flatten()
    all_target_values = targets_all.flatten()
    
    summary_stats = {
        'overall_summary': {
            'predicted': {
                'min': float(np.min(all_pred_values)),
                'max': float(np.max(all_pred_values)),
                'mean': float(np.mean(all_pred_values)),
                'std': float(np.std(all_pred_values)),
                'median': float(np.median(all_pred_values))
            },
            'ground_truth': {
                'min': float(np.min(all_target_values)),
                'max': float(np.max(all_target_values)),
                'mean': float(np.mean(all_target_values)),
                'std': float(np.std(all_target_values)),
                'median': float(np.median(all_target_values))
            }
        },
        'dataset_info': {
            'n_samples': int(preds_all.shape[0]),
            'n_genes': int(preds_all.shape[1]),
            'gene_list': gene_list
        }
    }
    
    # translated
    gene_stats['summary'] = summary_stats
    
    return gene_stats


def _test_scalar_metrics_from_perf(test_perf_dict):
    """translated test() translated perf dict translated（translated）。"""
    test_pearson = test_perf_dict["all"]["pearson_mean"]
    test_pcc200 = float("nan")
    test_pcc100 = float("nan")
    test_pcc50 = float("nan")
    test_mse = float("nan")
    test_mae = float("nan")
    test_pearson_items = test_perf_dict["all"].get("pearson_corrs", [])
    test_pearson_values = [x.get("pearson_corr", None) for x in test_pearson_items]
    test_pearson_values = [x for x in test_pearson_values if x is not None and not np.isnan(x)]
    test_pearson_values.sort(reverse=True)

    def _top_mean(values, k):
        if len(values) == 0:
            return float("nan")
        return float(np.mean(values[: min(k, len(values))]))

    test_pcc200 = _top_mean(test_pearson_values, 200)
    test_pcc100 = _top_mean(test_pearson_values, 100)
    test_pcc50 = _top_mean(test_pearson_values, 50)
    test_mse = test_perf_dict["all"].get("mse_overall", float("nan"))
    test_mae = test_perf_dict["all"].get("mae_overall", float("nan"))
    test_int_err = test_perf_dict["all"].get("integration_error_mse", float("nan"))
    return {
        "pearson_mean": test_pearson,
        "pcc200": test_pcc200,
        "pcc100": test_pcc100,
        "pcc50": test_pcc50,
        "mse": test_mse,
        "mae": test_mae,
        "integration_error_mse": test_int_err,
    }


def _build_global_spot_pool(train_dataset):
    """
    translated MultiHESTDataset translated spot translated，translated。
    translated:
      features_pool: [N, D]
      coords_pool:   [N, 2]
      labels_pool:   [N, C]
    """
    feat_list, coord_list, label_list = [], [], []
    for sp_data in train_dataset.sp_datasets:
        feat_list.append(sp_data.features)
        coord_list.append(sp_data.coords)
        label_list.append(sp_data.labels)
    if len(feat_list) == 0:
        raise ValueError("translated，translated spot translated。")
    features_pool = torch.cat(feat_list, dim=0)
    coords_pool = torch.cat(coord_list, dim=0)
    labels_pool = torch.cat(label_list, dim=0)
    return features_pool, coords_pool, labels_pool


def _build_shuffled_spot_batch_indices(total_spots, batch_size, device):
    """
    translated spot translated shuffle（translated），translated batch_size translated。
    """
    n = int(total_spots)
    if n <= 0:
        raise ValueError("translated spot translated，translated shuffle translated。")
    k = max(1, int(batch_size))
    perm = torch.randperm(n, device=device)
    return perm.split(k)


def _build_deranged_batch_permutation(batch_size, device):
    """
    translated（derangement）：translated i -> perm[i] translated perm[i] != i。
    translated batch_size<=1，translated None（translated）。
    """
    b = int(batch_size)
    if b <= 1:
        return None
    base = torch.arange(b, device=device)
    perm = torch.randperm(b, device=device)
    max_retry = 8
    retry = 0
    while retry < max_retry and torch.any(perm == base):
        perm = torch.randperm(b, device=device)
        retry += 1
    if torch.any(perm == base):
        # translated：translated。
        shift = int(torch.randint(1, b, (1,), device=device).item())
        perm = (base + shift) % b
    return perm


def _resolve_model_time_input(args, t_steps: torch.Tensor) -> torch.Tensor:
    """
    translated「translated」；translated z_t/x_t translated target translated t。
    """
    mode = str(getattr(args, "time_conditioning_mode", "full")).lower().strip()
    if mode == "full":
        return t_steps
    if mode == "fixed":
        t_fixed = float(getattr(args, "time_conditioning_fixed_value", 0.5))
        t_fixed = min(1.0, max(0.0, t_fixed))
        return torch.full_like(t_steps, t_fixed)
    if mode == "none":
        # no-time ablation：translated（0），translated。
        return torch.zeros_like(t_steps)
    raise ValueError(f"Unsupported time_conditioning_mode: {mode}")


def _sample_random_zinb_tokens_like(
    ref: torch.Tensor,
    zinb_prior_sampler: PriorSampler,
) -> torch.Tensor:
    """
    translated x1/z1 translated ZINB auxiliary token（log1p translated）。
    """
    sampled = zinb_prior_sampler.sample(tuple(ref.shape)).to(ref.device)
    sampled = torch.log(sampled + 1.0)
    return sampled.to(dtype=ref.dtype)


def _build_union_genepanel_with_cache(args, sample_ids):
    """
    translated train+test translated（union panel）translated union translated。
    translated:
      panel_json_path: translated {"genes":[...]} translated json translated
      mapping_npz_path: translated sample_id translated union translated（translated/translated）
      n_union_genes: translated
    """
    sample_ids = sorted(set(sample_ids))
    if len(sample_ids) == 0:
        raise ValueError("sample_ids translated，translated union gene panel。")

    cache_dir = (getattr(args, "genepanel_cache_dir", "") or "").strip()
    if not cache_dir:
        cache_dir = os.path.join(args.save_dir, "_genepanel_cache")
    os.makedirs(cache_dir, exist_ok=True)

    sig_src = "|".join(sample_ids) + f"|{args.dataset}|{args.source_dataroot}"
    sig = hashlib.md5(sig_src.encode("utf-8")).hexdigest()[:12]
    panel_json_path = os.path.join(cache_dir, f"union_genepanel_{args.dataset}_{sig}.json")
    mapping_npz_path = os.path.join(cache_dir, f"union_genepanel_map_{args.dataset}_{sig}.npz")

    if os.path.exists(panel_json_path) and os.path.exists(mapping_npz_path):
        with open(panel_json_path, "r") as f:
            panel_obj = json.load(f)
        genes = panel_obj.get("genes", [])
        if len(genes) == 0:
            raise ValueError(f"translated union panel translated: {panel_json_path}")
        return panel_json_path, mapping_npz_path, int(len(genes))

    try:
        import scanpy as sc
    except Exception as e:
        raise RuntimeError("translated union gene panel translated scanpy，translated。") from e

    union_genes = []
    union_gene_to_idx = {}
    sample_to_union_idx = {}
    print(f"translated union gene panel（train+test）: n_samples={len(sample_ids)}")
    for sample_id in sample_ids:
        h5ad_path = os.path.join(args.source_dataroot, args.dataset, f"adata/{sample_id}.h5ad")
        if not os.path.exists(h5ad_path):
            raise FileNotFoundError(f"translated h5ad: {h5ad_path}")
        adata = sc.read_h5ad(h5ad_path, backed="r")
        sample_genes = adata.var_names.tolist()
        if getattr(adata, "file", None) is not None:
            adata.file.close()
        sample_idx = []
        for g in sample_genes:
            if g not in union_gene_to_idx:
                union_gene_to_idx[g] = len(union_genes)
                union_genes.append(g)
            sample_idx.append(union_gene_to_idx[g])
        sample_to_union_idx[sample_id] = np.array(sample_idx, dtype=np.int32)

    panel_obj = {
        "dataset": args.dataset,
        "n_samples": len(sample_ids),
        "n_genes": len(union_genes),
        "genes": union_genes,
    }
    with open(panel_json_path, "w") as f:
        json.dump(panel_obj, f, ensure_ascii=False, indent=2)
    np.savez_compressed(mapping_npz_path, **sample_to_union_idx)
    print(
        f"union gene panel translated: n_genes={len(union_genes)} | "
        f"panel={panel_json_path} | mapping={mapping_npz_path}"
    )
    return panel_json_path, mapping_npz_path, int(len(union_genes))


def main(args, split_id, train_sample_ids, test_sample_ids, val_save_dir, checkpoint_save_dir):
    normalize_method = get_normalize_method(args.normalize_method)
    max_sampled_spots = int(getattr(args, "max_sampled_spots", 0))
    if max_sampled_spots <= 0:
        max_sampled_spots = None

    if bool(getattr(args, "use_union_genepanel", False)):
        panel_json_path, mapping_npz_path, n_union_genes = _build_union_genepanel_with_cache(
            args, list(train_sample_ids) + list(test_sample_ids)
        )
        args.gene_list = panel_json_path
        if int(args.n_genes) != int(n_union_genes):
            print(
                f"[union_genepanel] n_genes: {args.n_genes} -> {n_union_genes} "
                f"(translated train+test translated)"
            )
            args.n_genes = int(n_union_genes)
        args.genepanel_mapping_cache = mapping_npz_path
        if not bool(getattr(args, "use_latent_flow", False)):
            print(
                "[translated] translated use_latent_flow=False；translated union panel translated，translated "
                "use_latent_flow（translated MLP AE translated latent）。"
            )

    # translated：translated use_union_genepanel translated args.gene_list translated
    resolved_gene_list_path = _resolve_gene_list_path(
        args.source_dataroot, args.dataset, args.gene_list
    )

    gene_reweight_mode = getattr(args, "gene_reweight_mode", "none")
    gene_resample_mode = getattr(args, "gene_resample_mode", "none")
    use_gene_schedule = (gene_reweight_mode not in (None, "none")) or (
        gene_resample_mode == "rebalance_sparse_dense"
    )
    rho_np = base_w_np = is_sparse_np = None
    if use_gene_schedule:
        from utils.gene_rho_weighting import (
            load_or_compute_gene_rho,
            rho_to_base_weights,
            sparse_gene_mask_from_quantile,
        )

        cache_path = (getattr(args, "gene_rho_cache", None) or "").strip() or None
        rho_np, _genes = load_or_compute_gene_rho(
            train_sample_ids,
            args.embed_dataroot,
            args.source_dataroot,
            args.dataset,
            args.feature_encoder,
            args.gene_list,
            normalize_method,
            float(getattr(args, "gene_nonzero_eps", 0.0)),
            cache_path,
        )
        if rho_np.shape[0] != args.n_genes:
            print(
                f"[translated] translated rho translated {rho_np.shape[0]} translated n_genes={args.n_genes} translated，"
                f"translated n_genes translated {rho_np.shape[0]} translated。"
            )
            args.n_genes = int(rho_np.shape[0])
        if gene_reweight_mode in (None, "none"):
            base_w_np = np.ones(args.n_genes, dtype=np.float32)
        else:
            clip_m = float(getattr(args, "gene_weight_clip", 0.0))
            base_w_np = rho_to_base_weights(
                rho_np,
                gene_reweight_mode,
                float(getattr(args, "gene_rho_eps", 1e-6)),
                clip_m if clip_m > 0 else None,
            )
        is_sparse_np = sparse_gene_mask_from_quantile(
            rho_np, float(getattr(args, "gene_sparse_rho_quantile", 0.5))
        )
        n_sparse = int(np.sum(is_sparse_np))
        print(
            f"Gene schedule: reweight={gene_reweight_mode} resample={gene_resample_mode} | "
            f"rho in [{float(np.min(rho_np)):.4g}, {float(np.max(rho_np)):.4g}] | "
            f"sparse genes (by quantile)={n_sparse}/{args.n_genes}"
        )

    print("Dataset Loading")
    sample_id_paths = [
        HESTDatasetPath(
            name=sample_id,
            h5_path=os.path.join(args.embed_dataroot, args.dataset, args.feature_encoder, f"fp32/{sample_id}.h5"),
            h5ad_path=os.path.join(args.source_dataroot, args.dataset, f"adata/{sample_id}.h5ad"),
            gene_list_path=resolved_gene_list_path,
        ) for sample_id in train_sample_ids
    ]
    train_dataset = MultiHESTDataset(sample_id_paths, 
                                     distribution=args.patch_distribution, 
                                     normalize_method=normalize_method,
                                     sample_times=args.sample_times,
                                     spot_sampling_mode=args.spot_sampling_mode,
                                     max_sampled_spots=max_sampled_spots)
    observed_n_genes = int(train_dataset.sp_datasets[0].labels.shape[-1]) if len(train_dataset.sp_datasets) > 0 else int(args.n_genes)
    if int(args.n_genes) != observed_n_genes:
        print(
            f"[translated] translated gene translated={observed_n_genes} translated args.n_genes={args.n_genes} translated，"
            f"translated。"
        )
        args.n_genes = observed_n_genes
        if use_gene_schedule and base_w_np is not None and is_sparse_np is not None:
            if len(base_w_np) != observed_n_genes or len(is_sparse_np) != observed_n_genes:
                print(
                    "[translated] gene schedule translated，translated split translated（translated sparse/dense translated）。"
                )
                base_w_np = np.ones(observed_n_genes, dtype=np.float32)
                is_sparse_np = np.zeros(observed_n_genes, dtype=np.bool_)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=padding_batcher())
    use_random_spot_training = bool(getattr(args, "use_random_spot_training", False))
    random_sample_batchsize = int(getattr(args, "random_sample_batchsize", 128))
    random_test_every_n_epoch = max(
        1, int(getattr(args, "random_sample_test_every_n_epoch", 1))
    )
    features_pool = coords_pool = labels_pool = None
    if use_random_spot_training:
        features_pool, coords_pool, labels_pool = _build_global_spot_pool(train_dataset)
        steps_per_epoch_shuffle = int(
            np.ceil(float(features_pool.shape[0]) / max(1, random_sample_batchsize))
        )
        print(
            f"Shuffle-spot training enabled: pool_spots={features_pool.shape[0]}, "
            f"batchsize={random_sample_batchsize}, "
            f"steps_per_epoch={steps_per_epoch_shuffle} (full no-replacement traversal), "
            f"test_every_n_epoch={random_test_every_n_epoch}"
        )

    # using train_sample_ids for validation
    val_sample_id_paths = [
        HESTDatasetPath(
                name=sample_id,
                h5_path=os.path.join(args.embed_dataroot, args.dataset, args.feature_encoder, f"fp32/{sample_id}.h5"),
                h5ad_path=os.path.join(args.source_dataroot, args.dataset, f"adata/{sample_id}.h5ad"),
                gene_list_path=resolved_gene_list_path,
        ) for sample_id in test_sample_ids
    ]
    val_loaders = [
        torch.utils.data.DataLoader(
            HESTDataset(
                sample_id_path, distribution="constant_1.0", 
                normalize_method=normalize_method,
                sample_times=1,
                spot_sampling_mode=args.spot_sampling_mode
            ),
            batch_size=1, collate_fn=padding_batcher()
        ) for sample_id_path in val_sample_id_paths
    ]
    
    # create test set data loaders using test_sample_ids
    test_sample_id_paths = [
        HESTDatasetPath(
                name=sample_id,
                h5_path=os.path.join(args.embed_dataroot, args.dataset, args.feature_encoder, f"fp32/{sample_id}.h5"),
                h5ad_path=os.path.join(args.source_dataroot, args.dataset, f"adata/{sample_id}.h5ad"),
                gene_list_path=resolved_gene_list_path,
        ) for sample_id in test_sample_ids
    ]
    test_loaders = [
        torch.utils.data.DataLoader(
            HESTDataset(
                sample_id_path, distribution="constant_1.0", 
                normalize_method=normalize_method,
                sample_times=1,
                spot_sampling_mode=args.spot_sampling_mode
            ),
            batch_size=1, collate_fn=padding_batcher()
        ) for sample_id_path in test_sample_id_paths
    ]

    device = args.device
    # model = Denoiser(args).to(device)
    model = build_flow_denoiser(args).to(device)

    gene_base_w_t = None
    gene_is_sparse_t = None
    if use_gene_schedule and base_w_np is not None and is_sparse_np is not None:
        gene_base_w_t = torch.as_tensor(base_w_np, device=device, dtype=torch.float32)
        gene_is_sparse_t = torch.as_tensor(is_sparse_np, device=device, dtype=torch.bool)

    use_latent = getattr(args, "use_latent_flow", False)
    latent_ae = None
    latent_encoder_kind = str(getattr(args, "latent_encoder", "ae")).lower().strip()
    use_latent_vae = use_latent and (latent_encoder_kind == "vae")
    if use_latent:
        latent_cls = GeneLatentVAE if use_latent_vae else GeneLatentAE
        latent_ae = latent_cls(
            n_genes=args.n_genes,
            latent_dim=args.latent_dim,
            hidden_dim=getattr(args, "ae_mlp_hidden_dim", 256),
        ).to(device)
        print(f"Latent encoder: {latent_cls.__name__}")

    diffusier = Interpolant(
        args.prior_sampler, 
        device=device,
        total_count=torch.tensor([args.zinb_total_count], device=device),
        logits=torch.tensor([args.zinb_logits], device=device),
        zi_logits=args.zinb_zi_logits,
        normalize=args.prior_sampler != "gaussian",
        use_t_bounds=getattr(args, 'use_t_bounds', True),
        t_min=getattr(args, 't_min', 1e-3),
        t_max=getattr(args, 't_max', 0.999),
        alpha_schedule=getattr(args, 'alpha_schedule', 'linear'),
        t_schedule=getattr(args, 't_schedule', 'linear'),
        logit_normal_mu=getattr(args, 'logit_normal_mu', 0.0),
        logit_normal_sigma=getattr(args, 'logit_normal_sigma', 1.0),
        r=getattr(args, 'r', 1.0),
    )
    aux_zinb_sampler = PriorSampler(
        "zinb",
        total_count=torch.tensor([args.zinb_total_count], device=device),
        logits=torch.tensor([args.zinb_logits], device=device),
        zi_logits=args.zinb_zi_logits,
    )
    opt_params = list(model.parameters())
    if latent_ae is not None:
        opt_params += list(latent_ae.parameters())
    optimizer = torch.optim.Adam(opt_params, lr=args.lr)

    print("Training")
    sequence_token_pairing = str(
        getattr(args, "sequence_token_pairing", "paired")
    ).lower().strip()
    if sequence_token_pairing not in ("paired", "permuted"):
        raise ValueError(
            f"Unsupported sequence_token_pairing: {sequence_token_pairing}"
        )
    if sequence_token_pairing == "permuted":
        print(
            "[ablation] translated sequence token translated："
            "(h_i, x_t,j)/(h_i, z_t,j), i!=j。translated。"
        )
    sequence_aux_token_source = str(
        getattr(args, "sequence_aux_token_source", "target")
    ).lower().strip()
    if sequence_aux_token_source not in ("target", "zinb_random"):
        raise ValueError(
            f"Unsupported sequence_aux_token_source: {sequence_aux_token_source}"
        )
    if sequence_aux_token_source == "zinb_random":
        print(
            "[ablation] translated ZINB auxiliary token（translated target-derived）。"
        )
    time_conditioning_mode = str(
        getattr(args, "time_conditioning_mode", "full")
    ).lower().strip()
    if time_conditioning_mode not in ("full", "fixed", "none"):
        raise ValueError(
            f"Unsupported time_conditioning_mode: {time_conditioning_mode}"
        )
    if time_conditioning_mode != "full":
        if time_conditioning_mode == "fixed":
            t_fixed = float(getattr(args, "time_conditioning_fixed_value", 0.5))
            print(
                f"[ablation] translated（fixed）：translated t={t_fixed:.4f} translated；"
                "z_t translated target translated t。"
            )
        else:
            print(
                "[ablation] translated（none）：translated t=0；"
                "z_t translated target translated t。"
            )

    wandb_prefix = f"{args.dataset}/split{split_id}"
    # translated wandb run translated split translated step_metric「epoch」，translated epoch translated；translated split translated
    wandb_epoch_key = f"epoch_s{split_id}"
    if args.use_wandb:
        wandb.define_metric(wandb_epoch_key)
        for _m in (
            "loss",
            "train_fm_loss",
            "train_recon_loss",
            "train_kl_loss",
            "train_sequence_perm_applied_ratio",
            "train_sequence_perm_skipped_steps",
            "train_pcc_all",
            "test_pcc_all",
            "train_mse",
            "test_mse",
            "train_mae",
            "test_mae",
            "test_integration_error_mse",
        ):
            wandb.define_metric(f"{wandb_prefix}/{_m}", step_metric=wandb_epoch_key)
    best_pearson, best_val_dict = -1, None
    # translated（translated train translated val）translated；translated/translated，translated epoch
    best_test_at_best_val = None
    last_val_perf_dict = None
    early_stop_step = 0
    epoch_iter = tqdm(range(1, args.epochs + 1), ncols=100)
    for epoch in epoch_iter:
        avg_loss = 0
        avg_fm_loss = 0.0
        avg_recon_loss = 0.0
        n_recon_batches = 0
        avg_kl_loss = 0.0
        n_kl_batches = 0
        seq_perm_applied_steps = 0
        seq_perm_skipped_steps = 0
        model.train()
        random_spot_global_perm = None
        if use_random_spot_training:
            shuffle_batches = _build_shuffled_spot_batch_indices(
                total_spots=features_pool.shape[0],
                batch_size=random_sample_batchsize,
                device=features_pool.device,
            )
            if sequence_token_pairing == "permuted":
                random_spot_global_perm = _build_deranged_batch_permutation(
                    int(features_pool.shape[0]),
                    features_pool.device,
                )
            num_train_steps = len(shuffle_batches)
            train_iter = range(num_train_steps)
            train_loader_iter = None
        else:
            num_train_steps = max(1, len(train_loader))
            train_iter = range(num_train_steps)
            train_loader_iter = iter(train_loader)

        for step in train_iter:
            source_gene_exp = None
            if use_random_spot_training:
                idx = shuffle_batches[step]
                source_idx = None
                if random_spot_global_perm is not None:
                    source_idx = random_spot_global_perm.index_select(0, idx)
                batch = (
                    features_pool[idx].unsqueeze(0),
                    coords_pool[idx].unsqueeze(0),
                    labels_pool[idx].unsqueeze(0),
                )
                if source_idx is not None:
                    source_gene_exp = labels_pool[source_idx].unsqueeze(0)
            else:
                batch = next(train_loader_iter)
            batch = [x.to(device) for x in batch]
            img_features, coords, gene_exp = batch
            if source_gene_exp is not None:
                source_gene_exp = source_gene_exp.to(device)
            vae_mu = vae_logvar = None
            if use_latent:
                if use_latent_vae:
                    z1, vae_mu, vae_logvar = latent_ae.encode_with_stats(gene_exp)
                else:
                    z1 = latent_ae.encode(gene_exp)
                x1_flow = z1
            else:
                x1_flow = gene_exp

            # JIT only: t~sample_t, e~N(0,I), z=t*x+(1-t)*e
            t_steps = diffusier.get_timestep(x1_flow.shape[0], device=device).to(dtype=x1_flow.dtype)
            exp0 = torch.randn_like(x1_flow)
            noisy_exp = t_steps[:, None, None] * x1_flow + (1.0 - t_steps)[:, None, None] * exp0

            inference_kwargs = {"predict": True}
            try:
                sequence_token_input = noisy_exp
                sequence_clean_tokens = x1_flow
                model_t_steps = _resolve_model_time_input(args, t_steps)
                if sequence_token_pairing == "permuted":
                    if source_gene_exp is not None:
                        if use_latent:
                            if use_latent_vae:
                                z_source, _, _ = latent_ae.encode_with_stats(source_gene_exp)
                            else:
                                z_source = latent_ae.encode(source_gene_exp)
                            x1_source = z_source
                        else:
                            x1_source = source_gene_exp
                        noisy_source = (
                            t_steps[:, None, None] * x1_source
                            + (1.0 - t_steps)[:, None, None] * exp0
                        )
                        sequence_token_input = noisy_source
                        sequence_clean_tokens = x1_source
                        seq_perm_applied_steps += 1
                    else:
                        perm_idx = _build_deranged_batch_permutation(
                            noisy_exp.shape[0], noisy_exp.device
                        )
                        if perm_idx is None:
                            seq_perm_skipped_steps += 1
                        else:
                            sequence_token_input = noisy_exp.index_select(0, perm_idx)
                            sequence_clean_tokens = x1_flow.index_select(0, perm_idx)
                            seq_perm_applied_steps += 1
                if sequence_aux_token_source == "zinb_random":
                    sequence_token_input = _sample_random_zinb_tokens_like(
                        noisy_exp, aux_zinb_sampler
                    )
                    sequence_clean_tokens = _sample_random_zinb_tokens_like(
                        x1_flow, aux_zinb_sampler
                    )
                raw_pred = model.inference(
                    sequence_token_input,
                    img_features,
                    coords,
                    model_t_steps,
                    clean_sequence_tokens=sequence_clean_tokens,
                    **inference_kwargs,
                )
            except TypeError:
                raw_pred = model.inference(
                    sequence_token_input,
                    img_features,
                    coords,
                    model_t_steps,
                    predict=True,
                )
            
            # translated Δ = 1（translated，Δ > h）
            # Delta = 0.2
            # Delta_expanded = Delta
            
            # translated：v = -x_t/Δ + φ
            # translated v_true_original translated φ^*
            # v_true_original = -x_t/Δ + φ^*，translated φ^* = v_true_original + x_t/Δ
            # v_true_original = F.relu(v_true_original)
            # v_true_original = F.softplus(v_true_original)
            # translated：v^* = -x_t/Δ + φ^*
            # v_true = - noisy_exp / Delta_expanded + v_true_original


            # translated：translatedv_true_originaltranslatedReLU，translatedbias
            # translated，translatedv_truetranslated，translatedv_true_original
            # v_true_original = F.relu(v_true_original)  # translated：translatedbiastranslated
            # translatedsoftplustranslated，translated
            # phi_true = F.relu(v_true_original + noisy_exp / Delta_expanded)
            # # translated：v^* = -x_t/Δ + φ^*
            # v_true = - noisy_exp / Delta_expanded + phi_true


            # translated：φ^θ = softplus(u_θ) >= 0（translated）
            # phi_pred = F.softplus(v_pred)
            # phi_pred = F.relu(v_pred)
            
            # translated：v^θ = -x_t/Δ + φ^θ
            # v_pred = - noisy_exp / Delta_expanded + phi_pred
            # v_pred = - noisy_exp / diffusier.alpha(t_steps)[:, None, None] + F.softplus(v_pred)
            # S = args.n_sample_steps
            # dt = 1.0 / S
            # v_pred = - noisy_exp / dt + F.softplus(v_pred)
            # # u: raw network output same shape as v_target
            # # u = net(y_t, t, cond)
            # v = - y_t / delta_t + F.softplus(u)
            # loss = F.mse_loss(v, v_target)

            pad_mask = img_features.sum(-1) == 0

            w_eff_genes = None
            if (
                use_gene_schedule
                and gene_base_w_t is not None
                and gene_is_sparse_t is not None
                and gene_exp.shape[-1] == args.n_genes
            ):
                w_eff_genes = batch_gene_effective_weights(
                    gene_base_w_t,
                    gene_is_sparse_t,
                    gene_resample_mode,
                    float(getattr(args, "gene_dense_keep_prob", 0.3)),
                    device=gene_exp.device,
                    dtype=gene_exp.dtype,
                )

            # AE reconstruction term shared with JIT training
            recon_loss_term = None
            kl_loss_term = None
            if latent_ae is not None and use_latent:
                recon = latent_ae.decode(z1)
                if w_eff_genes is not None and recon.shape[-1] == args.n_genes:
                    mse_g_recon = per_gene_mean_sq_error(recon, gene_exp, pad_mask)
                    recon_loss_term = weighted_per_gene_scalar_loss(mse_g_recon, w_eff_genes)
                else:
                    recon_loss_term = F.mse_loss(recon[~pad_mask], gene_exp[~pad_mask])
                if use_latent_vae and vae_mu is not None and vae_logvar is not None:
                    kl_loss_term = GeneLatentVAE.kl_loss(
                        vae_mu, vae_logvar, keep_mask=(~pad_mask)
                    )

            denom = (1.0 - t_steps).clamp_min(1e-4)[:, None, None]
            # JIT: model predicts x_pred, then map to velocity
            x_pred = raw_pred
            v_true = (x1_flow - noisy_exp) / denom
            v_pred = (x_pred - noisy_exp) / denom

            use_gene_dim_fm = (not use_latent) and v_pred.shape[-1] == args.n_genes
            if use_gene_dim_fm and w_eff_genes is not None:
                mse_g_fm = per_gene_mean_sq_error(v_pred, v_true, pad_mask)
                mse_loss = weighted_per_gene_scalar_loss(mse_g_fm, w_eff_genes)
            else:
                mse_loss = model.loss_func(v_pred[~pad_mask], v_true[~pad_mask])
            avg_fm_loss += float(mse_loss.detach().cpu())

            loss = mse_loss
            if recon_loss_term is not None:
                loss = loss + getattr(args, "ae_recon_weight", 1.0) * recon_loss_term
                avg_recon_loss += float(recon_loss_term.detach().cpu())
                n_recon_batches += 1
            if kl_loss_term is not None:
                warmup_epochs = max(1, int(getattr(args, "latent_vae_kl_warmup_epochs", 50)))
                warmup_scale = min(1.0, float(epoch) / float(warmup_epochs))
                kl_weight = float(getattr(args, "latent_vae_kl_weight", 1e-3)) * warmup_scale
                loss = loss + kl_weight * kl_loss_term
                avg_kl_loss += float(kl_loss_term.detach().cpu())
                n_kl_batches += 1
            optimizer.zero_grad()
            model.zero_grad()
            if latent_ae is not None:
                latent_ae.zero_grad()
            loss.backward()
            clip_params = list(model.parameters())
            if latent_ae is not None:
                clip_params += list(latent_ae.parameters())
            torch.nn.utils.clip_grad_norm_(clip_params, args.clip_norm)
            optimizer.step()

            avg_loss += loss.cpu().item()

        avg_loss /= num_train_steps
        avg_fm_loss /= num_train_steps
        if n_recon_batches > 0:
            avg_recon_loss /= n_recon_batches
        if n_kl_batches > 0:
            avg_kl_loss /= n_kl_batches
        seq_perm_applied_ratio = (
            float(seq_perm_applied_steps) / float(num_train_steps)
            if num_train_steps > 0
            else 0.0
        )
        recon_disp = f"{avg_recon_loss:.6f}" if n_recon_batches > 0 else "n/a"
        kl_disp = f"{avg_kl_loss:.6f}" if n_kl_batches > 0 else "n/a"
        epoch_iter.set_description(
            f"ep{epoch} | loss(total)={avg_loss:.4f} | fm_loss={avg_fm_loss:.4f} | "
            f"recon_loss={recon_disp} | kl_loss={kl_disp}"
        )
        if sequence_token_pairing == "permuted" and seq_perm_skipped_steps > 0:
            print(
                f"[ablation] epoch {epoch}: translated batch_size=1 translated steps="
                f"{seq_perm_skipped_steps}/{num_train_steps}"
            )

        # translated：translated「translated」translated（translated best.pth，translated save_step=-1 translated）
        if not getattr(args, "save_best_checkpoint_only", True):
            if args.save_step > 0 and epoch % args.save_step == 0:
                ckpt = {"denoiser": model.state_dict()}
                if latent_ae is not None:
                    ckpt["latent_ae"] = latent_ae.state_dict()
                torch.save(ckpt, os.path.join(checkpoint_save_dir, f"{epoch}.pth"))

        if epoch % args.eval_step == 0 or epoch == args.epochs:
            val_perf_dict, pred_dump = test(
                args, diffusier, model, val_loaders, return_all=True, latent_ae=latent_ae
            )
            last_val_perf_dict = val_perf_dict
            val_improved = val_perf_dict["all"]["pearson_mean"] > best_pearson
            if val_improved:
                best_pearson = val_perf_dict["all"]['pearson_mean']
                best_val_dict = val_perf_dict

                ckpt_best = {
                    "epoch": epoch,
                    "best_pearson_mean": best_pearson,
                    "denoiser": model.state_dict(),
                }
                if latent_ae is not None:
                    ckpt_best["latent_ae"] = latent_ae.state_dict()
                best_ckpt_path = os.path.join(checkpoint_save_dir, "best.pth")
                torch.save(ckpt_best, best_ckpt_path)
                print(f"translated checkpoint: {best_ckpt_path} (PCC={best_pearson:.6f}, epoch={epoch})")

                for patch_name, dataset_res in val_perf_dict.items():
                    with open(os.path.join(val_save_dir, f'{patch_name}_results.json'), 'w') as f:
                        json.dump(dataset_res, f, sort_keys=True, indent=4, default=json_default)
                
                # translated
                gene_stats = calculate_gene_expression_stats(
                    pred_dump['preds_all'], 
                    pred_dump['targets_all'], 
                    val_loaders[0].dataset.gene_list
                )
                stats_file_path = os.path.join(val_save_dir, 'gene_expression_stats.json')
                with open(stats_file_path, 'w') as f:
                    json.dump(gene_stats, f, sort_keys=True, indent=4, default=json_default)
                
                print(f"translated: {stats_file_path}")
                print(f"translated {len(val_loaders[0].dataset.gene_list)} translated")
                
                # translatedPCCtranslatednpytranslated（translated）
                best_pcc_data = {
                    'predictions': pred_dump['preds_all'],
                    'targets': pred_dump['targets_all'],
                    'gene_list': val_loaders[0].dataset.gene_list,
                    'pearson_correlation': best_pearson,
                    'epoch': epoch
                }
                best_pcc_file = os.path.join(val_save_dir, 'best_pcc_data.npz')
                np.savez(best_pcc_file, **best_pcc_data)
                
                # translatedCSV：MSE、MAE、PCC-200、PCC-100、PCC-50（translatedPCCtranslated）
                try:
                    preds_all_np = pred_dump['preds_all']
                    targets_all_np = pred_dump['targets_all']

                    # translatedMSE/MAE
                    mse_overall = float(np.mean((preds_all_np - targets_all_np) ** 2))
                    mae_overall = float(np.mean(np.abs(preds_all_np - targets_all_np)))

                    # translatedPCC，translatedPCC-200/PCC-100/PCC-50
                    pearson_items = val_perf_dict["all"].get('pearson_corrs', [])
                    pearson_values = [x.get('pearson_corr', None) for x in pearson_items]
                    pearson_values = [x for x in pearson_values if x is not None and not np.isnan(x)]
                    pearson_values.sort(reverse=True)

                    def _top_mean(values, k):
                        if len(values) == 0:
                            return float('nan')
                        return float(np.mean(values[:min(k, len(values))]))

                    pcc200_mean = _top_mean(pearson_values, 200)
                    pcc100_mean = _top_mean(pearson_values, 100)
                    pcc50_mean = _top_mean(pearson_values, 50)

                    # translated（translated pcc(all) translated）
                    n_genes_effective = len(val_loaders[0].dataset.gene_list)

                    # translatedCSV（translated）
                    csv_path = os.path.join(val_save_dir, 'best_metrics.csv')
                    import csv
                    with open(csv_path, 'w', newline='') as f_csv:
                        writer = csv.writer(f_csv)
                        writer.writerow(['epoch', 'pearson_mean', 'mse', 'mae', 'pcc200', 'pcc100', 'pcc50', 'n_genes'])
                        writer.writerow([
                            epoch,
                            f"{best_pearson:.6f}",
                            f"{mse_overall:.6f}",
                            f"{mae_overall:.6f}",
                            f"{pcc200_mean:.6f}",
                            f"{pcc100_mean:.6f}",
                            f"{pcc50_mean:.6f}",
                            n_genes_effective,
                        ])

                    print(f"translatedPCCtranslatedCSV: {csv_path}")
                except Exception as e:
                    print(f"translatedPCCtranslatedCSVtranslated: {e}")

                print(f"translatedPCCtranslated: {best_pcc_file}")
                print(f"translated、translated、translated、PCCtranslatedepochtranslated")
                print(f"translatedPCC: {best_pearson:.4f} (Epoch {epoch})")
                # save_pkl(os.path.join(val_save_dir, 'inference_dump.pkl'), pred_dump)
                early_stop_step = 0

            else:
                early_stop_step += 1
                if (not getattr(args, "no_early_stop", False)) and early_stop_step >= 30:
                    print("Early stopping")
                    print(
                        "translated：translated「translated epoch」translated，translated "
                        "best_test_predictions.npz translated test_metrics_at_best_val.csv（translated）。"
                    )
                    if args.use_wandb:
                        train_recon_w = (
                            float(avg_recon_loss) if n_recon_batches > 0 else float("nan")
                        )
                        train_kl_w = (
                            float(avg_kl_loss) if n_kl_batches > 0 else float("nan")
                        )
                        wandb.log(
                            {
                                wandb_epoch_key: epoch,
                                f"{wandb_prefix}/loss": avg_loss,
                                f"{wandb_prefix}/train_fm_loss": avg_fm_loss,
                                f"{wandb_prefix}/train_recon_loss": train_recon_w,
                                f"{wandb_prefix}/train_kl_loss": train_kl_w,
                                f"{wandb_prefix}/train_sequence_perm_applied_ratio": seq_perm_applied_ratio,
                                f"{wandb_prefix}/train_sequence_perm_skipped_steps": float(seq_perm_skipped_steps),
                            }
                        )
                    break

            # translated：translated spot translated N translated epoch translated
            test_pearson = float("nan")
            test_integration_err = float("nan")
            test_pcc200 = float("nan")
            test_pcc100 = float("nan")
            test_pcc50 = float("nan")
            test_mse = float("nan")
            test_mae = float("nan")
            should_run_test = True
            if use_random_spot_training:
                should_run_test = (
                    epoch % random_test_every_n_epoch == 0 or epoch == args.epochs
                )
            if should_run_test:
                print(f"Epoch {epoch} translated...")
                test_perf_dict, test_pred_dump = test(
                    args, diffusier, model, test_loaders, return_all=True, latent_ae=latent_ae
                )
                if val_improved:
                    best_test_at_best_val = _test_scalar_metrics_from_perf(test_perf_dict)
                    best_test_at_best_val["epoch"] = epoch
                    best_test_file = os.path.join(val_save_dir, "best_test_predictions.npz")
                    np.savez(
                        best_test_file,
                        predictions=test_pred_dump["preds_all"],
                        targets=test_pred_dump["targets_all"],
                        gene_list=test_loaders[0].dataset.gene_list,
                        pearson_correlation=best_test_at_best_val["pearson_mean"],
                        epoch=epoch,
                        val_pearson_mean=best_pearson,
                    )
                    import csv
                    at_best_csv = os.path.join(val_save_dir, "test_metrics_at_best_val.csv")
                    with open(at_best_csv, "w", newline="") as f_csv:
                        w = csv.writer(f_csv)
                        w.writerow(
                            [
                                "epoch",
                                "pearson_mean",
                                "pcc200",
                                "pcc100",
                                "pcc50",
                                "mse",
                                "mae",
                                "integration_error_mse",
                            ]
                        )
                        w.writerow(
                            [
                                epoch,
                                f"{best_test_at_best_val['pearson_mean']:.6f}",
                                f"{best_test_at_best_val['pcc200']:.6f}",
                                f"{best_test_at_best_val['pcc100']:.6f}",
                                f"{best_test_at_best_val['pcc50']:.6f}",
                                f"{best_test_at_best_val['mse']:.6f}",
                                f"{best_test_at_best_val['mae']:.6f}",
                                f"{best_test_at_best_val.get('integration_error_mse', float('nan')):.8f}",
                            ]
                        )
                    print(f"translated: {best_test_file}")
                    print(
                        f"translated epoch translated PCC: {best_test_at_best_val['pearson_mean']:.4f} "
                        f"(translated {at_best_csv})"
                    )

                test_pearson = test_perf_dict["all"]['pearson_mean']
                test_integration_err = test_perf_dict["all"].get(
                    "integration_error_mse", float("nan")
                )

                try:
                    test_pearson_items = test_perf_dict["all"].get('pearson_corrs', [])
                    test_pearson_values = [x.get('pearson_corr', None) for x in test_pearson_items]
                    test_pearson_values = [x for x in test_pearson_values if x is not None and not np.isnan(x)]
                    test_pearson_values.sort(reverse=True)

                    def _top_mean(values, k):
                        if len(values) == 0:
                            return float('nan')
                        return float(np.mean(values[:min(k, len(values))]))

                    test_pcc200 = _top_mean(test_pearson_values, 200)
                    test_pcc100 = _top_mean(test_pearson_values, 100)
                    test_pcc50 = _top_mean(test_pearson_values, 50)
                    test_mse = test_perf_dict["all"].get('mse_overall', float('nan'))
                    test_mae = test_perf_dict["all"].get('mae_overall', float('nan'))

                    test_csv_path = os.path.join(val_save_dir, 'test_metrics.csv')
                    import csv
                    file_exists = os.path.exists(test_csv_path)
                    with open(test_csv_path, 'a', newline='') as f_csv:
                        writer = csv.writer(f_csv)
                        if not file_exists:
                            writer.writerow(
                                [
                                    'epoch',
                                    'pearson_mean',
                                    'pcc200',
                                    'pcc100',
                                    'pcc50',
                                    'mse',
                                    'mae',
                                    'integration_error_mse',
                                ]
                            )
                        writer.writerow([
                            epoch,
                            f"{test_pearson:.6f}",
                            f"{test_pcc200:.6f}",
                            f"{test_pcc100:.6f}",
                            f"{test_pcc50:.6f}",
                            f"{test_mse:.6f}",
                            f"{test_mae:.6f}",
                            f"{test_integration_err:.8f}",
                        ])

                    print(
                        f"translated Epoch {epoch} - PCC: {test_pearson:.4f}, PCC-200: {test_pcc200:.4f}, "
                        f"PCC-100: {test_pcc100:.4f}, PCC-50: {test_pcc50:.4f}, MSE: {test_mse:.6f}, MAE: {test_mae:.6f}, "
                        f"translated(translatedMSE): {test_integration_err:.8f}"
                    )
                except Exception as e:
                    print(f"translated: {e}")
            else:
                print(
                    f"Epoch {epoch} translated（random spot translated: translated {random_test_every_n_epoch} epoch translated）。"
                )

            if args.use_wandb:
                va = val_perf_dict["all"]
                train_recon_w = (
                    float(avg_recon_loss) if n_recon_batches > 0 else float("nan")
                )
                train_kl_w = (
                    float(avg_kl_loss) if n_kl_batches > 0 else float("nan")
                )
                wandb.log(
                    {
                        wandb_epoch_key: epoch,
                        f"{wandb_prefix}/loss": avg_loss,
                        f"{wandb_prefix}/train_fm_loss": avg_fm_loss,
                        f"{wandb_prefix}/train_recon_loss": train_recon_w,
                        f"{wandb_prefix}/train_kl_loss": train_kl_w,
                        f"{wandb_prefix}/train_sequence_perm_applied_ratio": seq_perm_applied_ratio,
                        f"{wandb_prefix}/train_sequence_perm_skipped_steps": float(seq_perm_skipped_steps),
                        f"{wandb_prefix}/train_pcc_all": va["pearson_mean"],
                        f"{wandb_prefix}/test_pcc_all": test_pearson,
                        f"{wandb_prefix}/train_mse": va.get("mse_overall", float("nan")),
                        f"{wandb_prefix}/test_mse": test_mse,
                        f"{wandb_prefix}/train_mae": va.get("mae_overall", float("nan")),
                        f"{wandb_prefix}/test_mae": test_mae,
                        f"{wandb_prefix}/test_integration_error_mse": float(
                            test_integration_err
                        ),
                    }
                )

        else:
            if args.use_wandb:
                train_recon_w = (
                    float(avg_recon_loss) if n_recon_batches > 0 else float("nan")
                )
                train_kl_w = (
                    float(avg_kl_loss) if n_kl_batches > 0 else float("nan")
                )
                wandb.log(
                    {
                        wandb_epoch_key: epoch,
                        f"{wandb_prefix}/loss": avg_loss,
                        f"{wandb_prefix}/train_fm_loss": avg_fm_loss,
                        f"{wandb_prefix}/train_recon_loss": train_recon_w,
                        f"{wandb_prefix}/train_kl_loss": train_kl_w,
                        f"{wandb_prefix}/train_sequence_perm_applied_ratio": seq_perm_applied_ratio,
                        f"{wandb_prefix}/train_sequence_perm_skipped_steps": float(seq_perm_skipped_steps),
                    }
                )

    # translated：translated「translated epoch」translated；translated test_metrics.csv translated（translated）
    test_final_results = {
        "pearson_mean": float("nan"),
        "pcc200": float("nan"),
        "pcc100": float("nan"),
        "pcc50": float("nan"),
        "mse": float("nan"),
        "mae": float("nan"),
        "integration_error_mse": float("nan"),
    }
    if best_test_at_best_val is not None:
        test_final_results["pearson_mean"] = best_test_at_best_val["pearson_mean"]
        test_final_results["pcc200"] = best_test_at_best_val["pcc200"]
        test_final_results["pcc100"] = best_test_at_best_val["pcc100"]
        test_final_results["pcc50"] = best_test_at_best_val["pcc50"]
        test_final_results["mse"] = best_test_at_best_val["mse"]
        test_final_results["mae"] = best_test_at_best_val["mae"]
        test_final_results["integration_error_mse"] = best_test_at_best_val.get(
            "integration_error_mse", float("nan")
        )
        print(
            f"translated（translated epoch={best_test_at_best_val['epoch']}）- PCC: {test_final_results['pearson_mean']:.4f}, "
            f"PCC-200: {test_final_results['pcc200']:.4f}, "
            f"PCC-100: {test_final_results['pcc100']:.4f}, "
            f"PCC-50: {test_final_results['pcc50']:.4f}, "
            f"MSE: {test_final_results['mse']:.6f}, "
            f"MAE: {test_final_results['mae']:.6f}, "
            f"translatedMSE: {test_final_results['integration_error_mse']:.8f}"
        )
    else:
        test_csv_path = os.path.join(val_save_dir, "test_metrics.csv")
        if os.path.exists(test_csv_path):
            try:
                import csv
                with open(test_csv_path, "r") as f_csv:
                    reader = csv.reader(f_csv)
                    rows = list(reader)
                    if len(rows) > 1:
                        last_row = rows[-1]
                        if len(last_row) >= 5:
                            test_final_results["pearson_mean"] = float(last_row[1])
                            test_final_results["pcc200"] = float(last_row[2])
                            test_final_results["pcc100"] = float(last_row[3])
                            test_final_results["pcc50"] = float(last_row[4])
                            if len(last_row) >= 7:
                                test_final_results["mse"] = float(last_row[5])
                                test_final_results["mae"] = float(last_row[6])
                            if len(last_row) >= 8:
                                test_final_results["integration_error_mse"] = float(
                                    last_row[7]
                                )
                            print(
                                f"translated（translated：test_metrics.csv translated）- PCC: {test_final_results['pearson_mean']:.4f}, "
                                f"PCC-200: {test_final_results['pcc200']:.4f}, "
                                f"PCC-100: {test_final_results['pcc100']:.4f}, "
                                f"PCC-50: {test_final_results['pcc50']:.4f}, "
                                f"MSE: {test_final_results['mse']:.6f}, "
                                f"MAE: {test_final_results['mae']:.6f}, "
                                f"translatedMSE: {test_final_results['integration_error_mse']:.8f}"
                            )
            except Exception as e:
                print(f"translated: {e}")

    # translated
    if best_val_dict is not None:
        val_out = best_val_dict["all"]
    elif last_val_perf_dict is not None:
        val_out = last_val_perf_dict["all"]
    else:
        val_out = {}
    return {
        'val': val_out,
        'test': test_final_results
    }


def run(args):
    # get train/test splits
    split_dir = os.path.join(args.source_dataroot, args.dataset, 'splits')
    splits = os.listdir(split_dir)
    n_splits = len(splits) // 2
    split_only_raw = str(getattr(args, "split_only", "")).strip().lower()
    if split_only_raw:
        if split_only_raw.startswith("split"):
            split_only_raw = split_only_raw[5:]
        try:
            target_split_ids = [int(split_only_raw)]
        except ValueError as e:
            raise ValueError(
                f"--split-only translated splitX translated，translated: {getattr(args, 'split_only', '')!r}"
            ) from e
    else:
        target_split_ids = list(range(n_splits))
    for split_id in target_split_ids:
        if split_id < 0 or split_id >= n_splits:
            raise ValueError(
                f"--split-only translated split{split_id} translated，translated split0~split{n_splits-1}"
            )
    print(f"translated splits: {target_split_ids}")
    all_split_results = []
    all_test_results = []  # translatedsplittranslated
    
    for i in target_split_ids:
        print(f"Running dataset {args.dataset} split {i}")

        train_df = pd.read_csv(os.path.join(split_dir, f'train_{i}.csv'))
        test_df = pd.read_csv(os.path.join(split_dir, f'test_{i}.csv'))

        train_sample_ids = train_df['sample_id'].tolist()
        test_sample_ids = test_df['sample_id'].tolist()

        kfold_save_dir = os.path.join(args.save_dir, f'split{i}')
        os.makedirs(kfold_save_dir, exist_ok=True)
        checkpoint_save_dir = os.path.join(kfold_save_dir, 'checkpoints')
        os.makedirs(checkpoint_save_dir, exist_ok=True)

        results = main(args, i, train_sample_ids, test_sample_ids, kfold_save_dir, checkpoint_save_dir)
        all_split_results.append(results['val'])
        all_test_results.append({
            'split_id': i,
            'pearson_mean': results['test']['pearson_mean'],
            'pcc200': results['test']['pcc200'],
            'pcc100': results['test']['pcc100'],
            'pcc50': results['test']['pcc50'],
            'mse': results['test'].get('mse', float('nan')),
            'mae': results['test'].get('mae', float('nan')),
            'integration_error_mse': results['test'].get(
                'integration_error_mse', float('nan')
            ),
        })

    kfold_results = merge_fold_results(all_split_results)
    with open(os.path.join(args.save_dir, f'results_kfold.json'), 'w') as f:
        p_corrs = kfold_results['pearson_corrs']
        p_corrs = sorted(p_corrs, key=itemgetter('mean'), reverse=True)
        kfold_results['pearson_corrs'] = p_corrs
        json.dump(kfold_results, f, sort_keys=True, indent=4, default=json_default)
    
    # translatedsplittranslated
    test_results_csv_path = os.path.join(args.save_dir, 'test_results_all_splits.csv')
    import csv
    with open(test_results_csv_path, 'w', newline='') as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(
            [
                'split_id',
                'pearson_mean',
                'pcc200',
                'pcc100',
                'pcc50',
                'mse',
                'mae',
                'integration_error_mse',
            ]
        )
        
        # translated
        valid_results = [r for r in all_test_results if not np.isnan(r['pearson_mean'])]
        if len(valid_results) > 0:
            avg_pearson = np.mean([r['pearson_mean'] for r in valid_results])
            avg_pcc200 = np.mean([r['pcc200'] for r in valid_results if not np.isnan(r['pcc200'])])
            avg_pcc100 = np.mean([r['pcc100'] for r in valid_results if not np.isnan(r['pcc100'])])
            avg_pcc50 = np.mean([r['pcc50'] for r in valid_results if not np.isnan(r['pcc50'])])
            avg_mse = np.mean([r['mse'] for r in valid_results if not np.isnan(r['mse'])])
            avg_mae = np.mean([r['mae'] for r in valid_results if not np.isnan(r['mae'])])
            int_vals = [
                r['integration_error_mse']
                for r in valid_results
                if not np.isnan(r.get('integration_error_mse', float('nan')))
            ]
            avg_int_err = float(np.mean(int_vals)) if len(int_vals) > 0 else float('nan')
        else:
            avg_pearson = float('nan')
            avg_pcc200 = float('nan')
            avg_pcc100 = float('nan')
            avg_pcc50 = float('nan')
            avg_mse = float('nan')
            avg_mae = float('nan')
            avg_int_err = float('nan')
        
        # translatedsplittranslated
        for result in all_test_results:
            writer.writerow([
                result['split_id'],
                f"{result['pearson_mean']:.6f}",
                f"{result['pcc200']:.6f}",
                f"{result['pcc100']:.6f}",
                f"{result['pcc50']:.6f}",
                f"{result['mse']:.6f}",
                f"{result['mae']:.6f}",
                f"{result.get('integration_error_mse', float('nan')):.8f}",
            ])
        
        # translated
        writer.writerow([
            'mean',
            f"{avg_pearson:.6f}",
            f"{avg_pcc200:.6f}",
            f"{avg_pcc100:.6f}",
            f"{avg_pcc50:.6f}",
            f"{avg_mse:.6f}",
            f"{avg_mae:.6f}",
            f"{avg_int_err:.8f}",
        ])
    
    print(f"\ntranslatedsplittranslated: {test_results_csv_path}")
    print(
        f"translated - PCC: {avg_pearson:.4f}, PCC-200: {avg_pcc200:.4f}, PCC-100: {avg_pcc100:.4f}, PCC-50: {avg_pcc50:.4f}, "
        f"MSE: {avg_mse:.6f}, MAE: {avg_mae:.6f}, translatedMSE: {avg_int_err:.8f}"
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument(
        '--deterministic',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='translated cuDNN translated（translated/translated，translated）；translated',
    )
    parser.add_argument('--datasets', nargs='+', default=["SKCM"], help=["PAAD", "CCRCC", "READ", "LYMPH_IDC", "PRAD", "COAD"])
    parser.add_argument(
        '--split-only',
        type=str,
        default='',
        help='translated split，translated: 0 translated split0；translated split',
    )
    parser.add_argument('--use_wandb', default=True)
    parser.add_argument('--source_dataroot', default="dataset")
    parser.add_argument('--embed_dataroot', type=str, default="dataset/embed_dataroot")
    parser.add_argument('--gene_list', type=str, default='var_200genes.json',help='hmhvg_50genes.json / var_50genes.json / var_200genes.json')
    parser.add_argument(
        '--use-union-genepanel',
        dest='use_union_genepanel',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='translated split translated train+test translated panel，translated --gene_list；'
        'translated 0。translated --use_latent_flow translated。',
    )
    parser.add_argument(
        '--genepanel-cache-dir',
        dest='genepanel_cache_dir',
        type=str,
        default='',
        help='union genepanel translated；translated save_dir/_genepanel_cache',
    )
    parser.add_argument('--save_dir', type=str, default="results_dir/")
    parser.add_argument('--feature_encoder', type=str, default='uni_v1_official', help="uni_v1_official | resnet50_trunc | ciga | gigapath")
    parser.add_argument('--normalize_method', type=str, default="log1p")
    parser.add_argument('--exp_code', type=str, default="test")
    
    # training hyperparameters
    parser.add_argument('--device', type=int, default=2)
    parser.add_argument('--sample_times', type=int, default=50, help='Number of times to sample patches from each image')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument(
        '--use-random-spot-training',
        dest='use_random_spot_training',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='translated spot translated shuffle（translated）translated batch translated',
    )
    parser.add_argument(
        '--random-sample-batchsize',
        dest='random_sample_batchsize',
        type=int,
        default=128,
        help='translated shuffle spot translated step translated spot translated（translated epoch translated）',
    )
    parser.add_argument(
        '--random-sample-test-every-n-epoch',
        dest='random_sample_test_every_n_epoch',
        type=int,
        default=1,
        help='translated spot translated（translated N translated epoch translated）',
    )
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--clip_norm', type=float, default=1.)
    parser.add_argument('--save_step', type=int, default=-1)
    parser.add_argument(
        '--save-best-checkpoint-only',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='translated True：translated PCC translated checkpoints/best.pth；'
        'translated --save_step translated {epoch}.pth（translated best.pth）',
    )
    parser.add_argument('--eval_step', type=int, default=1)
    parser.add_argument(
        '--no-early-stop',
        dest='no_early_stop',
        action='store_true',
        default=False,
        help='translated early stopping，translated epochs',
    )
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers for dataloader')
    parser.add_argument('--loss_func', type=str, default='mse', help="mse | mae | pearson")
    parser.add_argument('--patch_distribution', type=str, default='uniform')
    parser.add_argument(
        '--max-sampled-spots',
        dest='max_sampled_spots',
        type=int,
        default=0,
        help='translated patch translated spot translated；<=0 translated',
    )
    parser.add_argument(
        '--spot_sampling_mode',
        type=str,
        default='local',
        choices=['local', 'global'],
        help='spot translated：local=translated+translated；global=translated',
    )
    parser.add_argument(
        '--gene-reweight-mode',
        dest='gene_reweight_mode',
        type=str,
        default='inv_rho',
        choices=['none', 'inv_rho', 'inv_sqrt_rho'],
        help='translated rho_g translated：w∝1/(rho+eps) translated 1/sqrt(rho+eps)；none translated（translated resample）',
    )
    parser.add_argument(
        '--gene-rho-eps',
        dest='gene_rho_eps',
        type=float,
        default=1e-6,
        help='loss translated rho+eps，translated',
    )
    parser.add_argument(
        '--gene-nonzero-eps',
        dest='gene_nonzero_eps',
        type=float,
        default=0.0,
        help='translated rho translated spot translated > translated',
    )
    parser.add_argument(
        '--gene-weight-clip',
        dest='gene_weight_clip',
        type=float,
        default=0.0,
        help='>0 translated w_g translated，translated',
    )
    parser.add_argument(
        '--gene-rho-cache',
        dest='gene_rho_cache',
        type=str,
        default='',
        help='translated rho translated .npz（translated genes translated）；translated',
    )
    parser.add_argument(
        '--gene-resample-mode',
        dest='gene_resample_mode',
        type=str,
        default='none',
        choices=['none', 'rebalance_sparse_dense'],
        help='translated A：rho translated sparse translated，dense translated batch translated loss',
    )
    parser.add_argument(
        '--gene-sparse-rho-quantile',
        dest='gene_sparse_rho_quantile',
        type=float,
        default=0.5,
        help='rho translated sparse（translated batch translated loss）',
    )
    parser.add_argument(
        '--gene-dense-keep-prob',
        dest='gene_dense_keep_prob',
        type=float,
        default=0.3,
        help='dense translated batch translated loss translated',
    )
    parser.add_argument('--n_genes', type=int, default=200)
    parser.add_argument(
        '--use-gene-rank-embedding',
        dest='use_gene_rank_embedding',
        action='store_true',
        default=True,
        help='translated gene token translated rank-bin embedding（translated gene_indices translated）',
    )
    parser.add_argument(
        '--gene-rank-bins',
        dest='gene_rank_bins',
        type=int,
        default=32,
        help='rank embedding translated',
    )
    parser.add_argument(
        '--gene-rank-dropout-prob',
        dest='gene_rank_dropout_prob',
        type=float,
        default=0.3,
        help='translated rank translated，translated',
    )
    parser.add_argument(
        '--gene-rank-two-pass-uncond',
        dest='gene_rank_two_pass_uncond',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='translated rank（translated->translated rank->refine）',
    )
    parser.add_argument(
        '--img-cond-dropout-prob',
        dest='img_cond_dropout_prob',
        type=float,
        default=0.5,
        help='translated dropout translated（translated，translated）',
    )
    parser.add_argument(
        '--use-latent-flow',
        dest='use_latent_flow',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='translated MLP translated latent translated flow（translated/translated latent），translated decode translated；'
        'translated --no-use-latent-flow translated。',
    )
    parser.add_argument('--latent_dim', type=int, default=32, help='latent translated，translated n_genes translated')
    parser.add_argument('--ae_mlp_hidden_dim', type=int, default=256, help='GeneLatentAE translated')
    parser.add_argument(
        '--latent-encoder',
        dest='latent_encoder',
        type=str,
        default='vae',
        choices=['ae', 'vae'],
        help='latent translated：ae=translated MLP AE；vae=MLP VAE（translated KL translated）。',
    )
    parser.add_argument('--ae_recon_weight', type=float, default=0.1, help='translated（decode(encoder(x)) vs x）')
    parser.add_argument(
        '--latent-vae-kl-weight',
        dest='latent_vae_kl_weight',
        type=float,
        default=1e-3,
        help='latent_encoder=vae translated KL translated（warmup translated）。',
    )
    parser.add_argument(
        '--latent-vae-kl-warmup-epochs',
        dest='latent_vae_kl_warmup_epochs',
        type=int,
        default=10,
        help='latent_encoder=vae translated KL translated warmup translated epoch translated。',
    )
    
    # biological validity constraints
    parser.add_argument('--use_non_negative_constraint', action='store_true', default=True, help="Use non-negative constraint for biological validity")
    parser.add_argument('--lambda_barrier', type=float, default=1.0, help="Weight for non-negative barrier loss")

    # flow matching hyperparameters
    parser.add_argument('--n_sample_steps', type=int, default=50)
    parser.add_argument(
        '--test-integration-ref-steps',
        type=int,
        default=0,
        dest='test_integration_ref_steps',
        help='translated：translated MSE(pred_n_sample_steps, pred_ref)。'
        'translated 0=translated 2×n_sample_steps；-1=translated（translated，translated nan）；'
        'translated > n_sample_steps translated ref translated。',
    )
    parser.add_argument('--use_t_bounds', action='store_true', default=False, help='Use t bounds for random sampling (default: True)')
    parser.add_argument('--t_min', type=float, default=1e-3, help='Lower bound for random t sampling in training [0,1)')
    parser.add_argument('--t_max', type=float, default=0.999, help='Upper bound for random t sampling in training (0,1]')
    parser.add_argument('--prior_sampler', type=str, default="zinb", help="gaussian | uniform | zero | zinb")
    parser.add_argument('--zinb_logits', type=float, default=0.1)
    parser.add_argument('--zinb_total_count', type=float, default=1)
    parser.add_argument('--zinb_zi_logits', type=float, default=0., help="Prob for zero inflation")  # before sigmoid
    
    # translated
    parser.add_argument('--alpha_schedule', type=str, default='cos', help="linear | quad | cos | sigm")
    parser.add_argument(
        '--t_schedule',
        type=str,
        default='linear',
        choices=['linear', 'logit_normal'],
        help='translated t translated：linear=torch.rand[0,1]；logit_normal=sigmoid(N(μ,σ))',
    )
    parser.add_argument(
        '--time-conditioning-mode',
        type=str,
        default='full',
        choices=['full', 'fixed', 'none'],
        dest='time_conditioning_mode',
        help='translated：full=translated t（translated）；'
        'fixed=translated t；none=translated 0（translated）。'
        'translated：translated，translated z_t translated target。',
    )
    parser.add_argument(
        '--time-conditioning-fixed-value',
        type=float,
        default=0.5,
        dest='time_conditioning_fixed_value',
        help='time_conditioning_mode=fixed translated t（translated clamp translated [0,1]）。',
    )
    parser.add_argument('--logit_normal_mu', type=float, default=0.0, help='LOGIT_NORMAL translated')
    parser.add_argument('--logit_normal_sigma', type=float, default=1.0, help='LOGIT_NORMAL translated')
    parser.add_argument(
        '--r',
        type=float,
        default=1.0,
       
        help="Rectified flow translated r：t' = r t / (1+(r-1)t)；1.0 translated",
    )
    parser.add_argument(
        '--flow_loss_type',
        type=str,
        default='jit',
        choices=['jit'],
        help='Only JIT flow loss is supported.',
    )
    parser.add_argument(
        '--use-noisy-gene-latent-tokens',
        dest='use_noisy_gene_latent_tokens',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='translated gene/latent token（x_t）translated；'
        'translated，translated --no-use-noisy-gene-latent-tokens translated ablation。',
    )
    parser.add_argument(
        '--sequence-gene-token-source',
        type=str,
        default='noisy',
        choices=['noisy', 'clean'],
        help="translated sequence gene/latent token translated，translated："
        "noisy=translated x_t（translated）；clean=translated x_1/z_1（translated 'xt token translated' ablation）。",
    )
    parser.add_argument(
        '--sequence-token-pairing',
        type=str,
        default='paired',
        choices=['paired', 'permuted'],
        help="sequence token translated histology condition translated（translated）："
        "paired=translated；permuted=translated (h_i, x_t,j)/(h_i, z_t,j), i!=j，"
        "translated target-side signal translated。",
    )
    parser.add_argument(
        '--sequence-aux-token-source',
        type=str,
        default='target',
        choices=['target', 'zinb_random'],
        dest='sequence_aux_token_source',
        help='sequence auxiliary token translated：target=translated target-derived x_t/z_t（translated）；'
        'zinb_random=translated ZINB token（translated x1/z1），'
        'translated target-side translated generic noise+gating translated。',
    )
    parser.add_argument(
        '--token_training_mode',
        type=str,
        default='wsi_to_st',
        choices=['wsi_to_st'],
        help='train.py translated WSI->ST translated',
    )
    # model hyperparameters
    parser.add_argument(
        '--backbone',
        type=str,
        default="vpredictor",
        help="spatial_transformer: DitFlowDenoiser + mmDiT (model/dit.py, mmdit/); "
        "vpredictor: MMDiTTransformer (model/vpredictor.py). Alias: mm_dit -> vpredictor",
    )
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--pairwise_hidden_dim', type=int, default=128)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--attn_dropout', type=float, default=0.2)
    parser.add_argument('--n_neighbors', type=int, default=8)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--feature_dim', type=int, default=1024, help="uni:1024, ciga:512")
    parser.add_argument('--norm', type=str, default='layer', help="batch | layer")
    parser.add_argument('--activation', type=str, default='swiglu', help="relu | gelu | swiglu")
    parser.add_argument(
        '--time-modulation',
        dest='time_modulation',
        type=str,
        default='concat',
        choices=['concat', 'adaln'],
        help='translated：concat=translatedQKVtranslated；adaln=translatedAdaLNtranslated/translated',
    )
    parser.add_argument(
        '--use-gated-attention',
        dest='use_gated_attention',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='translated vpredictor translated gating translated；translated self-attention（ablation）。',
    )
    
    args = parser.parse_args()

    # train.py translated ST->WSI translated，translated WSI->ST。

    args.feature_dim = {
        "uni_v1_official": 1024,
        "gigapath": 1536,
        "ciga": 512,
        "genbio_pathfm": 4608,
    }[args.feature_encoder]

    set_random_seed(args.seed, deterministic=args.deterministic)

    # translated test+translated
    exp_code = f"test_{get_current_time()}"
    save_dir = os.path.join(args.save_dir, exp_code)
    os.makedirs(save_dir, exist_ok=True)
    
    if args.use_wandb:
        wandb.init(project="spatial_transcriptomics", name=exp_code)
        wandb.config.update(args)

    print(f"Save dir: {save_dir}")
    print(args)

    if args.datasets[0] == "all":
        args.datasets = ["LUNG", "COAD", "SKCM", "PAAD", "READ", "LYMPH_IDC", "PRAD", "CCRCC", "HCC"]
        # args.datasets = ["COAD","PRAD","READ"]
    # translated
    dataset_training_times = []
    # translatedCSVtranslated
    training_times_csv_path = os.path.join(save_dir, 'dataset_training_times.csv')
    
    for dataset in args.datasets:
        args.dataset = dataset
        args.save_dir = os.path.join(save_dir, dataset)
        os.makedirs(args.save_dir, exist_ok=True)

        with open(os.path.join(args.save_dir, 'config.json'), 'w') as f:
            json.dump(vars(args), f, sort_keys=True, indent=4, default=json_default)

        # translated
        training_start_time = time()
        training_start_time_str = get_current_time()
        print(f"\n{'='*80}")
        print(f"translated: {dataset}")
        print(f"translated: {training_start_time_str}")
        print(f"{'='*80}\n")
        
        run(args)
        
        # translated
        training_end_time = time()
        training_end_time_str = get_current_time()
        training_duration = training_end_time - training_start_time
        training_hours = int(training_duration // 3600)
        training_minutes = int((training_duration % 3600) // 60)
        training_seconds = int(training_duration % 60)
        
        time_info = {
            'dataset': dataset,
            'start_time': training_start_time_str,
            'end_time': training_end_time_str,
            'duration_seconds': training_duration,
            'duration_hours': training_hours,
            'duration_minutes': training_minutes,
            'duration_seconds_remainder': training_seconds,
            'duration_formatted': f"{training_hours:02d}:{training_minutes:02d}:{training_seconds:02d}"
        }
        
        dataset_training_times.append(time_info)
        
        # translated
        dataset_time_json_path = os.path.join(args.save_dir, 'training_time.json')
        with open(dataset_time_json_path, 'w') as f:
            json.dump(time_info, f, sort_keys=True, indent=4, default=json_default)
        print(f"translated {dataset} translated: {dataset_time_json_path}")
        
        # translatedCSVtranslated
        import csv
        file_exists = os.path.exists(training_times_csv_path)
        with open(training_times_csv_path, 'a', newline='') as f_csv:
            writer = csv.writer(f_csv)
            if not file_exists:
                writer.writerow(['dataset', 'duration_seconds', 'duration_formatted', 'start_time', 'end_time'])
            writer.writerow([
                time_info['dataset'],
                f"{time_info['duration_seconds']:.2f}",
                time_info['duration_formatted'],
                time_info['start_time'],
                time_info['end_time']
            ])
        print(f"translated {dataset} translatedCSVtranslated: {training_times_csv_path}")
        
        print(f"\n{'='*80}")
        print(f"translated {dataset} translated")
        print(f"translated: {training_hours:02d}:{training_minutes:02d}:{training_seconds:02d} ({training_duration:.2f} translated)")
        print(f"translated: {training_end_time_str}")
        print(f"{'='*80}\n")

    # translatedCSVtranslated
    total_duration = sum([t['duration_seconds'] for t in dataset_training_times])
    total_hours = int(total_duration // 3600)
    total_minutes = int((total_duration % 3600) // 60)
    total_seconds = int(total_duration % 60)
    
    # translatedCSVtranslated
    import csv
    with open(training_times_csv_path, 'a', newline='') as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow([
            'total',
            f"{total_duration:.2f}",
            f"{total_hours:02d}:{total_minutes:02d}:{total_seconds:02d}",
            '',  # start_timetranslated
            ''   # end_timetranslated
        ])
    
    print(f"\ntranslated: {training_times_csv_path}")
    print(f"translated: {total_hours:02d}:{total_minutes:02d}:{total_seconds:02d} ({total_duration:.2f} translated)")

    if args.use_wandb:
        wandb.finish()
