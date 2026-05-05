import torch
import numpy as np
from scipy.stats import pearsonr
import torch.nn.functional as F
import os
from flow.noise import PriorSampler


def _resolve_model_time_input(args, t_steps: torch.Tensor) -> torch.Tensor:
    """
    translated；translated/translated t。
    """
    mode = str(getattr(args, "time_conditioning_mode", "full")).lower().strip()
    if mode == "full":
        return t_steps
    if mode == "fixed":
        t_fixed = float(getattr(args, "time_conditioning_fixed_value", 0.5))
        t_fixed = min(1.0, max(0.0, t_fixed))
        return torch.full_like(t_steps, t_fixed)
    if mode == "none":
        return torch.zeros_like(t_steps)
    raise ValueError(f"Unsupported time_conditioning_mode: {mode}")


def _sample_random_zinb_tokens_like(ref: torch.Tensor, args) -> torch.Tensor:
    """
    translated ZINB token（translated x1/z1 translated），translated auxiliary token ablation。
    """
    sampler = getattr(args, "_aux_zinb_prior_sampler", None)
    if sampler is None:
        sampler = PriorSampler(
            "zinb",
            total_count=torch.tensor([args.zinb_total_count], device=ref.device),
            logits=torch.tensor([args.zinb_logits], device=ref.device),
            zi_logits=args.zinb_zi_logits,
        )
        setattr(args, "_aux_zinb_prior_sampler", sampler)
    sampled = sampler.sample(tuple(ref.shape)).to(ref.device)
    sampled = torch.log(sampled + 1.0)
    return sampled.to(dtype=ref.dtype)


def _flow_ode_step(
    raw_pred,
    exp_t1,
    t_cur=None,
    t_next=None,
):
    """
    JIT-only integration step:
      net predicts x_pred, v=(x_pred-z)/(1-t), z_{t+dt}=z_t + (t_next-t)*v
    """
    if t_cur is None or t_next is None:
        raise ValueError("JIT flow requires both t_cur and t_next.")
    denom = (1.0 - t_cur).clamp_min(1e-4)[:, None, None]
    v_pred = (raw_pred - exp_t1) / denom
    dt_local = (t_next - t_cur)[:, None, None]
    return exp_t1 + dt_local * v_pred


def _integrate_one_patch(
    args,
    diffusier,
    model,
    img_features,
    coords,
    exp_t0,
    num_steps,
    latent_ae,
    use_latent,
    collect_delta_t=True,
):
    """
    translated exp_t0（translated）translated [0,1] translated num_steps translated，translated（latent translated decode）。
    """
    exp_t1 = exp_t0.clone()
    S = int(num_steps)
    t_schedule_grid = diffusier.get_sampler_timesteps(S, device=args.device)
    delta_t_values = []
    for s in range(S):
        exp_t_prev = exp_t1.clone()
        t_tensor = t_schedule_grid[s].expand(exp_t1.shape[0]).contiguous()
        if s < S - 1:
            t_next_tensor = t_schedule_grid[s + 1].expand(exp_t1.shape[0]).contiguous()
        else:
            t_next_tensor = torch.ones_like(t_tensor)
        t_model_tensor = _resolve_model_time_input(args, t_tensor)
        sequence_aux_token_source = str(
            getattr(args, "sequence_aux_token_source", "target")
        ).lower().strip()
        if sequence_aux_token_source not in ("target", "zinb_random"):
            raise ValueError(
                f"Unsupported sequence_aux_token_source: {sequence_aux_token_source}"
            )
        sequence_token_input = exp_t1
        if sequence_aux_token_source == "zinb_random":
            sequence_token_input = _sample_random_zinb_tokens_like(exp_t1, args)
        inference_kwargs = {"predict": True}
        try:
            raw_pred = model.inference(
                sequence_token_input,
                img_features,
                coords,
                t_model_tensor,
                **inference_kwargs,
            )
        except TypeError:
            raw_pred = model.inference(
                sequence_token_input, img_features, coords, t_model_tensor, predict=True
            )
        exp_t1 = _flow_ode_step(
            raw_pred,
            exp_t1,
            t_cur=t_tensor,
            t_next=t_next_tensor,
        )
        if collect_delta_t:
            delta_t_norm = torch.norm(exp_t1 - exp_t_prev, p=2, dim=-1) ** 2
            delta_t_values.append(delta_t_norm.mean().item())
    sample = exp_t1
    if use_latent:
        sample = latent_ae.decode(sample)
    return sample, delta_t_values


def metric_func(preds_all: np.ndarray, y_test: np.ndarray, genes: list):
    errors = []
    mae_errors = []
    r2_scores = []
    pearson_corrs = []
    pearson_genes = []
    
    n_nan_genes = 0
    for i, target in enumerate(range(y_test.shape[1])):
        preds = preds_all[:, target]
        target_vals = y_test[:, target]

        errors.append(float(np.mean((preds - target_vals) ** 2)))
        mae_errors.append(float(np.mean(np.abs(preds - target_vals))))
        r2_scores.append(float(1 - np.sum((target_vals - preds) ** 2) / np.sum((target_vals - np.mean(target_vals)) ** 2)))
        pearson_corr, _ = pearsonr(target_vals, preds)
        pearson_corrs.append(pearson_corr)

        if np.isnan(pearson_corr):
            n_nan_genes += 1

        score_dict = {
            'name': genes[i],
            'pearson_corr': pearson_corr,
        }
        pearson_genes.append(score_dict)

    if n_nan_genes > 0:
        print(f"Warning: {n_nan_genes} genes have NaN Pearson correlation")

    # translated MSE translated MAE
    mse_overall = float(np.mean((preds_all - y_test) ** 2))
    mae_overall = float(np.mean(np.abs(preds_all - y_test)))

    return {'l2_errors': list(errors), 
            'mae_errors': list(mae_errors),
            'r2_scores': list(r2_scores),
            'pearson_corrs': pearson_genes,
            'pearson_mean': float(np.mean(pearson_corrs)),
            'pearson_std': float(np.std(pearson_corrs)),
            'l2_error_q1': float(np.percentile(errors, 25)),
            'l2_error_q2': float(np.median(errors)),
            'l2_error_q3': float(np.percentile(errors, 75)),
            'mae_error_q1': float(np.percentile(mae_errors, 25)),
            'mae_error_q2': float(np.median(mae_errors)),
            'mae_error_q3': float(np.percentile(mae_errors, 75)),
            'r2_score_q1': float(np.percentile(r2_scores, 25)),
            'r2_score_q2': float(np.median(r2_scores)),
            'r2_score_q3': float(np.percentile(r2_scores, 75)),
            'mse_overall': mse_overall,
            'mae_overall': mae_overall
        }


def _load_genepanel_mapping_cache(args):
    cache_path = (getattr(args, "genepanel_mapping_cache", "") or "").strip()
    if cache_path == "" or (not os.path.exists(cache_path)):
        return None
    try:
        z = np.load(cache_path, allow_pickle=False)
        return {k: z[k].astype(np.int64) for k in z.files}
    except Exception as e:
        print(f"[warning] translated genepanel mapping translated: {cache_path} | {e}")
        return None


def _subset_eval_genes(cur_pred: np.ndarray, cur_gt: np.ndarray, gene_list: list, valid_idx: np.ndarray | None):
    if valid_idx is None:
        return cur_pred, cur_gt, gene_list
    valid_idx = np.asarray(valid_idx, dtype=np.int64)
    valid_idx = valid_idx[(valid_idx >= 0) & (valid_idx < cur_pred.shape[1])]
    if valid_idx.size == 0:
        return cur_pred[:, :0], cur_gt[:, :0], []
    eval_genes = [gene_list[int(i)] for i in valid_idx]
    return cur_pred[:, valid_idx], cur_gt[:, valid_idx], eval_genes


@torch.no_grad()
def test(args, diffusier, model, loader_list, return_all=False, latent_ae=None):
    model.eval()
    if latent_ae is not None:
        latent_ae.eval()
    use_latent = getattr(args, "use_latent_flow", False) and latent_ae is not None
    all_pred, all_gt = [], []
    all_delta_t_values = []  # translated Δ_t translated
    all_integration_mse = []  # translated/translated（translated），translated
    res_dict = {}
    genepanel_mapping = _load_genepanel_mapping_cache(args)
    use_sample_panel_eval = bool(getattr(args, "use_union_genepanel", False)) and (
        genepanel_mapping is not None
    )
    S_main = int(args.n_sample_steps)
    ref_arg = int(getattr(args, "test_integration_ref_steps", 0) or 0)
    # 0：translated 2× translated（translated nan）；-1：translated（translated）；>0：translated（translated > n_sample_steps translated）
    if ref_arg < 0:
        S_ref = S_main
        compute_int_err = False
    elif ref_arg == 0:
        S_ref = max(S_main + 1, 2 * S_main)
        compute_int_err = S_main > 0 and S_ref > S_main
    else:
        S_ref = ref_arg
        compute_int_err = S_ref > S_main

    for loader in loader_list:
        cur_pred, cur_gt = [], []
        cur_integration_mse = []

        for step, batch in enumerate(loader):
            batch = [x.to(args.device) for x in batch]
            img_features, coords, labels = batch
            assert img_features.shape[0] == 1, "Batch size must be 1 for inference"

            if use_latent:
                B, M, _ = labels.shape
                prior_shape = (B, M, int(getattr(args, "latent_dim", 32)))
                prior = diffusier.sample_from_prior_latent(prior_shape).to(args.device)
            else:
                prior = diffusier.sample_from_prior(labels.shape).to(args.device)

            sample, delta_t_values = _integrate_one_patch(
                args,
                diffusier,
                model,
                img_features,
                coords,
                prior,
                S_main,
                latent_ae,
                use_latent,
                collect_delta_t=True,
            )

            if compute_int_err:
                sample_ref, _ = _integrate_one_patch(
                    args,
                    diffusier,
                    model,
                    img_features,
                    coords,
                    prior,
                    S_ref,
                    latent_ae,
                    use_latent,
                    collect_delta_t=False,
                )
                pad_mask = img_features.sum(-1) == 0
                int_mse = F.mse_loss(sample[~pad_mask], sample_ref[~pad_mask]).item()
                cur_integration_mse.append(int_mse)
                all_integration_mse.append(int_mse)

            if len(delta_t_values) > 0:
                all_delta_t_values.append(delta_t_values.copy())
            
            cur_pred.append(sample.squeeze(0).cpu().numpy())
            cur_gt.append(labels.squeeze(0).cpu().numpy())
        
        # test the performance on each dataset
        cur_pred = np.concatenate(cur_pred, axis=0)
        cur_gt = np.concatenate(cur_gt, axis=0)
        valid_idx = None
        if use_sample_panel_eval:
            valid_idx = genepanel_mapping.get(str(loader.dataset.name), None)
        cur_pred_eval, cur_gt_eval, eval_genes = _subset_eval_genes(
            cur_pred, cur_gt, loader.dataset.gene_list, valid_idx
        )
        if cur_pred_eval.shape[1] == 0:
            cur_res_dict = {
                "l2_errors": [],
                "mae_errors": [],
                "r2_scores": [],
                "pearson_corrs": [],
                "pearson_mean": float("nan"),
                "pearson_std": float("nan"),
                "l2_error_q1": float("nan"),
                "l2_error_q2": float("nan"),
                "l2_error_q3": float("nan"),
                "mae_error_q1": float("nan"),
                "mae_error_q2": float("nan"),
                "mae_error_q3": float("nan"),
                "r2_score_q1": float("nan"),
                "r2_score_q2": float("nan"),
                "r2_score_q3": float("nan"),
                "mse_overall": float("nan"),
                "mae_overall": float("nan"),
            }
        else:
            cur_res_dict = metric_func(cur_pred_eval, cur_gt_eval, eval_genes)
        cur_res_dict.update({'n_test': len(cur_gt)})
        cur_res_dict.update({'n_genes_eval': int(cur_pred_eval.shape[1])})
        if len(cur_integration_mse) > 0:
            cur_res_dict['integration_error_mse'] = float(np.mean(cur_integration_mse))
        else:
            cur_res_dict['integration_error_mse'] = float('nan')
        res_dict[loader.dataset.name] = cur_res_dict

        all_pred.append(cur_pred)
        all_gt.append(cur_gt)

    # test the performance on all datasets（translated union genepanel translated gene translated）
    all_pred = np.concatenate(all_pred, axis=0)
    all_gt = np.concatenate(all_gt, axis=0)
    if use_sample_panel_eval:
        agg_l2, agg_mae, agg_r2, agg_pearson = [], [], [], []
        agg_pearson_items = []
        total_sq_err, total_abs_err, total_elem = 0.0, 0.0, 0
        total_spots = 0
        for loader in loader_list:
            d = res_dict[loader.dataset.name]
            agg_l2.extend(d.get("l2_errors", []))
            agg_mae.extend(d.get("mae_errors", []))
            agg_r2.extend(d.get("r2_scores", []))
            cur_items = d.get("pearson_corrs", [])
            agg_pearson_items.extend(cur_items)
            agg_pearson.extend([x.get("pearson_corr", float("nan")) for x in cur_items])
            total_spots += int(d.get("n_test", 0))

        # translated loader translated MSE/MAE（translated gene translated）
        for loader in loader_list:
            # translated return_all translated，translated dataset translated
            # translated per-loader translated mse/mae（translated）
            d = res_dict[loader.dataset.name]
            mse_d = d.get("mse_overall", float("nan"))
            mae_d = d.get("mae_overall", float("nan"))
            n_spots = int(d.get("n_test", 0))
            n_genes_eval = int(d.get("n_genes_eval", 0))
            n_elem = n_spots * n_genes_eval
            if n_elem > 0 and not np.isnan(mse_d):
                total_sq_err += float(mse_d) * float(n_elem)
                total_elem += int(n_elem)
            if n_elem > 0 and not np.isnan(mae_d):
                total_abs_err += float(mae_d) * float(n_elem)

        cur_res_dict = {
            "l2_errors": list(agg_l2),
            "mae_errors": list(agg_mae),
            "r2_scores": list(agg_r2),
            "pearson_corrs": list(agg_pearson_items),
            "pearson_mean": float(np.mean(agg_pearson)) if len(agg_pearson) > 0 else float("nan"),
            "pearson_std": float(np.std(agg_pearson)) if len(agg_pearson) > 0 else float("nan"),
            "l2_error_q1": float(np.percentile(agg_l2, 25)) if len(agg_l2) > 0 else float("nan"),
            "l2_error_q2": float(np.median(agg_l2)) if len(agg_l2) > 0 else float("nan"),
            "l2_error_q3": float(np.percentile(agg_l2, 75)) if len(agg_l2) > 0 else float("nan"),
            "mae_error_q1": float(np.percentile(agg_mae, 25)) if len(agg_mae) > 0 else float("nan"),
            "mae_error_q2": float(np.median(agg_mae)) if len(agg_mae) > 0 else float("nan"),
            "mae_error_q3": float(np.percentile(agg_mae, 75)) if len(agg_mae) > 0 else float("nan"),
            "r2_score_q1": float(np.percentile(agg_r2, 25)) if len(agg_r2) > 0 else float("nan"),
            "r2_score_q2": float(np.median(agg_r2)) if len(agg_r2) > 0 else float("nan"),
            "r2_score_q3": float(np.percentile(agg_r2, 75)) if len(agg_r2) > 0 else float("nan"),
            "mse_overall": float(total_sq_err / total_elem) if total_elem > 0 else float("nan"),
            "mae_overall": float(total_abs_err / total_elem) if total_elem > 0 else float("nan"),
        }
        cur_res_dict.update({'n_test': total_spots})
    else:
        cur_res_dict = metric_func(all_pred, all_gt, loader_list[0].dataset.gene_list)
        cur_res_dict.update({'n_test': len(all_gt)})
    if len(all_integration_mse) > 0:
        cur_res_dict['integration_error_mse'] = float(np.mean(all_integration_mse))
    else:
        cur_res_dict['integration_error_mse'] = float('nan')

    delta_t_means = []
    if len(all_delta_t_values) > 0:
        # all_delta_t_values translated，translated Δ_t translated
        n_steps = len(all_delta_t_values[0])
        for step_idx in range(n_steps):
            step_values = [sample_deltas[step_idx] for sample_deltas in all_delta_t_values if step_idx < len(sample_deltas)]
            if len(step_values) > 0:
                delta_t_means.append(np.mean(step_values))
        
        cur_res_dict['delta_t_trajectory'] = delta_t_means  # translated Δ_t
        cur_res_dict['delta_t_mean'] = np.mean(delta_t_means) if len(delta_t_means) > 0 else float('nan')
        cur_res_dict['delta_t_std'] = np.std(delta_t_means) if len(delta_t_means) > 0 else float('nan')
        cur_res_dict['delta_t_max'] = np.max(delta_t_means) if len(delta_t_means) > 0 else float('nan')
        cur_res_dict['delta_t_min'] = np.min(delta_t_means) if len(delta_t_means) > 0 else float('nan')

    res_dict["all"] = cur_res_dict
    if return_all:
        return res_dict, {
            'preds_all': all_pred,
            'targets_all': all_gt,
            'delta_t_trajectory': delta_t_means,
        }
    return res_dict
