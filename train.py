"""Train and evaluate one configuration on every predefined split of a dataset.

    python train.py --config configs/latent.yaml --dataset COAD --seed 1
    python train.py --config configs/latent.yaml --dataset COAD --set formulation=regression
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from flowst.data import gene_weights, load_split, local_batches, n_splits, shuffled_batches
from flowst.flow import flow_step, masked_mse, sample
from flowst.metrics import evaluate
from flowst.model import GatedFlowModel, GeneVAE
from flowst.qwen import QwenFlowModel


def build(cfg, n_genes, feature_dim, device):
    dim = cfg["latent_dim"] if cfg["latent"] else n_genes
    regression = cfg["formulation"] == "regression"
    kw = dict(feature_dim=feature_dim, dim=dim, hidden_dim=cfg["hidden_dim"], n_layers=cfg["n_layers"],
              n_heads=cfg["n_heads"], dropout=cfg["dropout"], attn_dropout=cfg["attn_dropout"],
              img_dropout=cfg["img_dropout"])
    if cfg["backbone"] == "qwen":
        model = QwenFlowModel(**kw)
    else:
        model = GatedFlowModel(gated=cfg["backbone"] == "gated", gene_tokens=not regression,
                               xt_tokens=not regression and cfg["xt_mode"] != "none", **kw)
    return model.to(device), (GeneVAE(n_genes, cfg["latent_dim"]).to(device) if cfg["latent"] else None)


@torch.no_grad()
def predict(cfg, model, vae, slide, device, steps=None):
    img = slide.features[None].to(device)
    if cfg["formulation"] == "regression":
        t = torch.full((1,), 0.5, device=device)
        out = model(img, torch.zeros(1, img.shape[1], model.dim, device=device), t)
    else:
        out = sample(cfg, model, img, steps or cfg["ode_steps"])
    return (vae.decode(out) if vae is not None else out)[0].cpu().numpy()


def test(cfg, model, vae, slides, device):
    model.eval()
    if vae is not None:
        vae.eval()
    P = np.concatenate([predict(cfg, model, vae, s, device) for s in slides])
    Y = np.concatenate([s.expression.numpy() for s in slides])
    C = np.concatenate([s.coords.numpy() for s in slides])
    ids = np.concatenate([np.full(len(s), i) for i, s in enumerate(slides)])
    return evaluate(P, Y, C, ids), P


def train_split(cfg, split, out, device):
    train, test_slides, genes = load_split(cfg, split)
    model, vae = build(cfg, len(genes), train[0].features.shape[1], device)
    params = list(model.parameters()) + (list(vae.parameters()) if vae else [])
    opt = torch.optim.Adam(params, lr=cfg["lr"])
    weight = gene_weights(train).to(device) if cfg["gene_weighting"] else None
    best, best_epoch = {"pcc": -1.0}, 0
    out.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        if vae is not None:
            vae.train()
        batches = (local_batches(train, cfg["M"]) if cfg["sampling"] == "local"
                   else shuffled_batches(train, cfg["M"]))
        for img, y in batches:
            img, y = img.to(device), y.to(device)
            valid = (img.sum(-1) != 0).to(y.dtype)
            mu = logvar = None
            x1 = y
            if vae is not None:
                x1, mu, logvar = vae.encode(y)
            if cfg["formulation"] == "regression":
                t = torch.full((img.shape[0],), 0.5, device=device)
                pred = model(img, torch.zeros_like(x1), t)
                pred = vae.decode(pred) if vae is not None else pred
                loss = masked_mse((pred - y) ** 2, valid, weight)
            else:
                src = None
                if cfg["xt_mode"] == "permuted":  # x_t comes from other spots: pairing destroyed
                    src = x1[:, torch.roll(torch.arange(x1.shape[1], device=device), 1)]
                loss = flow_step(cfg, model, img, x1, None if vae is not None else weight, src)
            if vae is not None:
                recon = masked_mse((vae.decode(x1) - y) ** 2, valid, weight)
                warm = min(1.0, epoch / cfg["kl_warmup_epochs"])
                loss = loss + cfg["recon_weight"] * recon + cfg["kl_weight"] * warm * vae.kl(mu, logvar)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg["clip_norm"])
            opt.step()
        if epoch % cfg["eval_every"] == 0:
            metrics, _ = test(cfg, model, vae, test_slides, device)
            print(f"split {split} epoch {epoch} " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()), flush=True)
            # Checkpoint selection on the evaluation slides, following the STFlow pipeline.
            if metrics["pcc"] > best["pcc"]:
                best, best_epoch = metrics, epoch
                torch.save({"model": model.state_dict(), "vae": vae.state_dict() if vae else None,
                            "config": cfg, "genes": genes}, out / "best.pt")
            if epoch - best_epoch >= cfg["patience"]:
                break
    best["epoch"] = best_epoch
    json.dump(best, open(out / "metrics.json", "w"), indent=2)
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--split", type=int, default=None, help="run one split only")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config values")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    cfg.update(dataset=a.dataset, seed=a.seed)
    for item in a.set:
        key, value = item.split("=", 1)
        cfg[key] = yaml.safe_load(value)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    name = Path(a.config).stem + ("_" + "_".join(s.replace("=", "-") for s in a.set) if a.set else "")
    root = Path(cfg["save_dir"]) / a.dataset / f"{name}_seed{a.seed}"
    splits = [a.split] if a.split is not None else range(n_splits(cfg))
    results = [train_split(cfg, s, root / f"split{s}", a.device) for s in splits]
    summary = {k: float(np.mean([r[k] for r in results])) for k in results[0] if k != "epoch"}
    json.dump({"mean": summary, "splits": results}, open(root / "summary.json", "w"), indent=2)
    print(a.dataset, " ".join(f"{k}={v:.4f}" for k, v in summary.items()))


if __name__ == "__main__":
    main()
