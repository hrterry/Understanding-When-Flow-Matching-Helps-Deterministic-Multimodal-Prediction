"""Routing and time experiments on one compact token backbone (Section 4.3, Appendix E).

Arms (all share the backbone; the formulation and the role of x_t change):
  reg_tok        matched regression: target-side tokens are zeros, t fixed
  fm             flow, x_t as set-level tokens
  fm_wperm       per-spot fusion with x_t shuffled inside the group (pairing destroyed, set kept)
  fm_paired      per-spot fusion: each spot token receives its own x_t through a time gate
  fm_paired + xt_drop / selfcond   per-spot fusion with x_t dropout / self-conditioning

Data: `--data sim` is a synthetic task with known ground truth; `--data <cache.npz>` uses features
from analysis/cache_features.py.

    python analysis/simulation.py --arm fm --data sim --seed 0 --t-max 0.95 --loss v
"""
import argparse
import json
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_world(p, G, k, seed):
    g = np.random.default_rng(seed)
    return (g.normal(size=(p, k)) / math.sqrt(k), g.normal(size=(G, k)) / math.sqrt(k) * 1.5,
            g.normal(size=(G, p)) / math.sqrt(p) * 2.0)


def make_slides(world, n_slides, spots, k, seed, s_style=0.8, s_h=0.5, s_eps=0.5, dropout=0.3):
    """Latent programs z -> histology h (with slide style) and expression y (slide offset, zero inflation)."""
    A, W, Bm = world
    g = np.random.default_rng(seed)
    H, Y, S = [], [], []
    for s in range(n_slides):
        style = g.normal(size=A.shape[0]) * s_style
        z = g.normal(size=k) * 0.7 + g.normal(size=(spots, k))
        H.append(np.tanh(z @ A.T) + style + s_h * g.normal(size=(spots, A.shape[0])))
        mu = np.log1p(np.exp(z @ W.T + style @ Bm.T))
        Y.append(mu * (g.random(mu.shape) > dropout) + s_eps * g.normal(size=mu.shape))
        S.append(np.full(spots, s))
    f = lambda L: torch.tensor(np.concatenate(L), dtype=torch.float32)
    return f(H), f(Y), torch.tensor(np.concatenate(S))


def temb(t, dim=32):
    f = torch.exp(torch.arange(dim // 2, device=t.device) * (-math.log(1000.0) / (dim // 2)))
    a = t[:, None] * 100.0 * f[None]
    return torch.cat([a.sin(), a.cos()], -1)


class Attn(nn.Module):
    def __init__(self, d, dt, heads):
        super().__init__()
        self.h, self.hd = heads, d // heads
        self.qkv = nn.Linear(d + dt, 3 * d)
        self.proj = nn.Linear(d, d)
        self.g = nn.Parameter(torch.ones(4))

    def forward(self, x, te, M):
        B, N, D = x.shape
        q, k, v = self.qkv(torch.cat([x, te[:, None].expand(B, N, -1)], -1)) \
            .reshape(B, N, 3, self.h, self.hd).permute(2, 0, 3, 1, 4)
        a = (q @ k.transpose(-2, -1)) / math.sqrt(self.hd)
        s = torch.sigmoid(self.g)
        a = torch.cat([torch.cat([a[..., :M, :M] * s[0], a[..., :M, M:] * s[1]], -1),
                       torch.cat([a[..., M:, :M] * s[2], a[..., M:, M:] * s[3]], -1)], -2)
        return self.proj((a.softmax(-1) @ v).transpose(1, 2).reshape(B, N, D))


class Block(nn.Module):
    def __init__(self, d, dt, heads, drop):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = Attn(d, dt, heads)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(drop)

    def forward(self, x, te, M):
        x = x + self.drop(self.attn(self.n1(x), te, M))
        return x + self.drop(self.mlp(self.n2(x)))


class Net(nn.Module):
    def __init__(self, p, G, paired, d=128, L=4, heads=4, drop=0.1):
        super().__init__()
        self.paired = paired
        self.img = nn.Sequential(nn.Linear(p, d), nn.GELU(), nn.Linear(d, d))
        self.st = nn.Sequential(nn.Linear(G, d), nn.GELU(), nn.Linear(d, d))
        self.mod = nn.Parameter(torch.zeros(2, d))
        self.tm = nn.Sequential(nn.Linear(32, 32), nn.GELU())
        self.gate = nn.Linear(32, d)
        nn.init.zeros_(self.gate.weight); nn.init.zeros_(self.gate.bias)
        self.blocks = nn.ModuleList(Block(d, 32, heads, drop) for _ in range(L))
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, G))

    def forward(self, h, xt, t, keep=None):
        M = h.shape[1]
        te = self.tm(temb(t))
        hi, sx = self.img(h) + self.mod[0], self.st(xt)
        if keep is not None:
            sx = sx * keep[:, None, None]
        x = hi + torch.sigmoid(self.gate(te))[:, None] * sx if self.paired else torch.cat([hi, sx + self.mod[1]], 1)
        for b in self.blocks:
            x = b(x, te, M)
        return self.out(x[:, :M])


def groups(S, M, g, coords=None):
    """Groups of M spots from one slide: random, or spatially local when coords are given."""
    out = []
    for s in torch.unique(S).tolist():
        idx = (S == s).nonzero().squeeze(1)
        if coords is None:
            idx = idx[torch.randperm(len(idx), generator=g)]
            out.append(idx[:len(idx) // M * M].view(-1, M)); continue
        th = torch.rand(1, generator=g).item() * math.pi
        c = coords[idx]
        u = c[:, 0] * math.cos(th) + c[:, 1] * math.sin(th)
        v = -c[:, 0] * math.sin(th) + c[:, 1] * math.cos(th)
        idx, v = idx[u.argsort()], v[u.argsort()]
        strip = max(M, int(round(math.sqrt(len(idx) / M))) * M)
        for a in range(0, len(idx), strip):
            si = idx[a:a + strip][v[a:a + strip].argsort()]
            if len(si) < M:
                si = idx[-M:]
            for b in range(0, len(si), M):
                ch = si[b:b + M] if b + M <= len(si) else si[-M:]
                out.append(ch[None])
    return torch.cat(out)


def pcc(P, Y):
    P, Y = P - P.mean(0), Y - Y.mean(0)
    r = (P * Y).sum(0) / (P.norm(dim=0) * Y.norm(dim=0) + 1e-8)
    return r[Y.norm(dim=0) > 1e-6].mean().item()


@torch.no_grad()
def predict(net, arm, H, S, M, steps, dev, C=None):
    net.eval()
    grp = groups(S, M, torch.Generator().manual_seed(1234), C)
    G = net.out[-1].out_features
    P = torch.zeros(len(H), G)
    for i in range(0, len(grp), 256):
        gi = grp[i:i + 256]
        h = H[gi].to(dev)
        B, Mm = gi.shape
        if arm == "reg_tok":
            out = net(h, torch.zeros(B, Mm, G, device=dev), torch.full((B,), 0.5, device=dev))
        else:
            x = torch.randn(B, Mm, G, device=dev)
            for j in range(steps):
                t = torch.full((B,), j / steps, device=dev)
                xp = net(h, x, t)
                x = xp if j == steps - 1 else x + (xp - x) / (steps - j)
            out = x
        P[gi.reshape(-1).cpu()] = out.reshape(-1, G).float().cpu()
    cov = torch.zeros(len(H), dtype=torch.bool); cov[grp.reshape(-1)] = True
    return P, cov


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default="fm", choices=["reg_tok", "fm", "fm_wperm", "fm_paired"])
    ap.add_argument("--data", default="sim")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--slides", type=int, default=8, help="training slides (sim)")
    ap.add_argument("--t-max", type=float, default=0.95)
    ap.add_argument("--loss", default="v", choices=["v", "x"])
    ap.add_argument("--xt-drop", type=float, default=0.0)
    ap.add_argument("--selfcond", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--M", type=int, default=64)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev, Ctr, Cte = torch.device(a.device), None, None
    if a.data == "sim":
        world = make_world(64, 200, 8, 0)
        Htr, Ytr, Str = make_slides(world, a.slides, 2000, 8, a.seed * 100 + 1)
        Hte, Yte, Ste = make_slides(world, 4, 2048, 8, 999_000)
        Hva, Yva, Sva = make_slides(world, 2, 2048, 8, a.seed * 100 + 7)
    else:
        z = np.load(a.data)
        T = lambda k: torch.tensor(z[k])
        Htr, Ytr, Str, Hte, Yte, Ste, Ctr, Cte = (T(k) for k in ("H_train", "Y_train", "S_train", "H_test",
                                                                  "Y_test", "S_test", "C_train", "C_test"))
        Hva, Yva, Sva = Hte, Yte, Ste  # selection on the evaluation slides, as in the main protocol
    mu, sd = Htr.mean(0), Htr.std(0) + 1e-6
    Htr, Hte, Hva = (Htr - mu) / sd, (Hte - mu) / sd, (Hva - mu) / sd
    net = Net(Htr.shape[1], Ytr.shape[1], paired=a.arm in ("fm_paired", "fm_wperm")).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=5e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    gen, step, best = torch.Generator().manual_seed(a.seed), 0, (-1e9, None)
    H_d, Y_d = Htr.to(dev), Ytr.to(dev)
    while step < a.steps:
        grp = groups(Str, a.M, gen, Ctr)
        grp = grp[torch.randperm(len(grp), generator=gen)]
        for i in range(0, len(grp) - 15, 16):
            gi = grp[i:i + 16].to(dev)
            h, y = H_d[gi], Y_d[gi]
            B = h.shape[0]
            net.train()
            if a.arm == "reg_tok":
                loss = F.mse_loss(net(h, torch.zeros_like(y), torch.full((B,), 0.5, device=dev)), y)
            else:
                t, x0 = torch.rand(B, device=dev) * a.t_max, torch.randn_like(y)
                src = torch.roll(y, 1, dims=1) if a.arm == "fm_wperm" else y
                if a.selfcond > 0 and torch.rand(1).item() < a.selfcond:
                    with torch.no_grad():
                        src = net(h, x0, torch.zeros_like(t))
                xt = t[:, None, None] * src + (1 - t)[:, None, None] * x0
                keep = (torch.rand(B, device=dev) >= a.xt_drop).float() if a.xt_drop > 0 else None
                xp = net(h, xt, t, keep)
                w = 1.0 / (1 - t).clamp_min(1 - a.t_max) ** 2 if a.loss == "v" else torch.ones_like(t)
                loss = (w[:, None, None] * (xp - y) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            if step % 300 == 0:
                P, c = predict(net, a.arm, Hva, Sva, a.M, 10, dev, Cte if a.data != "sim" else None)
                v = pcc(P[c], Yva[c])
                if v > best[0]:
                    best = (v, {k: x.detach().clone() for k, x in net.state_dict().items()})
            if step >= a.steps:
                break
    net.load_state_dict(best[1])
    P, c = predict(net, a.arm, Hte, Ste, a.M, 50, dev, Cte)
    res = dict(vars(a), pcc=pcc(P[c], Yte[c]))
    if a.arm != "reg_tok":  # reliance on x_t: how much an oracle state would change the prediction
        grp = groups(Ste, a.M, torch.Generator().manual_seed(1234), Cte)
        O = torch.zeros_like(Yte)
        net.eval()
        with torch.no_grad():
            for i in range(0, len(grp), 256):
                gi = grp[i:i + 256]
                y = Yte[gi].to(dev)
                xt = 0.95 * y + 0.05 * torch.randn_like(y)
                O[gi.reshape(-1)] = net(Hte[gi].to(dev), xt, torch.full((y.shape[0],), 0.95, device=dev)).reshape(-1, y.shape[-1]).cpu()
        res["reliance"] = pcc(O[c], Yte[c]) - res["pcc"]
    print(json.dumps({k: res[k] for k in ("arm", "seed", "pcc") + (("reliance",) if "reliance" in res else ())}))
    if a.out:
        with open(a.out, "a") as f:
            f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
