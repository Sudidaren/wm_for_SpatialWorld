"""Train the VoI gate scoring head and evaluate offline interception."""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from extract_candidates import CAND_TYPES, TYPE_ID, load  # noqa: E402


class GateNet(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1))

    def forward(self, x):
        return self.mlp(x).squeeze(-1)


def split_episodes(ds, frac=0.8, seed=0):
    eps = sorted(set(ds["episode"]))
    rng = np.random.RandomState(seed)
    rng.shuffle(eps)
    n = int(len(eps) * frac)
    train_eps, val_eps = set(eps[:n]), set(eps[n:])
    tr = {k: np.asarray([ds[k][i] for i in range(len(ds["episode"]))
                         if ds["episode"][i] in train_eps])
          for k in ("X", "y", "ctype")}
    va = {k: np.asarray([ds[k][i] for i in range(len(ds["episode"]))
                         if ds["episode"][i] in val_eps])
          for k in ("X", "y", "ctype")}
    return tr, va


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/sudidaren/lightwm_phases/data/gate_fd.npz")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--out", default="/home/sudidaren/lightwm_phases/checkpoints/gate.pt")
    args = ap.parse_args()

    ds = load(args.data)
    tr, va = split_episodes(ds)
    dim = tr["X"].shape[1]
    model = GateNet(dim)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    Xt, yt = torch.from_numpy(tr["X"]), torch.from_numpy(tr["y"])
    Xv, yv = torch.from_numpy(va["X"]), torch.from_numpy(va["y"])
    print(f"train rows {len(Xt)} ({yt.mean():.3f} pos) | val rows "
          f"{len(Xv)} ({yv.mean():.3f} pos)")
    n = len(Xt)
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            logit = model(Xt[idx])
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logit, yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
        model.eval()
        with torch.no_grad():
            vlogit = model(Xv)
            acc = ((vlogit > 0) == (yv > 0.5)).float().mean()
            print(f"ep{ep} loss={tot / max(i + 1, 1):.4f} val_acc={acc:.3f}")
    torch.save(model.state_dict(), args.out)
    print("saved", args.out)
    eval_gate(model, va, ds)


def eval_gate(model, va, ds):
    """Offline interception at multiple thresholds vs rule baseline."""
    Xv, yv, cv = torch.from_numpy(va["X"]), va["y"], va["ctype"]
    model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(model(Xv)).numpy()
    total_err = ds["n_errors"] * len(va["X"]) / max(1, len(ds["X"]))
    print(f"\n== gate offline evaluation (val candidates={len(yv)}) ==")
    for th in (0.3, 0.5, 0.6, 0.7, 0.8):
        fire = scores >= th
        prevented = (fire & (yv > 0.5)).sum()
        fa = (fire & (yv <= 0.5)).sum()
        rule_hits = yv.sum()
        print(f"  th={th:.1f}: issued={fire.sum()} prevented={prevented:.0f} "
              f"false_alarm={fa:.0f} precision={prevented / max(1, fire.sum()):.2f} "
              f"recall={prevented / max(1, rule_hits):.2f}")
    print("rule-baseline recall (hits) =", yv.sum(), "of", len(yv),
          "candidate rows; total errors:", ds["n_errors"])


if __name__ == "__main__":
    main()
