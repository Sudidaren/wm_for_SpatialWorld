"""Train / evaluate the occupancy completion head."""

import argparse
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from occupancy import (  # noqa: E402
    OccupancyDataset,
    OccupancyUNet,
    collect_samples,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--max-episodes", type=int, default=0)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="/home/sudidaren/lightwm_phases/checkpoints")
    args = ap.parse_args()

    from shared.data_index import load_index
    index = load_index()
    cache = os.path.abspath(os.path.join(args.out, "..", "data",
                                         "occupancy_samples.pkl"))
    if os.path.exists(cache) and args.max_episodes == 0:
        print(f"loading occupancy samples from {cache}")
        with open(cache, "rb") as f:
            samples = pickle.load(f)
    else:
        samples = collect_samples(index, max_episodes=args.max_episodes)
        if args.max_episodes == 0:
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            with open(cache, "wb") as f:
                pickle.dump(samples, f, protocol=4)
    rng = np.random.RandomState(0)
    perm = rng.permutation(len(samples))
    n_val = max(1, len(samples) // 10)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    train_ds = OccupancyDataset([samples[i] for i in train_idx])
    val_ds = OccupancyDataset([samples[i] for i in val_idx])
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=0)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = OccupancyUNet().to(device)
    print(f"train {len(train_ds)} val {len(val_ds)} params "
          f"{sum(p.numel() for p in model.parameters())} device={device}")

    ckpt = os.path.join(args.out, "occupancy_best.pt")
    if args.eval_only:
        model.load_state_dict(torch.load(ckpt, map_location=device))
        return evaluate(model, val_loader, device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best = 1e9
    # class imbalance: unseen walkable cells are rare
    pos = sum(float(s["target"].sum()) for s in samples)
    tot = sum(float(s["target"].size) for s in samples)
    pos_w = min(50.0, max(1.0, (tot - pos) / max(pos, 1.0)))
    print(f"unseen-walkable positive rate: {pos/tot:.4f} -> pos_weight={pos_w:.1f}")
    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        tot = 0.0
        for bi, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            logit = model(x)
            loss = F.binary_cross_entropy_with_logits(
                logit, y.squeeze(1), pos_weight=torch.tensor(pos_w,
                                                             device=device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
        avg = tot / max(bi + 1, 1)
        print(f"[occupancy] ep{ep} loss={avg:.4f} time={time.time()-t0:.0f}s")
        val_iou, prec, rec = evaluate(model, val_loader, device)
        print(f"  val unseen-cell IoU={val_iou:.3f} prec={prec:.3f} "
              f"rec={rec:.3f}")
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        if f1 > best:
            best = f1
            torch.save(model.state_dict(), ckpt)
    print("best val unseen F1:", best)


def evaluate(model, loader, device):
    model.eval()
    tp = fp = fn = 0
    pred_pos = gt_pos = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            prob = torch.sigmoid(model(x))
            pred = prob > 0.5
            # evaluate only on cells not yet seen (the 'guess behind obstacle' part)
            seen = x[:, 1] > 0.5
            unseen = ~seen
            yy = y.squeeze(1) > 0.5
            tp += ((pred & unseen) & yy).sum().item()
            fp += ((pred & unseen) & ~yy).sum().item()
            fn += ((~pred & unseen) & yy).sum().item()
            pred_pos += (pred & unseen).sum().item()
            gt_pos += (yy & unseen).sum().item()
    iou = tp / max(tp + fp + fn, 1)
    prec = tp / max(pred_pos, 1)
    rec = tp / max(gt_pos, 1)
    return iou, prec, rec


if __name__ == "__main__":
    main()
