"""Pair-train the depth head on top of the local dense backbone (fast path).

Loads a dense checkpoint (e.g. checkpoints_local/dense_best.pt), freezes
everything except depth_head, and trains depth for a few epochs with the
same depth-supervised frames used by phase_b/train.py --task depth.

Output: <out>/dense_depth_best.pt  (full state dict: dense heads unchanged +
trained depth head) - a single-file "category + distance" checkpoint.

Usage (GPU machine):
  python phase_b/train_depth_paired.py --epochs 5 --batch 64 --amp \
      --ckpt checkpoints_local/dense_best.pt --out checkpoints_local
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from dataset import PerceptionDataset, collate_perception, load_depth  # noqa
from model import PerceptionModel  # noqa
from train import scale_invariant_depth  # noqa
from shared.data_index import load_index  # noqa


def object_weight_map(batch: dict, size: int, weight: float = 3.0):
    """Per-pixel weight map emphasizing object bounding boxes.

    Weight = 1 everywhere, `weight` inside boxes (boxes are normalized in the
    800x600 capture frame; depth GT is resized to `size`x`size` the same way
    the RGB is, so box pixel coords scale with 224/800 on x and 224/600 on y).
    """
    B = batch["boxes"].shape[0]
    wmap = torch.ones(B, 1, size, size, dtype=torch.float32)
    for i in range(B):
        n = int(batch["num_boxes"][i].item())
        for b in batch["boxes"][i][:n]:
            cx, cy, bw, bh = [float(v) for v in b]
            x0 = int(max(0, (cx - bw / 2.0) * size))
            x1 = int(min(size - 1, (cx + bw / 2.0) * size))
            y0 = int(max(0, (cy - bh / 2.0) * size))
            y1 = int(min(size - 1, (cy + bh / 2.0) * size))
            if x1 > x0 and y1 > y0:
                wmap[i, 0, y0:y1 + 1, x0:x1 + 1] = weight
    return wmap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints_local/dense_best.pt")
    ap.add_argument("--out", default="checkpoints_local")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--resolution", type=int, default=224)
    ap.add_argument("--depth-size", type=int, default=224,
                    help="train target resolution for depth GT (default 224)")
    ap.add_argument("--obj-weight", type=float, default=3.0,
                    help="loss weight multiplier inside object boxes")
    ap.add_argument("--splits", default=None,
                    help="split JSON (default: no-val split)")
    ap.add_argument("--variant", choices=["small", "base"], default="small")
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--amp", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    if device.startswith("cuda"):
        print("gpu:", torch.cuda.get_device_name(0))

    import dataset as ds_mod
    ds_mod.DEPTH_SIZE = args.depth_size
    index = load_index()
    num_types = len(index["object_types"])
    num_actions = len(index["actions"])
    num_errors = len(index["error_classes"])
    dinov2_name = ("dinov2_vits14" if args.variant == "small"
                   else "dinov2_vitb14")
    model = PerceptionModel(
        num_types=num_types, num_actions=num_actions, num_errors=num_errors,
        device="cpu", img_size=args.resolution,
        head_width=args.width, dinov2_name=dinov2_name).to(device)

    sd = torch.load(args.ckpt, map_location=device)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        raise SystemExit(f"checkpoint missing keys: {missing[:10]}")
    if unexpected:
        print("unexpected keys (ignored):", unexpected[:10])
    print(f"loaded {args.ckpt} ({len(sd)} tensors)")

    # freeze everything except depth_head (dense heads stay as in the ckpt)
    for p in model.parameters():
        p.requires_grad = False
    for p in model.depth_head.parameters():
        p.requires_grad = True
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params (depth head only): {n}")

    splits = args.splits or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "data",
        "splits_noval.json")
    ds = PerceptionDataset(index, limit=args.limit, seed=args.seed,
                           require_depth=True, split_path=splits)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True,
                        num_workers=args.workers,
                        collate_fn=collate_perception, pin_memory=True)
    print(f"depth frames: {len(ds)} | batches/epoch: {len(loader)}")
    print(f"depth target resolution: {args.depth_size}x{args.depth_size} "
          f"(object weight {args.obj_weight})")

    opt = torch.optim.AdamW(model.depth_head.parameters(),
                            lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda",
                                  enabled=args.amp and "cuda" in device)
    os.makedirs(args.out, exist_ok=True)
    best = 1e9
    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        tot = 0.0
        for bi, batch in enumerate(loader):
            rgb = batch["rgb"].to(device)
            gt = batch["depth"].to(device)
            with torch.autocast("cuda", enabled=args.amp and "cuda" in device):
                pred = model.depth(rgb)
                pred = F.interpolate(
                    pred, size=(args.depth_size, args.depth_size),
                                     mode="bilinear", align_corners=False)
                mask = (gt > 0.1).float()
                wm = mask * object_weight_map(
                    batch, args.depth_size, args.obj_weight).to(device)
                wl1 = (torch.abs(pred - gt) * wm).sum() / \
                    wm.sum().clamp(min=1.0)
                loss = scale_invariant_depth(pred, gt, mask) + wl1
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.depth_head.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            tot += float(loss.detach())
            if (bi + 1) % 100 == 0:
                print(f"  ep{ep} b{bi+1}/{len(loader)} loss={loss.item():.3f}",
                      flush=True)
        avg = tot / max(bi + 1, 1)
        print(f"[depth-paired] epoch {ep} avg_loss={avg:.3f} "
              f"time={time.time()-t0:.0f}s", flush=True)
        if avg < best:
            best = avg
            torch.save(model.state_dict(),
                       os.path.join(args.out, "dense_depth_best.pt"))

    # quick MAE sanity check on a sample of frames
    model.eval()
    errs = []
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if bi >= 4:
                break
            rgb = batch["rgb"].to(device)
            gt = batch["depth"].to(device)
            pred = model.depth(rgb)
            pred = F.interpolate(
                pred, size=(args.depth_size, args.depth_size),
                                 mode="bilinear", align_corners=False)
            mask = (gt > 0.1)
            e = (pred - gt).abs()[mask]
            wm = object_weight_map(batch, args.depth_size,
                                   args.obj_weight).to(device) > 1.0
            eo = (pred - gt).abs()[wm & mask]
            if e.numel():
                errs.append({"whole": e.float().mean().item()})
            if eo.numel():
                errs[-1]["obj"] = eo.float().mean().item()
    if errs:
        wh = np.mean([x["whole"] for x in errs])
        ob = np.mean([x.get("obj", float("nan")) for x in errs])
        print(f"sanity MAE over {len(errs)*args.batch} frames: "
              f"whole={wh:.3f} m | object-box={ob:.3f} m", flush=True)
    print("saved checkpoints_local/dense_depth_best.pt")


if __name__ == "__main__":
    main()
