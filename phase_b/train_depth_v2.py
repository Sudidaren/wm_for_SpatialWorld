#!/usr/bin/env python3
"""Retrain the monocular depth head with a scale-aware objective.

Why a v2: the shipped head was trained with

    loss = var(log pred - log gt) + 0.5 * mean|pred - gt|

The first term is **invariant to a global scale factor** (shifting every
log-prediction by a constant leaves a variance unchanged), so the head had
almost no incentive to be calibrated in metres.  Measured against recorded
ground truth the consequence is a systematic compression: on the held-out
rooms it over-predicts below 1 m (+0.18 m) and under-predicts above 2 m
(-0.31 m at 2-3 m, -0.87 m beyond 3 m); removing the per-distance bias alone
cuts the 2-3 m error from 0.34 m to 0.18 m.  A plain pixel-mean L1 does not
fix it either, because walls and floors near the camera dominate the mean.

v2 changes three things:

  * objective: mean|log p - log g| (scale-sensitive, still relative-error
    shaped) + lambda * mean|p - g| weighted by object boxes and by 1/depth so
    the far field has a voice;
  * resolution: defaults to 448.  Inference-time upscaling of the shipped
    336 model makes it *worse* (ratio 0.81 -> 0.76), i.e. the head is fused to
    its training scale, so the resolution has to be set at training time;
  * model selection: the epoch is picked by the object-centre depth MAE on a
    held-out split, which is the quantity the world model actually consumes,
    instead of the pixel-mean loss.

Training data is restricted to rooms the SpatialWorld evaluation never uses
(`data/splits_noneval.json`); the 81 evaluation rooms stay untouched and are
used afterwards as a clean held-out report set.

Usage (GPU machine):
  python phase_b/train_depth_v2.py --epochs 12 --batch 16 --amp \
      --ckpt checkpoints/small_objects_20260910/dense_depth_best.pt \
      --out checkpoints/depth_v2 --resolution 448
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from dataset import PerceptionDataset, collate_perception  # noqa: E402
from model import PerceptionModel  # noqa: E402
from shared.data_index import load_index  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def object_weight_map(batch: dict, size: int, weight: float = 3.0):
    """1 everywhere, `weight` inside GT object boxes."""
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


def depth_loss(pred, gt, obj_w, lam_metric=0.5, min_d=0.4, max_d=8.0):
    """Scale-aware log-L1 plus inverse-depth-weighted metric L1.

    ``mean|log p - log g|`` *does* penalise a global scale error (unlike the
    variance form used before), while still being gentle on large depths; the
    metric term keeps the absolute error small where it matters.
    """
    mask = ((gt > 0.1) & (gt < max_d * 1.5)).float()
    n = mask.sum().clamp(min=1.0)
    p = pred.clamp(min=0.05)
    g = gt.clamp(min=0.05)
    log_l1 = (torch.abs(torch.log(p) - torch.log(g)) * mask).sum() / n
    # far pixels get a comparable voice to near ones
    inv = (1.0 / g.clamp(min=min_d, max=max_d)) * mask
    w = obj_w * inv
    metric = (torch.abs(pred - gt) * w).sum() / w.sum().clamp(min=1.0)
    return log_l1 + lam_metric * metric, float(log_l1.detach()), float(metric.detach())


@torch.no_grad()
def validate_object_center_mae(model, loader, device, size, max_batches=60):
    """Median |pred - gt| at GT box centres -- the world model's own input."""
    model.eval()
    errs = []
    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break
        rgb = batch["rgb"].to(device)
        gt = batch["depth"].to(device)
        with torch.autocast("cuda", enabled=False):
            pred = model.depth(rgb)
            pred = F.interpolate(pred, size=(size, size),
                                 mode="bilinear", align_corners=False)
        B = gt.shape[0]
        for i in range(B):
            n = int(batch["num_boxes"][i].item())
            for b in batch["boxes"][i][:n]:
                cx, cy = float(b[0]) * size, float(b[1]) * size
                x = int(min(size - 1, max(0, round(cx))))
                y = int(min(size - 1, max(0, round(cy))))
                g = float(gt[i, 0, y, x])
                p = float(pred[i, 0, y, x])
                if 0.1 < g < 8.0 and np.isfinite(p):
                    errs.append(abs(p - g))
    return (float(np.median(errs)) if errs else float("nan"),
            float(np.mean(errs)) if errs else float("nan"),
            len(errs))


class EMA:
    def __init__(self, params, decay=0.999):
        self.decay = decay
        self.shadow = [p.detach().clone() for p in params]

    @torch.no_grad()
    def update(self, params):
        for s, p in zip(self.shadow, params):
            s.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)

    @torch.no_grad()
    def copy_to(self, params):
        for s, p in zip(self.shadow, params):
            p.copy_(s)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(
        ROOT, "checkpoints/small_objects_20260910/dense_depth_best.pt"))
    ap.add_argument("--out", default=os.path.join(ROOT, "checkpoints/depth_v2"))
    ap.add_argument("--splits", default=os.path.join(ROOT, "data/splits_noneval.json"))
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--resolution", type=int, default=448)
    ap.add_argument("--depth-size", type=int, default=448)
    ap.add_argument("--obj-weight", type=float, default=3.0)
    ap.add_argument("--lam-metric", type=float, default=0.5)
    ap.add_argument("--variant", choices=["small", "base"], default="small")
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--val-limit", type=int, default=1200)
    ap.add_argument("--jitter", type=float, default=0.0,
                    help="colour/brightness/contrast/gamma jitter strength on "
                         "training frames (0 = off). Use ~0.3 when the head has "
                         "to transfer to a room family it never trained on.")
    ap.add_argument("--scene-balanced", action="store_true",
                    help="sample scenes (not frames) uniformly, so a pool with "
                         "thousands of frames per house cannot drown a family "
                         "that only contributes a few hundred.")
    ap.add_argument("--epoch-samples", type=int, default=0,
                    help="frames per epoch (0 = one pass over the index); use "
                         "with --scene-balanced to control epoch length")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    ap.add_argument("--amp", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device} | resolution {args.resolution} | splits {args.splits}")

    import dataset as ds_mod
    ds_mod.DEPTH_SIZE = args.depth_size
    index = load_index()
    # The shipped checkpoint carries the 117-class detection vocabulary.  The
    # training index may have grown (newly collected pools can introduce new
    # object types), so take the class count from the checkpoint's own config
    # and keep the frozen heads byte-identical to it; only the depth head is
    # trained, so the class vocabulary does not matter for this run.
    num_types = len(index["object_types"])
    ckpt_json = args.ckpt + ".json"
    if os.path.isfile(ckpt_json):
        try:
            with open(ckpt_json) as fh:
                ckpt_meta = json.load(fh)
            if ckpt_meta.get("object_types"):
                num_types = len(ckpt_meta["object_types"])
                print(f"num_types from checkpoint config: {num_types} "
                      f"(index has {len(index['object_types'])})")
        except Exception:
            pass
    model = PerceptionModel(
        num_types=num_types, num_actions=len(index["actions"]),
        num_errors=len(index["error_classes"]), device="cpu",
        img_size=args.resolution, head_width=args.width,
        dinov2_name=("dinov2_vits14" if args.variant == "small"
                     else "dinov2_vitb14")).to(device)
    sd = torch.load(args.ckpt, map_location=device)
    own = model.state_dict()
    usable = {k: v for k, v in sd.items()
              if k in own and tuple(own[k].shape) == tuple(v.shape)}
    skipped = [k for k in sd if k not in usable]
    model.load_state_dict(usable, strict=False)
    print(f"init from {args.ckpt}: loaded {len(usable)} tensors, "
          f"skipped {len(skipped)} (shape/vocabulary mismatch)")

    for p in model.parameters():
        p.requires_grad = False
    for p in model.depth_head.parameters():
        p.requires_grad = True
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable (depth head only): {n:,}")

    train_ds = PerceptionDataset(index, limit=args.limit, seed=args.seed,
                                 require_depth=True, split_path=args.splits)
    train_ds.photometric_jitter = args.jitter
    # Hard gate: the evaluation rooms are the held-out report set and must never
    # be trained on.  A missing/renamed split file would otherwise make the
    # dataset silently fall back to "all frames", so verify the selection.
    eval_path = os.path.join(ROOT, "data", "eval_rooms.json")
    if os.path.isfile(eval_path):
        eval_rooms = set(json.load(open(eval_path))["rooms"])
        # Match the room id exactly.  A substring test is wrong here: the
        # evaluation set contains "FloorPlan20" and "FloorPlan21", which are
        # prefixes of the *training* rooms FloorPlan204/FloorPlan210-219, so
        # `r in scene` flagged 8832 innocent frames and the run refused to
        # start.
        def room_of(frame) -> str:
            return re.sub(r"_physics$", "",
                          str(frame.get("scene") or "").split("_")[0])

        offenders = [f for f in train_ds.frames if room_of(f) in eval_rooms]
        if offenders:
            raise SystemExit(
                f"refusing to train: {len(offenders)} frames come from evaluation "
                f"rooms (first: {offenders[0].get('scene')}). Check --splits.")
        print(f"leak check: 0/{len(train_ds.frames)} training frames from the "
              f"{len(eval_rooms)} evaluation rooms", flush=True)
    # validation wants a different scene set: reuse the same split file but the
    # val rooms are filtered in by swapping the split dict
    with open(args.splits) as fh:
        sp = json.load(fh)
    val_split = {"scene_splits": {
        "ai2thor_floorplans": {"train": sp["scene_splits"]["ai2thor_floorplans"]["val"]},
        "procthor_val": {"train": sp["scene_splits"]["procthor_val"]["val"]},
        "virtualhome": {"train": sp["scene_splits"]["virtualhome"]["val"]}}}
    val_path = os.path.join(args.out, "val_split.json")
    os.makedirs(args.out, exist_ok=True)
    with open(val_path, "w") as fh:
        json.dump(val_split, fh)
    val_ds = PerceptionDataset(index, limit=args.val_limit, seed=args.seed + 1,
                               require_depth=True, split_path=val_path)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers,
                              collate_fn=collate_perception, pin_memory=True)
    if args.scene_balanced:
        import collections as _collections
        counts = _collections.Counter(str(f.get("scene") or "") for f in train_ds.frames)
        weights = [1.0 / max(1, counts[str(f.get("scene") or "")])
                   for f in train_ds.frames]
        n_samples = args.epoch_samples or len(train_ds)
        sampler = torch.utils.data.WeightedRandomSampler(
            torch.tensor(weights, dtype=torch.double), n_samples, replacement=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch, sampler=sampler,
                                  num_workers=args.workers,
                                  collate_fn=collate_perception, pin_memory=True)
        print(f"scene-balanced sampling: {len(counts)} scenes, "
              f"{n_samples} frames/epoch")
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=4, collate_fn=collate_perception)
    print(f"train frames {len(train_ds)} | val frames {len(val_ds)}")

    opt = torch.optim.AdamW(model.depth_head.parameters(), lr=args.lr,
                            weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    amp_on = args.amp and device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_on)
    ema = EMA([p for p in model.depth_head.parameters()])

    best = (float("inf"), None)
    hist = []
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        for batch in train_loader:
            rgb = batch["rgb"].to(device, non_blocking=True)
            gt = batch["depth"].to(device, non_blocking=True)
            with torch.autocast("cuda", enabled=amp_on):
                pred = model.depth(rgb)
                pred = F.interpolate(pred, size=(args.depth_size, args.depth_size),
                                     mode="bilinear", align_corners=False)
                obj_w = object_weight_map(batch, args.depth_size,
                                          args.obj_weight).to(device)
                loss, l_log, l_met = depth_loss(pred, gt, obj_w, args.lam_metric)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.depth_head.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            ema.update([p for p in model.depth_head.parameters()])
            tot += float(loss); nb += 1
            if nb % 200 == 0:
                print(f"  ep{ep} {nb}/{len(train_loader)} loss={tot/nb:.4f} "
                      f"(log {l_log:.4f}, metric {l_met:.4f})", flush=True)
        sched.step()
        live = [p.detach().clone() for p in model.depth_head.parameters()]
        ema.copy_to([p for p in model.depth_head.parameters()])
        mae_med, mae_mean, ns = validate_object_center_mae(
            model, val_loader, device, args.depth_size)
        hist.append({"epoch": ep, "loss": tot / max(1, nb),
                     "val_obj_center_mae_median": mae_med,
                     "val_obj_center_mae_mean": mae_mean, "val_samples": ns,
                     "minutes": round((time.time() - t0) / 60, 1)})
        print(f"epoch {ep}: loss={tot/max(1,nb):.4f}  val obj-centre MAE "
              f"median {mae_med:.3f} m  mean {mae_mean:.3f} m  (n={ns})  "
              f"[{hist[-1]['minutes']} min]", flush=True)
        if mae_med < best[0]:
            best = (mae_med, ep)
            torch.save(model.state_dict(), os.path.join(args.out, "dense_depth_v2_best.pt"))
            print(f"  -> new best (median {mae_med:.3f} m), saved", flush=True)
        with torch.no_grad():
            for p, s in zip(model.depth_head.parameters(), live):
                p.copy_(s)

    meta = {"base_ckpt": args.ckpt, "splits": args.splits,
            "resolution": args.resolution, "epochs": args.epochs,
            "loss": "mean|log p - log g| + %.2f * (1/d)-weighted metric L1" % args.lam_metric,
            "obj_weight": args.obj_weight, "best_epoch": best[1],
            "best_val_obj_center_mae_median": best[0], "history": hist}
    with open(os.path.join(args.out, "train_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1, ensure_ascii=False)
    print(f"\nbest epoch {best[1]} (median object-centre MAE {best[0]:.3f} m)")
    print(f"checkpoint: {args.out}/dense_depth_v2_best.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
