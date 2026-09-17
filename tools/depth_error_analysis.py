#!/usr/bin/env python3
"""Root-cause analysis of the monocular depth head's error.

Splits the error into the parts that need different fixes:

  * systematic bias      -- median(pred-gt) per bucket; a scale/shift problem,
                            which multi-view triangulation can NOT average out
  * random scatter       -- spread after removing the bucket bias; this one
                            *is* reduced by triangulating several views
  * sampling error       -- bbox centre vs mask centroid vs mask median
  * error floor          -- how much depth GT depth itself varies across the
                            object's own mask (a perfect pixel-level model
                            still cannot beat that when we ask for "the"
                            depth of an object)
  * seen vs unseen rooms -- the head was trained on a scene split; the pool
                            read here mixes both, so the split is reported

Usage:
    python3 tools/depth_error_analysis.py --frames 400
    python3 tools/depth_error_analysis.py --frames 400 --keep-aspect
    python3 tools/depth_error_analysis.py --frames 400 --resolution 448
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CAMERA_Y = 0.675


def camera_basis(yaw, horizon=0.0):
    rad = math.radians(yaw)
    fwd_h = np.array([math.sin(rad), 0.0, math.cos(rad)])
    right = np.array([math.cos(rad), 0.0, -math.sin(rad)])
    up = np.array([0.0, 1.0, 0.0])
    th = math.radians(-float(horizon or 0.0))
    if abs(th) > 1e-9:
        c, s = math.cos(th), math.sin(th)
        return fwd_h * c + up * s, right, up * c - fwd_h * s
    return fwd_h, right, up


def mask_pixels(seg, bbox):
    """Pixels of the object's dominant instance colour inside its bbox."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(seg.shape[1], x2), min(seg.shape[0], y2)
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None
    crop = seg[y1:y2, x1:x2]
    flat = crop.reshape(-1, 3)
    colours, counts = np.unique(flat, axis=0, return_counts=True)
    for idx in np.argsort(-counts)[:3]:
        colour = colours[idx]
        if colour.max() == 0:
            continue
        local = np.all(crop == colour, axis=-1)
        ys, xs = np.nonzero(local)
        if len(xs) < 12:
            continue
        return xs + x1, ys + y1
    return None


def gt_position(object_id):
    parts = str(object_id).split("|")
    if len(parts) != 4:
        return None
    try:
        return np.array([float(parts[1]), float(parts[2]), float(parts[3])])
    except ValueError:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/d/lightwm_data")
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--frames", type=int, default=400)
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints/small_objects_20260910/dense_depth_best.pt"))
    ap.add_argument("--resolution", type=int, default=0)
    ap.add_argument("--keep-aspect", action="store_true",
                    help="resize preserving 4:3 instead of square squashing")
    ap.add_argument("--rooms", choices=("all", "train", "test"), default="all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from phase_b.perception_runtime import PerceptionRuntime
    rt = PerceptionRuntime(args.ckpt, device=device, zoom=False)
    native = rt.resolution
    if args.resolution:
        rt.resolution = int(args.resolution)
    print(f"depth head {args.ckpt}")
    print(f"native {native} -> using {rt.resolution}, keep_aspect={args.keep_aspect}, device={device}")

    # room split of the training pipeline (held-out test rooms)
    split = json.loads((ROOT / "data/splits_v2.json").read_text())
    test_rooms = set(split["scene_splits"]["ai2thor_floorplans"]["test"])
    train_rooms = set(split["scene_splits"]["ai2thor_floorplans"]["train"])

    rows = []
    frames_done = 0
    skip = collections.Counter()
    eps = sorted(p for p in (Path(args.root) / "episodes").iterdir()
                 if (p / "episode.json").exists())
    import random
    random.Random(args.seed).shuffle(eps)
    if args.rooms != "all":
        keep = test_rooms if args.rooms == "test" else train_rooms
        eps = [p for p in eps if p.name.split("__")[0].split("_")[0] in keep]
    print(f"episodes used: {len(eps)} (rooms={args.rooms})")
    for ep in eps:
        if frames_done >= args.frames:
            break
        meta = json.loads((ep / "episode.json").read_text())
        scene = str(meta.get("scene", ""))
        room = scene.split("_")[0]
        seen = "unseen" if room in test_rooms else ("seen" if room in train_rooms else "other")
        for frame in meta.get("frames", []):
            if frames_done >= args.frames:
                break
            agent = frame.get("agent") or {}
            pos = agent.get("position") or {}
            if pos.get("x") is None:
                skip["no_pose"] += 1
                continue
            try:
                rgb = np.asarray(Image.open(ep / frame["rgb"]).convert("RGB"))
                gtd = np.asarray(Image.open(ep / frame["depth"])).astype(np.float64)
                seg = np.asarray(Image.open(ep / frame["seg"]).convert("RGB"))
            except Exception:
                skip["unreadable"] += 1
                continue
            if gtd.ndim == 3:
                gtd = gtd[..., 0]
            gtd = gtd / 1000.0
            h, w = rgb.shape[:2]

            # --- run the head ---
            if args.keep_aspect:
                r = rt.resolution
                size = (r, int(round(r * w / h)))
                img = Image.fromarray(rgb).resize(size)
                x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255
                                     ).permute(2, 0, 1)[None].to(device)
                with torch.inference_mode(), torch.autocast(device.type, enabled=False):
                    d = rt.model.depth(x)
                pred = np.asarray(Image.fromarray(
                    d[0, 0].float().cpu().numpy()).resize((w, h), Image.BILINEAR))
            else:
                with torch.inference_mode():
                    out = rt(rgb)
                pred = out["depth"] if isinstance(out, dict) else out
                pred = np.asarray(pred if not isinstance(pred, torch.Tensor)
                                  else pred.cpu().numpy(), dtype=np.float64)
                if pred.shape != (h, w):
                    pred = np.asarray(Image.fromarray(pred).resize((w, h), Image.BILINEAR))
            frames_done += 1

            yaw = float((agent.get("rotation") or {}).get("y") or 0.0)
            horizon = float(agent.get("cameraHorizon") or 0.0)
            fwd, right, up = camera_basis(yaw, horizon)
            cam = np.array([pos["x"], (pos.get("y") or 0.0) + CAMERA_Y, pos["z"]])
            fx = (w / 2.0) / math.tan(math.radians(60.0) / 2.0)
            cx, cy = w / 2.0, h / 2.0

            for obj in frame.get("visible_objects") or []:
                gt = gt_position(obj.get("objectId"))
                if gt is None:
                    continue
                pix = mask_pixels(seg, obj.get("bbox") or [])
                if pix is None:
                    continue
                xs, ys = pix
                gvals = gtd[ys, xs]
                pvals = pred[ys, xs]
                ok = np.isfinite(gvals) & np.isfinite(pvals) & (gvals > 0.1) & (gvals < 8)
                if ok.sum() < 12:
                    skip["bad_depth"] += 1
                    continue
                gvals, pvals = gvals[ok], pvals[ok]
                d = gt - cam
                z_axis = float(np.dot(d, fwd))
                if z_axis <= 0.2:
                    continue
                ui = int(round(cx + fx * float(np.dot(d, right)) / z_axis))
                vi = int(round(cy - fx * float(np.dot(d, up)) / z_axis))
                rows.append(dict(
                    seen=seen, scene=scene, name=str(obj.get("objectId", "")).split("|")[0],
                    gt_pixel=float(np.median(gvals)),            # depth at the sampled pixels
                    pred_pixel=float(np.median(pvals)),
                    gt_center_png=(float(gtd[vi, ui])
                                   if 0 <= ui < w and 0 <= vi < h else float("nan")),
                    gt_geometric=z_axis,                          # true optical-axis depth
                    spread=float(np.percentile(gvals, 90) - np.percentile(gvals, 10)),
                    area=int(ok.sum()), horizon=horizon,
                    bbox=list(obj.get("bbox") or []),
                ))

    if not rows:
        print("no samples", skip); return 1
    def med(k, rs=None):
        rs = rows if rs is None else rs
        return statistics.median(r[k] for r in rs)

    err = [abs(r["pred_pixel"] - r["gt_pixel"]) for r in rows]
    bias = [r["pred_pixel"] - r["gt_pixel"] for r in rows]
    ratio = [r["pred_pixel"] / r["gt_pixel"] for r in rows]
    print(f"\nframes {frames_done}   objects {len(rows)}   skipped {dict(skip)}")
    print(f"single-pixel style metric: median|err| {statistics.median(err):.3f} m, "
          f"median bias {statistics.median(bias):+.3f} m, median ratio {statistics.median(ratio):.3f}")

    # ---- error floor: how much GT depth varies inside one object ----
    print("\n[误差下限] 物体自身 mask 内的 GT 深度离散度（对物体用像素级真值也吃不到这个）:")
    for lo, hi, lab in ((0, 1000, "<1k px"), (1000, 5000, "1k-5k px"),
                        (5000, 20000, "5k-20k px"), (20000, 10**9, ">20k px")):
        rs = [r for r in rows if lo <= r["area"] < hi]
        if rs:
            print(f"  {lab:<10} n={len(rs):>5}   中位 p90-p10 = {med('spread', rs):.3f} m")

    # ---- bias vs random ----
    print("\n[偏差 vs 随机] 按真值距离分桶:")
    print(f"  {'bucket':<10}{'n':>6}{'median|err|':>12}{'bias':>9}{'ratio':>8}"
          f"{'去掉中位偏差后|err|':>20}")
    buckets = collections.defaultdict(list)
    for r in rows:
        g = r["gt_pixel"]
        b = "<0.5m" if g < 0.5 else ("0.5-1m" if g < 1 else ("1-2m" if g < 2 else
                                      ("2-3m" if g < 3 else ">3m")))
        buckets[b].append(r)
    for b in ("<0.5m", "0.5-1m", "1-2m", "2-3m", ">3m"):
        rs = buckets.get(b)
        if not rs:
            continue
        bb = statistics.median(r["pred_pixel"] - r["gt_pixel"] for r in rs)
        rem = statistics.median(abs(r["pred_pixel"] - r["gt_pixel"] - bb) for r in rs)
        print(f"  {b:<10}{len(rs):>6}{med('gt_pixel', rs) and statistics.median(abs(r['pred_pixel']-r['gt_pixel']) for r in rs):>12.3f}"
              f"{bb:>+9.3f}{statistics.median(r['pred_pixel']/r['gt_pixel'] for r in rs):>8.3f}{rem:>20.3f}")

    # ---- sampling: bbox centre vs mask median, and vs geometric truth ----
    ok = [r for r in rows if math.isfinite(r["gt_center_png"]) and r["gt_center_png"] > 0.1]
    print(f"\n[采样口径] n={len(ok)}")
    print(f"  mask 中位深度  vs 框中心像素深度: 中位差 "
          f"{statistics.median(abs(r['gt_pixel']-r['gt_center_png']) for r in ok):.3f} m")
    print(f"  几何真值(物体中心沿光轴) vs mask 中位深度: 中位差 "
          f"{statistics.median(abs(r['gt_geometric']-r['gt_pixel']) for r in ok):.3f} m")
    print(f"  用几何真值当答案时，头部误差: 中位 "
          f"{statistics.median(abs(r['pred_pixel']-r['gt_geometric']) for r in ok):.3f} m "
          f"(bias {statistics.median(r['pred_pixel']-r['gt_geometric'] for r in ok):+.3f})")

    # ---- seen vs unseen rooms ----
    print("\n[房间] 训练见过的房间 vs 留出房间:")
    for tag in ("seen", "unseen", "other"):
        rs = [r for r in rows if r["seen"] == tag]
        if not rs:
            continue
        print(f"  {tag:<8} n={len(rs):>5}  中位|err| "
              f"{statistics.median(abs(r['pred_pixel']-r['gt_pixel']) for r in rs):.3f} m  "
              f"bias {statistics.median(r['pred_pixel']-r['gt_pixel'] for r in rs):+.3f}  "
              f"ratio {statistics.median(r['pred_pixel']/r['gt_pixel'] for r in rs):.3f}")

    # ---- horizon ----
    print("\n[俯仰] 相机 horizon 分桶:")
    for lo, hi, lab in ((-1, 5, "水平"), (5, 40, "低头"), (40, 95, "大幅低头")):
        rs = [r for r in rows if lo <= r["horizon"] < hi]
        if rs:
            print(f"  {lab:<8} n={len(rs):>5}  中位|err| "
                  f"{statistics.median(abs(r['pred_pixel']-r['gt_pixel']) for r in rs):.3f} m "
                  f"bias {statistics.median(r['pred_pixel']-r['gt_pixel'] for r in rs):+.3f}")

    # ---- worst classes ----
    print("\n[类别] 误差最大的（n≥20）:")
    per = collections.defaultdict(list)
    for r in rows:
        per[r["name"]].append(r)
    stats = [(statistics.median(abs(x["pred_pixel"]-x["gt_pixel"]) for x in v), k, len(v))
             for k, v in per.items() if len(v) >= 20]
    for m, k, n in sorted(stats, reverse=True)[:12]:
        print(f"  {k:<20}{m:.3f} m (n={n})")

    if args.json:
        Path(args.json).write_text(json.dumps({"rows": rows[:20000]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
