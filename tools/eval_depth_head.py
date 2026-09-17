#!/usr/bin/env python3
"""Measure the monocular depth head against recorded ground truth.

For every frame of the recorded episodes we take each visible object, find a
pixel that really belongs to it (modal instance colour inside its bbox in the
segmentation frame), and compare

    predicted depth at that pixel   (the shipped depth head)
    ground-truth depth at that pixel (AI2-THOR's depth frame, metric)

broken down by distance and by object size, and with the signed bias, so a
global scale/shift error is visible rather than hidden inside an MAE.

This is the number every later change has to move:

    python3 tools/eval_depth_head.py --frames 600
    python3 tools/eval_depth_head.py --frames 600 --resolution 448
    python3 tools/eval_depth_head.py --frames 600 --calibrate
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def object_pixel(seg, bbox):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(seg.shape[1], x2), min(seg.shape[0], y2)
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None
    crop = seg[y1:y2, x1:x2].reshape(-1, 3)
    colours, counts = np.unique(crop, axis=0, return_counts=True)
    for idx in np.argsort(-counts)[:3]:
        colour = colours[idx]
        if colour.max() == 0:
            continue
        mask = np.all(seg[y1:y2, x1:x2] == colour, axis=-1)
        ys, xs = np.nonzero(mask)
        if len(xs) < 4:
            continue
        return float(x1 + xs.mean()), float(y1 + ys.mean()), float(len(xs))
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/d/lightwm_data")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--frames", type=int, default=600, help="max frames to score")
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints/small_objects_20260910/dense_depth_best.pt"))
    ap.add_argument("--resolution", type=int, default=0, help="override the head's input size")
    ap.add_argument("--calibrate", action="store_true",
                    help="fit a global  a*pred+b  on the first half and score the second half")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from phase_b.perception_runtime import PerceptionRuntime
    rt = PerceptionRuntime(args.ckpt, device=device, zoom=False)
    native = rt.resolution
    if args.resolution:
        rt.resolution = int(args.resolution)
    print(f"depth head: {args.ckpt}")
    print(f"native resolution {native} -> using {rt.resolution}   device={device}")

    eps = sorted(p for p in (Path(args.root) / "episodes").iterdir()
                 if (p / "episode.json").exists())
    eps = sorted(eps, key=lambda p: ("sweep" not in p.name, p.name))[:args.episodes]

    rows = []
    skip = collections.Counter()
    seen_frames = 0
    for ep in eps:
        meta = json.loads((ep / "episode.json").read_text())
        for frame in meta.get("frames", []):
            if len(rows) >= args.frames * 12 or seen_frames >= args.frames:
                break
            agent = frame.get("agent") or {}
            if agent.get("position", {}).get("x") is None:
                skip["no_pose"] += 1
                continue
            try:
                rgb = np.asarray(Image.open(ep / frame["rgb"]).convert("RGB"))
                gt_depth = np.asarray(Image.open(ep / frame["depth"])).astype(np.float64)
                seg = np.asarray(Image.open(ep / frame["seg"]).convert("RGB"))
            except Exception:
                skip["unreadable"] += 1
                continue
            if gt_depth.ndim == 3:
                gt_depth = gt_depth[..., 0]
            gt_depth = gt_depth / 1000.0
            seen_frames += 1
            with torch.inference_mode():
                out = rt(rgb)
            pred = out["depth"] if isinstance(out, dict) else out
            if isinstance(pred, torch.Tensor):
                pred = pred.detach().cpu().numpy()
            pred = np.asarray(pred, dtype=np.float64)
            if pred.shape != gt_depth.shape:
                pred = np.asarray(Image.fromarray(pred).resize(
                    (gt_depth.shape[1], gt_depth.shape[0]), Image.BILINEAR))

            for obj in frame.get("visible_objects") or []:
                hit = object_pixel(seg, obj.get("bbox") or [])
                if hit is None:
                    continue
                u, v, area_px = hit
                ui, vi = int(round(u)), int(round(v))
                if not (0 <= ui < gt_depth.shape[1] and 0 <= vi < gt_depth.shape[0]):
                    continue
                g = float(gt_depth[vi, ui])
                p = float(pred[vi, ui])
                if not (0.1 < g < 8.0) or not np.isfinite(p):
                    skip["bad_depth"] += 1
                    continue
                rows.append(dict(gt=g, pred=p, err=abs(p - g), bias=p - g,
                                 horizon=float(agent.get("cameraHorizon") or 0.0),
                                 area=area_px, name=str(obj.get("objectId", "")).split("|")[0]))

    if not rows:
        print("no samples", skip); return 1
    if args.calibrate:
        half = len(rows) // 2
        fit, test = rows[:half], rows[half:]
        A = np.array([[r["pred"], 1.0] for r in fit])
        y = np.array([r["gt"] for r in fit])
        a, b = np.linalg.lstsq(A, y, rcond=None)[0]
        for r in test:
            r["err_cal"] = abs(a * r["pred"] + b - r["gt"])
        base = statistics.median(r["err"] for r in test)
        cal = statistics.median(r["err_cal"] for r in test)
        print(f"\ncalibration: pred -> {a:.4f}*pred {b:+.4f}  (fit on {len(fit)}, scored on {len(test)})")
        print(f"  median |err|  {base:.3f} m  ->  {cal:.3f} m   ({100*(base-cal)/base:+.1f}%)")

    def med(key, rs):
        return statistics.median(r[key] for r in rs)

    print(f"\nframes scored: {seen_frames}   object samples: {len(rows)}   skipped: {dict(skip)}")
    print(f"median |err|   {med('err', rows):.3f} m     mean |err| {statistics.mean(r['err'] for r in rows):.3f} m")
    print(f"median signed bias (pred-gt) {med('bias', rows):+.3f} m")
    print(f"median ratio pred/gt         {statistics.median(r['pred']/r['gt'] for r in rows):.3f}")

    print("\nby ground-truth distance:")
    print(f"{'bucket':<12}{'n':>7}{'median err':>12}{'median bias':>13}{'ratio':>8}")
    buckets = collections.defaultdict(list)
    for r in rows:
        g = r["gt"]
        b = "<0.5m" if g < 0.5 else ("0.5-1m" if g < 1 else
                                     ("1-2m" if g < 2 else ("2-3m" if g < 3 else ">3m")))
        buckets[b].append(r)
    for b in ("<0.5m", "0.5-1m", "1-2m", "2-3m", ">3m"):
        rs = buckets.get(b)
        if not rs:
            continue
        print(f"{b:<12}{len(rs):>7}{med('err', rs):>12.3f}{med('bias', rs):>+13.3f}"
              f"{statistics.median(r['pred']/r['gt'] for r in rs):>8.3f}")

    print("\nby object pixel area (smaller = smaller object):")
    print(f"{'bucket':<12}{'n':>7}{'median err':>12}{'median bias':>13}")
    buckets = collections.defaultdict(list)
    for r in rows:
        a = r["area"]
        b = "<500px" if a < 500 else ("500-2k" if a < 2000 else
                                      ("2k-8k" if a < 8000 else ">8k"))
        buckets[b].append(r)
    for b in ("<500px", "500-2k", "2k-8k", ">8k"):
        rs = buckets.get(b)
        if not rs:
            continue
        print(f"{b:<12}{len(rs):>7}{med('err', rs):>12.3f}{med('bias', rs):>+13.3f}")

    print("\nworst classes (median err, n>=15):")
    per = collections.defaultdict(list)
    for r in rows:
        per[r["name"]].append(r["err"])
    worst = [(statistics.median(v), k, len(v)) for k, v in per.items() if len(v) >= 15]
    for m, k, n in sorted(worst, reverse=True)[:10]:
        print(f"  {k:<22}{m:.3f} m  (n={n})")

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"resolution": rt.resolution, "samples": len(rows), "rows": rows[:20000]}))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
