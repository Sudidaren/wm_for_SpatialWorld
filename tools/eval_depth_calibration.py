#!/usr/bin/env python3
"""A/B the metric-scale calibration variants for the depth head.

All variants are evaluated on the held-out evaluation rooms only (never
trained on).  Every frame is pushed through the head once; each variant then
post-processes the same depth map, so the comparison costs one inference pass.

Variants
--------
``none``          raw head output
``oracle_scale``  per-frame scale fitted from the GT depth (debug ceiling for
                  *any* global per-frame correction -- not a legal method)
``gt_floor``      per-frame scale from the GT-identified floor pixels
                  (ceiling for a perfect floor detector)
``gt_floor_lin``  log-linear curve fitted on the GT-identified floor
``peak``          scale from the topmost peak of the implied-height histogram
``peak_lin``      log-linear curve fitted on that peak's pixel band
``band``          scale from the bottom image band (floor is under the camera)
``band_lin``      log-linear curve fitted on the bottom band
``pooled``        peak estimate pooled over a rolling window of frames

Usage:
    python3 tools/eval_depth_calibration.py --frames 300 \
        --ckpt checkpoints/depth_v2_20260918/dense_depth_v2_best.pt
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phase_b.depth_calib import (GroundCalibConfig, apply_scale_curve,  # noqa: E402
                                 fit_scale_curve, vertical_factor)


def logfit(pred: np.ndarray, true: np.ndarray, bins: int = 24,
           min_per_bin: int = 20, shrink: float = 0.0):
    """Robust log-linear fit ``log(true) = a * log(pred) + b``."""
    x, y = np.log(pred), np.log(true)
    order = np.argsort(x)
    x, y = x[order], y[order]
    n = len(x)
    xs, ys, ws = [], [], []
    for i in range(bins):
        lo, hi = int(i * n / bins), int((i + 1) * n / bins)
        if hi - lo < min_per_bin:
            continue
        xs.append(np.median(x[lo:hi]))
        ys.append(np.median(y[lo:hi]))
        ws.append(math.sqrt(hi - lo))
    if len(xs) < 3:
        return None
    xs, ys, ws = map(np.asarray, (xs, ys, ws))
    A = np.vstack([xs * ws, ws]).T
    coef, *_ = np.linalg.lstsq(A, ys * ws, rcond=None)
    a = (1 - shrink) * float(coef[0]) + shrink * 1.0
    b = float(np.median(y - a * x))
    return {"a": a, "b": b}


def apply_curve(pred, curves):
    if curves is None:
        return pred
    return apply_scale_curve(pred, curves)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/d/lightwm_data")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--ckpt", default=str(
        ROOT / "checkpoints/depth_v2_20260918/dense_depth_v2_best.pt"))
    ap.add_argument("--resolution", type=int, default=448)
    ap.add_argument("--fov", type=float, default=60.0)
    ap.add_argument("--convention", default="vertical",
                    choices=("vertical", "horizontal"))
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--rooms", default="test", choices=("test", "train", "all"))
    ap.add_argument("--depth-source", default="head", choices=("head", "da2"),
                    help="head = the project's trained depth head; da2 = the "
                         "public Depth-Anything-V2 Metric-Indoor-Small model "
                         "(zero-shot, metric, same cost class)")
    ap.add_argument("--da2-name",
                    default="depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    import torch
    from phase_b.perception_runtime import PerceptionRuntime

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = GroundCalibConfig(fov=args.fov, convention=args.convention,
                            stride=args.stride)
    H = cfg.cam_height

    eval_rooms = set(json.loads((ROOT / "data/eval_rooms.json").read_text())["rooms"])
    splits = json.loads((ROOT / "data/splits_noneval.json").read_text())
    split = splits["scene_splits"]["ai2thor_floorplans"]
    test_rooms = set(split["test"])
    train_rooms = set(split["train"]) | set(split["val"])
    keep = {"test": test_rooms, "train": train_rooms,
            "all": test_rooms | train_rooms}[args.rooms]

    eps = sorted(p for p in (Path(args.root) / "episodes").iterdir()
                 if (p / "episode.json").exists())
    random.Random(args.seed).shuffle(eps)
    eps = [p for p in eps if p.name.split("__")[0].split("_")[0] in keep]

    if args.depth_source == "da2":
        from transformers import DepthAnythingForDepthEstimation
        _da2 = DepthAnythingForDepthEstimation.from_pretrained(
            args.da2_name).to(device).eval()
        _MEAN = np.array([0.485, 0.456, 0.406], np.float32)
        _STD = np.array([0.229, 0.224, 0.225], np.float32)

        def _predict(rgb):
            im = Image.fromarray(rgb).resize((518, 518), Image.BILINEAR)
            x = ((np.asarray(im, np.float32) / 255.0 - _MEAN) / _STD
                 ).transpose(2, 0, 1)[None]
            with torch.no_grad():
                d = _da2(pixel_values=torch.from_numpy(x).to(device)).predicted_depth
            d = torch.nn.functional.interpolate(
                d[:, None].float(), size=rgb.shape[:2], mode="bilinear",
                align_corners=False)[0, 0]
            return d.cpu().numpy().astype(np.float32)

        class _DA2:
            resolution = 518

            def __call__(self, rgb):
                return {"depth": _predict(rgb), "detections": []}
        rt = _DA2()
        print(f"depth source: DA2 {args.da2_name}; rooms={args.rooms} "
              f"({len(eps)} episodes)")
    else:
        rt = PerceptionRuntime(args.ckpt, device=device, zoom=False)
        if args.resolution:
            rt.resolution = int(args.resolution)
        print(f"head {args.ckpt} @ {rt.resolution} on {device}; rooms={args.rooms} "
              f"({len(eps)} episodes)")

    variants = ("none", "oracle_scale", "gt_floor", "gt_floor_lin", "peak",
                "peak_lin", "band", "band_lin", "pooled")
    errs = collections.defaultdict(list)
    frame_rows = []
    hist = collections.defaultdict(list)
    n_frames = 0
    for ep in eps:
        if n_frames >= args.frames:
            break
        try:
            meta = json.loads((ep / "episode.json").read_text())
        except (OSError, ValueError):
            continue
        pool_k: list = []
        for fr in meta.get("frames", []):
            if n_frames >= args.frames:
                break
            ag = fr.get("agent") or {}
            pos = ag.get("position") or {}
            if pos.get("x") is None:
                continue
            try:
                rgb = np.asarray(Image.open(ep / fr["rgb"]).convert("RGB"))
                gtd = np.asarray(Image.open(ep / fr["depth"])).astype(np.float64)
                seg = np.asarray(Image.open(ep / fr["seg"]).convert("RGB"))
            except Exception:
                continue
            if gtd.ndim == 3:
                gtd = gtd[..., 0]
            gtd = gtd / 1000.0
            h, w = gtd.shape
            hz = float(ag.get("cameraHorizon") or 0.0)
            pred = np.asarray(rt(rgb)["depth"], dtype=np.float64)
            n_frames += 1

            c = vertical_factor(hz, width=w, height=h, cfg=cfg)
            c2 = c[:, None]
            z_floor = np.broadcast_to(
                np.where(c2 > 1e-3, H / np.maximum(c2, 1e-3), np.inf),
                pred.shape)

            # ---------- floor masks / labels ----------
            a_pred = pred * c2
            a_gt = gtd * c2
            floor_gt = (np.abs(a_gt - H) < 0.06) & (gtd > 0.4) & (gtd < 8)
            cand = {"none": None}
            # oracle
            m = np.isfinite(pred) & (gtd > 0.3) & (gtd < 8)
            if m.sum() > 100:
                k_or = float(np.exp(np.median(np.log(gtd[m]) - np.log(pred[m]))))
                cand["oracle_scale"] = {"a": 1.0, "b": -math.log(k_or)}
            # gt floor
            if floor_gt.sum() > 300:
                pf, gf = pred[floor_gt], gtd[floor_gt]
                k = float(np.exp(np.median(np.log(pf) - np.log(gf))))
                cand["gt_floor"] = {"a": 1.0, "b": -math.log(k)}
                fit = logfit(pf, gf)
                if fit:
                    cand["gt_floor_lin"] = fit
            # peak band from the prediction
            val = np.isfinite(a_pred) & (a_pred > 0.1) & (a_pred < 1.2)
            peak_ok = False
            if val.sum() > 800:
                av = a_pred[val]
                hist_, edges = np.histogram(av, bins=np.arange(0.05, 1.25, 0.01))
                hs = np.convolve(hist_, np.ones(5) / 5, "same")
                cen = 0.5 * (edges[:-1] + edges[1:])
                cands = np.nonzero(hs >= 0.30 * hs.max())[0]
                if len(cands):
                    pk = float(cen[cands[-1]])
                    band = val & (np.abs(a_pred - pk) < 0.07)
                    if band.sum() > 300:
                        pf = pred[band]
                        gf = z_floor[band]
                        k = float(np.median(pf / gf))
                        cand["peak"] = {"a": 1.0, "b": -math.log(k)}
                        fit = logfit(pf, gf)
                        if fit:
                            cand["peak_lin"] = fit
                        peak_ok = True
                        pool_k.append(k)
            # bottom band: the floor is directly under the camera
            v0 = int(h * 0.88)
            band = np.zeros_like(val)
            band[v0:, int(w * 0.10):int(w * 0.90)] = True
            band &= val
            if band.sum() > 200:
                pf = pred[band]
                gf = z_floor[band]
                k = float(np.median(pf / gf))
                cand["band"] = {"a": 1.0, "b": -math.log(k)}
                fit = logfit(pf, gf)
                if fit:
                    cand["band_lin"] = fit
            if len(pool_k) >= 3:
                k = float(np.median(pool_k))
                cand["pooled"] = {"a": 1.0, "b": -math.log(k)}

            # ---------- score ----------
            for obj in fr.get("visible_objects") or []:
                b = obj.get("bbox") or []
                if len(b) != 4:
                    continue
                x1, y1, x2, y2 = [int(v) for v in b]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 - x1 < 4 or y2 - y1 < 4:
                    continue
                crop = seg[y1:y2, x1:x2].reshape(-1, 3)
                colours, cnt = np.unique(crop, axis=0, return_counts=True)
                col = colours[int(np.argmax(cnt))]
                if col.max() == 0:
                    continue
                yy, xx = np.nonzero(np.all(seg == col, axis=-1))
                mm = (np.isfinite(pred[yy, xx]) & (gtd[yy, xx] > 0.3)
                      & (gtd[yy, xx] < 8))
                if mm.sum() < 12:
                    continue
                gt = float(np.median(gtd[yy, xx][mm]))
                for tag in variants:
                    cal = apply_curve(pred, cand.get(tag))
                    errs[tag].append(abs(float(np.median(cal[yy, xx][mm])) - gt))
            frame_rows.append({
                "episode": ep.name, "horizon": hz,
                "floor_gt_frac": float(floor_gt.mean()),
                "variants": {t: (cand.get(t) or {}).get("b") for t in variants},
            })

    if not errs["none"]:
        print("no samples")
        return 1
    print(f"\nframes {n_frames}   objects {len(errs['none'])}")
    base = float(np.median(errs["none"]))
    print(f"\n{'variant':16s}{'median |err|':>14s}{'mean':>8s}{'p90':>8s}"
          f"{'vs base':>10s}")
    for tag in variants:
        v = np.asarray(errs[tag])
        if not len(v):
            continue
        print(f"{tag:16s}{np.median(v):14.3f}{v.mean():8.3f}"
              f"{np.percentile(v, 90):8.3f}{base / max(1e-9, np.median(v)):9.2f}x")
    fl = np.asarray([r["floor_gt_frac"] for r in frame_rows])
    print(f"\n地面可见帧 (GT 地面像素 >2%): {100 * np.mean(fl > 0.02):.0f}%   "
          f"中位占比 {100 * np.median(fl):.1f}%")
    if args.json:
        Path(args.json).write_text(json.dumps({"frames": frame_rows}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
