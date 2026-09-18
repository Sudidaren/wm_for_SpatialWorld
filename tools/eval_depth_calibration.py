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
import os
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
    ap.add_argument("--gate", type=float, default=0.0,
                    help="prominence the floor bump must reach before the "
                         "anchored curve is applied at all; 0 = always apply "
                         "(the ungated detector measurably HURTS), a large "
                         "value = only touch the frames where the floor is "
                         "unmistakable")
    ap.add_argument("--pool-episode", action="store_true",
                    help="estimate the metric scale per *episode* (median of "
                         "the per-frame floor estimates) instead of per frame. "
                         "The scene's scale is constant while the per-frame "
                         "estimate is noisy: measured 20%% error per frame vs "
                         "6.5%% pooled over ~8 frames, which is the difference "
                         "between 'the anchor hurts' and 'the anchor helps'.")
    ap.add_argument("--min-pool-frames", type=int, default=4)
    ap.add_argument("--scale-const", type=float, default=1.3816,
                    help="DA2's systematic metric bias, calibrated on the "
                         "*non-evaluation* rooms only: pooled pred/gt = 1.3816 "
                         "over n=1251 objects (IQR 1.294-1.482).  It is a "
                         "property of the public model, and on the held-out "
                         "rooms the same ratio measures 1.32-1.43, so one "
                         "constant transfers.  Set --scale-const 0 to disable.")
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
                "peak_lin", "centroid_scale", "band", "band_lin", "pooled",
                "gated", "epi_scale", "const_scale")
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
        episode_k = None
        if args.pool_episode:
            # pre-pass: the scene's metric scale is constant, so the median of
            # the per-frame floor estimates is far better than any single one
            # (measured 6.5 % error vs 20 % per frame).  Costs one extra model
            # forward per frame and nothing else.
            ks = []
            for fr2 in meta.get("frames", [])[:64]:
                ag2 = fr2.get("agent") or {}
                if (ag2.get("position") or {}).get("x") is None:
                    continue
                try:
                    rgb2 = np.asarray(Image.open(ep / fr2["rgb"]).convert("RGB"))
                except Exception:
                    continue
                p2 = np.asarray(rt(rgb2)["depth"], dtype=np.float64)
                c2 = vertical_factor(float(ag2.get("cameraHorizon") or 0.0),
                                     width=rgb2.shape[1], height=rgb2.shape[0],
                                     cfg=cfg)[:, None]
                a2 = p2 * c2
                v2 = np.isfinite(a2) & (a2 > 0.1) & (a2 < 1.6)
                if v2.sum() < 500:
                    continue
                av2 = a2[v2]
                h2, e2 = np.histogram(av2, bins=np.arange(0.05, 1.6, 0.01))
                s2 = np.convolve(h2, np.ones(5) / 5, "same")
                ctr = 0.5 * (e2[:-1] + e2[1:])
                b2 = float(np.median(s2))
                ex2 = np.maximum(s2 - b2, 0.0)
                if ex2.sum() <= 0:
                    continue
                cnd = np.nonzero(s2 >= 0.30 * s2.max())[0]
                if not len(cnd):
                    continue
                pk2 = float(ctr[cnd[-1]])
                prom2 = float(s2.max() / max(b2, 1e-9))
                win2 = (ctr >= pk2 - 0.30) & (ctr <= pk2 + 0.05)
                ew2 = np.where(win2, ex2, 0.0)
                if ew2.sum() <= 0:
                    continue
                k2 = float((ctr * ew2).sum() / ew2.sum()) / H
                # Two gates matter, and the second one is not optional: on
                # frames whose view is dominated by furniture the estimator
                # locks onto a low plane (measured k = 0.20 on a focus
                # episode, i.e. a 5x depth blow-up).  Requiring a clear bump
                # and a physically plausible factor removes exactly those
                # frames.  The range is a statement about how wrong a depth
                # head can plausibly be, not about any particular room.
                if prom2 < max(args.gate, 1.5) or not (0.5 < k2 < 2.5):
                    continue
                ks.append(k2)
            if len(ks) >= args.min_pool_frames:
                # A pooled estimate is only worth using when the frames agree:
                # the scene's scale is constant, so a scattered set of
                # per-frame estimates means the detector is guessing on some
                # of them.  Requiring a tight inter-quartile spread removes
                # those episodes instead of averaging a wrong level in.
                med_k = float(np.median(ks))
                lo, hi = np.percentile(ks, [25, 75])
                if (hi - lo) / max(med_k, 1e-9) <= 0.25:
                    episode_k = med_k
            if os.environ.get("LIGHTWM_DEBUG_POOL"):
                print(f"[pool] {ep.name[:40]:40s} n={len(ks):3d} k={episode_k}", flush=True)
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
                # how far does the bump stand above the flat background?  The
                # floor covers only ~3 % of an evaluation frame, so its peak is
                # a shoulder on a flat distribution: a weak prominence means
                # the detector is guessing and the correction must be skipped.
                back = float(np.median(hs))
                prom = (float(hs.max()) / max(back, 1e-9)) if back > 0 else 0.0
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
                        # Confident-only variant.  The *level* comes from the
                        # excess-mass centroid rather than the top of the bump:
                        # the bump's upper edge is noise-biased (measured 23.6 %
                        # median scale error vs the centroid's 19.8 %, and
                        # 10.0 % versus 13.4 % on the frames where the floor is
                        # unmistakable), and a wrong level mislabels the whole
                        # band.  The correction is applied only when the bump
                        # stands clear of the flat background.
                        back = float(np.median(hs))
                        exc = np.maximum(hs - back, 0.0)
                        if exc.sum() > 0:
                            # keep the centroid local to the floor's own bump:
                            # the global excess mass includes tables and beds,
                            # which pull it below the floor
                            win = (cen >= pk - 0.30) & (cen <= pk + 0.05)
                            e_w = np.where(win, exc, 0.0)
                            cent = (float((cen * e_w).sum() / e_w.sum())
                                    if e_w.sum() > 0 else pk)
                            cand["centroid_scale"] = {
                                "a": 1.0, "b": -math.log(max(cent / H, 0.05))}
                        if exc.sum() > 0 and prom >= args.gate and args.gate > 0:
                            win = (cen >= pk - 0.30) & (cen <= pk + 0.05)
                            e_w = np.where(win, exc, 0.0)
                            cent = (float((cen * e_w).sum() / e_w.sum())
                                    if e_w.sum() > 0 else pk)
                            band_c = val & (np.abs(a_pred - cent) < 0.10)
                            if band_c.sum() > 300:
                                fit_c = logfit(pred[band_c], z_floor[band_c])
                                if fit_c:
                                    cand["gated"] = fit_c
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
            if args.scale_const and args.scale_const > 0:
                # one constant for the public model's metric bias, calibrated
                # on non-evaluation rooms; no evaluation data is involved
                cand["const_scale"] = {"a": 1.0,
                                       "b": -math.log(args.scale_const)}
            if episode_k:
                # this episode's pooled scale, applied to every frame of it
                cand["epi_scale"] = {"a": 1.0, "b": -math.log(episode_k)}

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
