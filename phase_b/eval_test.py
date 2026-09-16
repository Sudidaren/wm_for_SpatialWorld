"""Final TEST evaluation (no validation set protocol).

Evaluates the perception checkpoint on the held-out test rooms from
data/splits_noval.json:
  * detection P/R @ IoU>0.5 with optional class-match (recognition success),
    broken down by GT size bucket (on the 224px model scale);
  * object-center depth MAE (median too) using the paired dense_depth ckpt;
  * combined success = detected with IoU>0.5 + correct class + object-center
    depth error below a threshold (reported at 0.10 / 0.15 / 0.25 m).

Usage:
  python phase_b/eval_test.py --ckpt checkpoints_local/dense_best_ema.pt \
      [--depth-ckpt checkpoints_local/dense_depth_best.pt] [--max-frames 3000]
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from dataset import load_depth, load_rgb  # noqa: E402
from model import PerceptionModel  # noqa: E402
from shared.data_index import load_index  # noqa: E402
from train import decode_dense, iou2  # noqa: E402

SCALE = 224.0 / 800.0


def bucket_of(b) -> str:
    if not b or len(b) < 4:
        return "?"
    sz = max(b[2] - b[0], b[3] - b[1]) * SCALE
    if sz <= 16:
        return "A<=16"
    if sz <= 48:
        return "B17-48"
    if sz <= 144:
        return "C49-144"
    return "D>144"


def canonical(scene: str) -> str:
    m = re.match(r"(FloorPlan\d+)", str(scene or ""))
    return m.group(1) if m else ""


def load_test_frames(splits_path: str, index) -> list:
    with open(splits_path) as f:
        d = json.load(f)
    fp = d["scene_splits"]["ai2thor_floorplans"]
    test_rooms = set(fp["test"])
    out = []
    for fr in index["frames"]:
        s = str(fr.get("scene") or "")
        if s.startswith("procthor") or s.startswith("virtualhome"):
            continue
        if canonical(s) in test_rooms and fr.get("rgb"):
            out.append(fr)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",
                    default="checkpoints_local/dense_best_ema.pt")
    ap.add_argument("--depth-ckpt", default="")
    ap.add_argument("--splits", default=None)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--variant", choices=["small", "base"], default="small")
    ap.add_argument("--resolution", type=int, default=224)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--obj-thr", type=float, default=0.25)
    ap.add_argument("--nms-thr", type=float, default=0.5)
    ap.add_argument("--device", default=None)
    ap.add_argument("--require-cls", action="store_true",
                    help="TP requires predicted class == GT class "
                         "(recognition success); default localization only")
    args = ap.parse_args()

    splits = args.splits or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "splits_noval.json")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    index = load_index()
    frames = load_test_frames(splits, index)
    if args.max_frames:
        frames = frames[:: max(1, len(frames) // args.max_frames)][: args.max_frames]
    print(f"test frames: {len(frames)} (split {splits})")
    if not frames:
        raise SystemExit("no test frames")

    num_types = len(index["object_types"])
    dinov2_name = ("dinov2_vits14" if args.variant == "small"
                   else "dinov2_vitb14")
    model = PerceptionModel(
        num_types=num_types, device="cpu", img_size=args.resolution,
        head_width=args.width, dinov2_name=dinov2_name).to(device)
    sd = torch.load(args.ckpt, map_location=device)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print("missing keys (ok if depth head absent):", missing[:5])
    print(f"loaded dense {args.ckpt}")
    type2id = {t: i for i, t in enumerate(index["object_types"])}
    exclude = {type2id[t] for t in ("Floor", "Wall", "Ceiling", "Window")
               if t in type2id}

    model.eval()
    agg = collections.defaultdict(lambda: [0, 0, 0])  # bucket -> [tp, fp, fn]
    obj_depths = []  # (abs err)
    comb = {thr: {"n": 0, "ok": 0} for thr in (0.10, 0.15, 0.25)}
    with torch.no_grad():
        for fi, fr in enumerate(frames):
            rgb = torch.from_numpy(load_rgb(fr["rgb"], args.resolution))
            rgb = rgb.unsqueeze(0).permute(0, 3, 1, 2).to(device)
            d = model.dense_detect(rgb)
            boxes, clss, scores = decode_dense(
                d["objectness"], d["offset"], d["size"], d["class_logits"],
                grid=model.grid, obj_thr=args.obj_thr, nms_thr=args.nms_thr)
            bx = boxes[0].cpu().numpy() * args.resolution
            preds = []
            for b, c in zip(bx, clss[0].cpu().numpy()):
                if b[2] <= 0.01 or b[3] <= 0.01:
                    continue
                preds.append((b[0] - b[2] / 2, b[1] - b[3] / 2,
                              b[0] + b[2] / 2, b[1] + b[3] / 2, int(c)))
            gts = []
            for v in fr.get("visible", []) or []:
                t = type2id.get(v["type"])
                if t is None or t in exclude or v.get("bbox") is None:
                    continue
                b = v["bbox"]
                if b[2] <= b[0] or b[3] <= b[1]:
                    continue
                sx, sy = args.resolution / 800.0, args.resolution / 600.0
                gts.append((b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy,
                            t, bucket_of(b)))
            used = set()
            for pi, p in enumerate(preds):
                cand = []
                for gi, g in enumerate(gts):
                    if gi in used:
                        continue
                    if args.require_cls and p[4] != g[4]:
                        continue
                    iou = iou2(p[:4], g[:4])
                    if iou > 0.5:
                        cand.append((iou, gi))
                if not cand:
                    continue
                iou, gi = max(cand)
                used.add(gi)
            hit = set(used)
            for gi, g in enumerate(gts):
                bucket = g[5]
                tp = 1 if gi in hit else 0
                agg[bucket][0] += tp
                agg[bucket][2] += 1
            agg["ALL"][0] += len(hit)
            agg["ALL"][2] += len(gts)
            agg["ALL"][1] += max(0, len(preds) - len(hit))
            # object-center depth (GT bbox center on the 224 depth map)
            if args.depth_ckpt and fr.get("depth"):
                pred_d = model.depth(rgb)
                pred_d = torch.nn.functional.interpolate(
                    pred_d, size=(args.resolution, args.resolution),
                    mode="bilinear", align_corners=False)
                gt_d = torch.from_numpy(load_depth(fr["depth"],
                                                   args.resolution))
                gt_d = gt_d.to(device)
                for g in gts:
                    cx = int(round((g[0] + g[2]) / 2))
                    cy = int(round((g[1] + g[3]) / 2))
                    cx = min(max(cx, 0), args.resolution - 1)
                    cy = min(max(cy, 0), args.resolution - 1)
                    dgt = float(gt_d[cy, cx])
                    if dgt <= 0.1:
                        continue
                    err = abs(float(pred_d[0, 0, cy, cx]) - dgt)
                    obj_depths.append(err)
                    for thr in comb:
                        comb[thr]["n"] += 1
                        if gi in hit and err <= thr:
                            comb[thr]["ok"] += 1
            if (fi + 1) % 500 == 0:
                print(f"  {fi + 1}/{len(frames)}", flush=True)

    print("\n=== TEST detection P/R (IoU>0.5) ===")
    print("require-class-match:", args.require_cls)
    for bucket in ("ALL", "A<=16", "B17-48", "C49-144", "D>144"):
        if bucket not in agg:
            continue
        tp, fp, fn = agg[bucket]
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        print(f"  {bucket:6s}: P={p:.3f} R={r:.3f} (tp={tp} fp={fp} fn={fn})")
    if obj_depths:
        a = np.asarray(obj_depths)
        print("\n=== object-center depth MAE ===")
        print(f"  n_objects={len(a)} MAE={a.mean():.3f} m "
              f"median={np.median(a):.3f} m")
        for thr, c in comb.items():
            print(f"  combined success (IoU+cls) @depth<{thr:.2f}m: "
                  f"{c['ok']}/{c['n']} = {c['ok']/max(c['n'],1):.3f}")


if __name__ == "__main__":
    main()
