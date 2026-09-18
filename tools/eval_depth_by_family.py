#!/usr/bin/env python3
"""Per-family depth check: did the retrain actually fix the scale?

v2's failure was a *domain* failure: on room families the head never trained on
its predictions came out at 0.55-0.62x the truth.  The aggregate validation
MAE cannot show that (it is dominated by the families that were trained on), so
this measures the two things that matter, separately per family:

  * ``pred/gt``  -- a scale error shows up here as a ratio far from 1.0
  * ``|pred-gt|`` -- the absolute error, at the object's own pixels

Families are taken from the room id, and the split file decides what is
evaluation (never trained on), classic-non-eval (unseen by v2, same nature as
the evaluation domain), or trained-on.

Usage:
    python3 tools/eval_depth_by_family.py --ckpt checkpoints/depth_v3/dense_depth_v2_best.pt
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import statistics
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def mask_pixels(seg, bbox):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(seg.shape[1], x2), min(seg.shape[0], y2)
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None
    crop = seg[y1:y2, x1:x2]
    colours, counts = np.unique(crop.reshape(-1, 3), axis=0, return_counts=True)
    for i in np.argsort(-counts)[:3]:
        colour = colours[i]
        if colour.max() == 0:
            continue
        ys, xs = np.nonzero(np.all(crop == colour, axis=-1))
        if len(xs) < 12:
            continue
        return xs + x1, ys + y1
    return None


def family(room: str) -> str:
    m = re.match(r"FloorPlan(\d+)", room)
    if not m:
        return "other"
    n = int(m.group(1))
    return "classic" if n <= 30 else f"{n // 100}xx"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/d/lightwm_data")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--resolution", type=int, default=448)
    ap.add_argument("--frames-per-family", type=int, default=60,
                    help="episodes are walked until this many frames per "
                         "family have been scored")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    from phase_b.perception_runtime import PerceptionRuntime
    rt = PerceptionRuntime(args.ckpt, device=args.device, zoom=False)
    if args.resolution:
        rt.resolution = args.resolution

    ev = set(json.loads((ROOT / "data/eval_rooms.json").read_text())["rooms"])
    sp = json.loads((ROOT / "data/splits_noneval.json").read_text())
    s = sp["scene_splits"]["ai2thor_floorplans"]
    trained = set(s["train"]) | set(s["val"])

    stats = collections.defaultdict(lambda: {"ratio": [], "err": [], "n": 0})
    eps = sorted(p for p in (Path(args.root) / "episodes").iterdir()
                 if (p / "episode.json").exists())
    done = collections.Counter()
    for ep in eps:
        room = ep.name.split("__")[0].split("_")[0]
        fam = family(room)
        if done[fam] >= args.frames_per_family:
            continue
        if room in ev:
            tag = f"{fam}/评测房间(禁用训练)"
        elif room in trained and fam != "classic":
            tag = f"{fam}/训练过"
        elif fam == "classic":
            tag = f"{fam}/非评测(头没见过)"
        else:
            tag = f"{fam}/非评测"
        try:
            meta = json.loads((ep / "episode.json").read_text())
        except (OSError, ValueError):
            continue
        for fr in meta.get("frames", []):
            if done[fam] >= args.frames_per_family:
                break
            try:
                rgb = np.asarray(Image.open(ep / fr["rgb"]).convert("RGB"))
                gtd = np.asarray(Image.open(ep / fr["depth"])).astype(np.float64)
                seg = np.asarray(Image.open(ep / fr["seg"]).convert("RGB"))
            except Exception:
                continue
            if gtd.ndim == 3:
                gtd = gtd[..., 0]
            gtd /= 1000.0
            pred = np.asarray(rt(rgb)["depth"], dtype=np.float64)
            hit = False
            for obj in fr.get("visible_objects") or []:
                px = mask_pixels(seg, obj.get("bbox") or [])
                if px is None:
                    continue
                xs, ys = px
                g, p = gtd[ys, xs], pred[ys, xs]
                m = np.isfinite(g) & np.isfinite(p) & (g > 0.3) & (g < 8)
                if m.sum() < 20:
                    continue
                stats[tag]["ratio"].append(float(np.median(p[m] / g[m])))
                stats[tag]["err"].append(float(np.median(np.abs(p[m] - g[m]))))
                hit = True
            if hit:
                done[fam] += 1

    print(f"head: {args.ckpt} @ {rt.resolution}\n")
    print(f"{'家族':<26}{'物体数':>8}{'pred/gt':>10}{'中位|err|':>12}")
    print("-" * 58)
    for tag in sorted(stats):
        r, e = stats[tag]["ratio"], stats[tag]["err"]
        print(f"{tag:<26}{len(r):>8}{statistics.median(r):>10.3f}"
              f"{statistics.median(e):>11.3f}m")
    if args.json:
        Path(args.json).write_text(json.dumps(
            {k: {"n": len(v["ratio"]),
                 "ratio": statistics.median(v["ratio"]),
                 "err": statistics.median(v["err"])} for k, v in stats.items()},
            ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
