"""Evaluate the trained perception head: detection precision/recall (IoU>0.5)
and depth accuracy vs GT, plus 3D anchor quality when depth is available."""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from dataset import load_depth, load_rgb  # noqa: E402
from model import PerceptionModel, decode_dense  # noqa: E402


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix = max(0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / max(ua, 1e-9)


def main(ckpt: str, n_frames: int = 60, device: str = "cuda",
         head: str = "slots", variant: str = "small", resolution: int = 224,
         obj_thr: float = 0.35, nms_thr: float = 0.45,
         depth_ckpt: str = ""):
    import dataset as ds_mod
    ds_mod.IMG_SIZE = resolution
    from shared.data_index import load_index
    index = load_index()
    model = PerceptionModel(num_types=len(index["object_types"]),
                            num_actions=len(index["actions"]),
                            num_errors=len(index["error_classes"]),
                            device="cpu", img_size=resolution,
                            dinov2_name=("dinov2_vits14" if variant == "small"
                                         else "dinov2_vitb14")).to(device)
    state = torch.load(ckpt, map_location=device)
    cur = model.state_dict()
    filtered = {k: v for k, v in state.items()
                if k in cur and cur[k].shape == v.shape}
    skipped = len(state) - len(filtered)
    model.load_state_dict(filtered, strict=False)
    if skipped:
        print(f"[eval] skipped {skipped} incompatible checkpoint keys")
    if depth_ckpt and os.path.exists(depth_ckpt):
        dstate = torch.load(depth_ckpt, map_location=device)
        dfilt = {k: v for k, v in dstate.items()
                 if k.startswith("depth_head") and
                 k in cur and cur[k].shape == v.shape}
        model.load_state_dict(dfilt, strict=False)
        print(f"[eval] depth head loaded from {depth_ckpt}")
    model.eval()
    type2id = {t: i for i, t in enumerate(index["object_types"])}
    exclude = {type2id[t] for t in ("Floor", "Wall", "Ceiling", "Window")
               if t in type2id}
    frames = [f for f in index["frames"][::37] if f.get("rgb")
              and not f["rgb"].endswith("/")][:n_frames]
    tp = fp = fn = 0
    depth_mae = []
    with torch.no_grad():
        for fr in frames:
            rgb = torch.from_numpy(load_rgb(fr["rgb"])).permute(2, 0, 1)
            rgb = rgb.unsqueeze(0).to(device)
            if head == "dense":
                d = model.dense_detect(rgb)
                bboxes, clss, scores = decode_dense(
                    d["objectness"], d["offset"], d["size"], d["class_logits"],
                    obj_thr=obj_thr, nms_thr=nms_thr)
                bx = bboxes[0].cpu().numpy() * ds_mod.IMG_SIZE
                preds = [(b[0] - b[2] / 2, b[1] - b[3] / 2,
                          b[0] + b[2] / 2, b[1] + b[3] / 2)
                         for b in bx if b[2] > 0.01]
            else:
                o = model.perception(rgb)
                obj = torch.sigmoid(o["objectness"])[0].cpu().numpy()
                boxes = o["boxes"][0].cpu().numpy() * ds_mod.IMG_SIZE
                keep = obj > 0.5
                preds = [(b[0] - b[2] / 2, b[1] - b[3] / 2,
                          b[0] + b[2] / 2, b[1] + b[3] / 2)
                         for b in boxes[keep]]
            gts = []
            for v in fr.get("visible", []) or []:
                t = type2id.get(v["type"])
                if t is None or t in exclude or v.get("bbox") is None:
                    continue
                b = v["bbox"]
                sx, sy = ds_mod.IMG_SIZE / 800.0, ds_mod.IMG_SIZE / 600.0
                gts.append((b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy))
            used = set()
            for p in preds:
                best = max(((iou(p, g), i) for i, g in enumerate(gts)
                            if i not in used), default=(0, -1))
                if best[1] >= 0 and best[0] > 0.5:
                    tp += 1
                    used.add(best[1])
                else:
                    fp += 1
            fn += len(gts) - len(used)
            # depth accuracy on a few frames
            if len(depth_mae) < 12:
                d = model.depth(rgb).cpu()
                gt = torch.from_numpy(load_depth(fr["depth"])).unsqueeze(0)
                d = torch.nn.functional.interpolate(
                    d, size=(56, 56), mode="bilinear", align_corners=False)
                m = gt > 0.15
                if m.any():
                    depth_mae.append(
                        float((torch.abs(d - gt) * m).sum() / m.sum().clamp(min=1)))
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    print(f"n_frames={len(frames)}")
    print(f"detection: precision={prec:.3f} recall={rec:.3f} "
          f"(tp={tp} fp={fp} fn={fn})")
    if depth_mae:
        print(f"depth MAE (56x56, meters): {np.mean(depth_mae):.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",
                    default="/home/sudidaren/lightwm_phases/checkpoints/perception_best.pt")
    ap.add_argument("--n-frames", type=int, default=60)
    ap.add_argument("--head", choices=["slots", "dense"], default="slots")
    ap.add_argument("--variant", choices=["small", "base"], default="small")
    ap.add_argument("--resolution", type=int, default=224)
    ap.add_argument("--obj-thr", type=float, default=0.35)
    ap.add_argument("--nms-thr", type=float, default=0.45)
    ap.add_argument("--depth-ckpt", default="")
    args = ap.parse_args()
    main(args.ckpt, args.n_frames, head=args.head, variant=args.variant,
         resolution=args.resolution, obj_thr=args.obj_thr,
         nms_thr=args.nms_thr, depth_ckpt=args.depth_ckpt)
