"""Runtime perception adapter: one checkpoint -> detections + depth.

Loads a PerceptionModel state dict (e.g. checkpoints_local/dense_depth_best.pt
or dense_best.pt), runs dense decode + monocular depth on an RGB frame, and
returns:
  * detections: list of {type, score, bbox(x1,y1,x2,y2), center, size} in
    ORIGINAL image pixels
  * depth:      meters map at ORIGINAL image resolution (float32 HxW)

This is the "eyes" of the V1 world model: anchors are built only from these
detections + depth, so the model can only report objects it has seen.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from phase_b.model import PerceptionModel, decode_dense  # noqa: E402


def _default_types():
    # Deployment machines do not necessarily carry the raw training frames.
    # The checked-in inventory preserves the exact 117-class training
    # vocabulary in its class_coverage keys; data_index orders it
    # alphabetically when training the head.
    inventory = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "data", "inventory.json"))
    if os.path.exists(inventory):
        with open(inventory, encoding="utf-8") as f:
            pools = json.load(f).get("pools", {})
        for pool in pools.values():
            vocab = pool.get("class_coverage", {})
            if len(vocab) == 117:
                return sorted(vocab)
    from shared.data_index import load_index
    return list(load_index().get("object_types", []))


class PerceptionRuntime:
    def __init__(
        self,
        ckpt: str,
        variant: str = "small",
        resolution: int = 224,
        width: int = 256,
        num_types: int = 117,
        type_names: list[str] | None = None,
        obj_thr: float = 0.35,
        nms_thr: float = 0.5,
        device: str | None = None,
        zoom: bool | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        metadata = ckpt + '.json'
        if os.path.isfile(metadata):
            with open(metadata) as handle:
                info = json.load(handle)
            if type_names is None:
                type_names = info.get('object_types')
            if type_names:
                num_types = len(type_names)
            resolution = int(info.get('resolution', resolution))
        dinov2_name = "dinov2_vits14" if variant == "small" else "dinov2_vitb14"
        self.model = PerceptionModel(
            num_types=num_types, num_actions=0, num_errors=0,
            device="cpu", img_size=resolution,
            head_width=width, dinov2_name=dinov2_name,
            encoder_checkpoint=ckpt).to(self.device)
        sd = torch.load(ckpt, map_location=self.device, weights_only=True)
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        if missing:
            raise SystemExit(f"ckpt missing keys: {missing[:10]}")
        self.model.eval()
        self.resolution = resolution
        self.obj_thr = obj_thr
        self.nms_thr = nms_thr
        self.zoom = (os.environ.get('LIGHTWM_ZOOM', '0') == '1'
                     if zoom is None else zoom)
        metadata = ckpt + '.json'
        if type_names is None and os.path.isfile(metadata):
            with open(metadata) as handle:
                type_names = json.load(handle).get('object_types')
        self.type_names = type_names or _default_types()
        if len(self.type_names) < num_types:
            raise SystemExit("type_names shorter than num_types")
        print(f"[perception_runtime] loaded {ckpt} on {self.device} "
              f"({len(sd)} tensors)")

    @torch.no_grad()
    def __call__(self, rgb: np.ndarray) -> dict:
        """rgb: uint8 HxWx3 (BGR->RGB converted by caller or RGB ok)."""
        H, W = rgb.shape[:2]
        im = Image.fromarray(rgb).resize((self.resolution, self.resolution))
        x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0)
        x = x.permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.autocast("cuda", enabled=self.device.startswith("cuda")):
            out = self.model.dense_detect(x)
            depth = self.model.depth(x)  # (1,1,R,R) meters-ish
        depth = F.interpolate(depth, size=(H, W), mode="bilinear",
                              align_corners=False)[0, 0].float().cpu().numpy()
        boxes, clss, scores = decode_dense(
            out["objectness"], out["offset"], out["size"],
            out["class_logits"], grid=self.model.grid,
            img_size=self.resolution, obj_thr=self.obj_thr,
            nms_thr=self.nms_thr, max_det=100)
        dets = []
        for b, c, s in zip(boxes[0], clss[0], scores[0]):
            if s <= 0 or int(c) >= len(self.type_names):
                continue
            cx, cy, w, h = [float(v) for v in b.cpu()]
            x1, y1 = (cx - w / 2) * W, (cy - h / 2) * H
            x2, y2 = (cx + w / 2) * W, (cy + h / 2) * H
            dets.append({
                "type": self.type_names[int(c)],
                "class_id": int(c),
                "score": float(s),
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "center": [float(cx * W), float(cy * H)],
                "size": [float(w * W), float(h * H)],
            })
        if self.zoom:
            from phase_b.zoom import tile_windows, merge_detections
            for window in tile_windows(W, H):
                dets.extend(self.detect_crop(rgb, window))
            dets = merge_detections(dets, self.nms_thr)
        return {"detections": dets, "depth": depth, "rgb_shape": (H, W),
                "detection_views": 5 if self.zoom else 1}

    @torch.no_grad()
    def detect_crop(self, rgb, window):
        """Re-encode RGB crops; retain full-frame depth for metric geometry."""
        x1, y1, x2, y2 = window
        im = Image.fromarray(rgb[y1:y2, x1:x2]).resize(
            (self.resolution, self.resolution))
        x = torch.from_numpy(np.asarray(im, dtype=np.float32)/255).permute(2,0,1)
        with torch.autocast('cuda', enabled=self.device.startswith('cuda')):
            out = self.model.dense_detect(x[None].to(self.device))
        boxes, labels, scores = decode_dense(
            out['objectness'], out['offset'], out['size'], out['class_logits'],
            obj_thr=self.obj_thr, nms_thr=self.nms_thr, max_det=100)
        dets = []
        for b, c, score in zip(boxes[0].tolist(), labels[0].tolist(), scores[0].tolist()):
            if score <= 0:
                continue
            cx, cy, w, h = b
            a, b = max(0., cx-w/2), max(0., cy-h/2)
            c2, d = min(1., cx+w/2), min(1., cy+h/2)
            if c2 <= a or d <= b:
                continue
            a, c2 = x1+a*(x2-x1), x1+c2*(x2-x1)
            b, d = y1+b*(y2-y1), y1+d*(y2-y1)
            dets.append({'type': self.type_names[c], 'class_id': c, 'score': score,
                         'bbox': [a,b,c2,d], 'center': [(a+c2)/2,(b+d)/2],
                         'size': [c2-a,d-b], 'source': 'zoom'})
        return dets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints_local/dense_best.pt")
    ap.add_argument("--rgb", required=True)
    ap.add_argument("--obj-thr", type=float, default=0.35)
    args = ap.parse_args()
    runtime = PerceptionRuntime(args.ckpt, obj_thr=args.obj_thr)
    rgb = np.asarray(Image.open(args.rgb).convert("RGB"), dtype=np.uint8)
    res = runtime(rgb)
    print("detections:", len(res["detections"]))
    for d in res["detections"][:12]:
        print(f"  {d['type']:<16} score={d['score']:.2f} "
              f"bbox={d['bbox']}")
    print("depth shape:", res["depth"].shape,
          "range:", round(float(res["depth"].min()), 2),
          "-", round(float(res["depth"].max()), 2))


if __name__ == "__main__":
    main()
