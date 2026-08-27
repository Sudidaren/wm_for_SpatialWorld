"""Copy-paste augmentation for rare/small task objects.

The dataset has only 500-1200 samples of Egg / KeyChain / CreditCard, and at
224px they are smaller than one detector grid cell.  This module builds a
library of object cutouts (RGB + instance mask from the seg frames) and pastes
them into training frames at random plausible positions, generating new GT
boxes for free.
"""

from __future__ import annotations

import glob
import json
import os
import pickle
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

TARGET_TYPES = {
    "Egg", "KeyChain", "CreditCard", "Potato", "Apple", "Tomato", "Cup",
    "Mug", "Bowl", "Bread", "CellPhone", "Plate", "Knife", "ButterKnife",
    "Spatula", "SaltShaker", "PepperShaker", "SoapBottle", "Bottle",
    "Kettle", "Pan", "Toaster",
}
MAX_PER_TYPE = 25


def _dominant_mask(seg, b) -> Optional[np.ndarray]:
    y0, y1 = max(0, b[1]), min(seg.shape[0], b[3] + 1)
    x0, x1 = max(0, b[0]), min(seg.shape[1], b[2] + 1)
    crop = seg[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    colors, counts = np.unique(crop.reshape(-1, 3), axis=0, return_counts=True)
    c = colors[counts.argmax()]
    m = (crop[:, :, 0] == c[0]) & (crop[:, :, 1] == c[1]) & \
        (crop[:, :, 2] == c[2])
    return m


def build_library(data_root: str, cache: str =
                  "/home/sudidaren/lightwm_phases/data/copypaste_lib.pkl"
                  ) -> Dict[str, List[Dict]]:
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            return pickle.load(f)
    lib: Dict[str, List[Dict]] = {t: [] for t in TARGET_TYPES}
    eps = sorted(glob.glob(os.path.join(data_root, "episodes", "*",
                                        "episode.json")))
    for p in eps:
        d = json.load(open(p))
        ep = os.path.dirname(p)
        for fr in d.get("frames", []):
            if not fr.get("seg") or not fr.get("rgb"):
                continue
            seg = np.asarray(Image.open(os.path.join(ep, fr["seg"])))
            rgb = np.asarray(Image.open(os.path.join(ep, fr["rgb"])))
            for o in fr.get("visible_objects", []):
                nm = o.get("name", "")
                t = nm.split("_")[0] if "_" in nm else nm.split("|")[0]
                if t not in TARGET_TYPES or len(lib[t]) >= MAX_PER_TYPE:
                    continue
                b = o.get("bbox")
                if b is None or b[2] - b[0] < 8 or b[3] - b[1] < 8:
                    continue
                mask = _dominant_mask(seg, b)
                if mask is None or mask.sum() < 30:
                    continue
                y0, y1 = max(0, b[1]), min(seg.shape[0], b[3] + 1)
                x0, x1 = max(0, b[0]), min(seg.shape[1], b[2] + 1)
                patch = rgb[y0:y1, x0:x1].copy()
                lib[t].append({
                    "patch": patch, "mask": mask, "w": b[2] - b[0],
                    "h": b[3] - b[1],
                    "area_ratio": (b[2] - b[0]) * (b[3] - b[1]) / (800 * 600),
                })
            if all(len(lib[t]) >= MAX_PER_TYPE for t in TARGET_TYPES):
                break
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump(lib, f, protocol=4)
    print(f"[copy-paste] library: " +
          ", ".join(f"{t}:{len(lib[t])}" for t in TARGET_TYPES))
    return lib


def paste_into(rgb: np.ndarray, boxes: List[List[float]],
               classes: List[int], type2id, lib, rng: np.random.RandomState,
               max_objs: int = 3) -> Tuple[np.ndarray, List[List[float]],
                                           List[int]]:
    """Paste 0..max_objs library objects into rgb (800x600 uint8).
    Returns (new_rgb, new_boxes, new_classes) with added GT boxes."""
    candidates = [(t, o) for t, objs in lib.items() if objs
                  for o in objs]
    if not candidates:
        return rgb, boxes, classes
    n_paste = rng.randint(0, max_objs + 1)
    new_boxes = [list(b) for b in boxes]
    new_classes = list(classes)
    for _ in range(n_paste):
        t, o = candidates[rng.randint(len(candidates))]
        cid = type2id.get(t)
        if cid is None:
            continue
        scale = float(rng.uniform(0.8, 1.4))
        w = int(o["w"] * scale)
        h = int(o["h"] * scale)
        if w < 10 or h < 10 or w >= 700 or h >= 500:
            continue
        x = rng.randint(20, 800 - w - 20)
        y = rng.randint(20, 600 - h - 20)
        # skip heavy overlap with existing boxes
        cx, cy = x + w / 2, y + h / 2
        if any(_iou_box(cx, cy, w, h, b) > 0.4 for b in new_boxes):
            continue
        mask = np.asarray(Image.fromarray(o["mask"]).resize(
            (w, h), Image.NEAREST)) if (o["mask"].shape[1] != w or
                                        o["mask"].shape[0] != h) else o["mask"]
        patch = np.asarray(Image.fromarray(o["patch"]).resize(
            (w, h), Image.BILINEAR))
        if mask.shape[:2] != (h, w):
            continue
        m3 = np.repeat(mask[:, :, None], 3, axis=2).astype(bool)
        rgb[y:y + h, x:x + w][m3] = patch[m3]
        new_boxes.append([x, y, x + w, y + h])
        new_classes.append(cid)
    return rgb, new_boxes, new_classes


def _iou_box(cx, cy, w, h, b):
    bx1, by1, bx2, by2 = b
    ix = max(0, min(cx + w / 2, bx2) - max(cx - w / 2, bx1))
    iy = max(0, min(cy + h / 2, by2) - max(cy - h / 2, by1))
    inter = ix * iy
    ua = w * h + (bx2 - bx1) * (by2 - by1) - inter
    return inter / max(ua, 1e-9)
