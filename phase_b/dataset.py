"""PyTorch datasets for Phase B perception / depth / feasibility training."""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.data_index import load_index  # noqa: E402

IMG_SIZE = 224
DEPTH_SIZE = 56
MAX_BOXES = 30
EXCLUDE_TYPES = {"Floor", "Wall", "Ceiling", "Window"}
TASK_TYPES = {
    "Potato", "Plate", "Microwave", "Egg", "Pan", "KeyChain", "CreditCard",
    "Box", "Apple", "CellPhone", "Phone", "Tomato", "Bread", "Cup", "Bowl",
    "Mug", "Kettle", "Pot", "Toaster", "Knife", "ButterKnife", "Spatula",
    "SaltShaker", "PepperShaker", "SoapBottle", "Bottle",
}


def load_rgb(path: str, size: int = None) -> np.ndarray:
    size = size or IMG_SIZE
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB").resize((size, size)), dtype=np.float32) / 255.0


def load_depth(path: str, size: int = None) -> np.ndarray:
    size = size or DEPTH_SIZE
    with Image.open(path) as im:
        d = np.asarray(im.convert("I").resize((size, size), Image.NEAREST),
                       dtype=np.float32) / 1000.0
    return d


def has_image(fr: dict) -> bool:
    return bool(fr.get("rgb")) and not fr["rgb"].endswith("/")


class PerceptionDataset(Dataset):
    """Frame -> (rgb, gt boxes [cx,cy,w,h], object class ids, depth target)."""

    def __init__(self, index=None, limit: int = 0, seed: int = 0,
                 class_balanced: bool = False, copy_paste: bool = False,
                 require_depth: bool = False):
        self.index = index if index is not None else load_index()
        frames = self.index["frames"]
        self.types = self.index["object_types"]
        self.type2id = {t: i for i, t in enumerate(self.types)}
        self.exclude = {self.type2id[t] for t in EXCLUDE_TYPES if t in self.type2id}
        frames = [f for f in frames if has_image(f)]
        if require_depth:
            frames = [f for f in frames if f.get("depth")]
        if limit:
            rng = np.random.RandomState(seed)
            frames = [frames[i] for i in rng.choice(len(frames), limit, replace=False)]
        self.frames = frames
        self.weights = None
        self.copy_paste = copy_paste
        self._lib = None
        if class_balanced:
            self.weights = self._frame_weights()

    def _frame_weights(self) -> np.ndarray:
        """Frames containing task-critical or rare objects get higher
        sampling weight; furniture-heavy frames get lower weight."""
        w = np.ones(len(self.frames), dtype=np.float32)
        for i, fr in enumerate(self.frames):
            types = [o.get("type") for o in (fr.get("visible", []) or [])]
            n_task = sum(1 for t in types if t in TASK_TYPES)
            w[i] += 1.5 * n_task
            n_furn = sum(1 for t in types if t in
                         {"Cabinet", "Drawer", "Shelf", "CounterTop",
                          "StoveBurner", "StoveKnob", "Window", "Chair"})
            w[i] = max(0.35, w[i] - 0.25 * n_furn)
        return w

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i: int):
        fr = self.frames[i]
        with Image.open(fr["rgb"]) as im:
            rgb_full = np.array(im.convert("RGB"), dtype=np.uint8, copy=True)
        depth = (load_depth(fr["depth"]) if fr.get("depth")
                 else np.zeros((DEPTH_SIZE, DEPTH_SIZE), dtype=np.float32))
        boxes_raw, classes_raw = [], []
        for o in fr["visible"]:
            t = self.type2id.get(o["type"])
            if t is None or t in self.exclude:
                continue
            b = o["bbox"]
            if b is None or b[2] <= b[0] or b[3] <= b[1] or \
                    (b[2] - b[0]) < 2 or (b[3] - b[1]) < 2:
                continue
            boxes_raw.append(list(b))
            classes_raw.append(t)
        if self.copy_paste:
            if self._lib is None:
                from copy_paste import build_library
                import os as _os
                self._lib = build_library(
                    _os.environ.get("LIGHTWM_DATA_ROOT",
                                    "/mnt/d/lightwm_data"))
            import copy_paste as cp
            rgb_full, boxes_raw, classes_raw = cp.paste_into(
                rgb_full, boxes_raw, classes_raw, self.type2id, self._lib,
                np.random.RandomState(np.random.randint(2 ** 31)))
        rgb = np.asarray(Image.fromarray(rgb_full).resize(
            (IMG_SIZE, IMG_SIZE)), dtype=np.float32) / 255.0
        flip = bool(np.random.rand() < 0.5)
        if flip:
            rgb = np.ascontiguousarray(rgb[:, ::-1, :])
            depth = np.ascontiguousarray(depth[:, ::-1])
        sx, sy = 1.0 / 800.0, 1.0 / 600.0
        boxes, classes = [], []
        for b, t in zip(boxes_raw, classes_raw):
            x1, y1, x2, y2 = b
            cx = (x1 + x2) / 2 * sx
            if flip:
                cx = 1.0 - cx
            cy = (y1 + y2) / 2 * sy
            w = (x2 - x1) * sx
            h = (y2 - y1) * sy
            boxes.append([cx, cy, w, h])
            classes.append(t)
        if not boxes:
            boxes = [[0.0, 0.0, 0.0, 0.0]]
            classes = [0]
        boxes = np.asarray(boxes, dtype=np.float32)[:MAX_BOXES]
        classes = np.asarray(classes, dtype=np.int64)[:MAX_BOXES]
        n = boxes.shape[0]
        if n < MAX_BOXES:
            pad = np.zeros((MAX_BOXES - n, 4), dtype=np.float32)
            boxes = np.concatenate([boxes, pad])
            classes = np.concatenate([classes, np.zeros(MAX_BOXES - n, dtype=np.int64)])
        return {
            "rgb": torch.from_numpy(rgb).permute(2, 0, 1),
            "boxes": torch.from_numpy(boxes),
            "classes": torch.from_numpy(classes),
            "num_boxes": torch.tensor(n),
            "depth": torch.from_numpy(depth).unsqueeze(0),
        }


class FeasibilityDataset(Dataset):
    """Frame + action -> P(success) + error class (supervised by the simulator
    labels recorded for the action taken from this observation)."""

    def __init__(self, index=None, limit: int = 0, seed: int = 0,
                 use_fd: bool = True):
        self.index = index if index is not None else load_index()
        frames = self.index["frames"]
        self.actions = self.index["actions"]
        self.action2id = {a: i for i, a in enumerate(self.actions)}
        self.errors = self.index["error_classes"]
        self.err2id = {e: i for i, e in enumerate(self.errors)}
        self.types = self.index["object_types"]
        self.type2id = {t: i for i, t in enumerate(self.types)}
        # only frames with a meaningful action
        frames = [f for f in frames if f["action"] in self.action2id
                  and f["action"] not in ("Teleport", "Pass", "Done")
                  and has_image(f)]
        if limit:
            rng = np.random.RandomState(seed)
            frames = [frames[i] for i in rng.choice(len(frames), limit, replace=False)]
        # merge FD frames (richer error distribution, RGB + labels only)
        if use_fd:
            try:
                from shared.fd_index import load_fd_index
                fd = load_fd_index()
                fd_frames = [
                    f for f in fd["frames"]
                    if f["action"] in self.action2id and
                    f["error_class"] in self.err2id
                ]
                print(f"[FeasibilityDataset] merged {len(fd_frames)} FD frames")
                frames = frames + fd_frames
            except Exception as e:
                print(f"[FeasibilityDataset] FD merge skipped: {e}")
        self.frames = frames

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i: int):
        fr = self.frames[i]
        rgb = load_rgb(fr["rgb"])
        act_id = self.action2id[fr["action"]]
        arg_type = 0
        arg = fr.get("action_args", {}) or {}
        oid = arg.get("objectId") or arg.get("object_id") or ""
        if oid:
            arg_type = self.type2id.get(oid.split("_")[0], 0)
        return {
            "rgb": torch.from_numpy(rgb).permute(2, 0, 1),
            "action": torch.tensor(act_id, dtype=torch.long),
            "arg_type": torch.tensor(arg_type, dtype=torch.long),
            "success": torch.tensor(float(fr["success"])),
            "error_class": torch.tensor(self.err2id[fr["error_class"]],
                                        dtype=torch.long),
        }


def collate_perception(batch):
    out = {k: torch.stack([b[k] for b in batch]) for k in batch[0]}
    return out


def collate_feasibility(batch):
    out = {}
    for k in ("rgb", "action", "arg_type", "success", "error_class"):
        out[k] = torch.stack([b[k] for b in batch])
    return out
