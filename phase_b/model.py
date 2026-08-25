"""Phase B perception model.

Layers (trainable budget < 15M):
  - DINOv2-S (frozen, 21M inference-only)   -> patch features (16x16, dim 384)
  - 2-layer 1x1 conv adapter (384 -> 256 -> 256)
  - Slot Attention (16 slots, dim 256, 3 iters)
  - slot heads: box (cx,cy,w,h) + objectness + class logits
  - depth head: small conv decoder 384 -> 1 (56x56)
  - feasibility head: CLS + action emb -> P(success) + error class
"""

from __future__ import annotations

import math
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_dinov2(name: str = "dinov2_vits14", device="cpu"):
    """Load DINOv2 via transformers (weights from HF mirror when needed).

    Returns (encoder, feature_dim, patch_size).  The encoder output for a
    (B,3,H,W) input is (B, N_patches, D) patch features (CLS is dropped;
    transformers returns last_hidden_state with CLS at index 0).
    """
    from transformers import Dinov2Model
    hf_name = {"dinov2_vits14": "facebook/dinov2-small",
               "dinov2_vitb14": "facebook/dinov2-base"}.get(name, name)
    model = Dinov2Model.from_pretrained(hf_name)
    for p in model.parameters():
        p.requires_grad_(False)
    dim = model.config.hidden_size
    patch = model.config.patch_size
    return model.eval().to(device), dim, patch


class SlotAttention(nn.Module):
    def __init__(self, dim: int = 256, num_slots: int = 16, iters: int = 3):
        super().__init__()
        self.dim = dim
        self.num_slots = num_slots
        self.iters = iters
        self.slot_embed = nn.Parameter(torch.randn(1, num_slots, dim) * 0.02)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.gru = nn.GRUCell(dim, dim)
        self.mlp = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim),
                                 nn.ReLU(), nn.Linear(dim, dim))
        self.norm_in = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # feats: (B, N, D)
        B, N, D = feats.shape
        feats = self.norm_in(feats)
        k = self.k(feats)
        v = self.v(feats)
        slots = self.slot_embed.repeat(B, 1, 1)
        for _ in range(self.iters):
            slots_prev = slots
            q = self.q(self.norm_slots(slots))
            attn = torch.softmax(torch.einsum("bnd,bmd->bnm", q, k) / math.sqrt(D),
                                 dim=-1)
            updates = torch.einsum("bnm,bmd->bnd", attn, v)
            slots = self.gru(updates.reshape(-1, D), slots_prev.reshape(-1, D))
            slots = slots.reshape(B, -1, D)
            slots = slots + self.mlp(self.norm_slots(slots))
        return slots


class PerceptionModel(nn.Module):
    def __init__(self, num_types: int, dim: int = 256, num_slots: int = 16,
                 num_actions: int = 0, num_errors: int = 0,
                 dinov2_name: str = "dinov2_vits14", device="cpu",
                 img_size: int = 224, head_width: int = 256):
        super().__init__()
        self.num_slots = num_slots
        self.dim = dim
        self.img_size = img_size
        self.encoder, enc_dim, self.patch = build_dinov2(
            dinov2_name, device)
        self.grid = img_size // self.patch
        hw = head_width
        self.adapter = nn.Sequential(
            nn.Conv2d(enc_dim, hw, 1), nn.ReLU(), nn.Conv2d(hw, hw, 1))
        self.slots = SlotAttention(hw, num_slots)
        self.box_head = nn.Sequential(nn.LayerNorm(hw), nn.Linear(hw, 4))
        self.obj_head = nn.Sequential(nn.LayerNorm(hw), nn.Linear(hw, 1))
        self.cls_head = nn.Sequential(nn.LayerNorm(hw), nn.Linear(hw, num_types))
        # depth head: 384 -> 128 -> 64 -> 1, upsampling x4 to 56x56
        self.depth_head = nn.Sequential(
            nn.Conv2d(enc_dim, 160, 3, padding=1), nn.ReLU(),
            nn.Conv2d(160, 80, 3, padding=1), nn.ReLU(),
            nn.Conv2d(80, 40, 3, padding=1), nn.ReLU(),
            nn.Conv2d(40, 1, 1))
        if num_actions and num_errors:
            self.feas_head = nn.Sequential(
                nn.LayerNorm(enc_dim + 32 + 32),
                nn.Linear(enc_dim + 32 + 32, 256), nn.ReLU(),
                nn.Linear(256, 1 + num_errors))
            self.action_emb = nn.Embedding(num_actions, 32)
            self.arg_emb = nn.Embedding(4096, 32)
        else:
            self.feas_head = None
        self.enc_dim = enc_dim

        # dense CenterNet-style detector head (alternative to slots)
        self.dense_head = nn.Sequential(
            nn.Conv2d(hw, hw, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hw, hw, 3, padding=1), nn.ReLU())
        self.dense_obj = nn.Conv2d(hw, 1, 1)
        self.dense_off = nn.Conv2d(hw, 2, 1)
        self.dense_size = nn.Conv2d(hw, 2, 1)
        self.dense_cls = nn.Conv2d(hw, num_types, 1)

    def _patch_features(self, rgb: torch.Tensor):
        """Return (patch_feats (B,D,g,g), cls_token (B,enc_dim))."""
        with torch.no_grad():
            out = self.encoder(pixel_values=rgb).last_hidden_state
        cls = out[:, 0]
        patches = out[:, 1:]  # (B, g*g, enc_dim)
        B = patches.shape[0]
        g = int(round(math.sqrt(patches.shape[1])))
        patches = patches.reshape(B, g, g, self.enc_dim).permute(0, 3, 1, 2)
        return patches, cls

    def perception(self, rgb: torch.Tensor):
        patches, cls = self._patch_features(rgb)
        feat = self.adapter(patches)                       # (B,256,16,16)
        B, D, H, W = feat.shape
        slots = self.slots(feat.flatten(2).permute(0, 2, 1))  # (B,S,D)
        boxes = torch.sigmoid(self.box_head(slots))            # (B,S,4)
        obj = self.obj_head(slots).squeeze(-1)                 # (B,S)
        cls_logits = self.cls_head(slots)                      # (B,S,V)
        return {"slots": slots, "boxes": boxes, "objectness": obj,
                "class_logits": cls_logits, "cls": cls, "patch": feat}

    def depth(self, rgb: torch.Tensor) -> torch.Tensor:
        patches, _ = self._patch_features(rgb)
        d = self.depth_head(patches)   # (B,1,g,g)
        return F.interpolate(d, size=(self.img_size, self.img_size),
                             mode="bilinear", align_corners=False)

    def dense_detect(self, rgb: torch.Tensor):
        """CenterNet-style dense detection on the 16x16 patch grid.
        Returns dict with objectness (B,1,g,g), offset (B,2,g,g),
        size (B,2,g,g), class_logits (B,V,g,g)."""
        patches, _ = self._patch_features(rgb)
        feat = self.adapter(patches)
        f = self.dense_head(feat)
        return {
            "objectness": self.dense_obj(f),
            "offset": self.dense_off(f),
            "size": self.dense_size(f),
            "class_logits": self.dense_cls(f),
        }

    def feasibility(self, rgb: torch.Tensor, action: torch.Tensor,
                    arg_type: torch.Tensor):
        assert self.feas_head is not None
        _, cls = self._patch_features(rgb)
        a = self.action_emb(action)
        g = self.arg_emb(arg_type)
        x = torch.cat([cls, a, g], dim=-1)
        logits = self.feas_head(x)
        return logits[:, 0], logits[:, 1:]


def hungarian_match(boxes_pred: torch.Tensor, obj_pred: torch.Tensor,
                    boxes_gt: torch.Tensor, num_gt: torch.Tensor):
    """Optimal Hungarian matching (cost = -objectness + box L1), batched.
    Returns matched slot indices per sample, and the GT assignment for each
    slot (index into padded gt or -1)."""
    B, S, _ = boxes_pred.shape
    device = boxes_pred.device
    assign = torch.full((B, S), -1, dtype=torch.long, device=device)
    for b in range(B):
        n = int(num_gt[b].item())
        if n == 0:
            continue
        gt = boxes_gt[b, :n]  # (n,4) normalized
        cost = (torch.cdist(boxes_pred[b], gt, p=1)
                - obj_pred[b].unsqueeze(-1)).detach().cpu().numpy()
        try:
            from scipy.optimize import linear_sum_assignment
            si, gi = linear_sum_assignment(cost)
        except ImportError:
            si, gi = _hungarian_fallback(cost)
        for s, g in zip(si, gi):
            if g < n:
                assign[b, s] = g
    return assign


def _hungarian_fallback(cost: np.ndarray):
    """Min-cost matching without scipy (O(S^3) DP for small S)."""
    import numpy as np
    S, G = cost.shape
    INF = 1e9
    u = np.zeros(S + 1)
    v = np.zeros(G + 1)
    p = np.zeros(G + 1, dtype=int)
    way = np.zeros(G + 1, dtype=int)
    for i in range(1, S + 1):
        p[0] = i
        j0 = 0
        minv = np.full(G + 1, INF)
        used = np.zeros(G + 1, dtype=bool)
        while True:
            used[j0] = True
            i0, j1 = p[j0], 0
            delta = INF
            for j in range(1, G + 1):
                if not used[j]:
                    cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(G + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    return np.arange(S), p[1:] - 1


def giou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """boxes: (...,4) normalized cxcywh."""
    x1 = torch.min(boxes1[..., 0] - boxes1[..., 2] / 2,
                   boxes2[..., 0] - boxes2[..., 2] / 2)
    y1 = torch.min(boxes1[..., 1] - boxes1[..., 3] / 2,
                   boxes2[..., 1] - boxes2[..., 3] / 2)
    x2 = torch.max(boxes1[..., 0] + boxes1[..., 2] / 2,
                   boxes2[..., 0] + boxes2[..., 2] / 2)
    y2 = torch.max(boxes1[..., 1] + boxes1[..., 3] / 2,
                   boxes2[..., 1] + boxes2[..., 3] / 2)
    w1 = boxes1[..., 2] * boxes2[..., 2].clamp(min=1e-6)
    # fall back to IoU via standard computation
    inter = torch.clamp(torch.min(boxes1[..., 0] + boxes1[..., 2] / 2,
                                  boxes2[..., 0] + boxes2[..., 2] / 2) -
                        torch.max(boxes1[..., 0] - boxes1[..., 2] / 2,
                                  boxes2[..., 0] - boxes2[..., 2] / 2),
                        min=0) * torch.clamp(
        torch.min(boxes1[..., 1] + boxes1[..., 3] / 2,
                  boxes2[..., 1] + boxes2[..., 3] / 2) -
        torch.max(boxes1[..., 1] - boxes1[..., 3] / 2,
                  boxes2[..., 1] - boxes2[..., 3] / 2), min=0)
    area1 = boxes1[..., 2] * boxes1[..., 3]
    area2 = boxes2[..., 2] * boxes2[..., 3]
    union = area1 + area2 - inter + 1e-6
    iou = inter / union
    encl_w = torch.clamp(x2 - x1, min=0)
    encl_h = torch.clamp(y2 - y1, min=0)
    return iou - (encl_w * encl_h - union) / (encl_w * encl_h + 1e-6)


def dense_targets(boxes: torch.Tensor, classes: torch.Tensor,
                  num_boxes: torch.Tensor, grid: int = 16,
                  device="cpu", gauss_sigma: float = 0.8):
    """Convert normalized boxes (in [0,1] image coords) to dense 16x16 targets.
    Returns (obj_map (B,1,g,g), off_map (B,2,g,g), size_map (B,2,g,g),
    cls_map (B,g,g) as long, num_pos (B,))."""
    B, S, _ = boxes.shape
    obj = torch.zeros(B, 1, grid, grid, device=device)
    off = torch.zeros(B, 2, grid, grid, device=device)
    size = torch.zeros(B, 2, grid, grid, device=device)
    cls_map = torch.zeros(B, grid, grid, dtype=torch.long, device=device)
    num_pos = torch.zeros(B, dtype=torch.long, device=device)
    yy, xx = torch.meshgrid(torch.arange(grid, device=device),
                            torch.arange(grid, device=device), indexing="ij")
    for b in range(B):
        n = int(num_boxes[b].item())
        for s in range(n):
            cx, cy, w, h = boxes[b, s].tolist()
            if w < 0.005 or h < 0.005:
                continue
            gx = cx * grid
            gy = cy * grid
            # Gaussian blob around the center (standard CenterNet target)
            blob = torch.exp(-((xx - gx) ** 2 + (yy - gy) ** 2) /
                             (2 * gauss_sigma ** 2))
            obj[b, 0] = torch.maximum(obj[b, 0], blob)
            px = min(int(gx), grid - 1)
            py = min(int(gy), grid - 1)
            off[b, 0, py, px] = gx - px - 0.5
            off[b, 1, py, px] = gy - py - 0.5
            size[b, 0, py, px] = w
            size[b, 1, py, px] = h
            cls_map[b, py, px] = classes[b, s]
            num_pos[b] += 1
    return obj, off, size, cls_map, num_pos


def decode_dense(obj_map, off_map, size_map, cls_logits, grid: int = 16,
                 img_size: int = 224, obj_thr: float = 0.4,
                 nms_thr: float = 0.5, max_det: int = 30):
    """Decode dense outputs to (B, max_det, 4) boxes in [0,1] normalized
    image coords + class ids, with NMS."""
    B = obj_map.shape[0]
    out_boxes = torch.zeros(B, max_det, 4, device=obj_map.device)
    out_cls = torch.zeros(B, max_det, dtype=torch.long, device=obj_map.device)
    out_score = torch.zeros(B, max_det, device=obj_map.device)
    for b in range(B):
        o = torch.sigmoid(obj_map[b, 0])
        ys, xs = torch.nonzero(o > obj_thr, as_tuple=True)
        cands = []
        for y, x in zip(ys.tolist(), xs.tolist()):
            ox = (x + 0.5 + off_map[b, 0, y, x].item()) / grid
            oy = (y + 0.5 + off_map[b, 1, y, x].item()) / grid
            w = size_map[b, 0, y, x].item()
            h = size_map[b, 1, y, x].item()
            w = min(max(w, 0.01), 1.0)
            h = min(max(h, 0.01), 1.0)
            cls = int(cls_logits[b, :, y, x].argmax().item())
            score = float(o[y, x].item())
            cands.append([ox, oy, w, h, cls, score])
        cands.sort(key=lambda c: -c[5])
        keep = []
        for c in cands:
            if all(_iou(c[:4], k[:4]) < nms_thr for k in keep):
                keep.append(c)
            if len(keep) >= max_det:
                break
        for i, c in enumerate(keep):
            out_boxes[b, i] = torch.tensor(c[:4], device=obj_map.device)
            out_cls[b, i] = c[4]
            out_score[b, i] = c[5]
    return out_boxes, out_cls, out_score


def _iou(a, b):
    ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2
    ax2, ay2 = a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2
    bx2, by2 = b[0] + b[2] / 2, b[1] + b[3] / 2
    ix = max(0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / max(ua, 1e-9)
