"""Layer 4: learned occupancy completion.

Takes the agent's partial top-down walkability map (what floor has been seen
so far) and predicts the walkable cells that are currently occluded/unknown
(e.g. the open space behind a table).  Supervised by the episode-aggregated
floor observations (walkable = floor surface seen at y ~ 0).

Grid: world-aligned, 0.1 m cells, 80x80 (8m x 8m) around the scene center.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.geometry import unproject_mask  # noqa: E402

CELL = 0.1
GRID = 80
HALF = GRID * CELL / 2.0
MAX_DEPTH = 8.0


def _cell(p: np.ndarray) -> Tuple[int, int]:
    """World point -> (cx, cz) cell in the scene grid (origin at scene center)."""
    return int(np.floor((p[0] + HALF) / CELL)), int(np.floor((p[2] + HALF) / CELL))


def walkable_map_from_frame(depth: np.ndarray, agent_pos: dict, yaw: float,
                            horizon: float = 0.0
                            ) -> Tuple[np.ndarray, np.ndarray]:
    """Return (walkable mask 80x80, hit/obstacle mask 80x80) for one frame."""
    walk = np.zeros((GRID, GRID), dtype=bool)
    hit = np.zeros((GRID, GRID), dtype=bool)
    ys, us = np.mgrid[0:600:4, 0:800:4]
    zs = depth[ys, us].astype(np.float64)
    m = (zs > 0.3) & (zs < MAX_DEPTH)
    u, v, z = us[m].astype(float), ys[m].astype(float), zs[m]
    if len(z) == 0:
        return walk, hit
    pts = unproject_mask(u, v, z, agent_pos, yaw, horizon)
    # floor band: y near 0 (camera convention already accounts for height)
    floor = (np.abs(pts[:, 1]) < 0.15)
    for p in pts[floor]:
        c = _cell(p)
        if 0 <= c[0] < GRID and 0 <= c[1] < GRID:
            walk[c[0], c[1]] = True
    # obstacles: points above floor band and below ~1.8m (furniture/walls)
    obs = (pts[:, 1] > 0.2) & (pts[:, 1] < 1.8)
    for p in pts[obs]:
        c = _cell(p)
        if 0 <= c[0] < GRID and 0 <= c[1] < GRID:
            hit[c[0], c[1]] = True
    return walk, hit


def build_episode_samples(episode_frames, stride: int = 4) -> List[Dict]:
    """Aggregate walk/hit across an episode, then produce partial->full pairs.
    Returns samples: {partial (80,80), seen (80,80), target (80,80)}."""
    full_walk = np.zeros((GRID, GRID), dtype=bool)
    seen_walk = np.zeros((GRID, GRID), dtype=bool)
    all_frames = []
    for fr in episode_frames:
        if not fr.get("depth") or fr["depth"].endswith("/"):
            continue
        if not fr.get("agent") or not fr["agent"].get("position"):
            continue
        dep = np.asarray(Image.open(fr["depth"]).convert("I"),
                         dtype=np.float32) / 1000.0
        w, h = walkable_map_from_frame(dep, fr["agent"]["position"],
                                       fr["agent"]["yaw"],
                                       fr["agent"].get("horizon", 0.0) or 0.0)
        all_frames.append((w, h))
        full_walk |= w
    samples = []
    for i, (w, h) in enumerate(all_frames):
        seen_walk |= w
        if i % stride != 0 and i != len(all_frames) - 1:
            continue
        samples.append({
            "partial": seen_walk.astype(np.float32),
            "seen": (seen_walk | h).astype(np.float32),
            "target": full_walk.astype(np.float32),
        })
    return samples


class OccupancyUNet(nn.Module):
    """Small U-Net: 2 channels (partial walkability, seen mask) -> walkability
    probability.  ~1.6M params at ch=64."""

    def __init__(self, ch: int = 64):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(2, ch, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU())
        self.pool = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(nn.Conv2d(ch, ch * 2, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(ch * 2, ch * 2, 3, padding=1), nn.ReLU())
        self.enc3 = nn.Sequential(nn.Conv2d(ch * 2, ch * 4, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(ch * 4, ch * 4, 3, padding=1), nn.ReLU())
        self.up2 = nn.ConvTranspose2d(ch * 4, ch * 2, 2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(ch * 4, ch * 2, 3, padding=1), nn.ReLU())
        self.up1 = nn.ConvTranspose2d(ch * 2, ch, 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(ch * 2, ch, 3, padding=1), nn.ReLU())
        self.head = nn.Conv2d(ch, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d2 = self.up2(e3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.head(d1).squeeze(1)


class OccupancyDataset(torch.utils.data.Dataset):
    def __init__(self, samples: List[Dict]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        x = np.stack([s["partial"], s["seen"]])
        return (torch.from_numpy(x),
                torch.from_numpy(s["target"]).unsqueeze(0))


def collect_samples(index, max_episodes: int = 0,
                    stride: int = 4) -> List[Dict]:
    from collections import OrderedDict
    by_ep: "OrderedDict[str, list]" = OrderedDict()
    for fr in index["frames"]:
        if fr.get("action") == "Teleport" or not fr.get("rgb"):
            continue
        by_ep.setdefault(fr["episode"], []).append(fr)
    samples = []
    eps = list(by_ep.items())
    if max_episodes:
        eps = eps[:max_episodes]
    for ei, (name, frames) in enumerate(eps):
        samples.extend(build_episode_samples(frames, stride=stride))
        if (ei + 1) % 100 == 0:
            print(f"occupancy samples: {ei+1}/{len(eps)} episodes, "
                  f"{len(samples)} samples")
    return samples
