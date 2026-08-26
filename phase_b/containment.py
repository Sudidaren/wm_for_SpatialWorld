"""Learned containment head (on/in) for the scene graph.

Supervision: AI2-THOR metadata `visible_objects[].meta.parentReceptacles`
(the object is ON/IN that receptacle).  Features: object type one-hot +
receptacle type one-hot + relative 3D geometry.  A small MLP predicts
P(contained); the on/in split is geometric (object center above the
receptacle top -> on, below -> in).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.data_index import load_index  # noqa: E402


def extract(max_episodes: int = 0) -> dict:
    """Read raw episode.json files: positive (obj, rec) pairs from
    parentReceptacles; negatives from nearby furniture not in parents."""
    index = load_index()
    types = index["object_types"]
    t2i = {t: i for i, t in enumerate(types)}
    gt_files = {os.path.basename(g).replace(".json", ""): g
                for g in glob.glob(os.path.join(
                    os.environ.get("LIGHTWM_DATA_ROOT",
                                   "/mnt/d/lightwm_data"),
                    "scene_gt", "*.json"))}
    eps = sorted(glob.glob(
        os.path.join(os.environ.get("LIGHTWM_DATA_ROOT", "/mnt/d/lightwm_data"),
                     "episodes", "*", "episode.json")))
    if max_episodes:
        eps = eps[:max_episodes]
    X, y = [], []
    for p in eps:
        d = json.load(open(p))
        gpath = gt_files.get(d.get("scene"))
        gt = {}
        if gpath:
            g = json.load(open(gpath))
            for o in g.get("objects", []):
                gt[o["objectId"]] = o.get("position")
        for fr in d.get("frames", []):
            vis = fr.get("visible_objects", []) or []
            objs = {}
            for o in vis:
                objs[o.get("name")] = o
            for oid, o in objs.items():
                meta = o.get("meta", {}) or {}
                pr = meta.get("parentReceptacles") or []
                if not pr:
                    continue
                oi = t2i.get(_otype(o.get("name", "")))
                if oi is None:
                    continue
                op = meta.get("position")
                for rid in pr:
                    rt = rid.split("|")[0]
                    ri = t2i.get(rt)
                    if ri is None or op is None:
                        continue
                    rpos = gt.get(rid)
                    if rpos is None:
                        continue
                    dx = op["x"] - rpos["x"]
                    dy = op["y"] - rpos["y"]
                    dz = op["z"] - rpos["z"]
                    X.append(_feat(oi, ri, dx, dy, dz, len(types),
                                   op["y"]))
                    y.append(1.0)
            # negatives: furniture visible but NOT a parent
            furn = [(o, _otype(o.get("name", ""))) for o in vis
                    if _otype(o.get("name", "")) in t2i and _otype(
                        o.get("name", "")) in {
                        "CounterTop", "Cabinet", "Drawer", "Shelf", "Table",
                        "Dresser", "Desk", "SideTable"} and
                    (o.get("meta") or {}).get("position")]
            for oid, o in objs.items():
                oi = t2i.get(_otype(o.get("name", "")))
                op = (o.get("meta") or {}).get("position")
                if oi is None or op is None:
                    continue
                pr = set((o.get("meta") or {}).get("parentReceptacles") or [])
                for fo, ft in furn:
                    fp = fo["meta"]["position"]
                    if fo.get("name") in pr or _otype(
                            fo.get("name", "")) in {
                        t.split("|")[0] for t in pr}:
                        continue
                    dx = op["x"] - fp["x"]
                    dy = op["y"] - fp["y"]
                    dz = op["z"] - fp["z"]
                    if math_hypot(dx, dz) > 2.5:
                        continue
                    X.append(_feat(oi, t2i[ft], dx, dy, dz, len(types),
                                   op["y"]))
                    y.append(0.0)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    return {"X": X, "y": y, "types": types}


def math_hypot(a, b):
    return float(np.hypot(a, b))


def _feat(oi, ri, dx, dy, dz, n_types, obj_y=0.0):
    f = np.zeros(n_types * 2 + 5, dtype=np.float32)
    f[oi] = 1.0
    f[n_types + ri] = 1.0
    f[2 * n_types] = float(dx)
    f[2 * n_types + 1] = float(dy)
    f[2 * n_types + 2] = float(dz)
    f[2 * n_types + 3] = float(np.hypot(dx, dz))
    f[2 * n_types + 4] = float(obj_y)
    return f


def _otype(name: str) -> str:
    return name.split("_")[0] if "_" in name else name


class ContainmentNet(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim, 128), nn.ReLU(),
                                 nn.Linear(128, 64), nn.ReLU(),
                                 nn.Linear(64, 1))

    def forward(self, x):
        return self.mlp(x).squeeze(-1)


def main(max_episodes: int = 0, epochs: int = 15, seed: int = 0):
    ds = extract(max_episodes)
    X, y = ds["X"], ds["y"]
    print(f"samples={len(X)} positive_rate={y.mean():.3f}")
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(X))
    nv = max(1, len(X) // 10)
    tr, va = perm[nv:], perm[:nv]
    model = ContainmentNet(X.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    Xt = torch.from_numpy(X[tr])
    yt = torch.from_numpy(y[tr])
    Xv = torch.from_numpy(X[va])
    yv = torch.from_numpy(y[va])
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        pos_w = torch.tensor(
            max(1.0, (1 - yt.mean()) / max(yt.mean(), 1e-3)))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            model(Xt), yt, pos_weight=pos_w)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(Xv)) > 0.5
        tp = ((p == 1) & (yv > 0.5)).sum().item()
        fp = ((p == 1) & (yv <= 0.5)).sum().item()
        fn = ((p == 0) & (yv > 0.5)).sum().item()
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    print(f"containment head val: precision={prec:.3f} recall={rec:.3f} "
          f"n={len(yv)} pos={int((yv > 0.5).sum())}")
    torch.save(model.state_dict(),
               "/home/sudidaren/lightwm_phases/checkpoints/containment.pt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-episodes", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=15)
    args = ap.parse_args()
    main(args.max_episodes, args.epochs)
