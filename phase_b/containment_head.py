"""Load the trained containment MLP and expose it as a SceneGraph hook."""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from phase_b.containment import ContainmentNet, _feat  # noqa: E402
from shared.data_index import load_index  # noqa: E402


class ContainmentHead:
    def __init__(self, ckpt: str = None, thr: float = 0.65):
        index = load_index()
        self.types = index["object_types"]
        self.t2i = {t: i for i, t in enumerate(self.types)}
        self.n_types = len(self.types)
        ckpt = ckpt or "/home/sudidaren/lightwm_phases/checkpoints/containment.pt"
        self.model = ContainmentNet(self.n_types * 2 + 5)
        self.model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        self.model.eval()
        self.thr = thr

    def __call__(self, obj_type: str, rec_type: str, rel: np.ndarray,
                 dist: float, obj_y: float = 0.9):
        oi = self.t2i.get(obj_type)
        ri = self.t2i.get(rec_type)
        if oi is None or ri is None:
            return None
        dx, dy, dz = float(rel[0]), float(rel[1]), float(rel[2])
        x = torch.from_numpy(_feat(oi, ri, dx, dy, dz, self.n_types,
                                   obj_y)).unsqueeze(0)
        with torch.no_grad():
            p = torch.sigmoid(self.model(x)).item()
        if p < self.thr:
            return None
        # object center above the receptacle top -> on, else in
        return "on" if dy > 0.15 else "in"


if __name__ == "__main__":
    h = ContainmentHead()
    print("on plate/counter:", h("Plate", "CounterTop",
                                 np.array([0.0, 0.05, 0.0]), 0.1))
    print("in apple/cabinet:", h("Apple", "Cabinet",
                                 np.array([0.0, -0.3, 0.0]), 0.4))
    print("far microwave/counter:", h("Microwave", "CounterTop",
                                      np.array([0.0, 0.4, 3.0]), 3.0))
