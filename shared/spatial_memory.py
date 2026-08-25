"""Object-centric 3D spatial memory + scene graph.

Layer 2 (persistent object anchors): every detected object gets a 3D world
anchor fused across views (multi-view association by proximity + type +
appearance), with confidence that decays with distance/freshness.

Layer 3 (topology / scene graph): pairwise spatial relations computed from
anchors (on/in/left/right/above + distances), queried from any agent pose.

This is the module that guarantees "the world model always knows where the
microwave is after turning around, in which direction and how high".
"""

from __future__ import annotations

import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from geometry import camera_position, rel_direction, unproject

ASSOC_RADIUS = 0.6          # m: anchors closer than this (same type) merge
CONF_PER_VIEW = 0.15        # confidence added per consistent view
MAX_CONF = 0.95
FURNITURE = {
    "CounterTop", "Cabinet", "Drawer", "Shelf", "Fridge", "StoveBurner",
    "Sink", "Dresser", "Desk", "Table", "SideTable", "CoffeeTable",
    "DiningTable", "Bed", "Sofa", "Chair", "Toilet", "Bathtub",
}


class ObjectAnchor:
    __slots__ = ("oid", "obj_type", "pos", "conf", "n_views", "last_step",
                 "first_step", "appearance", "states", "height_band")

    def __init__(self, oid: str, obj_type: str, pos: np.ndarray,
                 step: int, appearance=None):
        self.oid = oid
        self.obj_type = obj_type
        self.pos = np.asarray(pos, dtype=float)
        self.conf = CONF_PER_VIEW
        self.n_views = 1
        self.last_step = step
        self.first_step = step
        self.appearance = appearance
        self.states: Dict[str, bool] = {}
        self.height_band = "mid"


class SpatialMemory:
    def __init__(self):
        self.anchors: Dict[str, ObjectAnchor] = {}
        self.step = 0

    # ------------------------------------------------------------------
    def update(
        self,
        detections: List[Dict[str, Any]],
        agent_pos: dict,
        yaw: float,
        depth: Optional[np.ndarray] = None,
        step: Optional[int] = None,
        horizon: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Fuse one frame's detections into persistent anchors.

        detections: [{type, bbox (x1,y1,x2,y2) in 800x600, appearance?}]
        depth: GT/predicted 800x600 depth (m) used to lift bbox centers to 3D.
        Returns the list of updated anchors (for logging).
        """
        if step is not None:
            self.step = step
        else:
            self.step += 1
        for det in detections:
            b = det["bbox"]
            uc = (b[0] + b[2]) / 2
            vc = (b[1] + b[3]) / 2
            z = self._depth_at(depth, uc, vc, b)
            if z is None or z <= 0.15:
                continue
            pos = unproject(uc, vc, z, agent_pos, yaw, horizon)
            # height sanity: below floor or absurdly high -> skip
            if pos[1] < -0.2 or pos[1] > 4.0:
                continue
            key = self._associate(det["type"], pos, det.get("appearance"))
            a = self.anchors[key]
            # EMA position update weighted by confidence
            w = a.conf / (a.conf + CONF_PER_VIEW)
            a.pos = w * a.pos + (1 - w) * pos
            a.conf = min(MAX_CONF, a.conf + CONF_PER_VIEW)
            a.n_views += 1
            a.last_step = self.step
            if det.get("appearance") is not None:
                if a.appearance is None:
                    a.appearance = det["appearance"]
                else:
                    a.appearance = 0.8 * a.appearance + 0.2 * det["appearance"]
        return self.summary(agent_pos, yaw, horizon)

    def _depth_at(self, depth, uc, vc, bbox) -> Optional[float]:
        if depth is None:
            return None
        y0, y1 = int(max(bbox[1], 0)), int(min(bbox[3] + 1, depth.shape[0]))
        x0, x1 = int(max(bbox[0], 0)), int(min(bbox[2] + 1, depth.shape[1]))
        patch = depth[y0:y1, x0:x1]
        if patch.size == 0:
            return None
        vals = patch[patch > 0.15]
        if vals.size == 0:
            return None
        return float(np.median(vals))

    def _associate(self, obj_type: str, pos: np.ndarray,
                   appearance) -> str:
        best, best_d = None, ASSOC_RADIUS
        for key, a in self.anchors.items():
            if a.obj_type != obj_type:
                continue
            d = float(np.linalg.norm(a.pos - pos))
            if d < best_d:
                # appearance gate (if available): cosine > 0.7
                if appearance is not None and a.appearance is not None:
                    cos = float(np.dot(appearance, a.appearance) /
                                (np.linalg.norm(appearance) *
                                 np.linalg.norm(a.appearance) + 1e-9))
                    if cos < 0.7:
                        continue
                best, best_d = key, d
        if best is not None:
            return best
        key = f"{obj_type}#{len(self.anchors)}"
        self.anchors[key] = ObjectAnchor(key, obj_type, pos, self.step,
                                         appearance)
        return key

    # ------------------------------------------------------------------
    def query(self, obj_type: str, agent_pos: dict, yaw: float,
              horizon: float = 0.0,
              max_age: int = 10 ** 9) -> Optional[Dict[str, Any]]:
        """Best-known location of an object type from the current pose.
        Returns dict with direction/height/distance bands + confidence,
        ready for hint rendering.  None if never seen or too stale."""
        cands = [a for a in self.anchors.values()
                 if a.obj_type == obj_type and
                 self.step - a.last_step <= max_age]
        if not cands:
            return None
        best = max(cands, key=lambda a: a.conf)
        rel, dist, (yaw_d, pitch_d) = rel_direction(
            best.pos, agent_pos, yaw, horizon)
        return {
            "type": obj_type,
            "oid": best.oid,
            "world_pos": best.pos.tolist(),
            "distance_m": round(dist, 2),
            "distance_band": self._dist_band(dist),
            "yaw_deg": round(yaw_d, 1),
            "pitch_deg": round(pitch_d, 1),
            "direction": self._dir_words(yaw_d, pitch_d),
            "height_band": self._height_band(best.pos[1], pitch_d),
            "confidence": round(best.conf, 2),
            "n_views": best.n_views,
            "last_step": best.last_step,
        }

    def query_all(self, agent_pos: dict, yaw: float,
                  horizon: float = 0.0) -> List[Dict]:
        return [q for t in {a.obj_type for a in self.anchors.values()}
                if (q := self.query(t, agent_pos, yaw, horizon))]

    def scene_graph(self, max_pairs: int = 50) -> List[Dict[str, Any]]:
        """Pairwise relations between anchors (Layer 3)."""
        keys = list(self.anchors)
        edges = []
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = self.anchors[keys[i]], self.anchors[keys[j]]
                d = float(np.linalg.norm(a.pos - b.pos))
                if d > 6.0:
                    continue
                dx, dy, dz = b.pos - a.pos
                vert = "above" if dy > 0.25 else ("below" if dy < -0.25 else "level")
                horiz = ("left" if dx > 0.35 else "right" if dx < -0.35
                         else "front" if dz > 0.35 else "back" if dz < -0.35
                         else "beside")
                on = (abs(dy) < 0.5 and math.hypot(dx, dz) < 0.8
                      and a.obj_type in FURNITURE)
                edges.append({
                    "a": a.obj_type, "b": b.obj_type,
                    "relation": "on" if on else f"{vert}+{horiz}",
                    "distance_m": round(d, 2),
                    "conf": round(min(a.conf, b.conf), 2),
                })
        edges.sort(key=lambda e: -e["conf"])
        return edges[:max_pairs]

    def summary(self, agent_pos: dict, yaw: float,
                horizon: float = 0.0) -> List[Dict]:
        return self.query_all(agent_pos, yaw, horizon)

    # ------------------------------------------------------------------
    @staticmethod
    def _dist_band(d: float) -> str:
        return "near" if d < 1.0 else ("mid" if d < 3.0 else "far")

    @staticmethod
    def _dir_words(yaw_d: float, pitch_d: float) -> str:
        y = ("front" if abs(yaw_d) <= 30 else
             "right" if yaw_d > 30 else "left")
        if yaw_d > 150 or yaw_d < -150:
            y = "back"
        p = "up" if pitch_d > 12 else ("down" if pitch_d < -12 else "level")
        return f"{y}-{p}" if p != "level" else y

    @staticmethod
    def _height_band(y: float, pitch_d: float) -> str:
        if y > 1.3:
            return "high"
        if y > 0.55:
            return "mid"
        return "low"

    def save(self, path: str) -> None:
        data = {
            "step": self.step,
            "anchors": [
                {"oid": a.oid, "type": a.obj_type, "pos": a.pos.tolist(),
                 "conf": a.conf, "n_views": a.n_views,
                 "last_step": a.last_step, "first_step": a.first_step,
                 "states": a.states}
                for a in self.anchors.values()
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=1)

    def load(self, path: str) -> None:
        with open(path) as f:
            data = json.load(f)
        self.step = data["step"]
        self.anchors = {}
        for a in data["anchors"]:
            obj = ObjectAnchor(a["oid"], a["type"], a["pos"], a["first_step"])
            obj.conf, obj.n_views, obj.last_step = (
                a["conf"], a["n_views"], a["last_step"])
            obj.states = a.get("states", {})
            self.anchors[obj.oid] = obj
