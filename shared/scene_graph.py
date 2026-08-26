"""Scene graph with a real graph structure (Phase B upgrade).

Compared with the ad-hoc `scene_graph()` list in spatial_memory.py, this:
  - stores adjacency incrementally (only re-computes edges of changed anchors);
  - uses a spatial grid index for O(1) neighbour lookup instead of O(n^2);
  - supports containment relations (on/in) from a learned head or GT/causal
    memory;
  - exposes query API: objects_on(), nearest(), inside(), edges().
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

CELL = 1.0
MAX_EDGE_DIST = 6.0
FURNITURE = {
    "CounterTop", "Cabinet", "Drawer", "Shelf", "Fridge", "StoveBurner",
    "Sink", "Dresser", "Desk", "Table", "SideTable", "CoffeeTable",
    "DiningTable", "Bed", "Sofa", "Chair", "Toilet", "Bathtub", "Box",
}


class GraphNode:
    __slots__ = ("anchor_id", "obj_type", "pos", "conf")

    def __init__(self, anchor_id: str, obj_type: str, pos: np.ndarray,
                 conf: float):
        self.anchor_id = anchor_id
        self.obj_type = obj_type
        self.pos = np.asarray(pos, dtype=float)
        self.conf = conf


class SceneGraph:
    def __init__(self, containment_head=None):
        self.nodes: Dict[str, GraphNode] = {}
        self.adj: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._grid: Dict[Tuple[int, int], List[str]] = {}
        self.containment = containment_head  # callable(obj_type, rec_type, rel) -> "on"/"in"/None

    # -- node management ----------------------------------------------------
    def upsert_node(self, anchor_id: str, obj_type: str, pos, conf: float):
        if anchor_id in self.nodes:
            n = self.nodes[anchor_id]
            if np.linalg.norm(n.pos - np.asarray(pos)) < 0.05 and \
                    abs(n.conf - conf) < 0.01:
                return False
            self._remove_node(anchor_id)
        self.nodes[anchor_id] = GraphNode(anchor_id, obj_type, pos, conf)
        self._grid.setdefault(self._cell(pos), []).append(anchor_id)
        return True

    def remove_node(self, anchor_id: str):
        self._remove_node(anchor_id)

    def _remove_node(self, anchor_id: str):
        self.nodes.pop(anchor_id, None)
        cell = self._cell_of(anchor_id)
        if cell and anchor_id in self._grid.get(cell, []):
            self._grid[cell].remove(anchor_id)
        for other in list(self.adj.pop(anchor_id, {})):
            self.adj.get(other, {}).pop(anchor_id, None)

    def _cell_of(self, anchor_id: str) -> Optional[Tuple[int, int]]:
        n = self.nodes.get(anchor_id)
        return self._cell(n.pos) if n else None

    @staticmethod
    def _cell(pos) -> Tuple[int, int]:
        return int(np.floor(pos[0] / CELL)), int(np.floor(pos[2] / CELL))

    # -- incremental edge update -------------------------------------------
    def update_edges_for(self, anchor_id: str, max_dist: float = MAX_EDGE_DIST):
        """Recompute edges of one anchor against nearby anchors only."""
        n = self.nodes.get(anchor_id)
        if n is None:
            return
        self.adj.setdefault(anchor_id, {})
        for oid in list(self.adj[anchor_id]):
            self.adj[oid].pop(anchor_id, None)
            self.adj[anchor_id].pop(oid, None)
        cx, cz = self._cell(n.pos)
        for dcx in (-1, 0, 1):
            for dcz in (-1, 0, 1):
                for oid in self._grid.get((cx + dcx, cz + dcz), []):
                    if oid == anchor_id or oid not in self.nodes:
                        continue
                    other = self.nodes[oid]
                    d = float(np.linalg.norm(n.pos - other.pos))
                    if d > max_dist:
                        continue
                    rel = self._relation(n, other, d)
                    self.adj[anchor_id][oid] = rel
                    self.adj.setdefault(oid, {})[anchor_id] = rel

    def _relation(self, a: GraphNode, b: GraphNode, d: float) -> Dict[str, Any]:
        dx, dy, dz = b.pos - a.pos
        vert = "above" if dy > 0.25 else ("below" if dy < -0.25 else "level")
        horiz = ("left" if dx > 0.35 else "right" if dx < -0.35
                 else "front" if dz > 0.35 else "back" if dz < -0.35
                 else "beside")
        rel = f"{vert}+{horiz}"
        if self.containment is not None:
            c = self.containment(a.obj_type, b.obj_type,
                                 np.array([dx, dy, dz]), d, a.pos[1])
            if c:
                rel = c
        elif abs(dy) < 0.5 and math.hypot(dx, dz) < 0.8 and \
                a.obj_type in FURNITURE:
            rel = "on"
        return {"relation": rel, "distance_m": round(d, 2),
                "conf": round(min(a.conf, b.conf), 2)}

    # -- queries ------------------------------------------------------------
    def objects_on(self, anchor_id: str) -> List[str]:
        return [oid for oid, e in self.adj.get(anchor_id, {}).items()
                if e["relation"] == "on"]

    def inside(self, container_id: str) -> List[str]:
        return [oid for oid, e in self.adj.get(container_id, {}).items()
                if e["relation"] == "in"]

    def nearest(self, pos, types: Optional[List[str]] = None,
                max_dist: float = 3.0) -> Optional[str]:
        cx, cz = self._cell(pos)
        best, bd = None, max_dist
        for dcx in (-1, 0, 1):
            for dcz in (-1, 0, 1):
                for oid in self._grid.get((cx + dcx, cz + dcz), []):
                    n = self.nodes.get(oid)
                    if n is None or (types and n.obj_type not in types):
                        continue
                    d = float(np.linalg.norm(n.pos - np.asarray(pos)))
                    if d < bd:
                        best, bd = oid, d
        return best

    def edges(self, max_pairs: int = 100) -> List[Dict[str, Any]]:
        out = []
        seen = set()
        for a, others in self.adj.items():
            for b, e in others.items():
                key = tuple(sorted((a, b)))
                if key in seen:
                    continue
                seen.add(key)
                out.append({"a": self.nodes[a].obj_type, "b": self.nodes[b].obj_type,
                            **e})
        out.sort(key=lambda e: -e["conf"])
        return out[:max_pairs]

    def sync(self, anchors: Dict[str, Any]):
        """Sync from a SpatialMemory.anchors dict (anchor_id -> ObjectAnchor)."""
        changed = []
        for aid, a in anchors.items():
            if self.upsert_node(aid, a.obj_type, a.pos, a.conf):
                changed.append(aid)
        for aid in list(self.nodes):
            if aid not in anchors:
                self.remove_node(aid)
                changed.append(aid)
        for aid in changed:
            self.update_edges_for(aid)
