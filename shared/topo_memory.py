"""Topological memory: a route graph built from loop-closure place keyframes.

Nodes = places (world position + step).  Edges = movement between consecutive
places, stored as odometry deltas (locally accurate even after loop-closure
corrections).  Query: "how do I get from A to B" -> a short BFS path over the
place graph, rendered as walking hints.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np


class TopoNode:
    __slots__ = ("pid", "world_pos", "step")

    def __init__(self, pid: str, world_pos, step: int):
        self.pid = pid
        self.world_pos = np.asarray(world_pos, dtype=float)
        self.step = step


class TopoMemory:
    def __init__(self, edge_dist_thr: float = 4.0):
        self.nodes: Dict[str, TopoNode] = {}
        self.order: List[str] = []          # creation order
        self.edges: Dict[str, List[str]] = {}   # pid -> [pids]
        self.edge_info: Dict[Tuple[str, str], Dict] = {}
        self.edge_dist_thr = edge_dist_thr

    # -- construction -------------------------------------------------------
    def add_place(self, pid: str, world_pos, step: int):
        if pid in self.nodes:
            return
        self.nodes[pid] = TopoNode(pid, world_pos, step)
        self.order.append(pid)
        self.edges.setdefault(pid, [])
        if len(self.order) >= 2:
            prev = self.order[-2]
            d = float(np.linalg.norm(self.nodes[prev].world_pos -
                                     self.nodes[pid].world_pos))
            if d < self.edge_dist_thr:
                self._add_edge(prev, pid, d)
            else:
                # long jump (teleport / correction): no edge
                pass

    def _add_edge(self, a: str, b: str, d: float):
        self.edges.setdefault(a, []).append(b)
        self.edges.setdefault(b, []).append(a)
        self.edge_info[(a, b)] = {"distance_m": round(d, 2)}
        self.edge_info[(b, a)] = {"distance_m": round(d, 2)}

    # -- queries ------------------------------------------------------------
    def nearest(self, world_pos, max_dist: float = 3.0) -> Optional[str]:
        best, bd = None, max_dist
        for pid, n in self.nodes.items():
            d = float(np.linalg.norm(n.world_pos - np.asarray(world_pos)))
            if d < bd:
                best, bd = pid, d
        return best

    def route(self, start_pid: str, goal_pid: str
              ) -> Optional[List[str]]:
        """BFS path over the place graph."""
        if start_pid not in self.nodes or goal_pid not in self.nodes:
            return None
        if start_pid == goal_pid:
            return [start_pid]
        q = deque([start_pid])
        prev = {start_pid: None}
        while q:
            cur = q.popleft()
            if cur == goal_pid:
                break
            for nb in self.edges.get(cur, []):
                if nb not in prev:
                    prev[nb] = cur
                    q.append(nb)
        if goal_pid not in prev:
            return None
        path = []
        cur = goal_pid
        while cur is not None:
            path.append(cur)
            cur = prev[cur]
        path.reverse()
        return path

    def route_hint(self, from_pos, to_pos, max_jump: float = 3.0) -> str:
        """Human hint: how to walk from from_pos to to_pos via the graph."""
        a = self.nearest(from_pos)
        b = self.nearest(to_pos)
        if a is None or b is None:
            return ""
        path = self.route(a, b)
        if not path or len(path) < 2:
            return ""
        steps = []
        total = 0.0
        for i in range(len(path) - 1):
            x, y = path[i], path[i + 1]
            d = self.edge_info.get((x, y), {}).get("distance_m", 0.0)
            total += d
            steps.append(d)
        # direction of the first edge relative to current heading: skipped for
        # v1 (we report distances only; heading comes from geometry hints)
        dir_words = _dir_between(self.nodes[path[0]].world_pos,
                                 self.nodes[path[-1]].world_pos)
        return (f"沿路线{dir_words}走约 {total:.1f} m"
                f"（途经 {len(path) - 2} 个节点）")

    def stats(self) -> Dict:
        return {"nodes": len(self.nodes), "edges": sum(len(v) for v in
                                                       self.edges.values()) // 2}


def _dir_between(a: np.ndarray, b: np.ndarray) -> str:
    dx = b[0] - a[0]
    dz = b[-1] - a[-1]
    if abs(dx) < 0.3 and abs(dz) < 0.3:
        return "前方"
    if abs(dx) > abs(dz):
        return "左方" if dx > 0 else "右方"
    return "前方" if dz > 0 else "后方"
