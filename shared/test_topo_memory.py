"""Topological memory: graph build + BFS route + hint."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from topo_memory import TopoMemory  # noqa: E402


def test():
    t = TopoMemory()
    pts = [(0.0, 0.0), (0.0, 1.0), (0.0, 2.0), (0.0, 3.0), (1.0, 3.0),
           (2.0, 3.0), (2.0, 2.0)]
    for i, (x, z) in enumerate(pts):
        t.add_place(f"p{i}", np.array([x, 0.9, z]), i)
    s = t.stats()
    assert s["nodes"] == 7 and s["edges"] == 6, s
    path = t.route("p0", "p6")
    assert path == ["p0", "p1", "p2", "p3", "p4", "p5", "p6"], path
    h = t.route_hint(np.array([0.0, 0.9, 0.0]), np.array([2.0, 0.9, 2.0]))
    print("hint:", h)
    assert "走约" in h
    print(s, "| route OK | hint OK")
    print("OK")


if __name__ == "__main__":
    test()
