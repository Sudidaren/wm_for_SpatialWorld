"""Scene graph: incremental edges + queries."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from scene_graph import SceneGraph  # noqa: E402


def test():
    g = SceneGraph()
    g.upsert_node("plate#1", "Plate", np.array([0.5, 1.02, -1.0]), 0.9)
    g.upsert_node("counter#1", "CounterTop", np.array([0.5, 0.98, -1.0]), 0.9)
    g.upsert_node("microwave#1", "Microwave", np.array([0.5, 1.45, -1.0]), 0.8)
    for a in ("plate#1", "counter#1", "microwave#1"):
        g.update_edges_for(a)
    print("objects_on(counter):", g.objects_on("counter#1"))
    assert "plate#1" in g.objects_on("counter#1")
    near = g.nearest(np.array([0.4, 0.9, -0.9]), types=["CounterTop"])
    assert near == "counter#1"
    print("edges:", g.edges())
    print("OK")


if __name__ == "__main__":
    test()
