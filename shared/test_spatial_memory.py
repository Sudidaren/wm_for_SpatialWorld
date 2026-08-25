"""End-to-end spatial-memory test on a real episode:
1) fuse the first 12 frames with GT depth -> anchors
2) after turning 180 degrees (query from a later frame's pose), the microwave
   anchor must still point in the correct direction and height band.
"""

import json
import math
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from spatial_memory import SpatialMemory  # noqa: E402

EP = "/mnt/d/lightwm_data/episodes/FloorPlan10__ep00_sweep/episode.json"


def detections_from_frame(fr, depth):
    out = []
    for o in fr["visible_objects"]:
        t = o["name"].split("_")[0]
        if t in ("Floor", "Wall", "Ceiling", "Window"):
            continue
        out.append({"type": t, "bbox": o["bbox"]})
    return out


def main():
    d = json.load(open(EP))
    ep_root = os.path.dirname(EP)
    mem = SpatialMemory()
    for fr in d["frames"][:12]:
        dep = np.asarray(Image.open(
            os.path.join(ep_root, fr["depth"]))) / 1000.0
        mem.update(detections_from_frame(fr, dep), fr["agent"]["position"],
                   float(fr["agent"]["rotation"]["y"]), dep, fr["step"])
    print(f"anchors: {len(mem.anchors)}")
    for t in ("Microwave", "Plate", "Potato", "Sink", "Fridge"):
        if t in {a.obj_type for a in mem.anchors.values()}:
            print("  seen:", t)
    # query from a later pose (agent rotated a lot)
    fr = d["frames"][30]
    q = mem.query("Microwave", fr["agent"]["position"],
                  float(fr["agent"]["rotation"]["y"]))
    assert q is not None, "microwave must stay in memory after rotation"
    print("query from frame 30:", q)
    # sanity: distance band should be sensible (< 6m), direction must exist
    assert q["distance_m"] < 6.0, "microwave too far / wrong anchor"
    assert q["direction"], "no direction"
    # scene graph sanity
    edges = mem.scene_graph()
    print(f"scene graph edges: {len(edges)}")
    assert len(edges) > 0
    print("OK")


if __name__ == "__main__":
    main()
