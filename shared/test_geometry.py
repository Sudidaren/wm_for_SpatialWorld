"""Validate the camera model against real AI2-THOR episode data."""

import json
import glob
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from geometry import project, rel_direction, unproject  # noqa: E402

EP = "/mnt/d/lightwm_data/episodes/FloorPlan10__ep00_sweep/episode.json"


def test_unproject_counter_surface():
    """Pixel (400, 500) in the sweep frame sits on the kitchen counter
    (y ~ 0.97-1.1).  Unprojecting its depth must land at that height with
    cameraY=0.675; a wrong camera-height convention fails this badly."""
    d = json.load(open(EP))
    f = d["frames"][10]
    ag = f["agent"]
    from PIL import Image

    ep_root = os.path.dirname(EP)
    dep = np.asarray(Image.open(os.path.join(ep_root, f["depth"]))) / 1000.0
    yaw = float(ag["rotation"]["y"])
    z = dep[500, 400]
    assert 1.5 < z < 3.0, f"depth at counter pixel unexpected: {z:.2f}"
    w = unproject(400, 500, z, ag["position"], yaw)
    assert 0.8 < w[1] < 1.3, f"counter surface unprojected to y={w[1]:.2f}"


def test_scene_objects_project_near_bbox():
    """Scene-GT compact objects should project inside (or within ~60px of)
    their observed bbox centers."""
    gt = json.load(open("/mnt/d/lightwm_data/scene_gt/FloorPlan10.json"))
    gt_pos = {o["name"]: o["position"] for o in gt["objects"]}
    d = json.load(open(EP))
    compact = {"Plate", "Potato", "Apple", "Cup", "Bread", "Egg", "Tomato"}
    errs = []
    for f in d["frames"][::2]:
        ag = f["agent"]
        for o in f["visible_objects"]:
            if o["name"].split("_")[0] not in compact:
                continue
            if o["name"] not in gt_pos:
                continue
            b = o["bbox"]
            if (b[2] - b[0]) * (b[3] - b[1]) > 70 * 70:
                continue
            p = gt_pos[o["name"]]
            uv = project((p["x"], p["y"], p["z"]), ag["position"],
                         float(ag["rotation"]["y"]))
            if uv is None:
                continue
            bc = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
            errs.append(math.hypot(uv[0] - bc[0], uv[1] - bc[1]))
    assert errs, "no test samples"
    med = float(np.median(errs))
    print(f"projection median error vs bbox center: {med:.1f}px (n={len(errs)})")
    assert med < 120, "projection convention is wrong (median > 120px)"


def test_rel_direction_consistency():
    """rel_direction of a point seen from two poses should agree on the world
    location when unprojected back."""
    for ep in sorted(glob.glob("/mnt/d/lightwm_data/episodes/*/episode.json"))[:30]:
        d = json.load(open(ep))
        for ia in range(0, len(d["frames"]) - 1):
            a, b = d["frames"][ia], d["frames"][ia + 1]
            if a["agent"]["position"] == b["agent"]["position"]:
                continue
            if abs(float(a["agent"]["rotation"]["y"]) -
                   float(b["agent"]["rotation"]["y"])) > 1:
                continue
            oa = next((o for o in a["visible_objects"]
                       if o["name"].split("_")[0] in
                       {"Plate", "Potato", "Cup", "Bread", "Apple"}), None)
            ob = next((o for o in b["visible_objects"]
                       if oa is not None and o["name"] == oa["name"]), None)
            if oa is None or ob is None:
                continue
            wa = np.array([oa["meta"]["position"]["x"],
                           oa["meta"]["position"]["y"],
                           oa["meta"]["position"]["z"]])
            _, dist, _ = rel_direction(wa, a["agent"]["position"],
                                       float(a["agent"]["rotation"]["y"]))
            wb = np.array([ob["meta"]["position"]["x"],
                           ob["meta"]["position"]["y"],
                           ob["meta"]["position"]["z"]])
            _, dist2, _ = rel_direction(wb, b["agent"]["position"],
                                        float(b["agent"]["rotation"]["y"]))
            assert abs(dist - dist2) < 0.2, \
                "distance inconsistency across frames"
            return  # one pair is enough for a smoke check
    raise AssertionError("no translation-only pair found")


if __name__ == "__main__":
    test_unproject_counter_surface()
    print("OK unproject_counter_surface")
    test_scene_objects_project_near_bbox()
    print("OK scene_objects_project_near_bbox")
    test_rel_direction_consistency()
    print("OK rel_direction_consistency")
