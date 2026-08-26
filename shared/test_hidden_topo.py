"""End-to-end: topological route from a real sweep + hidden-state inference
from the keys-task action log."""

import glob
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from hidden_state import HiddenState  # noqa: E402
from loop_closure import LoopClosure  # noqa: E402
from spatial_memory import SpatialMemory  # noqa: E402

DATA = os.environ.get("LIGHTWM_DATA_ROOT", "/mnt/d/lightwm_data")


def _dets(fr):
    return [{"type": o["name"].split("_")[0], "bbox": o["bbox"]}
            for o in fr.get("visible_objects", [])
            if o["name"].split("_")[0] not in
            ("Floor", "Wall", "Ceiling", "Window")]


def test_topo_from_sweep():
    eps = sorted(glob.glob(os.path.join(DATA, "episodes", "*",
                                        "episode.json")))
    for p in eps[:40]:
        d = json.load(open(p))
        ep = os.path.dirname(p)
        frames = [f for f in d["frames"]
                  if f.get("depth") and f.get("agent") and
                  f["agent"].get("position") and f.get("rgb")]
        if len(frames) < 30:
            continue
        lc = LoopClosure(SpatialMemory(), device="cpu")
        for i, fr in enumerate(frames[:40]):
            dep = np.asarray(Image.open(os.path.join(ep, fr["depth"]))) / 1000.0
            op = {"x": fr["agent"]["position"]["x"],
                  "z": fr["agent"]["position"]["z"], "y": 0.9}
            lc.observe(None, _dets(fr), op,
                       float(fr["agent"]["rotation"]["y"]), dep, i,
                       use_fp=False)
        if lc.topo.stats()["nodes"] >= 6:
            a = lc.topo.nearest(lc.topo.nodes[lc.topo.order[0]].world_pos)
            b = lc.topo.nearest(lc.topo.nodes[lc.topo.order[-1]].world_pos)
            path = lc.topo.route(a, b)
            hint = lc.topo.route_hint(
                lc.topo.nodes[lc.topo.order[0]].world_pos,
                lc.topo.nodes[lc.topo.order[-1]].world_pos)
            print(f"topo: nodes={lc.topo.stats()['nodes']} "
                  f"path_len={len(path) if path else 0} hint='{hint}'")
            assert hint
            return
    raise AssertionError("no suitable sweep episode found")


def test_hidden_state_keys():
    hs = HiddenState()
    found = False
    for p in glob.glob(os.path.join(DATA, "episodes", "*", "episode.json")):
        d = json.load(open(p))
        for fr in d["frames"]:
            act = fr.get("action") or {}
            name = act.get("name") if isinstance(act, dict) else act
            if name == "PutObject" and fr.get("action_success"):
                args = act.get("args", {}) if isinstance(act, dict) else {}
                oid = args.get("objectId") or ""
                if "Box" in oid:
                    found = True
                    break
        if found:
            for fr in d["frames"]:
                act = fr.get("action") or {}
                name = act.get("name") if isinstance(act, dict) else act
                args = act.get("args", {}) if isinstance(act, dict) else {}
                hs.note_action(name, args, bool(fr.get("action_success")),
                               fr.get("step", 0))
            break
    assert found, "no keys task episode with PutObject(Box) found"
    loc = hs.where_is("KeyChain")
    assert loc is not None and "Box" in loc["container"], loc
    print(f"hidden: KeyChain -> {loc['container']} (step {loc['step']}, "
          f"opened_since={loc['opened_since']})")
    print("OK")


if __name__ == "__main__":
    test_topo_from_sweep()
    test_hidden_state_keys()
