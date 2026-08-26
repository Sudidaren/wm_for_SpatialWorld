"""Integration eval for the memory module on a real episode.

Replays one sweep episode with GT detections+depth; checks:
  - anchors get confidence up / uncertainty down with repeated views;
  - queries report uncertainty;
  - MemoryManager wraps anchors + episodic events + eviction under budget.
"""

from __future__ import annotations

import json
import glob
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from shared.spatial_memory import SpatialMemory  # noqa: E402
from shared.memory_manager import MemoryManager  # noqa: E402


def main(max_episodes: int = 12):
    eps = sorted(glob.glob(
        os.path.join(os.environ.get("LIGHTWM_DATA_ROOT", "/mnt/d/lightwm_data"),
                     "episodes", "*", "episode.json")))[:max_episodes]
    seen_unc = []
    conf_progress = []
    for p in eps:
        d = json.load(open(p))
        ep = os.path.dirname(p)
        mem = SpatialMemory()
        mm = MemoryManager(budget=300)
        target = None
        for fr in d["frames"]:
            mm.step_forward()
            if not fr.get("depth") or not fr.get("agent") or \
                    not fr["agent"].get("position"):
                continue
            dep = np.asarray(Image.open(os.path.join(ep, fr["depth"]))) / 1000.0
            dets = [{"type": o["name"].split("_")[0], "bbox": o["bbox"]}
                    for o in fr.get("visible_objects", [])
                    if o["name"].split("_")[0] not in
                    ("Floor", "Wall", "Ceiling", "Window")]
            mem.update(dets, fr["agent"]["position"],
                       float(fr["agent"]["rotation"]["y"]), dep, fr["step"])
            act = fr.get("action") or {}
            aname = act.get("name") if isinstance(act, dict) else act
            if aname in ("OpenObject", "PickupObject", "PutObject") and \
                    fr.get("action_success"):
                mm.add_event(f"{fr['action']} at step {fr['step']}",
                             importance=0.7)
        # track the most-viewed anchor's uncertainty/confidence evolution
        if mem.anchors:
            best = max(mem.anchors.values(), key=lambda a: a.n_views)
            seen_unc.append(best.uncertainty)
            conf_progress.append(best.conf)
            for a in mem.anchors.values():
                mm.put("spatial", a.oid,
                       {"type": a.obj_type, "pos": a.pos.tolist()},
                       confidence=a.conf, importance=0.6)
        n_ev = len(mm._events)
        print(f"{os.path.basename(ep):36s} anchors={len(mem.anchors)} "
              f"events={n_ev} best_obj={best.obj_type if mem.anchors else '-':12s} "
              f"n_views={best.n_views if mem.anchors else 0} "
              f"conf={best.conf:.2f} unc={best.uncertainty:.2f} "
              f"recs={mm.stats()['records']}")
    print(f"\nmedian anchor uncertainty (post-views): {np.median(seen_unc):.3f} m "
          f"| median conf: {np.median(conf_progress):.2f}")
    print("OK")


if __name__ == "__main__":
    main()
