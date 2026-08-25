"""Post-process coverage episodes: keep only interactable objects that exist
in the scene GT (matching the original lightwm_data semantics), so the
type vocabulary stays clean (no Wall/Door/CeilingLight instance IDs)."""

from __future__ import annotations

import glob
import json
import os

ROOT = "/mnt/d/lightwm_data_cov"


def main():
    gts = {}
    for p in glob.glob(os.path.join(ROOT, "scene_gt", "*.json")):
        g = json.load(open(p))
        gts[g["scene"]] = {o["objectId"] for o in g["objects"]}
    for ep in sorted(glob.glob(os.path.join(ROOT, "episodes", "*",
                                            "episode.json"))):
        d = json.load(open(ep))
        allowed = gts.get(d["scene"], set())
        n_before = n_after = 0
        for f in d["frames"]:
            vis = f.get("visible_objects", []) or []
            n_before += len(vis)
            f["visible_objects"] = [o for o in vis
                                    if o.get("objectId") in allowed]
            n_after += len(f["visible_objects"])
        with open(ep, "w") as f:
            json.dump(d, f)
        print(f"{os.path.basename(os.path.dirname(ep))}: "
              f"{n_before} -> {n_after} visible objects")


if __name__ == "__main__":
    main()
