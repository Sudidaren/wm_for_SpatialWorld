"""M0 step 3: build data/scene_registry.json from the frame indexes.

Registry fields per scene:
  family, pools (per-pool episode/frame counts), has_fd,
  alfred_label (null until oct21 alignment), split (null until make_splits).

Also writes data/official120_missing.json: AI2-THOR's official 120 FloorPlan
list (4 families x 30) minus the ones present in our data -> collection queue.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import pickle
import re
import sys


def pool_of(path: str) -> str:
    if "_virtualhome" in path:
        return "virtualhome"
    if "_procthor" in path:
        return "procthor"
    if "_objviews" in path:
        return "objviews"
    if "_cov" in path:
        return "cov"
    if "fd_benchmark" in path:
        return "fd"
    return "lightwm_data"


def family(scene: str) -> str:
    m = re.search(r"FloorPlan(\d+)", scene or "")
    if not m:
        if (scene or "").startswith("procthor"):
            return "procthor-val"
        if (scene or "").startswith("virtualhome"):
            return "virtualhome"
        return "unknown"
    n = int(m.group(1))
    for lo, hi in ((1, 30), (201, 230), (301, 330), (401, 430)):
        if lo <= n <= hi:
            return f"FloorPlan {lo}-{hi}"
    return f"FloorPlan other"


FAMILIES = {
    "FloorPlan 1-30": range(1, 31),
    "FloorPlan 201-230": range(201, 231),
    "FloorPlan 301-330": range(301, 331),
    "FloorPlan 401-430": range(401, 431),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/frame_index.pkl")
    ap.add_argument("--fd-index", default="data/fd_index.pkl")
    args = ap.parse_args()

    def load(p):
        if not os.path.exists(p):
            return None
        with open(p, "rb") as f:
            return pickle.load(f)

    idx = load(args.index)
    fd = load(args.fd_index)
    if idx is None:
        sys.exit("frame_index missing")

    scenes = collections.defaultdict(lambda: {
        "family": None, "pools": collections.defaultdict(
            lambda: {"episodes": set(), "frames": 0}),
        "has_fd": False, "alfred_label": None, "split": None,
        "episodes": set(), "frames": 0})

    for fr in idx.get("frames", []):
        if not fr.get("rgb") or not fr.get("scene"):
            continue
        sc = fr["scene"]
        if sc.startswith("FloorPlan"):
            sc = re.match(r"(FloorPlan\d+)", sc).group(1)
        p = pool_of(fr["rgb"])
        rec = scenes[sc]
        rec["family"] = family(sc)
        rec["pools"][p]["frames"] += 1
        if fr.get("episode"):
            rec["pools"][p]["episodes"].add(fr["episode"])
            rec["episodes"].add(fr["episode"])
        rec["frames"] += 1

    if fd:
        for fr in fd.get("frames", []):
            if not fr.get("scene"):
                continue
            m = re.match(r"(FloorPlan\d+)", fr["scene"])
            if not m:
                continue
            sc = m.group(1)
            rec = scenes[sc]
            rec["has_fd"] = True
            rec["family"] = family(sc)
            if fr.get("episode"):
                rec["pools"]["fd"]["episodes"].add(fr["episode"])
                rec["episodes"].add(fr["episode"])
            rec["pools"]["fd"]["frames"] += 1
            rec["frames"] += 1

    out = {}
    for sc in sorted(scenes):
        rec = scenes[sc]
        rec["pools"] = {p: {"episodes": len(v["episodes"]), "frames": v["frames"]}
                        for p, v in sorted(rec["pools"].items())}
        rec["episodes"] = len(rec["episodes"])
        out[sc] = rec

    collected = set(out)
    missing = {}
    for fam, rng in FAMILIES.items():
        official = [f"FloorPlan{i}" for i in rng]
        absent = [s for s in official if s not in collected]
        if absent:
            missing[fam] = absent

    os.makedirs("data", exist_ok=True)
    with open("data/scene_registry.json", "w") as f:
        json.dump({"scenes": out, "meta": {
            "n_scenes": len(out),
            "n_ai2thor": sum(1 for s in out
                             if s.startswith("FloorPlan")),
            "n_procthor": sum(1 for s in out if s.startswith("procthor")),
            "n_virtualhome": sum(1 for s in out
                                 if s.startswith("virtualhome")),
        }, "official120_missing": missing},
            f, ensure_ascii=False, indent=1)
    with open("data/official120_missing.json", "w") as f:
        json.dump(missing, f, ensure_ascii=False, indent=1)

    print(f"[registry] scenes={len(out)} "
          f"(ai2thor={sum(1 for s in out if s.startswith('FloorPlan'))}, "
          f"procthor={sum(1 for s in out if s.startswith('procthor'))}, "
          f"virtualhome={sum(1 for s in out if s.startswith('virtualhome'))})")
    for fam, lst in missing.items():
        print(f"[registry] missing {fam}: {len(lst)} -> {lst[:6]}...")
    print("[registry] wrote data/scene_registry.json and "
          "data/official120_missing.json")


if __name__ == "__main__":
    main()
