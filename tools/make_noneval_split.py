#!/usr/bin/env python3
"""Write data/splits_noneval.json from whatever pools are present.

Train/val come only from scenes the SpatialWorld evaluation never uses:
  * AI2-THOR: every room except the 81 in data/eval_rooms.json;
  * ProcTHOR: houses from the procTHOR-10k **val** partition (the evaluation
    runs on the train partition) and the build's own FloorPlan_Val* houses;
  * VirtualHome scenes (the evaluation does not run VirtualHome at all).

The 81 evaluation rooms are written to `test` so a later report can measure the
perception head on exactly the rooms that matter, without ever training on them.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shared.data_index import load_index  # noqa: E402


def canon(scene: str) -> str:
    return re.sub(r"_physics$", "", str(scene))


def main() -> int:
    eval_rooms = set(json.loads((ROOT / "data/eval_rooms.json").read_text())["rooms"])
    idx = load_index()
    ai, procthor, vh = set(), set(), set()
    for fr in idx["frames"]:
        s = str(fr.get("scene") or "")
        if not s:
            continue
        if s.startswith("procthor") or s.startswith("FloorPlan_Val"):
            procthor.add(s)
        elif s.startswith("virtualhome"):
            vh.add(s)
        else:
            ai.add(canon(s))
    non_eval = sorted(r for r in ai if r not in eval_rooms)
    leaked = sorted(r for r in ai if r in eval_rooms)
    # Hold out one room per family.  The evaluation draws on several AI2-THOR
    # families that look nothing alike, so a validation set taken from one
    # family alone would select a head that is calibrated for that family only.
    def family(room: str) -> str:
        n = int(re.match(r"FloorPlan(\d+)", room).group(1))
        return "classic" if n <= 30 else f"{n // 100}xx"

    by_family: dict = {}
    for r in non_eval:
        by_family.setdefault(family(r), []).append(r)
    val_rooms = sorted(rooms[-1] for rooms in by_family.values() if len(rooms) >= 2)
    train_rooms = [r for r in non_eval if r not in val_rooms]
    pt = sorted(procthor)
    pt_val = pt[-5:]
    pt_train = [h for h in pt if h not in pt_val]
    vh_sorted = sorted(vh)
    vh_val = vh_sorted[-1:] if len(vh_sorted) > 1 else []
    vh_train = [h for h in vh_sorted if h not in vh_val]
    split = {
        "version": "noneval-v1",
        "note": ("Trains only on scenes the SpatialWorld evaluation never uses; "
                 "the 81 evaluation rooms stay in `test` as a clean held-out set."),
        "scene_splits": {
            "ai2thor_floorplans": {"train": train_rooms, "val": val_rooms,
                                   "test": sorted(eval_rooms)},
            "procthor_val": {"train": pt_train, "val": pt_val, "test": []},
            "virtualhome": {"train": vh_train, "val": vh_val, "test": []},
        },
        "counts": {
            "ai2thor_non_eval_rooms_with_data": len(non_eval),
            "ai2thor_train": len(train_rooms), "ai2thor_val": len(val_rooms),
            "ai2thor_test_eval_rooms": len(eval_rooms),
            "eval_rooms_seen_in_pools": len(leaked),
            "procthor_train": len(pt_train), "procthor_val": len(pt_val),
            "virtualhome_train": len(vh_train),
        },
    }
    out = ROOT / "data/splits_noneval.json"
    out.write_text(json.dumps(split, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")
    print(json.dumps(split["counts"], indent=1, ensure_ascii=False))
    if leaked:
        print(f"WARNING: evaluation rooms present in the pools (must stay in test): "
              f"{len(leaked)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
