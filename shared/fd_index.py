"""Index FD benchmark trajectories + their RGB frames for feasibility training
and gate candidate extraction.  FD frames have NO depth/bbox; they provide
RGB + authoritative action/error labels with a richer error distribution
(not_in_view 58%, hand_occupied 18%) than the LightWM collection."""

from __future__ import annotations

import glob
import json
import os
import pickle
import re
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(__file__))
from data_index import classify_error  # noqa: E402

_FD_ROOT = os.environ.get("LIGHTWM_FD_ROOT",
                          "/mnt/d/fd_benchmark_full_20260811_224644")
FD_GLOB = os.path.join(_FD_ROOT, "ai2thor", "run_20260811_224645",
                       "*", "episode_*.json")
DEFAULT_FD_INDEX = "/home/sudidaren/lightwm_phases/data/fd_index.pkl"


def parse_action(s: str):
    m = re.match(r"^([A-Za-z]+)(?:\(([^)]*)\))?$", (s or "").strip())
    if not m:
        return None
    return m.group(1), m.group(2)


def build_fd_index(out_path: str = DEFAULT_FD_INDEX) -> Dict[str, Any]:
    frames: List[Dict] = []
    for p in sorted(glob.glob(FD_GLOB)):
        d = json.load(open(p))
        dirn = os.path.dirname(p)
        pngs = {}
        for x in glob.glob(os.path.join(dirn, "step_*.png")):
            m = re.search(r"step_(\d+)_", os.path.basename(x))
            if m:
                pngs[int(m.group(1))] = x
        for t in d.get("trajectory", []):
            a = parse_action(t.get("action_string") or "")
            if not a:
                continue
            step = t.get("step")
            rgb = pngs.get(step)
            if not rgb:
                continue
            err = t.get("error_message") or ""
            frames.append({
                "episode": os.path.basename(dirn),
                "scene": d.get("scene", ""),
                "step": step,
                "rgb": rgb,
                "action": a[0],
                "action_args": {"objectId": a[1] or ""},
                "success": not bool(err),
                "error_class": classify_error(err),
                "error_message": err,
            })
    actions = sorted({f["action"] for f in frames})
    errors = sorted({f["error_class"] for f in frames})
    idx = {"frames": frames, "actions": actions, "error_classes": errors}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(idx, f, protocol=4)
    print(f"FD index: {len(frames)} frames, {len(actions)} actions, "
          f"{len(errors)} errors")
    return idx


def load_fd_index(path: str = DEFAULT_FD_INDEX) -> Dict[str, Any]:
    if not os.path.exists(path):
        return build_fd_index(path)
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    build_fd_index()
