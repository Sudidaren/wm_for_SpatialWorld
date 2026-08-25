"""Extract VoI-gate training data from FD trajectories (offline replay).

For every action step we record which candidate hints the rule pool would
have issued BEFORE the action, with a feature vector and the label
"did this hint prevent an actual error of its class".

This mirrors gate_sim.py but produces a learnable dataset instead of counts.
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

FD_GLOB = ("/mnt/d/fd_benchmark_full_20260811_224644/ai2thor/"
           "run_20260811_224645/*/episode_*.json")
LIGHTWM_STEPS = "/home/sudidaren/lightwm_test/*/run_*/ai2thor*/steps.jsonl"

CAND_TYPES = [
    "hand_occupied", "not_holding", "look_limit", "repeat",
    "niv_retry", "blocked_retry",
]
TYPE_ID = {t: i for i, t in enumerate(CAND_TYPES)}

INTERACTIONS = {
    "PickupObject", "PutObject", "DropHandObject", "ThrowObject", "OpenObject",
    "CloseObject", "ToggleObjectOn", "ToggleObjectOff", "SliceObject",
    "BreakObject", "CookObject", "DirtyObject", "CleanObject",
    "FillObjectWithLiquid", "EmptyLiquidFromObject", "UseUpObject",
    "PushObject", "PullObject",
}
MOVES = {"MoveAhead", "MoveBack", "MoveLeft", "MoveRight"}


def parse_action(s: str):
    s = (s or "").strip()
    m = re.match(r"^([A-Za-z]+)(?:\(([^)]*)\))?$", s)
    if not m:
        return None
    return {"name": m.group(1), "arg": m.group(2)}


def classify_error(err: str) -> Optional[str]:
    e = (err or "").lower()
    if "hand already has an object" in e:
        return "hand_occupied"
    if "not holding" in e or "isn't holding" in e or "agent isn't holding" in e:
        return "not_holding"
    if "can't look down beyond" in e or "can't look up beyond" in e:
        return "look_limit"
    if "not in view" in e:
        return "not_in_view"
    if "blocking agent" in e or "blocked by" in e:
        return "blocked"
    return "other"


def candidate_features(name: str, holding: bool, horizon: float,
                       n_fail_recent: int, same_as_prev_failed: bool,
                       moved_since_fail: bool, step: int, max_steps: int,
                       recent_success: float) -> np.ndarray:
    f = np.zeros(len(CAND_TYPES) + 8, dtype=np.float32)
    return f


def step_features(name: str, arg: str, holding: bool, horizon: float,
                  n_fail_recent: int, same_as_prev_failed: bool,
                  moved_since_fail: bool, step: int, max_steps: int,
                  recent_success: float) -> np.ndarray:
    f = np.zeros(len(CAND_TYPES) + 8, dtype=np.float32)
    if name in TYPE_ID:
        f[TYPE_ID[name]] = 1.0
    f[len(CAND_TYPES) + 0] = float(holding)
    f[len(CAND_TYPES) + 1] = float(np.clip(horizon / 30.0, -1, 1))
    f[len(CAND_TYPES) + 2] = float(np.clip(n_fail_recent / 5.0, 0, 1))
    f[len(CAND_TYPES) + 3] = float(same_as_prev_failed)
    f[len(CAND_TYPES) + 4] = float(moved_since_fail)
    f[len(CAND_TYPES) + 5] = float(np.clip(step / max(1, max_steps), 0, 1))
    f[len(CAND_TYPES) + 6] = float(recent_success)
    f[len(CAND_TYPES) + 7] = float(name in MOVES)
    return f


def extract(glob_path: str = FD_GLOB) -> Dict:
    """Returns dataset dict with per-candidate rows and per-step records."""
    X, y, ctype, ep_ids, steps = [], [], [], [], []
    per_ep = []  # (episode, n_actions, n_errors, n_intercepted_by_rule)
    n_actions = n_errors = 0
    for p in sorted(glob.glob(glob_path)):
        d = json.load(open(p))
        holding = False
        horizon = 0.0
        prev_raw = None
        prev_err = ""
        prev_name = None
        moved_since_fail = True
        ep_actions = ep_errors = ep_hits = 0
        recent = []
        for t in d.get("trajectory", []):
            a = parse_action(t.get("action_string") or "")
            if not a:
                continue
            raw = (t.get("action_string") or "").strip()
            name, arg = a["name"], a["arg"]
            err = t.get("error_message") or ""
            cls = classify_error(err) if err else None
            n_actions += 1
            ep_actions += 1
            if cls:
                n_errors += 1
                ep_errors += 1
            n_fail_recent = sum(1 for r in recent[-5:] if r == 0)
            same_as_prev_failed = bool(prev_err) and raw == prev_raw
            recent_success = (sum(recent[-10:]) / max(1, len(recent[-10:])))
            feats = step_features(name, arg or "", holding, horizon,
                                  n_fail_recent, same_as_prev_failed,
                                  moved_since_fail, t.get("step", 0),
                                  d.get("max_steps", 50), recent_success)

            # candidates that fire BEFORE this action
            cands = []
            if name == "PickupObject" and holding:
                cands.append("hand_occupied")
            if name in ("DropHandObject", "PutObject", "ThrowObject") and not holding:
                cands.append("not_holding")
            if name == "LookDown" and horizon - _deg(arg) < -30:
                cands.append("look_limit")
            if name == "LookUp" and horizon + _deg(arg) > 30:
                cands.append("look_limit")
            if prev_err and raw == prev_raw:
                cands.append("repeat")
            if prev_err and classify_error(prev_err) == "not_in_view" and \
                    name in INTERACTIONS and not moved_since_fail:
                cands.append("niv_retry")
            if prev_err and classify_error(prev_err) == "blocked" and \
                    name in MOVES and not moved_since_fail:
                cands.append("blocked_retry")

            for c in cands:
                x = feats.copy()
                x[TYPE_ID[c]] = 1.0
                prevented = (cls == c) or (c == "repeat" and cls is not None)
                X.append(x)
                y.append(float(prevented))
                ctype.append(TYPE_ID[c])
                ep_ids.append(os.path.basename(os.path.dirname(p)))
                steps.append(t.get("step", 0))
                if prevented:
                    ep_hits += 1

            # execute state update
            if err:
                if name == "PickupObject":
                    holding = False
                prev_err = err
                moved_since_fail = False
                recent.append(0)
            else:
                if name in INTERACTIONS:
                    holding = name in ("PickupObject",)
                    prev_err = ""
                    moved_since_fail = True
                if name in MOVES:
                    moved_since_fail = True
                recent.append(1)
            prev_raw = raw
            prev_name = name
            if name in ("LookUp", "LookDown"):
                horizon = float(np.clip(
                    horizon + (_deg(arg) if name == "LookUp" else -_deg(arg)),
                    -30, 30))
        per_ep.append((os.path.basename(os.path.dirname(p)), ep_actions,
                       ep_errors, ep_hits))
    return {
        "X": np.asarray(X, dtype=np.float32),
        "y": np.asarray(y, dtype=np.float32),
        "ctype": np.asarray(ctype, dtype=np.int64),
        "episode": ep_ids,
        "step": steps,
        "per_episode": per_ep,
        "n_actions": n_actions,
        "n_errors": n_errors,
    }


def _recover_error(mem_hint: str) -> Tuple[Optional[str], str]:
    """Parse '上一个动作 X 失败：[error]' / '上一个动作 X 成功' from mem_hint."""
    h = mem_hint or ""
    m = re.search(r"上一个动作\s+([A-Za-z]+)\s+失败[:：]?\s*\[?(.*?)\]?$",
                  h, re.MULTILINE)
    if m:
        return m.group(1), (m.group(2) or "").strip()
    if "成功" in h:
        m2 = re.search(r"上一个动作\s+([A-Za-z]+)\s+成功", h)
        return (m2.group(1) if m2 else None), ""
    return None, ""


def extract_lightwm(glob_path: str = LIGHTWM_STEPS) -> Dict:
    """Extract gate candidates from real LightWM runtime logs (steps.jsonl).
    Errors are recovered from the templated mem_hint text."""
    X, y, ctype, ep_ids, steps = [], [], [], [], []
    per_ep = []
    n_actions = n_errors = 0
    for p in sorted(glob.glob(glob_path)):
        holding = False
        horizon = 0.0
        prev_raw = None
        prev_err = ""
        moved_since_fail = True
        recent = []
        ep_actions = ep_errors = ep_hits = 0
        for line in open(p):
            t = json.loads(line)
            a = parse_action(t.get("action_string") or "")
            if not a:
                continue
            raw = (t.get("action_string") or "").strip()
            name, arg = a["name"], a["arg"]
            prev_name, prev_err_text = _recover_error(t.get("mem_hint") or "")
            err = prev_err_text if prev_name is not None and prev_err_text else ""
            cls = classify_error(err) if err else None
            n_actions += 1
            ep_actions += 1
            if cls:
                n_errors += 1
                ep_errors += 1
            n_fail_recent = sum(1 for r in recent[-5:] if r == 0)
            same_as_prev_failed = bool(prev_err) and raw == prev_raw
            recent_success = sum(recent[-10:]) / max(1, len(recent[-10:]))
            feats = step_features(name, arg or "", holding, horizon,
                                  n_fail_recent, same_as_prev_failed,
                                  moved_since_fail, t.get("step", 0),
                                  50, recent_success)
            cands = []
            if name == "PickupObject" and holding:
                cands.append("hand_occupied")
            if name in ("DropHandObject", "PutObject", "ThrowObject") and not holding:
                cands.append("not_holding")
            if prev_err and raw == prev_raw:
                cands.append("repeat")
            if prev_err and classify_error(prev_err) == "not_in_view" and \
                    name in INTERACTIONS and not moved_since_fail:
                cands.append("niv_retry")
            if prev_err and classify_error(prev_err) == "blocked" and \
                    name in MOVES and not moved_since_fail:
                cands.append("blocked_retry")
            for c in cands:
                x = feats.copy()
                x[TYPE_ID[c]] = 1.0
                prevented = (cls == c) or (c == "repeat" and cls is not None)
                X.append(x)
                y.append(float(prevented))
                ctype.append(TYPE_ID[c])
                ep_ids.append(os.path.basename(os.path.dirname(
                    os.path.dirname(p))))
                steps.append(t.get("step", 0))
                if prevented:
                    ep_hits += 1
            # state update
            if err:
                holding = False
                prev_err = err
                moved_since_fail = False
                recent.append(0)
            else:
                holding = name in ("PickupObject",)
                prev_err = ""
                moved_since_fail = True
                recent.append(1)
            prev_raw = raw
        per_ep.append((os.path.basename(p), ep_actions, ep_errors, ep_hits))
    return {
        "X": np.asarray(X, dtype=np.float32),
        "y": np.asarray(y, dtype=np.float32),
        "ctype": np.asarray(ctype, dtype=np.int64),
        "episode": ep_ids,
        "step": steps,
        "per_episode": per_ep,
        "n_actions": n_actions,
        "n_errors": n_errors,
    }


def _deg(arg) -> float:
    if arg:
        try:
            return float(arg)
        except ValueError:
            pass
    return 30.0


def save(dataset: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path, X=dataset["X"], y=dataset["y"], ctype=dataset["ctype"],
        episode=np.asarray(dataset["episode"]), step=np.asarray(dataset["step"]),
        per_episode=np.asarray(
            [(a, b, c, d) for a, b, c, d in dataset["per_episode"]],
            dtype=object),
        n_actions=dataset["n_actions"], n_errors=dataset["n_errors"])


def load(path: str) -> Dict:
    z = np.load(path, allow_pickle=True)
    return {
        "X": z["X"], "y": z["y"], "ctype": z["ctype"],
        "episode": list(z["episode"]), "step": list(z["step"]),
        "per_episode": list(z["per_episode"]),
        "n_actions": int(z["n_actions"]), "n_errors": int(z["n_errors"]),
    }


if __name__ == "__main__":
    ds = extract()
    lw = extract_lightwm()
    print(f"FD: actions={ds['n_actions']} errors={ds['n_errors']} "
          f"rows={len(ds['X'])} pos={ds['y'].sum():.0f}")
    print(f"LightWM logs: actions={lw['n_actions']} errors={lw['n_errors']} "
          f"rows={len(lw['X'])} pos={lw['y'].sum():.0f}")
    for k in ("X", "y", "ctype", "episode", "step"):
        ds[k] = np.concatenate([ds[k], lw[k]])
    ds["per_episode"] = ds["per_episode"] + lw["per_episode"]
    ds["n_actions"] += lw["n_actions"]
    ds["n_errors"] += lw["n_errors"]
    print(f"merged: rows={len(ds['X'])} pos={ds['y'].sum():.0f}")
    out = "/home/sudidaren/lightwm_phases/data/gate_fd.npz"
    save(ds, out)
    print("saved", out)
