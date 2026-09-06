#!/usr/bin/env python3
"""Experiment A: pure map-reasoning probe on ai2thor00000.

Phase 1: BFS-walk FloorPlan17 (with the task init LookDown) at 0.25 m to build
         a perfect occupancy grid, including object positions from metadata.
Phase 2: One text-only GPT-5 call: given the ASCII map, output a full action
         sequence to put the Lettuce into the GarbageCan.
Phase 3: Play the returned sequence back in the simulator and report success.

Run with the ai2thor venv, e.g.:
  envs/ai2thor/.venv/bin/python -u envs/ai2thor/probe_expA.py
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ai2thor.controller import Controller
from actions.parser import parse_action_string
from mllm_base_agent.llm.messages import HumanMessage, SystemMessage
from mllm_base_agent.llm.provider import get_vlm

SCENE = "FloorPlan17"
TASK_DESC = "I found the lettuce was rotten; please help me throw it in the trash."
CELL = 0.25
API_BASE = os.environ.get("LIGHTWM_API_BASE", "https://apic1.ohmycdn.com/v1")

MOVES = [
    ("MoveAhead", "MoveBack"),
    ("MoveBack", "MoveAhead"),
    ("MoveLeft", "MoveRight"),
    ("MoveRight", "MoveLeft"),
]
DIRS = {
    # AI2-THOR at yaw=0: MoveAhead increases z, MoveBack decreases z
    "MoveAhead": (0.0, CELL),
    "MoveBack": (0.0, -CELL),
    "MoveLeft": (-CELL, 0.0),
    "MoveRight": (CELL, 0.0),
}

OBJECT_LEGEND = {
    "Lettuce": "L",
    "GarbageCan": "G",
    "CounterTop": "C",
    "Cabinet": "K",
    "Sink": "N",
    "StoveBurner": "V",
}


def _cell(x: float, z: float):
    return (round(x / CELL), round(z / CELL))


def build_occupancy(max_steps: int = 8000):
    """DFS-walk every reachable floor cell, physically moving the agent along
    the tree and backtracking by reverse moves, so the agent is always at the
    current node. Records occupancy + object positions."""
    c = Controller(
        scene=SCENE,
        width=256,
        height=192,
        gridSize=CELL,
        renderDepthImage=False,
        renderInstanceSegmentation=False,
        visibilityDistance=1.0,
    )
    c.reset(scene=SCENE)
    c.step(action="LookDown", degrees=30)  # task init.json
    meta = c.last_event.metadata
    ag = meta["agent"]
    start = (ag["position"]["x"], ag["position"]["z"])

    objects: dict = {}
    for o in meta.get("objects", []):
        t = o.get("objectType")
        p = o.get("position") or {}
        if t and p.get("x") is not None:
            objects.setdefault(t, []).append((p["x"], p["z"]))

    visited = {_cell(*start)}
    blocked = set()
    n_steps = 0

    def target_cell(x, z, action):
        dx, dz = DIRS[action]
        return _cell(x + dx, z + dz)

    # stack entries: (position, action-iterator, back_action)
    stack = [(start, iter(MOVES), None)]
    while stack and n_steps < max_steps:
        (x, z), it, back = stack[-1]
        advanced = False
        for action, rev in it:
            tcell = target_cell(x, z, action)
            if tcell in visited or tcell in blocked:
                continue
            ev = c.step(action=action, moveMagnitude=CELL)
            n_steps += 1
            ok = bool(ev.metadata.get("lastActionSuccess", False))
            if ok:
                ag2 = ev.metadata["agent"]
                nx, nz = ag2["position"]["x"], ag2["position"]["z"]
                cell = _cell(nx, nz)
                if cell not in visited:
                    visited.add(cell)
                    stack.append(((nx, nz), iter(MOVES), rev))
                    advanced = True
                    break
                # already visited: step back to the current node
                c.step(action=rev, moveMagnitude=CELL)
                n_steps += 1
            else:
                blocked.add(tcell)
        if not advanced:
            stack.pop()
            if back is not None:
                c.step(action=back, moveMagnitude=CELL)
                n_steps += 1
    c.stop()
    return start, objects, visited, blocked, n_steps


def render_map(start, objects, visited, blocked):
    def obj_cells():
        for t, pts in objects.items():
            if t in OBJECT_LEGEND:
                for (px, pz) in pts:
                    yield _cell(px, pz), OBJECT_LEGEND[t]

    ocells = list(obj_cells())
    xs = [c[0] for c in visited | blocked] + [cx for (cx, _cz), _ch in ocells]
    zs = [c[1] for c in visited | blocked] + [cz for (_cx, cz), _ch in ocells]
    x0, x1 = min(xs), max(xs)
    z0, z1 = min(zs), max(zs)

    label = {}
    for (cx, cz), ch in ocells:
        label[(cx, cz)] = ch
    for b in blocked:
        label.setdefault(b, "#")
    for v in visited:
        label.setdefault(v, ".")
    label[_cell(*start)] = "S"

    rows = []
    for z in range(z1, z0 - 1, -1):  # top row = largest z
        row = "".join(label.get((x, z), "?") for x in range(x0, x1 + 1))
        rows.append(row)
    return "\n".join(rows), (x0, x1, z0, z1), label


def ask_gpt5(map_text: str, out_dir: Path, retries: int = 3) -> str:
    vlm = get_vlm(
        provider="openai",
        model_name="gpt-5",
        temperature=1.0,  # this relay/model only supports the default (1)
        max_tokens=8192,  # GPT-5 burns tokens on reasoning; 2048 ran out silently
        base_url=API_BASE,
    )
    system = (
        "You are a navigation planner for an embodied agent in a grid kitchen. "
        "Reason carefully about the map and output a single valid action sequence. "
        "Output ONLY the action sequence, space-separated, with no explanation."
    )
    user = f"""Task: {TASK_DESC}

Here is a top-down grid map of the kitchen. Each cell is 0.25 m.
Legend:
  S = your start position (you are facing UP, toward the top of the map)
  L = Lettuce (pick it up)
  G = GarbageCan (put the Lettuce into it)
  C = CounterTop (obstacle), K = Cabinet (obstacle), N = Sink, V = StoveBurner
  # = wall or blocked cell
  . = walkable floor
  ? = inside the room bounds but not walkable (furniture/wall interior)

Map:
{map_text}

Grid positions: {_coord_note(map_text)} (row 0 is the top of the map).

Rules:
- MoveAhead / MoveBack / MoveLeft / MoveRight move 0.25 m by default; you may
  specify a magnitude, e.g. MoveAhead(0.5), MoveRight(1). You may chain several
  moves in the same direction into one larger move only along clear straight lines.
- RotateLeft / RotateRight turn 90 degrees and change which way you face.
- PickupObject(Lettuce) works only within 1 m of the Lettuce.
- PutObject(GarbageCan) works only within 1 m of the GarbageCan, and only while
  holding the Lettuce.
- PickupObject and PutObject also require the target to be IN YOUR FIELD OF VIEW:
  rotate to face the target before the interaction, and confirm you are facing it
  in the same step you interact.
- End the sequence with Done.

Output a single line like:
RotateRight(90) MoveRight(0.5) MoveAhead(0.25) PickupObject(Lettuce) MoveLeft(1) PutObject(GarbageCan) Done"""
    (out_dir / "prompt.txt").write_text(
        f"SYSTEM:\n{system}\n\nUSER:\n{user}", encoding="utf-8"
    )
    last = ""
    for attempt in range(retries):
        resp = vlm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        last = str(resp.content or "")
        if last.strip():
            return last
        print(f"  empty response (attempt {attempt + 1}), retrying...", flush=True)
    return last


def parse_route(text: str) -> list:
    text = re.sub(r"```[a-z]*", "", text)
    tokens = re.findall(r"[A-Za-z]+(?:\([^)]*\))?", text)
    out = []
    for t in tokens:
        if t.upper() in ("DONE", "FAIL"):
            out.append({"action_type": "task_completion", "action_name": t.upper()})
            break
        try:
            out.append(parse_action_string(t))
        except Exception:
            continue
    return out


def _coord_note(map_text: str) -> str:
    """Explicit row/col positions of S/L/G so distances are unambiguous."""
    rows = map_text.splitlines()
    found = []
    for ch, name in (("S", "start"), ("L", "Lettuce"), ("G", "GarbageCan")):
        for r, line in enumerate(rows):
            for c, cell in enumerate(line):
                if cell == ch:
                    found.append(f"{name} at row {r}, col {c}")
    return "; ".join(found)


def check_success(meta) -> bool:
    for o in meta.get("objects", []):
        if o.get("objectType") == "Lettuce":
            for pr in o.get("parentReceptacles", []) or []:
                if str(pr).split("|")[0] == "GarbageCan":
                    return True
    return False


def playback(route: list, out_dir: Path) -> dict:
    from mllm_base_agent.environments.ai2thor.wrapper import AI2ThorEnvWrapper

    env = AI2ThorEnvWrapper(
        scene=SCENE,
        width=512,
        height=384,
        output_dir=str(out_dir),
    )
    env.configure_task(
        target_object_types=["Lettuce"],
        success_predicate=lambda obj: False,
        target_description="throw the rotten lettuce into the trash",
    )
    obs = env.reset(TASK_DESC, scene=SCENE)
    env.step_with_action_dict(
        {"action_type": "navigation", "action_name": "LookDown", "degrees": 30}
    )
    log = []
    success = False
    for i, ad in enumerate(route):
        try:
            observation, err = env.step_with_action_dict(ad)
        except Exception as exc:
            log.append({"action": ad, "error": f"exception: {exc}"})
            break
        log.append(
            {
                "action": ad,
                "error_message": err,
                "success": err is None,
            }
        )
        if ad.get("action_name") in ("DONE", "FAIL"):
            break
        if observation is not None and check_success(observation.metadata):
            success = True
            break
    if obs is not None and check_success(obs.metadata):
        success = True
    env.close()
    return {"success": success, "n_actions": len(log), "log": log}


def main() -> None:
    reuse = None
    if len(sys.argv) > 1:
        reuse = Path(sys.argv[1])
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("/home/sudidaren/lightwm_test/expA") / f"run_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if reuse is not None and (reuse / "map.txt").exists():
        print(f"== Reusing map from {reuse} ==", flush=True)
        map_text = (reuse / "map.txt").read_text(encoding="utf-8")
        objects = json.loads((reuse / "objects.json").read_text(encoding="utf-8"))
        print(map_text)
        (out_dir / "map.txt").write_text(map_text, encoding="utf-8")
        (out_dir / "objects.json").write_text(
            json.dumps(objects, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    else:
        print("== Phase 1: building occupancy map (BFS) ==", flush=True)
        start, objects, visited, blocked, n_steps = build_occupancy()
        map_text, bounds, _label = render_map(start, objects, visited, blocked)
        print(f"BFS steps: {n_steps} | reachable cells: {len(visited)} | blocked: {len(blocked)}")
        print(map_text)
        print(f"objects: { {k: len(v) for k, v in objects.items()} }", flush=True)
        (out_dir / "map.txt").write_text(map_text, encoding="utf-8")
        (out_dir / "objects.json").write_text(
            json.dumps(objects, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print("\n== Phase 2: GPT-5 route planning (text-only) ==", flush=True)
    route_text = ask_gpt5(map_text, out_dir)
    print("--- GPT-5 output ---")
    print(route_text, flush=True)
    (out_dir / "gpt5_route.txt").write_text(route_text, encoding="utf-8")

    route = parse_route(route_text)
    print("\nparsed route:", [str(a) for a in route], flush=True)
    (out_dir / "route.json").write_text(
        json.dumps([str(a) for a in route], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n== Phase 3: playback ==", flush=True)
    result = playback(route, out_dir)
    print("playback success:", result["success"], "| actions executed:", result["n_actions"])
    for item in result["log"]:
        a = item["action"]
        if isinstance(a, dict):
            name = a.get("action_name")
            obj = a.get("object_type") or a.get("magnitude") or a.get("degrees") or ""
            desc = f"{name}({obj})"
        else:
            desc = str(a)
        print(f"  {desc} -> {'OK' if item['success'] else 'FAIL: ' + str(item['error_message'])[:70]}")
    (out_dir / "playback.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nresults saved under {out_dir}", flush=True)


if __name__ == "__main__":
    main()
