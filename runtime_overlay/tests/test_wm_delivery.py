#!/usr/bin/env python
"""Delivery verification for the WingmanWM runtime.

Run inside a SpatialWorld checkout that has the LightWM overlay applied::

    python tests/test_wm_delivery.py            # from the repo root

It needs no GPU, no simulator and no network: the perception runtime and the
VLM are stubbed.  Two kinds of check:

  A. Information-isolation audit (static)
     The world-model path must never touch a simulator-only channel.  The
     module docstrings are stripped before matching, so prose that *describes*
     the isolation does not count as a violation.

  B. Functional smoke (behavioural)
     With a stubbed perception runtime and a stubbed VLM, the runtime must
     actually produce: the per-step memory readout, the target-position hint
     (when enabled), the CheckState summary, and world-model anchors +
     dead-reckoned pose.

Every failure prints what was expected and what was found, so this doubles as
the evidence that the published repo reproduces the intended behaviour.
"""

from __future__ import annotations

import ast
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

AGENT_DIR = REPO / "mllm_base_agent" / "agent"

#: every file the world-model path is allowed to consist of
WM_FILES = (
    "world_model.py",
    "memory_probe.py",
    "target_priority.py",
    "object_query.py",
    "self_observation.py",
    "plan.py",
)

#: Simulator-only *attributes*: reaching any of these means the WM read
#: something the frozen baseline agent cannot see.  Matched as attribute
#: accesses, so a docstring or an error message that merely mentions the word
#: is not a hit.
FORBIDDEN_ATTRS = (
    (r"\.metadata\b", "simulator per-step metadata"),
    (r"\.controller\b", "simulator controller handle"),
    (r"\.segmentation\w*\b", "simulator instance segmentation"),
    (r"\.instance_seg\w*\b", "simulator instance segmentation"),
    (r"\.depth_frame\b|\.gt_depth\b|\.depth_gt\b", "ground-truth depth"),
    (r"\.visible_objects\b|\.objects_meta\b|\.obj_meta\b", "ground-truth objects"),
    (r"\.golden\w*\b", "golden action plan"),
)

#: Names that only exist on the task-truth side of the harness.
FORBIDDEN_NAMES = r"\b(target_object_types|success_condition|success_conditions|golden_action|golden_steps)\b"

#: Parameters that would hand the WM the environment, its raw metadata, or a
#: simulator-derived quantity.  The WM's own per-step dict is deliberately
#: called ``wm_metadata`` so it can never be confused with
#: ``observation.metadata``.
FORBIDDEN_PARAMS = {"env", "environment", "controller", "metadata", "sim",
                    "sim_pose", "ground_truth"}

_RESULTS = []


def check(label, fn):
    try:
        fn()
    except AssertionError as exc:
        _RESULTS.append((label, False, str(exc)))
        print(f"FAIL {label}: {exc}")
    except Exception as exc:                                  # noqa: BLE001
        _RESULTS.append((label, False, f"{type(exc).__name__}: {exc}"))
        print(f"FAIL {label}: {type(exc).__name__}: {exc}")
    else:
        _RESULTS.append((label, True, ""))
        print(f"ok   {label}")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _code_only(path: Path) -> str:
    """Source with comments and docstrings removed."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body:
            first = body[0]
            if (isinstance(first, ast.Expr)
                    and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)):
                body.pop(0)
    return ast.unparse(tree)


def meta(agent_xy=(0.0, 0.0), yaw=0.0, objects=(), held=None):
    return {
        "agent": {"position": {"x": agent_xy[0], "y": 0.9, "z": agent_xy[1]},
                  "rotation": {"x": 0.0, "y": yaw, "z": 0.0}},
        "objects": [
            {"objectType": t, "position": {"x": x, "y": y, "z": z},
             "visible": vis, "distance": dist, "sigma": sig}
            for (t, x, y, z, vis, dist, sig) in objects
        ],
        "inventoryObjects": ([{"objectId": "det|" + held + "|1",
                               "objectType": held}] if held else []),
    }


class StubVLM:
    """Answers the one-shot object-extraction call with a fixed JSON list."""

    def __init__(self, objects):
        self.objects = list(objects)
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        payload = '{"objects": [' + ", ".join(
            '{"env": "%s", "words": "%s"}' % (o, o.lower())
            for o in self.objects) + "]}"
        return type("R", (), {"content": payload})


class StubPerception:
    """Callable perception runtime: detector boxes + a metric depth map.

    It returns exactly the interface ``WorldModel.observe`` consumes -- a
    dict with ``detections`` (each carrying a pixel ``center`` and a ``type``)
    and a metric ``depth`` map -- so no simulator and no checkpoint are needed.
    """

    def __init__(self, ckpt=None):
        self.depth_value = 2.0

    def __call__(self, rgb):
        import numpy as np

        h, w = rgb.shape[:2]
        depth = np.full((h, w), self.depth_value, dtype=np.float32)
        return {
            "detections": [
                {"type": "Apple", "center": (w // 2, h // 2), "score": 0.9},
            ],
            "depth": depth,
        }

    def embed(self, rgb):
        import numpy as np

        v = np.ones(8, dtype=np.float32)
        return v / np.linalg.norm(v)


# --------------------------------------------------------------------------
# A. information-isolation audit
# --------------------------------------------------------------------------

def test_every_wm_file_exists():
    missing = [f for f in WM_FILES if not (AGENT_DIR / f).exists()]
    assert not missing, f"missing runtime files: {missing}"


def test_no_simulator_channel_in_wm_code():
    hits = []
    for name in WM_FILES:
        code = _code_only(AGENT_DIR / name)
        for pattern, why in FORBIDDEN_ATTRS:
            m = re.search(pattern, code)
            if m:
                hits.append(f"{name}: {why!r} matched {m.group(0)!r}")
        m = re.search(FORBIDDEN_NAMES, code)
        if m:
            hits.append(f"{name}: task truth {m.group(0)!r}")
        if re.search(r"\berror_message\b", code):
            hits.append(f"{name}: simulator action error string")
    assert not hits, "simulator channel reached the WM path:\n  " + "\n  ".join(hits)


def test_wm_functions_are_not_handed_the_environment():
    """No WM entry point takes env / controller / metadata as a parameter."""
    bad = []
    for name in WM_FILES:
        tree = ast.parse((AGENT_DIR / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            args = list(node.args.posonlyargs) + list(node.args.args) + \
                list(node.args.kwonlyargs)
            for arg in args:
                if arg.arg in FORBIDDEN_PARAMS:
                    bad.append(f"{name}:{node.lineno} {node.name}(... {arg.arg}=...)")
    assert not bad, ("the environment object is being passed into the WM:\n  "
                     + "\n  ".join(bad))


def test_wm_does_not_import_the_environment():
    bad = []
    for name in WM_FILES:
        tree = ast.parse((AGENT_DIR / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                mods = [node.module or ""]
            for mod in mods:
                if "environments" in mod or mod.startswith("envs"):
                    bad.append(f"{name}: imports {mod}")
    assert not bad, "WM imports the environment:\n  " + "\n  ".join(bad)


def test_target_hint_has_no_object_vocabulary():
    """The object names may only come from the model, never from this file."""
    from mllm_base_agent.agent import target_priority as tp

    src = (AGENT_DIR / "target_priority.py").read_text(encoding="utf-8")
    # the extraction prompt legitimately shows the *format* with example tokens
    prompt = tp.EXTRACTION_PROMPT
    stripped = src.replace(prompt, "")
    tokens = re.findall(r'"([A-Z][A-Za-z]{3,})"|\'([A-Z][A-Za-z]{3,})\'', stripped)
    names = {a or b for a, b in tokens}
    assert not names, f"object-name literals found in target_priority.py: {sorted(names)}"


def test_runtime_never_reads_task_truth():
    """No runtime module may consult the task definition or golden actions."""
    bad = []
    for name in WM_FILES + ("runner.py",):
        code = _code_only(AGENT_DIR / name)
        for pattern in (r"target_object_types", r"success_condition",
                        r"\bgolden\w*", r"task\.json"):
            if re.search(pattern, code):
                bad.append(f"{name}: {pattern}")
    assert not bad, "task truth reached the runtime:\n  " + "\n  ".join(bad)


# --------------------------------------------------------------------------
# B. functional smoke
# --------------------------------------------------------------------------

def test_memory_readout_lines():
    from mllm_base_agent.agent.memory_probe import MemoryProbe

    probe = MemoryProbe(task_description="open my laptop")
    block = probe.update(
        wm_metadata=meta(objects=[("Laptop", 0.0, 0.9, 2.0, True, 2.0, 0.2)]),
        action_name="MoveAhead", object_type=None,
        blocked=True, action_ok=False,
    )
    assert "手持：空手" in block, block
    assert "移动提示" in block, block
    assert "上一个动作：MoveAhead（失败" in block, block

    held = probe.update(wm_metadata=meta(held="Egg"), action_name="PickupObject",
                        object_type="Egg", blocked=False, action_ok=True)
    assert "手持：Egg" in held, held
    assert "上一个动作：PickupObject（成功" in held, held


def test_target_hint_off_by_default():
    from mllm_base_agent.agent.memory_probe import MemoryProbe

    probe = MemoryProbe(task_description="slice the apple")
    probe.set_vlm(StubVLM(["Apple"]))
    probe.set_target_hint({"enabled": False})
    block = probe.update(
        wm_metadata=meta(objects=[("Apple", 0.0, 0.9, 2.0, False, 2.0, 0.2)]),
        action_name="RotateLeft", blocked=False, action_ok=True)
    assert "任务相关记忆" not in block, block
    assert "Apple" not in block, block
    assert "手持：空手" in block, block


def test_target_hint_reports_detected_object():
    from mllm_base_agent.agent.memory_probe import MemoryProbe

    vlm = StubVLM(["Apple"])
    probe = MemoryProbe(task_description="slice the apple")
    probe.set_vlm(vlm)
    probe.set_target_hint({"enabled": True, "limit": 6, "names_limit": 5,
                           "max_dist": 3.0, "max_sigma": 0.5})
    block = probe.update(
        wm_metadata=meta(objects=[("Apple", 0.0, 0.9, 2.0, False, 2.0, 0.2)]),
        action_name="RotateLeft", blocked=False, action_ok=True)
    assert vlm.calls == 1, f"one extra text call expected, got {vlm.calls}"
    assert "任务相关记忆" in block, block
    assert "Apple" in block, block
    assert "约 2.0m" in block, block


def test_check_state_returns_summary_once():
    from mllm_base_agent.agent import object_query as oq

    st = {"config": {"memory_probe": {"object_query": {"enabled": True}}},
          "step_count": 0}
    mem = oq.memory_of(st)
    mem.observe(meta(objects=[("Laptop", 0.0, 0.9, 2.0, True, 2.0, 0.3)]),
                step=3)
    assert oq.render_block(st, step=3) == "", "nothing before the request"
    oq.apply_query(st, "__all__", 3)
    summary = oq.render_block(st, step=4)
    assert "Laptop" in summary, summary
    assert oq.render_block(st, step=5) == "", "the summary is shown once"


def test_world_model_builds_anchors_and_dead_reckons():
    import numpy as np

    from mllm_base_agent.agent.world_model import WorldModel

    wm = WorldModel(move_magnitudes={"MoveAhead": 0.5, "MoveBack": 0.5,
                                     "MoveLeft": 0.5, "MoveRight": 0.5,
                                     "MoveSmall": 0.25, "MoveMedium": 0.5,
                                     "MoveLarge": 1.0})
    wm.attach_perception(StubPerception())
    frame = np.zeros((600, 800, 3), dtype=np.uint8)

    out = wm.observe(None, action={"action_name": "MoveAhead"},
                     moved=True, action_ok=True, frame=frame)
    assert wm._slots, "no object anchor was created from the detections"
    apple = next(iter(wm._slots.values()))
    assert apple["type"] == "Apple", apple
    pos = apple["pos"]
    assert all(np.isfinite(pos)), pos
    assert pos[2] > 0.5, f"anchor should sit in front of the agent: {pos}"
    assert abs(wm._pose[2] - 0.25) < 1e-6, wm._pose

    assert set(out) >= {"agent", "objects", "inventoryObjects"}, sorted(out)
    obj = out["objects"][0]
    for key in ("objectType", "position", "visible", "distance",
                "last_seen_step", "seen_count", "sigma", "state", "contents"):
        assert key in obj, f"metadata contract lost key {key!r}: {sorted(obj)}"


def test_world_model_refuses_simulator_pose():
    from mllm_base_agent.agent.world_model import WorldModel

    try:
        WorldModel(pose_from_action_log=False)
    except ValueError:
        return
    raise AssertionError("pose_from_action_log=False must be refused")


def test_world_model_checkpoint_round_trip():
    """save() must produce complete JSON even with numpy inside a slot."""
    import json
    import tempfile

    import numpy as np

    from mllm_base_agent.agent.world_model import WorldModel

    wm = WorldModel(move_magnitudes={"MoveAhead": 0.5, "MoveSmall": 0.25})
    wm.attach_perception(StubPerception())
    frame = np.zeros((600, 800, 3), dtype=np.uint8)
    wm.observe(None, action={"action_name": "MoveAhead"}, moved=True,
               action_ok=True, frame=frame)
    wm.observe(None, action={"action_name": "MoveAhead"}, moved=True,
               action_ok=True, frame=frame)
    assert wm._slots, "precondition: at least one anchor exists"
    assert any(slot.get("obs") for slot in wm._slots.values()), \
        "precondition: the slot kept multi-view rays (numpy inside)"

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "wm.json")
        wm.save(path)                                  # must not raise
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)                        # must be complete JSON
        assert raw["_slots"], raw.keys()
        assert not os.path.exists(path + ".tmp"), "temp file left behind"

        back = WorldModel.load(path)
        assert set(back._slots) == set(wm._slots), (set(back._slots), set(wm._slots))
        assert back._slot_seq == wm._slot_seq, (back._slot_seq, wm._slot_seq)
        assert back._step == wm._step
        for oid, slot in wm._slots.items():
            for i in range(3):
                assert abs(back._slots[oid]["pos"][i] - slot["pos"][i]) < 1e-9


def test_done_is_never_intercepted():
    from mllm_base_agent.agent import object_query as oq

    assert oq.parse_query("DONE") is None
    assert oq.parse_query("EndTask(DONE)") is None
    assert oq.parse_query("MoveAhead(1)") is None


def main():
    print(f"repo: {REPO}")
    print(f"python: {sys.version.split()[0]}\n")
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        check(name, fn)
    failed = [r for r in _RESULTS if not r[1]]
    print(f"\n{len(_RESULTS) - len(failed)}/{len(_RESULTS)} checks passed")
    if failed:
        print("FAILED: " + ", ".join(r[0] for r in failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
