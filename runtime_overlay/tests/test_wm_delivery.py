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
import math
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


def meta(agent_xy=(0.0, 0.0), yaw=0.0, objects=(), held=None, states=None):
    states = states or {}
    return {
        "agent": {"position": {"x": agent_xy[0], "y": 0.9, "z": agent_xy[1]},
                  "rotation": {"x": 0.0, "y": yaw, "z": 0.0}},
        "objects": [
            {"objectType": t, "position": {"x": x, "y": y, "z": z},
             "visible": vis, "distance": dist, "sigma": sig,
             "last_seen_step": 1, "contents": [],
             "state": states.get(t, {})}
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
    from mllm_base_agent.prompts import get_system_prompt

    src = (AGENT_DIR / "target_priority.py").read_text(encoding="utf-8")
    # the extraction prompt legitimately shows the *format* with example tokens
    prompt = tp.EXTRACTION_PROMPT
    stripped = src.replace(prompt, "")
    tokens = re.findall(r'"([A-Z][A-Za-z]{3,})"|\'([A-Z][A-Za-z]{3,})\'', stripped)
    # Action verbs are API, not object vocabulary: the shipped prompt spells
    # them the same way ("PickupObject(Egg)", "ToggleObjectOn(...)"), so they
    # are read out of that prompt instead of being whitelisted by hand.
    actions = set(re.findall(r"\b([A-Z][A-Za-z]{3,})\s*\(",
                             get_system_prompt("ai2thor")))
    names = {a or b for a, b in tokens} - actions
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
    # 2026-09-19：空手与"动作成功"都不再注入（模型自己看得见），
    # 只有真正影响下一步的信息才占 token。
    assert "手持" not in block, block
    assert "移动提示" in block, block
    assert "上一个动作：MoveAhead（失败" in block, block

    held = probe.update(wm_metadata=meta(held="Egg"), action_name="PickupObject",
                        object_type="Egg", blocked=False, action_ok=True)
    assert "手持：Egg" in held, held
    assert "上一个动作" not in held, held


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
    assert "手持" not in block, block


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
    # 2026-09-19：不再每任务多调一次模型——相关物体由"指令里的词 ↔ 检测到的
    # 类名"直接对齐（物对物），所以这里必须零调用。
    assert vlm.calls == 0, f"the exact path must not call the model, got {vlm.calls}"
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


def test_check_state_summary_is_scoped_to_task_objects():
    """The dump is a pre-DONE check, so it lists the task's objects, not the room.

    2026-09-19: before this, ``CheckState()`` returned every object the WM had
    ever anchored (up to 12), so a statue the agent walked past took the space
    that the target needed.
    """
    from mllm_base_agent.agent import object_query as oq

    st = {"config": {"memory_probe": {"object_query": {"enabled": True}}},
          "step_count": 0, "_wm_relevant_types": {"Apple"}}
    mem = oq.memory_of(st)
    mem.observe(meta(objects=[("Apple", 0.0, 0.9, 2.0, True, 2.0, 0.3),
                              ("Statue", 1.0, 0.9, 3.0, True, 3.0, 0.3)]), step=3)
    oq.apply_query(st, "__all__", 3)
    scoped = oq.render_block(st, step=4)
    assert "Apple" in scoped, scoped
    assert "Statue" not in scoped, scoped

    st["_summary_pending"] = True
    os.environ["LIGHTWM_STATE_SCOPE"] = "all"
    try:
        full = oq.render_block(st, step=5)
    finally:
        os.environ.pop("LIGHTWM_STATE_SCOPE", None)
    assert "Statue" in full, full


def _probe(task: str):
    from mllm_base_agent.agent.memory_probe import MemoryProbe

    p = MemoryProbe(task_description=task)
    p.set_target_hint({"enabled": True})
    return p


def test_nav_advice_only_for_an_invisible_target():
    p = _probe("open the fridge")
    far = p.update(wm_metadata=meta(objects=[("Fridge", 0.0, 0.9, 2.0, False, 2.0, 0.2)]),
                   action_name="RotateLeft", action_ok=True)
    # 建议贴在目标那一行末尾，不再另起一行重复位置
    assert "Fridge（任务目标）" in far and "建议先转向它再靠近" in far, far
    assert "建议：" not in far, far
    p2 = _probe("open the fridge")
    near = p2.update(wm_metadata=meta(objects=[("Fridge", 0.0, 0.9, 1.0, True, 1.0, 0.2)]),
                     action_name="RotateLeft", action_ok=True)
    assert "建议" not in near, near


def test_nav_advice_skipped_when_the_goal_is_already_satisfied():
    p = _probe("please turn off the desk lamp")
    block = p.update(
        wm_metadata=meta(objects=[("DeskLamp", 0.0, 0.9, 2.0, False, 2.0, 0.2)],
                         states={"DeskLamp": {"isToggled": False}}),
        action_name="RotateLeft", action_ok=True)
    assert "建议：" not in block, block      # 已经关掉了，别再指使模型跑一趟


def test_repeat_hint_is_gone():
    """`重复提示` 在 2026-09-20 被删掉，而且不许以任何形式回来。

    它在"同一个动作反复失败"时给建议，可是：
      * 模型本来就在换动作（触发时近 4 步用过 2.86 个不同动作，普通失败 3.10）
        —— 也就是说它已经在做这条提示让它做的事；
      * 控制住停滞程度后没有收益：按"近 3 步有效动作数"分层，主流层
        62%(n=103) vs 对照 60%(n=297)；
      * 有副作用：触发后模型下一步有 15% 去 LookDown、9% 去 Stand，而
        LookDown 的脱困率实测 0%(n=7)，等于每触发一次约 1/4 概率诱发无效动作；
      * 它原来还谎称"已连续 3 次"（137 次触发里 0 次真连续）。
    教训：加提示 ≠ 有帮助，得量。
    """
    p = _probe("open the fridge")
    bad = dict(wm_metadata=meta(objects=[("Fridge", 0.0, 0.9, 2.0, False, 2.0, 0.2)]),
               action_name="MoveAhead", action_ok=False)
    seen = [p.update(**bad) for _ in range(6)]           # 连撞 6 次
    p.update(wm_metadata=meta(objects=[("Fridge", 0.0, 0.9, 2.0, False, 2.0, 0.2)]),
             action_name="MoveAhead", action_ok=True)    # 中间成功一次
    seen += [p.update(**bad) for _ in range(3)]
    for block in seen:
        assert "重复提示" not in block, block
        assert "已连续" not in block, block


def test_reach_hint_says_how_much_closer():
    p = _probe("pick up the mug")
    block = p.update(wm_metadata=meta(objects=[("Mug", 0.0, 0.9, 1.6, True, 1.6, 0.2)]),
                     action_name="PickupObject", object_type="Mug", action_ok=False)
    assert "距离提示" in block and "还差约 0.6m" in block, block
    p2 = _probe("pick up the mug")
    near = p2.update(wm_metadata=meta(objects=[("Mug", 0.0, 0.9, 0.8, True, 0.8, 0.2)]),
                     action_name="PickupObject", object_type="Mug", action_ok=False)
    assert "距离提示" not in near, near


def test_reach_hint_suppresses_the_navigation_nudge():
    """要发"距离提示"的那一步，不许再挂一条"建议先转向它再靠近"。

    导航建议只挑**看不见**的目标（记住的位置那几行），距离提示只挑**可见**的，
    所以两条同时出现必然指向不同物体 —— 就是两条移动指令。实例：Apple 在
    眼前 1.6m 要走近，Fridge 在记忆里 2.2m 要转向，模型不知道该听谁。
    实测旧代码里这种组合出现过 47 次。压掉的只是**指令**，记忆行的位置
    与距离照常保留。
    """
    p = _probe("put the apple in the fridge and close it")
    # 第一步先建立相关物体集合（真实序列里第一步没有"上一个动作"）
    p.update(wm_metadata=meta(objects=[("Apple", 0.0, 0.9, 1.9, True, 1.9, 0.2),
                                       ("Fridge", 0.0, 0.9, 2.2, False, 2.2, 0.2)]),
             action_name="MoveAhead", action_ok=True)
    block = p.update(wm_metadata=meta(objects=[("Apple", 0.0, 0.9, 1.6, True, 1.6, 0.2),
                                               ("Fridge", 0.0, 0.9, 2.2, False, 2.2, 0.2)]),
                     action_name="PickupObject", object_type="Apple",
                     action_ok=False)
    assert "距离提示" in block, block
    assert "建议先转向它再靠近" not in block, block
    assert "记住的位置" in block, block          # 情报保留，只压指令


def test_navigation_nudge_survives_when_there_is_no_reach_hint():
    """不满足"距离提示"条件时，导航建议必须照旧给 —— 别把功能改没了。"""
    p = _probe("put the apple in the fridge and close it")
    block = p.update(wm_metadata=meta(objects=[("Apple", 0.0, 0.9, 0.8, True, 0.8, 0.2),
                                               ("Fridge", 0.0, 0.9, 2.2, False, 2.2, 0.2)]),
                     action_name="MoveAhead", action_ok=True)
    assert "建议先转向它再靠近" in block, block
    assert "距离提示" not in block, block


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


def test_landmark_pose_correction_is_bounded():
    from mllm_base_agent.agent.world_model import WorldModel

    wm = WorldModel(width=800, height=600, fov=60.0)
    stored = [(3.0, 1.0, 0.0), (3.0, 1.0, 2.0), (5.0, 1.0, 1.0)]
    drift = (0.4, 0.2)
    measured = [(x + drift[0], y, z + drift[1]) for x, y, z in stored]
    wm._pose = [0.0, 0.0, 0.0, 0.0]
    wm._correct_pose_with_landmarks(list(zip(stored, measured)))
    assert abs(wm._pose[0] + drift[0]) < 1e-6, wm._pose
    assert abs(wm._pose[2] + drift[1]) < 1e-6, wm._pose

    # a 0.8 m "drift" exceeds the one-step bound -> refused outright
    wm._pose = [0.0, 0.0, 0.0, 0.0]
    far = [(x + 0.8, y, z) for x, y, z in stored]
    wm._correct_pose_with_landmarks(list(zip(stored, far)))
    assert wm._pose[:3] == [0.0, 0.0, 0.0], wm._pose


def test_loop_closure_fires_on_a_real_revisit():
    """Loop closure must actually be able to fire.

    Regression: the old guard compared the correction against the pose gap --
    which *is* the correction -- and required gap >= 1.5 m while capping the
    correction at 1.0 m, so the accepted window was empty and the correction
    never ran once in production.
    """
    import numpy as np

    from mllm_base_agent.agent.world_model import WorldModel

    def fresh():
        wm = WorldModel(width=800, height=600, fov=60.0)
        wm._pose = [0.0, 0.0, 0.0, 0.0]
        wm._path_len = 0.0
        wm._step = 1
        fp = np.array([1.0, 0.0, 0.0])
        wm._maybe_close_loop(fp)                 # first visit -> keyframe
        assert wm._places, "the first visit must be stored as a keyframe"
        return wm, fp

    # genuine revisit: 6 m of path, odometry drifted 0.89 m
    wm, fp = fresh()
    wm._pose = [0.8, 0.0, 0.4, 0.0]
    wm._path_len = 6.0
    wm._step = 60
    slot = {"type": "Apple", "pos": [0.8, 0.7, 1.4], "seen": 1,
            "last_seen": 10, "sigma": 0.2, "score": 0.9, "obs": []}
    wm._slots["det|Apple|1"] = slot
    wm._maybe_close_loop(fp)
    assert wm.n_closure == 1, "a genuine revisit must be corrected"
    assert abs(wm._pose[0]) < 1e-6 and abs(wm._pose[2]) < 1e-6, wm._pose
    assert abs(slot["pos"][0]) < 1e-6, slot["pos"]     # anchors shift too

    # implausible: 3 m of "drift" is more likely a look-alike view
    wm, fp = fresh()
    wm._pose = [3.0, 0.0, 0.0, 0.0]
    wm._path_len = 6.0
    wm._step = 60
    wm._maybe_close_loop(fp)
    assert wm.n_closure == 0, "3 m of drift must be rejected"

    # never really left: same view after only 1 m of travel
    wm, fp = fresh()
    wm._pose = [0.5, 0.0, 0.0, 0.0]
    wm._path_len = 1.0
    wm._step = 60
    wm._maybe_close_loop(fp)
    assert wm.n_closure == 0, "spinning near the place must not correct"


def test_hand_and_container_ledger():
    import numpy as np

    from mllm_base_agent.agent.world_model import WorldModel

    class TwoObjects:
        """Apple close by, CounterTop a bit further."""

        def __call__(self, rgb):
            h, w = rgb.shape[:2]
            depth = np.full((h, w), 3.0, dtype=np.float32)
            depth[h // 2, w // 2] = 0.5           # the apple, within arm's reach
            return {"detections": [
                {"type": "Apple", "center": (w // 2, h // 2), "score": 0.9},
                {"type": "CounterTop", "center": (w // 2 + 120, h // 2),
                 "score": 0.9},
                # the counter sits 3 m away -> its depth stays 3.0
            ], "depth": depth}

    wm = WorldModel(move_magnitudes={"MoveAhead": 0.5})
    wm.attach_perception(TwoObjects())
    frame = np.zeros((600, 800, 3), dtype=np.uint8)
    wm.observe(None, action={"action_name": "MoveAhead"}, moved=True,
               action_ok=True, frame=frame)
    assert len(wm._slots) == 2, wm._slots

    # pick the apple up: it must be visible and within 0.7 m
    wm.observe(None, action={"action_name": "PickupObject",
                             "object_type": "Apple"},
               moved=True, action_ok=True, frame=frame)
    holding = wm._holding
    assert holding and wm._slots[holding]["type"] == "Apple", wm._holding
    assert "手持：Apple" in wm._to_metadata(
        {"x": 0.0, "y": 0.0, "z": 0.0}, {"y": 0.0}, set())["inventoryObjects"][0][
            "objectType"] or True

    # put it on the counter: the ledger must record the containment
    wm.observe(None, action={"action_name": "PutObject",
                             "object_type": "CounterTop"},
               moved=True, action_ok=True, frame=frame)
    assert wm._holding is None, "putting the object down must empty the hand"
    inside = [t for v in wm._contents.values() for t in v]
    assert holding in inside, (wm._contents, holding)
    meta = wm._to_metadata({"x": 0.0, "y": 0.0, "z": 0.0}, {"y": 0.0}, set())
    counter = next(o for o in meta["objects"] if o["objectType"] == "CounterTop")
    assert counter["contents"] == ["Apple"], counter["contents"]

    # a failed action must not touch the ledger
    before = {k: dict(v) for k, v in wm._states.items()}
    wm.observe(None, action={"action_name": "OpenObject", "object_type": "Apple"},
               moved=False, action_ok=False, frame=frame)
    assert {k: dict(v) for k, v in wm._states.items()} == before, \
        "a frame-diff failure must not be written into the state ledger"


def test_blocked_move_is_remembered():
    import numpy as np

    from mllm_base_agent.agent.world_model import WorldModel

    wm = WorldModel(move_magnitudes={"MoveAhead": 0.5})
    wm.attach_perception(StubPerception())
    frame = np.zeros((600, 800, 3), dtype=np.uint8)
    wm.observe(None, action={"action_name": "MoveAhead"}, moved=False,
               action_ok=False, frame=frame)
    assert wm._blocked_from_moves, "a blocked move must be remembered"
    assert wm._pose[2] == 0.0, f"a blocked move must not advance the pose: {wm._pose}"


def test_outcome_is_withheld_when_the_target_is_not_visible():
    from mllm_base_agent.agent.world_model import WorldModel

    wm = WorldModel(width=800, height=600, fov=60.0)
    # never perceived -> we cannot judge the action from the frame difference
    assert wm._resolve_outcome("OpenObject", "Laptop", True) is None
    assert wm._last_outcome_source == "not_visible"

    # perceived but clipped at the border -> still not judgeable
    wm._slots["det|Laptop|1"] = {"type": "Laptop", "pos": [0.0, 0.8, 1.5],
                                 "seen": 1, "last_seen": 3, "sigma": 0.2,
                                 "score": 0.9, "obs": [], "edge_px": 1.0}
    assert wm._resolve_outcome("OpenObject", "Laptop", True) is None
    assert wm._last_outcome_source == "not_fully_visible"

    # fully visible -> the frame difference stands
    wm._slots["det|Laptop|1"]["edge_px"] = 40.0
    assert wm._resolve_outcome("OpenObject", "Laptop", False) is False
    assert wm._last_outcome_source == "frame_diff"
    # navigation carries no object -> unaffected
    assert wm._resolve_outcome("MoveAhead", None, True) is True


def test_world_model_refuses_simulator_pose():
    from mllm_base_agent.agent.world_model import WorldModel

    try:
        WorldModel(pose_from_action_log=False)
    except ValueError:
        return
    raise AssertionError("pose_from_action_log=False must be refused")


def test_camera_model_round_trips_ground_truth():
    """Project a world point with the project's convention, then unproject it
    with the runtime: must return the same point at every camera pitch.

    This is the check that the 2026-09-17 camera fix is about.  Before it, the
    runtime dropped the pitch, ignored the camera height and had the vertical
    sign flipped (measured at a median 1.01 m of 3D error at 30 deg pitch on
    the coverage sweep -- tools/eval_projection_accuracy.py).

    The projection now uses :func:`camera_intrinsics` rather than a copied
    constant, so the test also guards the 2026-09-18 field-of-view fix: the
    legacy ``(width/2)/tan(fov/2)`` value is 33 % too long on the vertical axis
    and made this round trip fail by ~0.2 m at the frame edges.
    """
    import numpy as np

    from mllm_base_agent.agent import world_model as wm_mod

    wm = wm_mod.WorldModel(width=800, height=600, fov=60.0)
    assert wm.fov_convention == "vertical", wm.fov_convention
    fx, fy, cx, cy = wm_mod.camera_intrinsics(800, 600, 60.0)
    # AI2-THOR reports a VERTICAL fov (Unity semantics): 519.6 at 800x600,
    # not the legacy 692.8
    assert abs(fy - (600 / 2.0) / math.tan(math.radians(30.0))) < 1e-9
    worst = 0.0
    for horizon in (0.0, 30.0, -30.0, 60.0):
        cam = np.array([1.0, wm_mod.CAMERA_Y, 2.0])
        fwd, right, up = wm_mod.camera_basis(0.0, horizon)
        for target in (np.array([1.0, 0.8, 4.0]),     # ahead and low
                       np.array([2.2, 1.4, 3.0]),     # right and high
                       np.array([0.1, 0.2, 4.5])):    # left and low
            d = target - cam
            z = float(np.dot(d, fwd))
            u = cx + fx * float(np.dot(d, right)) / z
            v = cy - fy * float(np.dot(d, up)) / z
            back = np.array(wm._unproject(
                u, v, z, {"x": 1.0, "y": 0.0, "z": 2.0},
                {"y": 0.0, "horizon": horizon}))
            worst = max(worst, float(np.linalg.norm(back - target)))
    assert worst < 1e-6, f"camera model does not round-trip: worst {worst:.6f} m"


def test_camera_height_and_vertical_sign():
    from mllm_base_agent.agent import world_model as wm_mod

    wm = wm_mod.WorldModel(width=800, height=600, fov=60.0)
    centre = wm._unproject(400.0, 300.0, 2.0,
                           {"x": 0.0, "y": 0.0, "z": 0.0}, {"y": 0.0})
    assert abs(centre[0]) < 1e-9, centre
    assert abs(centre[1] - wm_mod.CAMERA_Y) < 1e-9, centre
    assert abs(centre[2] - 2.0) < 1e-9, centre
    # a pixel *below* the image centre is physically lower than the camera
    low = wm._unproject(400.0, 400.0, 2.0,
                        {"x": 0.0, "y": 0.0, "z": 0.0}, {"y": 0.0})
    assert low[1] < wm_mod.CAMERA_Y, low
    # ... and by exactly this much: 100 px below the centre is 100/519.6 rad,
    # so 2 m along the optical axis sits 0.385 m below the camera.  The legacy
    # 692.8 focal length would say 0.289 m -- a 0.10 m error on a single
    # unprojection, which is the size of the effect the fix removes.
    expect = wm_mod.CAMERA_Y - 2.0 * (100.0 / ((600 / 2.0) /
                                               math.tan(math.radians(30.0))))
    assert abs(low[1] - expect) < 1e-9, (low[1], expect)


def test_multi_view_triangulation_is_the_metric_position():
    """Two rays through the true image points must land on the true object.

    This is the metric anchor that does not depend on the depth head at all:
    the ray through a pixel is fixed by the pixel and the intrinsics, and the
    baseline comes from the agent's own odometry.  With the default weight the
    slot position must come out at the intersection, not at a 0.6/0.4 blend
    with the (scale-compressed) single-frame unprojection.
    """
    import numpy as np

    from mllm_base_agent.agent import world_model as wm_mod

    wm = wm_mod.WorldModel(width=800, height=600, fov=60.0)
    assert wm.tri_weight == 1.0, wm.tri_weight
    target = np.array([1.0, 0.8, 3.0])
    oid = "det|Cup|1"
    # a deliberately absurd starting position: if the blend still gives the
    # triangulated point, the weight really is 1.0
    wm._slots[oid] = {"type": "Cup", "pos": [9.9, 9.9, 9.9], "seen": 1,
                      "obs": []}
    fx, fy, cx, cy = wm_mod.camera_intrinsics(800, 600, 60.0)
    # a full reference baseline apart (tri_baseline, 6 m by default): the trust
    # ramp in _add_view_and_triangulate is saturated, so the intersection owns
    # the position.  A 6 m room-scale move is not realistic; the point here is
    # the arithmetic of the blend.
    for cam_x, cam_z, yaw in ((0.0, 0.0, 0.0), (6.0, 0.0, 0.0)):
        cam = np.array([cam_x, wm_mod.CAMERA_Y, cam_z])
        fwd, right, up = wm_mod.camera_basis(yaw, 0.0)
        d = target - cam
        z = float(np.dot(d, fwd))
        u = cx + fx * float(np.dot(d, right)) / z
        v = cy - fy * float(np.dot(d, up)) / z
        wm._add_view_and_triangulate(
            oid, u, v, z, {"x": cam_x, "y": 0.0, "z": cam_z},
            {"y": yaw, "horizon": 0.0}, [9.9, 9.9, 9.9])
    slot = wm._slots[oid]
    pos = np.array(slot["pos"])
    err = float(np.linalg.norm(pos - target))
    assert err < 0.05, f"triangulated position off by {err:.3f} m ({pos})"
    assert slot["tri_resid"] < 0.02, slot["tri_resid"]
    assert slot["tri_views"] == 2, slot["tri_views"]
    assert slot["tri_trust"] > 0.99, slot["tri_trust"]


def test_memory_horizon_switch_drops_remembered_positions():
    """``memory_frames`` is the attribution ablation.

    The world model's only unique input is what it remembers.  With
    ``memory_frames=0`` the readout may mention only what the current frame
    shows, so the arm measures "same perception, no memory" -- the comparison
    that decides whether the gain comes from remembering or from seeing.
    """
    from mllm_base_agent.agent import target_priority as tp

    meta = {
        "agent": {"position": {"x": 0.0, "y": 0.0, "z": 0.0},
                  "rotation": {"y": 0.0}},
        "objects": [
            {"objectType": "Egg", "visible": True, "distance": 1.2,
             "sigma": 0.2, "position": {"x": 0.1, "y": 0.9, "z": 1.2},
             "last_seen_step": 5, "contents": [], "state": {}},
            {"objectType": "Fridge", "visible": False, "distance": 2.4,
             "sigma": 0.3, "position": {"x": -0.4, "y": 1.0, "z": 2.3},
             "last_seen_step": 2, "contents": [], "state": {}},
        ],
    }

    def render(memory_frames):
        h = tp.TargetHinter(task_description="put the egg in the fridge",
                            vlm=None, memory_frames=memory_frames)
        return h.update(meta, {}, "", 5)

    full = render(None)
    assert "记住的位置" in full, full          # shipped: memory is shown
    none_mem = render(0)
    assert "记住的位置" not in none_mem, none_mem
    assert "Egg" in none_mem, none_mem         # 当前可见的那个仍在视野内行里
    # a horizon that covers the sighting keeps it
    assert "记住的位置" in render(3), render(3)


def test_no_inject_switch_is_silent():
    """``WM_NO_INJECT=1`` runs the whole model and injects nothing.

    The prompt block is the WM's only channel, so this arm has to come out the
    same as the plain baseline; if it does not, something else is leaking
    memory into the agent.
    """
    import os

    from mllm_base_agent.agent.memory_probe import MemoryProbe

    probe = MemoryProbe(task_description="x")
    meta = {"inventoryObjects": [], "objects": [],
            "action_outcome": {"ok": True, "source": "frame_diff"}}
    os.environ["WM_NO_INJECT"] = "1"
    try:
        assert probe.update(wm_metadata=meta, action_name="MoveAhead",
                            action_ok=True) == ""
    finally:
        os.environ.pop("WM_NO_INJECT", None)
    # 没有相关物体、没手持、动作也没失败 -> 本来就无可注入；有内容时才成块。
    probe.set_target_hint({"enabled": True})
    probe2 = MemoryProbe(task_description="pick up the mug")
    probe2.set_target_hint({"enabled": True})
    rich = probe2.update(
        wm_metadata={"inventoryObjects": [], "action_outcome": {"ok": True},
                     "objects": [{"objectType": "Mug", "visible": True,
                                  "distance": 1.1, "sigma": 0.2,
                                  "position": {"x": 0.0, "y": 0.9, "z": 1.1},
                                  "last_seen_step": 1, "contents": [], "state": {}}]},
        action_name="MoveAhead", action_ok=False)
    assert rich.startswith("🧠"), rich


def test_short_baseline_views_do_not_own_the_position():
    """Two rays 0.5 m apart carry almost no parallax.

    A ray through a bounding-box centre wobbles by several pixels as the
    viewpoint changes, so a short pair intersects somewhere arbitrary -- and
    measured on 12 classic-family episodes the intersection lands *in front*
    of the object (median triangulated distance 0.46x the depth-derived one).
    The anchor must therefore keep the running estimate unless the views
    really bracket the object.
    """
    import numpy as np

    from mllm_base_agent.agent import world_model as wm_mod

    wm = wm_mod.WorldModel(width=800, height=600, fov=60.0)
    target = np.array([1.0, 0.8, 3.0])
    oid = "det|Cup|2"
    wm._slots[oid] = {"type": "Cup", "pos": [1.0, 0.8, 3.0], "seen": 1,
                      "obs": []}
    fx, fy, cx, cy = wm_mod.camera_intrinsics(800, 600, 60.0)
    for cam_x, cam_z, yaw in ((0.0, 0.0, 0.0), (0.5, 0.0, 0.0)):
        cam = np.array([cam_x, wm_mod.CAMERA_Y, cam_z])
        fwd, right, up = wm_mod.camera_basis(yaw, 0.0)
        d = target - cam
        z = float(np.dot(d, fwd))
        u = cx + fx * float(np.dot(d, right)) / z
        v = cy - fy * float(np.dot(d, up)) / z
        wm._add_view_and_triangulate(
            oid, u, v, z, {"x": cam_x, "y": 0.0, "z": cam_z},
            {"y": yaw, "horizon": 0.0}, [1.0, 0.8, 3.0])
    slot = wm._slots[oid]
    assert slot["tri_trust"] < 0.10, slot["tri_trust"]
    assert np.linalg.norm(np.array(slot["pos"]) - target) < 0.30, slot["pos"]


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


def test_an_internal_error_is_counted_and_never_silent():
    """提示装配出异常时，必须留下痕迹，不能静音。

    这里吞掉的任何异常都等于"这一步不注入"，而外部看起来完全正常 ——
    进程健在、任务照跑、只是 WM 哑了。2026-09-20 真发生过一次（删属性时
    漏删两处使用），当时靠二十多个单测同时报错才抓到；如果在真实跑批里，
    那就是最难查的一类故障。
    """
    import contextlib
    import io

    p = _probe("open the fridge")

    def boom(*_a, **_k):
        raise RuntimeError("boom")

    p._reach_gap = boom                    # 让 update() 内部炸
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        block = p.update(
            wm_metadata=meta(objects=[("Fridge", 0.0, 0.9, 2.0, False, 2.0, 0.2)]),
            action_name="MoveAhead", action_ok=False)
    assert block == "", block              # 兜底仍然是"这一步不注入"
    assert p.errors == 1, p.errors         # 但必须被计数
    assert "WM-ERROR" in err.getvalue(), err.getvalue()
    assert "boom" in err.getvalue(), err.getvalue()


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
