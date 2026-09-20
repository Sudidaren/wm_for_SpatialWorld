#!/usr/bin/env python
"""被挡住的那一步：只给一条脱困指令，不跟其它移动建议打架。

实测（5529 个有提示的步）：530 次被挡提示里有 **193 次**同一段还挂着
"——建议先转向它再靠近"，也就是一段话里给了**两条互相冲突的移动指令**
（占被挡提示的 36%）。被挡的时候模型下一步还继续平移的比例是 54%，
而原文明确写着"不要连续重复同一个被挡的动作"。

这里锁三件事：
  1. 被挡 → 压掉"建议靠近"（只压这一步），但**位置和距离照常给**；
  2. 没被挡 → 导航建议照旧（不能把已有功能改没）；
  3. 被挡文案只点**一个**动作，且连续被挡时方向左右交替。

    python tests/test_blocked_hint.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from mllm_base_agent.agent.memory_probe import MemoryProbe   # noqa: E402


def _meta():
    """一个看不见的任务目标（Laptop，1.6m）＋一个看得见的东西。"""
    return {
        "agent": {"position": {"x": 0.0, "y": 0.0, "z": 0.0},
                  "rotation": {"x": 0.0, "y": 0.0, "z": 0.0}},
        "objects": [
            {"objectId": "det|Laptop|0", "objectType": "Laptop",
             "objectTypeDisplay": "Laptop", "position": {"x": 0.0, "y": 1.0, "z": 1.6},
             "visible": False, "distance": 1.6, "edge_px": 200.0,
             "box": [0, 0, 10, 10], "last_seen_step": 3, "seen_count": 2,
             "sigma": 0.2, "state": {}, "contents": []},
        ],
        "inventoryObjects": [],
    }


def _probe() -> MemoryProbe:
    p = MemoryProbe(task_description="open my laptop")
    p.set_target_hint({"enabled": True, "limit": 6, "names_limit": 5,
                       "max_dist": 3.0, "max_sigma": 0.5})
    return p


def test_blocked_suppresses_the_competing_nav_suggestion():
    block = _probe().update(wm_metadata=_meta(), action_name="MoveAhead",
                            blocked=True, action_ok=False)
    assert "移动提示：" in block, block
    assert "建议先转向它再靠近" not in block, block


def test_blocked_keeps_the_position_and_distance():
    """压掉的是"往哪走"的建议，不是位置情报 —— 距离必须还在。"""
    block = _probe().update(wm_metadata=_meta(), action_name="MoveAhead",
                            blocked=True, action_ok=False)
    assert "记住的位置" in block, block
    assert "1.6m" in block, block


def test_not_blocked_still_gives_the_nav_suggestion():
    """没有被挡时必须照旧给建议 —— 不能把已有功能改没了。"""
    block = _probe().update(wm_metadata=_meta(), action_name="MoveAhead",
                            blocked=False, action_ok=True)
    assert "建议先转向它再靠近" in block, block


def test_blocked_text_names_exactly_one_action():
    block = _probe().update(wm_metadata=_meta(), action_name="MoveAhead",
                            blocked=True, action_ok=False)
    # 只有一个转身动作，且不再并列两个方向让模型自己拼
    assert "下一步只做这一件事" in block, block
    got = [d for d in ("RotateLeft(90)", "RotateRight(90)") if d in block]
    assert len(got) == 1, (got, block)
    assert "再 MoveAhead" not in block, block


def test_consecutive_blocked_alternates_the_direction():
    p = _probe()
    first = p.update(wm_metadata=_meta(), action_name="MoveAhead",
                     blocked=True, action_ok=False)
    second = p.update(wm_metadata=_meta(), action_name="MoveAhead",
                      blocked=True, action_ok=False)
    a = "RotateLeft(90)" if "RotateLeft(90)" in first else "RotateRight(90)"
    b = "RotateLeft(90)" if "RotateLeft(90)" in second else "RotateRight(90)"
    assert a != b, (first, second)


def test_streak_resets_after_a_normal_step():
    p = _probe()
    first = p.update(wm_metadata=_meta(), action_name="MoveAhead",
                     blocked=True, action_ok=False)
    p.update(wm_metadata=_meta(), action_name="MoveAhead",
             blocked=False, action_ok=True)
    again = p.update(wm_metadata=_meta(), action_name="MoveAhead",
                     blocked=True, action_ok=False)
    a = "RotateLeft(90)" if "RotateLeft(90)" in first else "RotateRight(90)"
    b = "RotateLeft(90)" if "RotateLeft(90)" in again else "RotateRight(90)"
    assert a == b, (first, again)


def test_blocked_suppresses_the_reach_hint_but_keeps_it_otherwise():
    """交互失败 + 目标还远时那条"要先走到 1m 以内"，被挡时也不该同时出现。"""
    meta = _meta()
    # 任务目标 Laptop 就在视野里、但离 1.6m —— 正是"距离提示"要触发的场景
    meta["objects"][0]["visible"] = True
    blocked = _probe().update(wm_metadata=meta, action_name="PickupObject",
                              object_type="Laptop", blocked=True, action_ok=False)
    assert "要先走到" not in blocked, blocked
    normal = _probe().update(wm_metadata=meta, action_name="PickupObject",
                             object_type="Laptop", blocked=False, action_ok=False)
    assert "要先走到" in normal, normal


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"ok   {name}")
        except AssertionError as exc:
            fails += 1
            print(f"FAIL {name}: {exc}")
        except Exception as exc:                        # noqa: BLE001
            fails += 1
            print(f"ERR  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{'FAILED' if fails else 'passed'}")
    raise SystemExit(1 if fails else 0)
