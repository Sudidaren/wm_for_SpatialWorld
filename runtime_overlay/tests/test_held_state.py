#!/usr/bin/env python
"""手持状态：检测器看不见手里的东西，不代表手里没有东西。

原来的判定只在本帧"还检测得到"的槽位里找手边的物体：

    for oid in self._slots_of_type(object_type, visible_ids):
        ...
    if nearest is not None and best <= 0.7:
        self._holding = nearest

而东西一旦拿到手上就基本检测不到了（画面底部、被身体挡住）—— 恰好在最需要
判断的那一刻候选集是空的。2026-09-20 的 311 批实测：20 个出现
``Hand already has an object`` 的任务里，**14 个 WM 一次都没记到手持有**；
129 局里只有 30 局报过"手持：X"。

后果不止是"手持"那一行不出现：``target_priority`` 里本来就有
``if tp == holding: continue``（不给自己手上的东西报"记住的位置/建议靠近"），
它只在 holding 非空时生效 —— 于是模型被反复指使去够自己手里的东西。

    python tests/test_held_state.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from mllm_base_agent.agent.target_priority import TargetHinter   # noqa: E402
from mllm_base_agent.agent.world_model import WorldModel         # noqa: E402


def _wm_with_egg(distance: float = 1.2) -> WorldModel:
    wm = WorldModel(fov=60.0, width=800, height=600)
    wm._slots["det|Egg|0"] = {
        "type": "Egg", "pos": [0.0, 0.8, distance], "last_seen": 3,
        "seen": 2, "sigma": 0.2, "edge_px": 300.0, "box": [0, 0, 10, 10],
    }
    return wm


def _pickup(wm, visible, action_ok):
    wm._update_action_log("PickupObject", "Egg",
                          visible_ids={"det|Egg|0"} if visible else set(),
                          agent_pos={"x": 0.0, "z": 0.0}, action_ok=action_ok)


# --------------------------------------------------------------------------
# 老行为不能退化
# --------------------------------------------------------------------------
def test_visible_and_within_arm_reach_still_works():
    wm = _wm_with_egg(0.6)
    _pickup(wm, visible=True, action_ok=True)
    assert wm._holding == "det|Egg|0", wm._holding


def test_unknown_frame_diff_stays_conservative():
    """帧差未知（目标没完全入画）时不许放宽 —— 宁可不记，也不记错。"""
    wm = _wm_with_egg(1.2)
    _pickup(wm, visible=False, action_ok=None)
    assert wm._holding is None, wm._holding


def test_out_of_reach_is_not_claimed():
    """帧差变了也够不着：2.5 m 明显超出伸手范围，不能记成拿到了。"""
    wm = _wm_with_egg(2.5)
    _pickup(wm, visible=False, action_ok=True)
    assert wm._holding is None, wm._holding


# --------------------------------------------------------------------------
# 这次修的就是这一条
# --------------------------------------------------------------------------
def test_invisible_but_in_reach_is_credited_when_frame_changed():
    """检测器看不见了（拿在手上就是这样），但帧差确认动作生效、且它就在伸手
    范围内 —— 必须记成手持。"""
    wm = _wm_with_egg(1.2)
    _pickup(wm, visible=False, action_ok=True)
    assert wm._holding == "det|Egg|0", wm._holding


def test_failed_pickup_never_reaches_here():
    """帧差说这一步没生效：函数开头就 return，状态一个都不许动。"""
    wm = _wm_with_egg(1.2)
    _pickup(wm, visible=False, action_ok=False)
    assert wm._holding is None, wm._holding


def test_put_clears_the_hand():
    wm = _wm_with_egg(1.2)
    _pickup(wm, visible=False, action_ok=True)
    assert wm._holding is not None
    wm._update_action_log("PutObject", "Fridge", visible_ids=set(),
                          agent_pos={"x": 0.0, "z": 0.0}, action_ok=True)
    assert wm._holding is None, wm._holding


def test_holding_without_seeing_does_not_stay_a_navigation_target():
    """修好①之后②③自动消失：手上那个东西不该再收到"记住的位置/建议靠近"。

    这条是端到端的 —— 之前 1807 条"建议先转向它再靠近"里有 480 条指向
    已交互目标，靠的就是 target_priority 里那道 ``tp == holding`` 的保护，
    而它只在 holding 非空时才生效。
    """
    wm = _wm_with_egg(1.2)
    _pickup(wm, visible=False, action_ok=True)
    assert wm._holding is not None
    md = wm._to_metadata({"x": 0.0, "y": 0.0, "z": 0.0},
                         {"x": 0.0, "y": 0.0, "z": 0.0}, set())
    held_type = md["inventoryObjects"][0]["objectType"]
    assert held_type == "Egg", md["inventoryObjects"]

    hinter = TargetHinter(task_description="Take the egg out of the fridge.")
    block = hinter.update(md, {"Egg": (3, "PickupObject", True)},
                          held_type, 4)
    assert "建议先转向它再靠近" not in block, block
    assert "记住的位置" not in block, block


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
