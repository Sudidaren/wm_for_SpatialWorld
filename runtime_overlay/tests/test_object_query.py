#!/usr/bin/env python
"""Self-checks for mllm_base_agent.agent.object_query (state-check only).

Run:  python tests/test_object_query.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mllm_base_agent.agent import object_query as oq  # noqa: E402


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


def state_with(enabled=True):
    return {
        "config": {"memory_probe": {"object_query": {"enabled": enabled}}},
        "step_count": 0,
    }


def test_parse_action():
    assert oq.parse_query("CheckState()") == "__all__"
    assert oq.parse_query("check state") == "__all__"
    assert oq.parse_query("Query(all)") == "__all__"
    # 旧习惯（问具体物体）也按同一次状态查询处理，避免浪费一次解析重试
    assert oq.parse_query("Query(laptop)") == "laptop"
    assert oq.parse_query("MoveAhead(1)") is None
    assert oq.parse_query("DONE") is None
    assert oq.extract_action_string(
        "<THINK>x</THINK><ACTION>CheckState()</ACTION>") == "CheckState()"


def test_nothing_is_injected_per_step():
    st = state_with()
    mem = oq.memory_of(st)
    mem.observe(meta(objects=[("Laptop", 0.0, 0.9, 2.0, True, 2.0, 0.3)]), step=3)
    # 没有请求汇总时，每一步都不产生任何提示块（通用 query 已砍掉）
    assert oq.render_block(st, step=4) == ""
    assert oq.render_block(st, step=5) == ""


def test_check_state_returns_summary_once():
    st = state_with()
    action = oq.apply_query(st, "all", step=4)
    assert action["action_name"] == "CheckState"
    assert action["action_type"] == "internal_noop"
    assert st["should_continue"] is True
    mem = oq.memory_of(st)
    mem.observe(meta(agent_xy=(0.0, 0.0), yaw=0.0,
                     objects=[("Laptop", 0.0, 0.9, 2.0, True, 2.0, 0.3)],
                     held="Egg"), step=4)
    first = oq.render_block(st, step=5)
    assert "WM 物体状态汇总" in first and "Laptop" in first, first
    assert "手持：Egg" in first, first
    assert oq.render_block(st, step=6) == ""      # 只给一次


def test_summary_recomputes_bearing_for_the_current_pose():
    st = state_with()
    mem = oq.memory_of(st)
    # 第 3 步：物体在正前方 2m（yaw=0，+z 方向）
    mem.observe(meta(agent_xy=(0.0, 0.0), yaw=0.0,
                     objects=[("Laptop", 0.0, 0.9, 2.0, False, 2.0, 0.3)]), step=3)
    # 第 8 步：走过物体（agent z=3，物体 z=2），此时它应该在正后方
    mem.observe(meta(agent_xy=(0.0, 3.0), yaw=0.0,
                     objects=[("Laptop", 0.0, 0.9, 2.0, False, 2.0, 0.3)]), step=8)
    oq.apply_query(st, "all", step=8)
    text = oq.render_block(st, step=9)
    assert "正后方" in text, text
    assert "step 8 看到过" in text, text


def test_summary_marks_interactions():
    st = state_with()
    mem = oq.memory_of(st)
    mem.observe(meta(objects=[("Fridge", 0.0, 0.9, 1.5, True, 1.5, 0.2)]), step=2)
    mem.record_interaction(4, "OpenObject", "Fridge", True)
    oq.apply_query(st, "all", step=4)
    text = oq.render_block(st, step=5)
    assert "Fridge" in text and "已打开" in text, text


def test_disabled_by_default():
    st = {"config": {"memory_probe": {}}, "step_count": 0}
    assert oq.config_of(st) == {}
    assert oq.render_block(st, 0) == ""


def test_prompt_protocol_when_enabled():
    from mllm_base_agent.agent import runner

    def prompt_text(enabled):
        st = {
            "config": {"env": {"type": "ai2thor"},
                       "task": {"instruction": "open the laptop"},
                       "memory_probe": {"object_query": {"enabled": enabled}}},
            "task_prompt": "open the laptop",
            "step_count": 0,
        }
        msgs = runner._build_messages(st, "data:image/png;base64,AAAA")
        return " ".join(p.get("text", "") for p in msgs[-1].content
                        if isinstance(p, dict) and p.get("type") == "text")

    assert "CheckState()" not in prompt_text(False)
    assert "CheckState()" in prompt_text(True)


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"ok   {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
