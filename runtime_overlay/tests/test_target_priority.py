#!/usr/bin/env python
"""Self-checks for the model-named target hint (target_priority.py).

No object vocabulary and no alias table may appear anywhere in this path: the
model names the task objects itself, and those runtime strings are matched
against the perception head's runtime output.

Run:  python tests/test_target_priority.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mllm_base_agent.agent import target_priority as tp  # noqa: E402


class StubVLM:
    """Echoes a fixed JSON answer for the one-shot object extraction (v2格式)."""

    def __init__(self, objects, legacy=False):
        self.objects = objects
        self.legacy = legacy
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        if self.legacy:
            payload = '{"objects": [' + ", ".join(
                f'"{o}"' for o in self.objects) + "]}"
        else:
            payload = '{"objects": [' + ", ".join(
                '{"env": "%s", "words": "%s"}' % (o, o.lower())
                for o in self.objects) + "]}"
        return type("R", (), {"content": payload})


def obj(otype, dist=1.0, visible=False, x=0.0, z=1.0, last=1, sigma=0.2,
        state=None, contents=None):
    return {"objectType": otype, "distance": dist, "visible": visible,
            "position": {"x": x, "y": 0.9, "z": z}, "last_seen_step": last,
            "sigma": sigma, "state": state or {}, "contents": contents or []}


def meta(objects, agent_xy=(0.0, 0.0), yaw=0.0, held=None):
    return {"agent": {"position": {"x": agent_xy[0], "y": 0.9, "z": agent_xy[1]},
                      "rotation": {"x": 0.0, "y": yaw, "z": 0.0}},
            "objects": objects,
            "inventoryObjects": ([{"objectId": "det|" + held + "|1",
                                   "objectType": held}] if held else [])}


# ---- no vocabulary in the path --------------------------------------

def test_no_object_vocabulary_in_source():
    import inspect

    src = inspect.getsource(tp)
    for token in ('"GarbageCan"', '"Laptop"', "'GarbageCan'", "_ALIASES"):
        assert token not in src, f"词表残留: {token}"


# ---- model-named extraction and matching ----------------------------

def test_model_names_come_from_the_model_once():
    vlm = StubVLM(["GarbageCan", "Lettuce"])
    h = tp.TargetHinter("throw the lettuce in the trash", vlm=vlm)
    assert h.model_names() == ["GarbageCan", "Lettuce"]
    h.model_names()
    assert vlm.calls == 1, vlm.calls          # 只问一次


def test_legacy_string_format_still_parsed():
    vlm = StubVLM(["GarbageCan"], legacy=True)
    h = tp.TargetHinter("throw it in the trash", vlm=vlm)
    assert h.entries() == [("GarbageCan", "GarbageCan")], h.entries()


def test_words_fallback_when_env_name_differs():
    # 模型写 env=Fridge，但检测头报的是 Refrigerator（或反之）时的退回逻辑
    class V:
        calls = 0
        def invoke(self, messages):
            V.calls += 1
            return type("R", (), {"content":
                '{"objects": [{"env": "Refrigerator", "words": "fridge"}]}'})
    h = tp.TargetHinter("open the fridge", vlm=V())
    # 检测头输出 Fridge，env 名 "Refrigerator" 匹配不上，退回 words "fridge"
    text = h.update(meta([obj("Fridge", dist=1.5)]), {}, "", 1)
    assert "Fridge" in text and "目前没见过" not in text, text


def test_matching_is_case_and_substring_based():
    got = tp.match_names(["garbage can", "CellPhone"],
                         ["GarbageCan", "CellPhone", "Desk"])
    assert got["garbage can"] == "GarbageCan", got
    assert got["CellPhone"] == "CellPhone", got


def test_hint_reports_only_detected_matches():
    vlm = StubVLM(["GarbageCan", "Lettuce"])
    # mode="llm" is the *legacy* path (ask the VLM to name the objects); the
    # default is now the deterministic object-to-object match below.
    h = tp.TargetHinter("throw the lettuce in the trash", vlm=vlm, mode="llm")
    text = h.update(meta([obj("GarbageCan", dist=2.0)]), {}, "", 1)
    assert "GarbageCan" in text, text
    assert "Lettuce（任务目标·模型自述）：目前没见过" in text, text


def test_hint_skips_visible_and_memory_only():
    vlm = StubVLM(["Laptop", "Fridge"])
    h = tp.TargetHinter("open the laptop and put the apple in the fridge",
                        vlm=vlm)
    text = h.update(meta([obj("Laptop", dist=1.0, visible=True),
                          obj("Fridge", dist=2.0, x=0.0, z=2.0)]), {}, "", 1)
    assert "视野内：Laptop" in text, text
    assert "任务相关记忆（当前看不见的）" in text, text
    assert "- Laptop（" not in text, text
    assert "记住的位置在正前方约 2.0m" in text, text


def test_position_is_repeated_every_step():
    """2026-09-19：取消了"位置同上"压缩，每步都照常给完整位置。"""
    vlm = StubVLM(["Laptop"])
    h = tp.TargetHinter("open the laptop", vlm=vlm)
    m = meta([obj("Laptop", dist=2.0, x=0.0, z=2.0)])
    assert "约 2.0m" in h.update(m, {}, "", 1)
    again = h.update(m, {}, "", 2)
    assert "位置同上" not in again, again
    assert "约 2.0m" in again, again
    # 走近了（目标在正前方 1.0m）——仍然每步给完整位置
    moved = h.update(meta([obj("Laptop", dist=1.0, x=0.0, z=1.0)],
                          agent_xy=(0.0, 0.0)), {}, "", 3)
    assert "约 1.0m" in moved, moved


def test_quota_and_settled_demotion():
    vlm = StubVLM(["DeskLamp"])
    h = tp.TargetHinter("please turn off the desk lamp", vlm=vlm)
    lamp = obj("DeskLamp", state={"isToggled": False})
    # 已经关掉（WM 账本）-> 不再出现在每步块里
    assert h.update(meta([lamp]), {}, "", 1) == ""

    vlm2 = StubVLM(["Laptop"])
    h2 = tp.TargetHinter("find the laptop", vlm=vlm2, limit=6)
    objects = [obj("Laptop"), obj("Pan"), obj("Mug"), obj("Bowl"), obj("Statue")]
    text = h2.update(meta(objects), {}, "", 1)
    body = [l for l in text.splitlines() if l.startswith("- ")]
    assert len(body) <= 3, text                # tier1 + tier3/4 各 1


def test_held_object_reported_without_position():
    vlm = StubVLM(["Egg"])
    h = tp.TargetHinter("crack the egg", vlm=vlm, mode="llm")
    text = h.update(meta([obj("Pan")], held="Egg"), {}, "Egg", 1)
    assert "Egg（手持/容器内容" in text, text


# --- the deterministic object-to-object path (default since 2026-09-19) ----

def test_exact_match_uses_only_names_in_the_instruction():
    h = tp.TargetHinter("put the apple in the fridge")
    text = h.update(meta([obj("Apple", dist=1.0), obj("Fridge", dist=2.0, x=0.0, z=2.0),
                          obj("GarbageCan", dist=3.0, x=0.0, z=3.0)]), {}, "", 1)
    assert "Fridge" in text, text
    assert "GarbageCan" not in text, text          # not named -> not relevant


def test_exact_match_handles_compounds_and_plurals():
    h = tp.TargetHinter("turn off the desk lamps and slice the tomatoes")
    h.update(meta([obj("DeskLamp", dist=2.0, x=0.0, z=2.0),
                   obj("Tomato", dist=1.0, x=0.0, z=1.0),
                   obj("Potato", dist=1.0, x=0.0, z=1.0)]), {}, "", 1)
    assert h.relevant(["DeskLamp", "Tomato", "Potato"]) == {"DeskLamp", "Tomato"}


def test_exact_match_needs_no_vlm_call():
    class Boom:
        def invoke(self, *a, **kw):
            raise AssertionError("the exact path must not call the model")

    h = tp.TargetHinter("open the laptop", vlm=Boom())
    h.update(meta([obj("Laptop", dist=2.0, x=0.0, z=2.0)]), {}, "", 1)
    assert h.relevant(["Laptop"]) == {"Laptop"}


def test_relevance_persists_after_the_object_leaves_the_view():
    h = tp.TargetHinter("open the laptop")
    h.update(meta([obj("Laptop", dist=2.0, x=0.0, z=2.0)]), {}, "", 1)
    # 第二步检测列表里没有 Laptop 了，但相关性必须保留（规则 3）
    assert h.relevant(["Desk", "Mug"]) == {"Laptop"}


def test_exact_mode_emits_no_never_seen_lines():
    h = tp.TargetHinter("throw the lettuce in the trash")
    text = h.update(meta([obj("GarbageCan", dist=2.0, x=0.0, z=2.0)]), {}, "", 1)
    assert "没见过" not in text, text


def test_far_away_tail_only_compares_the_same_object():
    """「你在远离它」不能拿 A 物体的距离去比 B 物体的距离。

    实测 616 次"你在远离它"里有 167 次（27%）是跨物体比较 —— 上一步建议
    靠近 Egg，这一步的候选换成 Microwave，两个不同的距离一减就宣布
    "你在远离它"。所以只有指向同一个物体时才可以比较。
    """
    h = tp.TargetHinter("open the laptop and turn on the desk lamp")
    # 第一步：最近的任务目标是 Laptop(1.0m)
    h.update(meta([obj("Laptop", dist=1.0, x=0.0, z=1.0),
                   obj("DeskLamp", dist=3.0, x=0.0, z=3.0)]), {}, "", 1)
    # 第二步：最近的目标换成了另一个物体，而且更远 —— 必须**不**报"远离"
    text = h.update(meta([obj("DeskLamp", dist=2.5, x=0.0, z=2.5)]), {}, "", 2)
    assert "你在远离它" not in text, text


def test_far_away_tail_still_fires_for_the_same_object():
    """同一个物体、确实变远了 —— 这条提醒要保留（别把功能改没了）。"""
    h = tp.TargetHinter("open the laptop")
    h.update(meta([obj("Laptop", dist=1.0, x=0.0, z=1.0)]), {}, "", 1)
    text = h.update(meta([obj("Laptop", dist=1.8, x=0.0, z=1.8)]), {}, "", 2)
    assert "你在远离它" in text, text


def test_far_away_tail_needs_an_unrefreshed_anchor():
    """锚点被重新观测刷新过就不比 —— 那时差值里混着深度噪声，比不得。

    只有两次的 last_seen_step 相同，才说明两边量的是同一个锚点，距离差
    只来自 agent 自己的移动（里程计精确），这时提醒才是可信的。
    """
    h = tp.TargetHinter("open the laptop")
    h.update(meta([obj("Laptop", dist=1.0, x=0.0, z=1.0, last=1)]), {}, "", 1)
    # 同一个物体、距离也变大了，但中间重新看到过它（last 从 1 变成 2）
    text = h.update(meta([obj("Laptop", dist=1.8, x=0.0, z=1.8, last=2)]),
                    {}, "", 2)
    assert "你在远离它" not in text, text


def test_far_away_tail_cannot_print_the_same_distance_twice():
    """增量必须大到两个显示值一定不同（打印只有 1 位小数）。

    实测有 28 条提示写成"你在远离它，上一步 0.5m → 现在 0.5m" —— 阈值
    0.05m 小于显示精度 0.1m。增量 >=0.15m 时两个一位小数必定不同。
    """
    h = tp.TargetHinter("open the laptop")
    h.update(meta([obj("Laptop", dist=0.46, x=0.0, z=0.46, last=1)]),
             {}, "", 1)
    text = h.update(meta([obj("Laptop", dist=0.52, x=0.0, z=0.52, last=1)]),
                    {}, "", 2)
    assert "你在远离它" not in text, text      # 0.5 -> 0.5 这种一律不发


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
