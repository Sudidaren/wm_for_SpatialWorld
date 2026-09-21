#!/usr/bin/env python3
"""把当前这版 WM 实际注入给 MLLM 的提示**原样打印出来**。

交接文档里的提示表有两份（09-15 / 09-19），其中 09-15 那张已经过时（有 5 条
代码里已经不存在）。这个脚本不问文档，直接驱动 shipped 的
MemoryProbe + TargetHinter，把每种触发条件下**真实的字符串**打出来。

    python3 tools/show_wm_prompt.py
"""

from __future__ import annotations

import sys

sys.path.insert(0, '/home/sudidaren/SpatialWorld')

from mllm_base_agent.agent.memory_probe import MemoryProbe  # noqa: E402


def meta(objects, held=None, x=0.0, z=0.0, yaw=0.0):
    return {
        "agent": {"position": {"x": x, "y": 0.9, "z": z},
                  "rotation": {"x": 0.0, "y": yaw, "z": 0.0}},
        "objects": objects,
        "inventoryObjects": ([{"objectId": f"det|{held}|1", "objectType": held,
                               "objectTypeDisplay": held}] if held else []),
    }


def obj(tp, dist, visible=False, last=3, sigma=0.2, x=0.0, z=None, state=None):
    return {"objectId": f"det|{tp}|0", "objectType": tp,
            "objectTypeDisplay": tp,
            "position": {"x": x, "y": 0.9, "z": dist if z is None else z},
            "visible": visible, "distance": dist, "edge_px": 200.0,
            "box": [0, 0, 10, 10], "last_seen_step": last, "seen_count": 2,
            "sigma": sigma, "state": state or {}, "contents": []}


def probe(task):
    p = MemoryProbe(task_description=task)
    p.set_target_hint({"enabled": True, "limit": 6, "names_limit": 5,
                       "max_dist": 3.0, "max_sigma": 0.5})
    return p


def show(title, note, block):
    print("=" * 78)
    print(f"【{title}】")
    print(f"  触发条件：{note}")
    print("-" * 78)
    print(block if block else "（本步不注入任何内容）")
    print()


def main() -> None:
    # 1) 目标在视野里、但还在 1m 之外，交互又失败了 -> 距离提示
    p = probe("pick up the mug")
    show("目标可见 + 交互失败 + 还太远",
         "目标可见、距离 1.6m > 1m 门槛，且上一步 PickupObject 没让画面变化",
         p.update(wm_metadata=meta([obj("Mug", 1.6, visible=True)]),
                  action_name="PickupObject", object_type="Mug",
                  blocked=False, action_ok=False))

    # 2) 目标看不见，但记忆里有位置 -> 任务相关记忆 + 建议靠近
    p = probe("open the laptop")
    show("目标看不见、但记得它在哪",
         "Laptop 不在当前帧，但 WM 记着它的位置（step 3 看到过）",
         p.update(wm_metadata=meta([obj("Laptop", 1.6, visible=False)]),
                  action_name="MoveAhead", blocked=False, action_ok=True))

    # 3) 手里拿着东西 —— 只报手持，不再给位置
    p = probe("throw the lettuce in the trash")
    show("手里拿着东西",
         "inventoryObjects 里有 Lettuce；同一物体即使这一帧被检测到，也不再报位置",
         p.update(wm_metadata=meta([obj("Lettuce", 0.3, visible=True),
                                    obj("GarbageCan", 2.0, visible=False)],
                                   held="Lettuce"),
                  action_name="MoveAhead", blocked=False, action_ok=True))

    # 4) 被挡住：压掉"建议靠近"和"距离提示"，只留脱困
    p = probe("open the laptop")
    show("上一步被挡住了",
         "目标不可见、本来会发「建议先转向它再靠近」，但 blocked=True 把它压掉一步",
         p.update(wm_metadata=meta([obj("Laptop", 1.6, visible=False)]),
                  action_name="MoveAhead", blocked=True, action_ok=False))

    # 5) 同一个动作反复失败 -> 重复提示（措辞是"最近 3 次"，不是"连续 3 次"）
    p = probe("open the fridge")
    args = dict(wm_metadata=meta([obj("Fridge", 2.0, visible=False)]),
                action_name="MoveAhead", object_type=None,
                blocked=False, action_ok=False)
    for _ in range(2):
        p.update(**args)
    show("同一个动作反复不灵",
         "最近 6 步里该动作最近 3 次出现都没让画面变化（不要求背靠背）",
         p.update(**args))

    # 6) 什么都没有可说的那一步
    p = probe("open the laptop")
    show("没什么可说的",
         "目标可见且就在手边、动作成功、没被挡 —— 空手那行也已经不再注入",
         p.update(wm_metadata=meta([obj("Laptop", 0.6, visible=True)]),
                  action_name="MoveAhead", blocked=False, action_ok=True))


if __name__ == "__main__":
    main()
