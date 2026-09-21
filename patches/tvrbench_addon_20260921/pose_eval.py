# -*- coding: utf-8 -*-
"""TVRBench 判定：位姿口径（官方参考）+ 可见性口径（本 addon 的正式判据）。

与 spatialworld_eval v1.0 的判定完全独立：这里不动冻结文件，只提供新条件类型
`agent_pose` / `objects_visible`，以后接 v1.1 时再挂上去。

两个口径的关系（2026-09-21 定）：
  * `agent_pose` —— TVRBench 官方口径，位置 0.01 m / 朝向 1° / 俯仰 1°。
    但动作粒度是 0.25 m / 45° / 30°，所以它实质要求"落在同一格、同一角度"，
    差一次 LookDown 就是 0 分。留作**参考指标**，不当作成败。
  * `objects_visible` —— 正式判据：**目标位姿能看到的那些"可判别锚点"，在
    最终画面里是否也都在**。锚点清单在生成任务时就用同口径量好了
    （见 audit_anchors.py），判据本身只读模拟器渲染出的实例掩码，
    属于裁判权限；**任何情况下都不进 agent 的 prompt**。

`objects_visible` 必须用**实例分割掩码**：实测同一帧下 `object["visible"]`
只报 1 个物体，而掩码能给出 21 个（Chair/DiningTable/Plate/Vase…），两者不等价。
掩码由判定端按需打开（controller.step(action="Pass",
renderInstanceSegmentation=True)），不增加 episode 内的渲染成本。
"""

from __future__ import annotations

import math

#: 硬结构面：几乎在任何位置都能看到，任何口径都不该当锚点
HARD_STRUCTURAL = {
    "Wall", "WallBase", "Floor", "Ceiling", "Doorway", "Window", "Room",
    "BackSplash", "StandardWallTileHeight", "StandardWallSize", "RoomDecor",
    "LightSwitch", "CeilingLight", "Door",
}
#: 软装饰：loose 口径里算锚点，strict 口径里剔掉
SOFT_DECOR = {"Painting", "Poster", "Mirror", "Curtains", "Blinds", "Rug", "WallDecor"}

#: 目标位姿下掩码占比 >=0.5% 才算"可判别锚点"（同 audit_anchors.py）
ANCHOR_MIN_FRAC = 0.005
#: area1pct 口径：每个锚点至少有一个实例占画面 >=1%
AREA1PCT_MIN_FRAC = 0.01

MASK_HINT = ('objects_visible 需要实例掩码：判定端先执行 '
             'controller.step(action="Pass", renderInstanceSegmentation=True)')

# TVRBench 官方默认容差（scripts/eval_va.py → is_exact_match 默认参数）
POS_TOL = 0.01     # 米
ROT_TOL = 1.0      # 度
HOR_TOL = 1.0      # 度


def position_error(state: dict, target: dict) -> float:
    dx = state["position"]["x"] - target["position"]["x"]
    dz = state["position"]["z"] - target["position"]["z"]
    return math.sqrt(dx * dx + dz * dz)


def rotation_error(state: dict, target: dict) -> float:
    diff = abs(state["rotation_y"] - target["rotation_y"]) % 360
    return min(diff, 360 - diff)


def horizon_error(state: dict, target: dict) -> float:
    return abs(state["horizon"] - target["horizon"])


def is_exact_match(state: dict, target: dict,
                   pos_tol: float = POS_TOL, rot_tol: float = ROT_TOL,
                   hor_tol: float = HOR_TOL) -> bool:
    return (position_error(state, target) < pos_tol
            and rotation_error(state, target) < rot_tol
            and horizon_error(state, target) < hor_tol)


def state_from_metadata(metadata: dict) -> dict:
    agent = metadata["agent"]
    return {
        "position": {"x": agent["position"]["x"], "y": agent["position"]["y"],
                     "z": agent["position"]["z"]},
        "rotation_y": float(agent["rotation"]["y"]),
        "horizon": float(agent["cameraHorizon"]),
    }


def check_agent_pose(condition: dict, metadata: dict) -> bool:
    """spatialworld 风格的成功条件入口。

    condition = {"type": "agent_pose",
                 "target": {"position": {...}, "rotation_y": .., "horizon": ..},
                 "pos_tol": 0.01, "rot_tol": 1.0, "hor_tol": 1.0,
                 "require_stop": True}
    metadata  = ai2thor event.metadata（取 agent.position / rotation / cameraHorizon）
    """
    state = state_from_metadata(metadata)
    target = condition.get("target") or condition
    ok = is_exact_match(
        state, target,
        pos_tol=condition.get("pos_tol", POS_TOL),
        rot_tol=condition.get("rot_tol", ROT_TOL),
        hor_tol=condition.get("hor_tol", HOR_TOL),
    )
    if condition.get("require_stop", True):
        ok = ok and bool(metadata.get("_stopped", False))
    return ok


# ---------------------------------------------------------------------------
# 可见性判定（正式判据）
# ---------------------------------------------------------------------------
def type_fracs(event, metadata: dict) -> dict:
    """{objectType: 该类型在画面里的最大掩码占比}。

    与 audit_anchors.py 的 type_fracs 同口径：遍历 event.instance_masks，
    用 metadata 里的 objectId → objectType 映射归并，取同类型的最大值。
    没有掩码（没开 renderInstanceSegmentation）时返回空 dict。
    """
    masks = getattr(event, "instance_masks", None) or {}
    if not masks:
        return {}
    id2type = {o["objectId"]: o.get("objectType", "") for o in metadata.get("objects", [])}
    out: dict = {}
    for oid, mask in masks.items():
        ty = id2type.get(oid)
        if not ty:
            continue
        h, w = mask.shape[:2]
        frac = float(mask.sum()) / float(h * w)
        if frac > out.get(ty, 0.0):
            out[ty] = frac
    return out


def strict_anchors_from(fracs: dict) -> dict:
    """按生成锚点时的同一口径筛出 strict 锚点集合（剔硬结构面 + 软装饰）。"""
    loose = {k: v for k, v in fracs.items()
             if k not in HARD_STRUCTURAL and v >= ANCHOR_MIN_FRAC}
    return {k: v for k, v in loose.items() if k not in SOFT_DECOR}


def check_objects_visible(condition: dict, fracs: dict) -> bool:
    """condition = {"type": "objects_visible",
                     "anchors": ["Book", "Sofa", ...],
                     "mode": "presence" | "area1pct",
                     "min_area": 0.01,          # 仅 area1pct 用
                     "require_stop": false}

    fracs = type_fracs(event, metadata)。
    """
    anchors = [a for a in (condition.get("anchors") or []) if a]
    if not anchors:
        return False
    mode = (condition.get("mode") or "presence").lower()
    if mode == "area1pct":
        need = float(condition.get("min_area", AREA1PCT_MIN_FRAC))
        return all(fracs.get(a, 0.0) >= need for a in anchors)
    return all(fracs.get(a, 0.0) > 0.0 for a in anchors)


def judge(condition: dict, metadata: dict, event=None, stopped: Optional[bool] = None) -> bool:
    """统一入口：按 condition["type"] 分派到两个口径之一。"""
    ctype = (condition or {}).get("type")
    md = metadata
    if stopped is not None:
        md = {**metadata, "_stopped": bool(stopped)}
    if ctype == "objects_visible":
        ok = check_objects_visible(condition, type_fracs(event, metadata))
    elif ctype == "agent_pose":
        ok = check_agent_pose(condition, md)
    else:
        raise ValueError(f"未知条件类型: {ctype}")
    if condition.get("require_stop", True):
        ok = ok and bool(md.get("_stopped", False))
    return ok
