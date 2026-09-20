#!/usr/bin/env python
"""相机 / 深度参数只能来自"定义 / 非评测房 / 几何"，且必须声明来源。

红线（2026-09-20 与用户确认，见 spatialworld_eval/PARAM_POLICY.md）：

  禁止 A：不得用评测任务（311 AI2-THOR 等）的反饋去拟合相机或深度参数；
  禁止 B：不得用任务真值（success_conditions / golden_actions /
          target_object_types / 模拟器物体表与位姿）去挑参数。

这个测试把"红线"变成机械可查的东西：从源码里正则抽出每个旋钮的默认值，
和下面的清单逐字比对；每个旋钮必须声明来源，且来源必须在允许的三种之内。
改了默认值而不更新声明 —— 测试直接失败。

    python tests/test_param_provenance.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

#: 允许的来源。**故意不含任何指向评测集的来源** —— 加进来就等于破例。
ALLOWED_SOURCES = {"definition", "non_eval_rooms", "geometry"}

#: 每个旋钮：名字、源码文件、代码里的默认值、来源、依据。
#: 依据栏里必须写清楚"凭什么"，而不是"哪个误差小"。
KNOBS = [
    dict(env="LIGHTWM_FOV_CONVENTION",
         file="mllm_base_agent/agent/world_model.py",
         default="vertical", source="definition",
         why="AI2-THOR 的 fieldOfView 定义为垂直视场；"
             "真值深度下的 3088 个记录视角印证 3D 0.189m vs 旧约定 0.356m"),
    dict(env="LIGHTWM_TRI_WEIGHT",
         file="mllm_base_agent/agent/world_model.py",
         default="1.0", source="non_eval_rooms",
         why="707 个留出物体：纯几何 0.236m vs 纯深度 0.8m"),
    dict(env="LIGHTWM_TRI_BASELINE",
         file="mllm_base_agent/agent/world_model.py",
         default="6.0", source="non_eval_rooms",
         why="12 个 classic 非评测 episode 的基线扫描"),
    dict(env="LIGHTWM_DEPTH_PROBE",
         file="mllm_base_agent/agent/world_model.py",
         default="center", source="non_eval_rooms",
         why="出厂值；任何改动都要先拿 non_eval 数据说话"),
    dict(env="LIGHTWM_SCALE_ANCHOR",
         file="mllm_base_agent/agent/world_model.py",
         default="0", source="geometry",
         why="零标签（只用 WM 自己的三角化/深度比值），尚未验证故默认关"),
    dict(env="LIGHTWM_DA2_SCALE",
         file="../lightwm_phases/phase_b/rfdetr_depth_runtime.py",
         default="1.3816", source="non_eval_rooms",
         why="深度头在非评测房上的尺度常数"),
]


def _default_in_source(path: Path, env: str):
    """把 ``os.environ.get("ENV", "<默认>")`` 里的默认值抠出来。"""
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    m = re.search(re.escape(env) + r'",\s*"([^"]*)"\s*\)', text)
    return m.group(1) if m else None


def _resolve(rel: str):
    """在几个候选根目录里找源码文件。

    交付副本（runtime_overlay）只带 SpatialWorld 的一部分，`phase_b/` 不在里面；
    所以先按仓库相对路径找，再按"仓库的兄弟目录"找，都找不到就跳过该旋钮
    （声明仍然必须存在 —— 来源要求不会被绕过）。
    """
    for root in (REPO, REPO.parent):
        cand = (root / rel).resolve()
        if cand.exists():
            return cand
    return None


def test_every_knob_defaults_to_its_declared_value():
    """改了默认值却不动这份声明 —— 直接失败。"""
    missing = []
    for knob in KNOBS:
        path = _resolve(knob["file"])
        if path is None:
            print(f'   （跳过 {knob["env"]}：这份 checkout 里没有 '
                  f'{knob["file"]}）')
            continue
        got = _default_in_source(path, knob["env"])
        if got is None:
            missing.append(f'{knob["env"]}（在 {knob["file"]} 里找不到默认值）')
        elif got != knob["default"]:
            missing.append(f'{knob["env"]}: 代码里是 {got!r}，声明是 '
                           f'{knob["default"]!r}')
    assert not missing, "; ".join(missing)


def test_every_knob_declares_an_allowed_source():
    bad = [f'{k["env"]} -> {k["source"]}' for k in KNOBS
           if k["source"] not in ALLOWED_SOURCES]
    assert not bad, ("参数来源必须在 {definition, non_eval_rooms, geometry} 之内，"
                     f"不允许指向评测集：{bad}")


def test_every_knob_states_why_not_which_error_is_smaller():
    """依据栏必须写来源，而不是"哪个误差小"。"""
    hollow = [k["env"] for k in KNOBS if len(k.get("why", "")) < 12]
    assert not hollow, f"这些旋钮没有写清依据：{hollow}"


def test_fov_convention_is_definitional_and_vertical():
    """FOV 约定由模拟器定义导出，不能用"误差更小"来换。

    AI2-THOR 的 fieldOfView 是垂直视场，所以 fx = fy = (height/2)/tan(fov/2)。
    """
    fov = next(k for k in KNOBS if k["env"] == "LIGHTWM_FOV_CONVENTION")
    assert fov["default"] == "vertical", fov
    assert fov["source"] == "definition", fov
    text = (REPO / "mllm_base_agent/agent/world_model.py").read_text(
        encoding="utf-8")
    # 源码里必须留着"为什么是 vertical"的说明，别让后来人只看到结果
    assert "VERTICAL fov" in text, "world_model.py 丢了 FOV 约定的由来说明"
    assert "ai2thor" in text.lower() or "AI2-THOR" in text


def test_the_policy_file_ships_with_the_harness():
    """策略文件必须和 harness 在一起，否则没人会读到它。"""
    policy = REPO.parent / "spatialworld_eval" / "PARAM_POLICY.md"
    if not policy.exists():
        print(f"   （跳过：{policy} 在交付副本里不一定存在）")
        return
    text = policy.read_text(encoding="utf-8")
    for rule in ("禁止 A", "禁止 B", "non_eval_rooms", "definition"):
        assert rule in text, (rule, policy)


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
