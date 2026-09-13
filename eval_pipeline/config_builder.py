# -*- coding: utf-8 -*-
"""Build the per-task YAML config that is handed to the official runner.

原则：官方配置不动。我们只做三件事：
  1. 以每个场景官方 config_close_*.yaml 为基底；
  2. 覆盖 model.vlm（换成选中的 GPT/Gemini/Qwen 预设）；
  3. wingman_llm 时追加 memory_probe / world_model 块（targets 自动取自该
     任务的 target_object_types），并打开 depth / instance-segmentation 渲染；
     headless 时按官方 run_benchmark 的做法把 env.platform 设为 CloudRendering。
"""

from __future__ import annotations

import copy
import os
import zlib
from pathlib import Path

import yaml

import eval_config as cfg
from env_spec import EnvSpec, TaskInfo


def _base_yaml(spec: EnvSpec) -> dict:
    path = spec.config_path
    if not path.exists():
        raise FileNotFoundError(f"官方基底配置不存在: {path}")
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data


def _apply_model_block(data: dict) -> None:
    model_cfg = cfg.resolve_llm_model()
    data.setdefault("model", {})
    data["model"]["vlm"] = copy.deepcopy(model_cfg)
    # 官方基底配置里 planner 指向 localhost:11435；本地没有该服务时会让
    # agent 卡在子目标分解。默认复用主模型，避免死等本地服务。
    if getattr(cfg, "PLANNER_FROM_MAIN_MODEL", True):
        planner = copy.deepcopy(model_cfg)
        planner["max_tokens"] = min(int(planner.get("max_tokens", 512)), 512)
        planner["temperature"] = 0.2
        planner["client_max_retries"] = 2
        data["model"]["planner"] = planner


def _apply_context_block(data: dict) -> None:
    ctx = data.setdefault("context_management", {})
    ctx["enable_subgoal_decomposition"] = bool(
        getattr(cfg, "ENABLE_SUBGOAL_DECOMPOSITION", False))


def _apply_wingman_block(data: dict, task: TaskInfo) -> None:
    wm = cfg.WINGMAN_OPTIONS
    targets = [t for t in task.target_types if t]
    data.setdefault("env", {})
    # WorldModel 的 V1 眼睛：真实感知头（RGB+depth），不用语义分割作弊。
    data["env"]["render_depth"] = True
    data["env"]["render_instance_segmentation"] = True
    data["memory_probe"] = {
        "enabled": True,
        "targets": targets or None,
        "navigation_directive": bool(wm.get("navigation_directive", True)),
        "interact_soft_dist": wm.get("interact_soft_dist", 1.2),
        "done_gate": bool(wm.get("done_gate", True)),
        "world_model": {
            "enabled": True,
            "pose_from_action_log": bool(wm.get("pose_from_action_log", True)),
            "pose_initial": wm.get("pose_initial", "origin"),
            "perception_ckpt": cfg.PERCEPTION_CKPT,
            "variant": wm.get("variant", "small"),
            "resolution": int(wm.get("resolution", 224)),
            "width": int(wm.get("width", 256)),
            "obj_thr": float(wm.get("obj_thr", 0.35)),
        },
    }


def _apply_headless(data: dict, spec: EnvSpec) -> None:
    if not cfg.HEADLESS:
        return
    if not spec.supports_headless_flag:
        # VirtualHome 用自己的模拟器运行时，不套 CloudRendering。
        return
    data.setdefault("env", {})
    data["env"]["platform"] = "CloudRendering"


def _apply_virtualhome_port(data: dict, spec: EnvSpec, task: TaskInfo) -> None:
    if spec.name != "virtualhome":
        return
    base = int(getattr(cfg, "VIRTUALHOME_PORT_BASE", 8100))
    port = base + (zlib.crc32(task.task_id.encode("utf-8")) % 800)
    data.setdefault("env", {})
    data["env"]["port"] = str(port)
    # 必须提供 X display（不带冒号），否则 wrapper 走 no_graphics 启动，
    # Unity 后端返回 502。'0' 表示 DISPLAY=:0（WSLg）。
    display = os.environ.get("DISPLAY", ":0") or ":0"
    data["env"]["x_display"] = display.lstrip(":")


def build_task_config(spec: EnvSpec, task: TaskInfo,
                      run_dir: Path) -> Path:
    """生成该任务的运行配置并落盘，返回路径。"""
    data = _base_yaml(spec)
    _apply_model_block(data)
    _apply_context_block(data)
    if cfg.PROFILE == "wingman_llm":
        _apply_wingman_block(data, task)
    _apply_headless(data, spec)
    _apply_virtualhome_port(data, spec, task)
    config_dir = run_dir / "task_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    out = config_dir / f"{task.task_id}.yaml"
    with out.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, allow_unicode=True, sort_keys=False)
    return out


def dry_model_sanity() -> None:
    """启动前快速检查：LLM 预设字段齐全。"""
    m = cfg.resolve_llm_model()
    for key in ("provider", "model_name", "temperature", "max_tokens"):
        assert key in m, f"LLM 预设缺少字段: {key}"
