# -*- coding: utf-8 -*-
"""Scene registry + task manifest for the three SpatialWorld home domains.

任务清单的唯一来源是 SpatialWorld 仓库：
  - data/<env>/tasks/<task_id>/task.json （指令、场景、golden actions、
    成功条件、目标物体）
  - task_classification_detail.csv     （category / task_type 官方标签）
评测编号（task_id）与官方 CSV / 数据目录完全一致。
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from eval_config import SPATIALWORLD_ROOT
from official_compat import load_task_metadata


@dataclass
class EnvSpec:
    name: str
    task_root: Path
    config_path: Path
    venv_python: Path
    supports_headless_flag: bool = True


@dataclass
class TaskInfo:
    env: str
    task_id: str
    scene: str = ""
    instruction: str = ""
    category: str = ""
    task_type: str = ""
    golden_steps: Optional[int] = None
    target_types: list = field(default_factory=list)
    task_json_path: Path = None


def env_specs() -> dict[str, EnvSpec]:
    root = Path(SPATIALWORLD_ROOT)
    cfgs = _config_map()
    pythons = _python_map()
    specs: dict[str, EnvSpec] = {}
    for env in ("ai2thor", "procthor", "virtualhome"):
        if env not in cfgs or env not in pythons:
            continue
        specs[env] = EnvSpec(
            name=env,
            task_root=root / "data" / env / "tasks",
            config_path=Path(cfgs[env]),
            venv_python=Path(pythons[env]),
            supports_headless_flag=env in ("ai2thor", "procthor"),
        )
    return specs


def _config_map() -> dict[str, str]:
    from eval_config import BASE_CONFIG

    return dict(BASE_CONFIG)


def _python_map() -> dict[str, str]:
    from eval_config import ENV_VENV_PYTHON

    return dict(ENV_VENV_PYTHON)


def classification_map() -> dict[str, list[dict]]:
    """task_id -> [ {environment, category, task_type, instruction}, ... ].

    同一个 task_id 可能同时出现在单 agent 与 multi-agent 两套协议里
    （AI2-THOR 29 个、ProcTHOR 7 个），所以这里按 task_id 收集所有行，
    不能只保留最后一行（否则单 agent 集合会被少算 36 个）。
    """
    path = Path(SPATIALWORLD_ROOT) / "task_classification_detail.csv"
    out: dict[str, list[dict]] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            tid = (row.get("task_id") or "").strip()
            if tid:
                out.setdefault(tid, []).append({
                    "environment": (row.get("environment") or "").strip(),
                    "category": (row.get("category") or "").strip(),
                    "task_type": (row.get("task_type") or "").strip(),
                    "instruction": (row.get("instruction") or "").strip(),
                })
    return out


def discover_tasks(env: str, spec: EnvSpec,
                   classification: dict) -> list[TaskInfo]:
    out: list[TaskInfo] = []
    if not spec.task_root.is_dir():
        return out
    for folder in sorted(spec.task_root.iterdir()):
        task_json = folder / "task.json"
        if not task_json.is_file():
            continue
        try:
            data = json.loads(task_json.read_text(encoding="utf-8"))
        except Exception:
            continue
        tid = str(data.get("task_id") or folder.name)
        rows = classification.get(tid, [])
        envs = {r.get("environment") for r in rows}
        if rows and env not in envs:
            continue
        cls = next((r for r in rows if r.get("environment") == env),
                   rows[0] if rows else {})
        # 与官方 load_task_action_count / load_task_metadata 同口径：
        # 优先 task.json 的 golden_actions.steps，否则数非空动作（含 DONE/FAIL）。
        meta = load_task_metadata(task_json)
        g_steps = meta["golden_action_count"]
        out.append(TaskInfo(
            env=env,
            task_id=tid,
            scene=str(data.get("scene") or data.get("scene_index") or ""),
            instruction=str(
                data.get("instruction")
                or data.get("target_description")
                or data.get("task_name")
                or cls.get("instruction")
                or ""),
            category=cls.get("category", ""),
            task_type=cls.get("task_type", ""),
            golden_steps=g_steps,
            target_types=list(data.get("target_object_types") or []),
            task_json_path=task_json,
        ))
    return out


def select_tasks(scenes: list[str], task_ids: list[str] | None,
                 id_filter: str = "") -> list[TaskInfo]:
    """按配置选择任务：空 task_ids = 全部，否则精确匹配；id_filter 正则再筛。"""
    specs = env_specs()
    cls = classification_map()
    wanted = set(task_ids or [])
    pat = re.compile(id_filter) if id_filter else None
    tasks: list[TaskInfo] = []
    for env in scenes:
        for t in discover_tasks(env, specs[env], cls):
            if wanted and t.task_id not in wanted:
                continue
            if pat and not pat.search(t.task_id):
                continue
            tasks.append(t)
    if wanted:
        missing = sorted(wanted - {t.task_id for t in tasks})
        if missing:
            raise ValueError(
                "以下任务 id 找不到（检查所属场景是否已开启）: "
                + ", ".join(missing))
    return tasks
