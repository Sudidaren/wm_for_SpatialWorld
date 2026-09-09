# -*- coding: utf-8 -*-
"""与 SpatialWorld 官方 run_benchmark 逻辑逐函数对齐的兼容实现。

来源（仅作行为对照，不改动官方仓库）：
    scripts/ai2thor/run_benchmark.py
    scripts/procthor/run_benchmark.py
    scripts/virtualhome/run_benchmark.py

这里复刻的函数：
    - find_result_json          （官方目录查找规则，含 run_* 重试目录）
    - load_task_metadata        （instruction / golden 计数/文本）
    - read_result_status_info   （task_result / failure_type / fail_reason）
    - determine_failure_reason  （默认 external_error）
    - decide_csv_status_from_result（Completed true/false/null 判定）
    - extract_actual_actions    （log.json / episode json 动作序列提取）
    - extract_token_stats_from_result_json
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional


def find_result_json(task_path: Path) -> Optional[Path]:
    """Official ai2thor run_benchmark.find_result_json 的移植。"""
    task_path = Path(task_path)
    log_file = task_path / "log.json"
    if log_file.exists():
        return log_file

    episode_files = sorted(task_path.glob("episode_*.json"))
    if episode_files:
        return episode_files[0]

    # run_task 的多进程重试目录兼容（官方逻辑）
    for run_dir in sorted(task_path.glob("run_*")):
        if not run_dir.is_dir():
            continue
        task_subdir_with_retry = run_dir / task_path.name
        if task_subdir_with_retry.exists():
            log_file = task_subdir_with_retry / "log.json"
            if log_file.exists():
                return log_file
            episode_files = sorted(
                task_subdir_with_retry.glob("episode_*.json"))
            if episode_files:
                return episode_files[0]

        task_name = task_path.name
        match = re.match(
            r"^(ai2thor\d+|procthor\d+|virtualhome\d+|carla\d+)"
            r"(?:_retry_\d+)?$", task_name)
        if match:
            task_id = match.group(1)
            task_subdir = run_dir / task_id
            if task_subdir.exists():
                log_file = task_subdir / "log.json"
                if log_file.exists():
                    return log_file
                episode_files = sorted(task_subdir.glob("episode_*.json"))
                if episode_files:
                    return episode_files[0]

        for subdir in sorted(run_dir.iterdir()):
            if not subdir.is_dir():
                continue
            log_file = subdir / "log.json"
            if log_file.exists():
                return log_file
            episode_files = sorted(subdir.glob("episode_*.json"))
            if episode_files:
                return episode_files[0]

        log_file = run_dir / "log.json"
        if log_file.exists():
            return log_file
        episode_files = sorted(run_dir.glob("episode_*.json"))
        if episode_files:
            return episode_files[0]

    return None


def load_task_metadata(task_json_path: Path) -> dict:
    """Official load_task_metadata：golden 计数优先用 task.json 的 steps 字段。"""
    info = {
        "instruction": None,
        "golden_action_count": None,
        "golden_action_text": None,
    }
    try:
        data = json.loads(Path(task_json_path).read_text(encoding="utf-8"))
    except Exception:
        return info
    info["instruction"] = data.get("instruction") or data.get("task_name")
    golden_actions = data.get("golden_actions")
    actions_list = None
    if isinstance(golden_actions, dict):
        steps = golden_actions.get("steps")
        if isinstance(steps, int):
            info["golden_action_count"] = steps
        actions = golden_actions.get("actions")
        if isinstance(actions, list):
            actions_list = [str(a).strip() for a in actions if str(a).strip()]
        if info["golden_action_count"] is None:
            info["golden_action_count"] = len(actions_list or [])
    elif isinstance(golden_actions, list):
        actions_list = [str(a).strip() for a in golden_actions if str(a).strip()]
        info["golden_action_count"] = len(actions_list)
    elif isinstance(golden_actions, str):
        actions_list = [x.strip() for x in golden_actions.split(",") if x.strip()]
        info["golden_action_count"] = len(actions_list)
    if actions_list:
        info["golden_action_text"] = " | ".join(actions_list)
    return info


def read_result_status_info(result_json: Optional[Path]) -> dict:
    """Official read_result_status_info。"""
    info = {
        "has_result_json": bool(result_json and Path(result_json).exists()),
        "task_result": None,
        "failure_type": None,
        "fail_reason": None,
    }
    if not result_json or not Path(result_json).exists():
        return info
    try:
        data = json.loads(Path(result_json).read_text(encoding="utf-8"))
    except Exception as exc:
        info["fail_reason"] = f"Failed to parse result JSON: {exc}"
        return info
    metadata = data.get("metadata", {}) if isinstance(data, dict) else {}
    task_result = metadata.get("task_result")
    if task_result in ("success", "failure"):
        info["task_result"] = task_result
    elif (isinstance(data, dict)
          and Path(result_json).name.startswith("episode_")):
        success = data.get("success")
        if success is True:
            info["task_result"] = "success"
        elif success is False:
            info["task_result"] = "failure"
    failure_type = data.get("failure_type") if isinstance(data, dict) else None
    if not failure_type:
        failure_type = metadata.get("failure_type")
    info["failure_type"] = failure_type
    fail_reason = metadata.get("fail_reason")
    if not fail_reason and isinstance(data, dict):
        fail_reason = data.get("fail_reason")
    info["fail_reason"] = fail_reason
    return info


def determine_failure_reason(result_json: Optional[Path]) -> str:
    """Official determine_failure_reason：JSON 有 failure_type 用之，否则 external。"""
    if result_json and Path(result_json).exists():
        info = read_result_status_info(result_json)
        if info["failure_type"]:
            return info["failure_type"]
    return "external_error"


def decide_csv_status_from_result(result_info: dict,
                                  fallback_failure_type: str) -> Optional[str]:
    """Official decide_csv_status_from_result（含 HTTP 错误码等细规则）。"""
    task_result = result_info.get("task_result")
    failure_type = result_info.get("failure_type") or fallback_failure_type
    fail_reason = result_info.get("fail_reason") or ""

    if task_result == "success":
        return "true"

    if fail_reason:
        if re.search(
            r"(?:\berror\s+code\b|\bstatus\s*code\b)\s*:\s*"
            r"(?:401|403|404|408|409|422|429|5\d{2})\b",
            fail_reason, re.I,
        ):
            return None
        low = fail_reason.lower()
        if "does not exist" in low and "model" in low:
            return None

    if fail_reason:
        low = fail_reason.lower()
        if ("model claimed done" in low or "done but" in low
                or "consecutive" in low and "action failures" in low
                or "maximum step" in low or "max_steps" in low
                or "reached maximum" in low
                or "model determined" in low
                or "refused to continue" in low
                or "called with invalid argument" in low
                and "expected arguments" in low
                or "invalid action parameters:" in low):
            return "false"

    if failure_type in ("api_error", "env_error", "external_error"):
        return None
    if failure_type in ("parse_error", "action_error", "model_error"):
        return "false"
    if task_result == "failure":
        return None
    return None


def extract_actual_actions(result_json: Optional[Path]) -> dict:
    """Official extract_actual_actions。"""
    info = {"actual_action_count": None, "actual_action_text": None}
    if not result_json or not Path(result_json).exists():
        return info
    try:
        data = json.loads(Path(result_json).read_text(encoding="utf-8"))
    except Exception:
        return info
    actions = []
    name = Path(result_json).name
    if isinstance(data, dict) and name == "log.json":
        messages = data.get("messages", [])
        if isinstance(messages, list):
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                action_executed = msg.get("action_executed")
                if action_executed:
                    actions.append(str(action_executed).strip())
    if not actions and isinstance(data, dict) and name.startswith("episode_"):
        trajectory = data.get("trajectory", [])
        if isinstance(trajectory, list):
            for step in trajectory:
                if not isinstance(step, dict):
                    continue
                action_string = step.get("action_string")
                if action_string:
                    actions.append(str(action_string).strip())
        if not actions:
            action_sequence = data.get("action_sequence")
            if isinstance(action_sequence, str) and action_sequence.strip():
                actions = [x.strip() for x in action_sequence.split("->")
                           if x.strip()]
    if actions:
        info["actual_action_count"] = len(actions)
        info["actual_action_text"] = " | ".join(actions)
    return info


def extract_token_stats_from_result_json(
        result_json: Optional[Path]) -> dict:
    """Official extract_token_stats_from_result_json。"""
    empty = {"prompt_tokens": None, "completion_tokens": None,
             "total_tokens": None, "api_calls": None}
    if not result_json or not Path(result_json).exists():
        return empty
    try:
        data = json.loads(Path(result_json).read_text(encoding="utf-8"))
    except Exception:
        return empty
    metadata = data.get("metadata", {}) if isinstance(data, dict) else {}
    token_usage = metadata.get("token_usage", {})
    if not isinstance(token_usage, dict):
        return empty

    def _int(key):
        value = token_usage.get(key)
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    prompt = _int("prompt_tokens")
    completion = _int("completion_tokens")
    total = _int("total_tokens")
    calls = _int("api_calls")
    if total is None and (prompt is not None or completion is not None):
        total = (prompt or 0) + (completion or 0)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "api_calls": calls,
    }


def read_steps_and_max(result_json: Optional[Path]) -> dict:
    """从官方结果文件读取 total_steps / max_steps（log.json metadata 或 episode）。"""
    out = {"total_steps": None, "max_steps": None}
    if not result_json or not Path(result_json).exists():
        return out
    try:
        data = json.loads(Path(result_json).read_text(encoding="utf-8"))
    except Exception:
        return out
    if Path(result_json).name == "log.json":
        meta = data.get("metadata", {}) if isinstance(data, dict) else {}
        out["total_steps"] = meta.get("total_steps")
        out["max_steps"] = meta.get("max_steps")
    elif isinstance(data, dict):
        out["total_steps"] = data.get("step_count")
        out["max_steps"] = data.get("max_steps")
    return out
