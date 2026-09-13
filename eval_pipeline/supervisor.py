# -*- coding: utf-8 -*-
"""SpatialWorld 三类家庭任务统一评测调度器。

用法：
    python supervisor.py                      # 全部按 eval_config.py 执行
    python supervisor.py --dry-run            # 只打印任务计划，不启动模拟器
    python supervisor.py --scenes ai2thor,procthor
    python supervisor.py --profile llm
    python supervisor.py --task ai2thor03073 --force
    python supervisor.py --workers 2

设计原则：
    - 每个任务调用官方 run_task（10+2g 步数预算、终态校验、log.json /
      episode_*.json / 逐帧图），官方逻辑不改；
    - 状态判定（Completed true/false/null）与 official run_benchmark 的
      decide_csv_status_from_result / determine_failure_reason 对齐
      （见 official_compat.py）；
    - api/env 级故障自动重试；每任务进度写 state.json，可断点续跑；
    - 可选 golden 复核：AI2-THOR 走官方 evaluate_actions_ai2thor.py；
      ProcTHOR / VirtualHome 走仓库内对应 evaluate_actions_*.py。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import select
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval_config as cfg
import official_compat as oc
from config_builder import build_task_config, dry_model_sanity
from env_spec import EnvSpec, TaskInfo, select_tasks, env_specs

VERIFY_LOCK = threading.Lock()
EXTERNAL_TYPES = {"api_error", "env_error", "external_error", "external"}


# --------------------------------------------------------------------------
# 官方 runner 调用
# --------------------------------------------------------------------------
def run_official_task(spec: EnvSpec, task: TaskInfo, config_path: Path,
                      run_output_dir: Path,
                      timeout_sec: int) -> subprocess.CompletedProcess:
    """调用官方 run_task。

    注意官方 run_task 只在 --output-dir 的 basename 以 ``worker_`` 开头时
    （ai2thor / virtualhome）或单任务时（procthor）使用指定目录，否则会自建
    outputs/run_<ts>/...。这里统一用 ``worker_<n>`` 前缀确保输出受控。
    """
    if cfg.PROFILE == "rl":
        rl_script = Path(__file__).resolve().parent / "rl_agents.py"
        cmd = [
            str(spec.venv_python), "-u", str(rl_script),
            "--env", spec.name,
            "--task-json", str(task.task_json_path),
            "--config", str(config_path),
            "--outdir", str(run_output_dir),
            "--policy-import", cfg.RL_POLICY_IMPORT,
            "--checkpoint", cfg.RL_CHECKPOINT,
        ]
        if cfg.HEADLESS:
            cmd.append("--headless")
    else:
        cmd = [
            str(spec.venv_python),
            "-u",
            "-m",
            f"scripts.{spec.name}.work.run_task",
            "--config", str(config_path),
            "--tasks", task.task_id,
            "--output-dir", str(run_output_dir),
        ]
    env = dict(os.environ)
    env["VLM_API_TIMEOUT"] = str(cfg.LLM_API_TIMEOUT_SEC)
    return _run_stream(cmd, cwd=cfg.SPATIALWORLD_ROOT,
                       total_timeout=timeout_sec,
                       stall_timeout=cfg.STALL_TIMEOUT_SEC,
                       env=env,
                       log_path=run_output_dir / "run_stream.log")


def _run_stream(cmd: list, cwd: str, total_timeout: float,
                stall_timeout: float, env: Optional[dict] = None,
                log_path: Optional[Path] = None):
    """流式执行子进程并加看门狗：超时或长时间无输出时杀整个进程组。

    解决"模型/API 中途卡死、进程不退出"：卡死按外部故障处理（结果 null，
    可自动重试），不再干等数小时。
    """
    proc = subprocess.Popen(
        cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        start_new_session=True, env=env,
    )
    out_chunks, err_chunks = [], []
    log_handle = None
    if log_path:
        try:
            log_handle = open(log_path, "a", encoding="utf-8")
        except Exception:
            log_handle = None
    last_output = time.time()
    start = time.time()
    stalled = timed_out = False
    streams = {proc.stdout.fileno(): proc.stdout,
               proc.stderr.fileno(): proc.stderr}
    while proc.poll() is None or streams:
        elapsed = time.time() - start
        if elapsed >= total_timeout:
            timed_out = True
            break
        if time.time() - last_output >= stall_timeout:
            stalled = True
            break
        wait = max(0.1, min(total_timeout - elapsed,
                            stall_timeout - (time.time() - last_output),
                            5.0))
        try:
            ready, _, _ = select.select(list(streams), [], [], wait)
        except (OSError, ValueError):
            break
        for fd in ready:
            stream = streams[fd]
            chunk = os.read(fd, 65536)
            if chunk:
                last_output = time.time()
                decoded = chunk.decode("utf-8", errors="replace")
                if stream is proc.stdout:
                    out_chunks.append(decoded)
                else:
                    err_chunks.append(decoded)
                if log_handle:
                    try:
                        log_handle.write(decoded)
                        log_handle.flush()
                    except Exception:
                        pass
            else:
                streams.pop(fd, None)
    if proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), 9)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.kill()
            except Exception:
                pass
    try:
        proc.wait(timeout=10)
    except Exception:
        pass
    stdout = "".join(out_chunks)
    stderr = "".join(err_chunks)
    if log_handle:
        try:
            log_handle.close()
        except Exception:
            pass
    return _ProcResult(returncode=proc.returncode, stdout=stdout,
                       stderr=stderr, stalled=stalled, timed_out=timed_out)


class _ProcResult:
    def __init__(self, returncode, stdout, stderr, stalled, timed_out):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.stalled = stalled
        self.timed_out = timed_out


# --------------------------------------------------------------------------
# golden 复核（官方现有可执行脚本；官方 run_benchmark 引用的
# evaluate_action_sequence.py 在当前仓库缺失，故按仓库实际脚本实现）
# --------------------------------------------------------------------------
def _latest_json(root: Path, pattern: str) -> Optional[Path]:
    hits = sorted(root.glob(pattern))
    return hits[-1] if hits else None


def _json_ok(path: Optional[Path]) -> bool:
    if not path or not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return bool(data.get("success")) and data.get("evaluation_score") == 1.0


def _verify_ai2thor(spec: EnvSpec, task: TaskInfo, config_path: Path) -> bool:
    cmd = [
        str(spec.venv_python), "scripts/evaluate_actions_ai2thor.py",
        "--task", task.task_id, "--config", str(config_path),
    ]
    if cfg.HEADLESS:
        cmd.append("--headless")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=1800, cwd=cfg.SPATIALWORLD_ROOT)
    except Exception:
        return False
    if proc.returncode != 0:
        return False
    out_root = Path(cfg.SPATIALWORLD_ROOT) / "outputs"
    eval_dirs = sorted(out_root.glob("action_eval_*"))
    if not eval_dirs:
        return False
    return _json_ok(_latest_json(
        eval_dirs[-1], "action_sequence_result_*.json"))


def _verify_procthor(spec: EnvSpec, task: TaskInfo, config_path: Path) -> bool:
    verify_root = (Path(cfg.SPATIALWORLD_ROOT) / "outputs"
                   / f"verify_{task.task_id}_{int(time.time())}")
    verify_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(spec.venv_python), "scripts/evaluate_actions_procthor.py",
        "--task", task.task_id, "--output-dir", str(verify_root),
    ]
    if cfg.HEADLESS:
        cmd.append("--headless")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=1800, cwd=cfg.SPATIALWORLD_ROOT)
    except Exception:
        return False
    if proc.returncode != 0:
        return False
    eval_dirs = sorted(verify_root.glob("action_eval_*"))
    if not eval_dirs:
        return False
    return _json_ok(_latest_json(
        eval_dirs[-1], "action_sequence_result_*.json"))


def _verify_virtualhome(spec: EnvSpec, task: TaskInfo,
                        config_path: Path) -> bool:
    verify_root = (Path(cfg.SPATIALWORLD_ROOT) / "outputs"
                   / f"verify_{task.task_id}_{int(time.time())}")
    verify_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(spec.venv_python), "scripts/evaluate_actions_virtualhome.py",
        "--task", task.task_id, "--config", str(config_path),
        "--output-dir", str(verify_root),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=1800, cwd=cfg.SPATIALWORLD_ROOT)
    except Exception:
        return False
    if proc.returncode != 0:
        return False
    return _json_ok(_latest_json(
        verify_root, "**/vh_action_result_*.json"))


def verify_golden_replay(spec: EnvSpec, task: TaskInfo,
                         config_path: Path) -> bool:
    if spec.name == "ai2thor":
        return _verify_ai2thor(spec, task, config_path)
    if spec.name == "procthor":
        return _verify_procthor(spec, task, config_path)
    if spec.name == "virtualhome":
        return _verify_virtualhome(spec, task, config_path)
    return False


# --------------------------------------------------------------------------
# 单任务执行（重试 / 状态判定 / 结果落盘）
# --------------------------------------------------------------------------
def _write_no_result_diag(task_dir: Path, task_id: str, detail: dict) -> None:
    try:
        (task_dir / "run_error.txt").write_text(
            json.dumps({
                "task_id": task_id,
                "time": datetime.now().isoformat(),
                "detail": detail,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8")
    except Exception:
        pass


def run_single_task(spec: EnvSpec, task: TaskInfo, run_dir: Path,
                    config_root: Path) -> dict:
    base_dir = run_dir / spec.name / task.task_id
    base_dir.mkdir(parents=True, exist_ok=True)
    task_cfg = build_task_config(spec, task, config_root)

    outcome = {
        "task_id": task.task_id,
        "env": spec.name,
        "scene": task.scene,
        "category": task.category,
        "task_type": task.task_type,
        "completed": None,
        "success": None,
        "status": "pending",
        "attempts": 0,
        "duration_sec": None,
        "golden_steps": task.golden_steps,
        "golden_action": None,
        "instruction": task.instruction,
        "actual_actions": None,
        "actual_actions_count": None,
        "actual_steps": None,
        "max_steps": None,
        "fail_reason": None,
        "failure_type": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "api_calls": None,
        "attempts_detail": [],
    }
    meta = oc.load_task_metadata(task.task_json_path)
    outcome["golden_steps"] = meta["golden_action_count"]
    outcome["golden_action"] = meta["golden_action_text"]
    if not outcome["instruction"]:
        outcome["instruction"] = meta["instruction"]

    max_tries = 1 + int(cfg.MAX_EXTERNAL_RETRIES)
    t0 = time.time()
    final = None  # (status, detail)
    for attempt in range(max_tries):
        # basename 必须以 worker_ 开头，官方 run_task 才会使用该输出目录
        attempt_dir = base_dir / f"worker_{attempt}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        detail = {"attempt": attempt, "ok": False}
        try:
            proc = run_official_task(
                spec, task, task_cfg, attempt_dir, cfg.TASK_TIMEOUT_SEC)
            detail["returncode"] = proc.returncode
            detail["stdout_tail"] = (proc.stdout or "")[-2000:]
            detail["stderr_tail"] = (proc.stderr or "")[-2000:]
            if getattr(proc, "stalled", False) or getattr(proc, "timed_out", False):
                reason = ("stalled: no output for "
                          f"{cfg.STALL_TIMEOUT_SEC}s"
                          if getattr(proc, "stalled", False)
                          else f"timeout > {cfg.TASK_TIMEOUT_SEC}s")
                detail["error"] = reason
                outcome["attempts_detail"].append(detail)
                outcome["attempts"] += 1
                _write_no_result_diag(base_dir, task.task_id, detail)
                final = ("external", detail)
                continue
        except subprocess.TimeoutExpired:
            detail["error"] = f"timeout > {cfg.TASK_TIMEOUT_SEC}s"
            outcome["attempts_detail"].append(detail)
            outcome["attempts"] += 1
            _write_no_result_diag(base_dir, task.task_id, detail)
            final = ("external", detail)
            continue
        except Exception as exc:
            detail["error"] = str(exc)
            outcome["attempts_detail"].append(detail)
            outcome["attempts"] += 1
            _write_no_result_diag(base_dir, task.task_id, detail)
            final = ("external", detail)
            continue

        outcome["attempts"] += 1
        # 官方 run_task 的输出层级随环境不同：
        #   AI2-THOR / VirtualHome -> <attempt>/<task_id>/...
        #   ProcTHOR 单任务        -> <attempt>/...（结果直接写在 attempt 下）
        result_path = (oc.find_result_json(attempt_dir / task.task_id)
                       or oc.find_result_json(attempt_dir))
        detail["result_path"] = str(result_path) if result_path else None
        outcome["attempts_detail"].append(detail)

        if not result_path:
            outcome.update({
                "status": "failed_external",
                "failure_type": "external_error",
                "fail_reason": "no result json produced",
            })
            _write_no_result_diag(base_dir, task.task_id, detail)
            final = ("external", detail)
            continue

        status = oc.read_result_status_info(result_path)
        steps = oc.read_steps_and_max(result_path)
        tokens = oc.extract_token_stats_from_result_json(result_path)
        actions = oc.extract_actual_actions(result_path)
        outcome.update({
            "actual_steps": steps["total_steps"],
            "max_steps": steps["max_steps"],
            "prompt_tokens": tokens["prompt_tokens"],
            "completion_tokens": tokens["completion_tokens"],
            "total_tokens": tokens["total_tokens"],
            "api_calls": tokens["api_calls"],
            "actual_actions": actions["actual_action_text"],
            "actual_actions_count": actions["actual_action_count"],
            "failure_type": status["failure_type"],
            "fail_reason": status["fail_reason"],
        })

        if status["task_result"] == "success":
            replay_ok = True
            if cfg.VERIFY_GOLDEN_REPLAY:
                with VERIFY_LOCK:
                    replay_ok = verify_golden_replay(spec, task, task_cfg)
            # 与官方一致：agent 成功时 CSV Completed=true（复核失败仍记 true，
            # 仅追加诊断；官方 run_benchmark 同款行为）。
            outcome.update({
                "completed": "true",
                "success": True,
                "status": "success",
                "fail_reason": ("golden replay failed"
                                if not replay_ok else None),
                "failure_type": ("verify_error"
                                 if not replay_ok else None),
            })
            final = ("success", detail)
            break

        # 失败判定（与官方 decide_csv_status_from_result 完全一致）
        fallback = oc.determine_failure_reason(result_path)
        csv_status = oc.decide_csv_status_from_result(status, fallback)
        if csv_status == "false":
            outcome.update({
                "completed": "false",
                "success": False,
                "status": "failed_model",
            })
            final = ("model", detail)
            break

        # None => null（api/env/外部故障，或未知原因）；外部故障才自动重试
        outcome.update({
            "status": "failed_external",
            "failure_type": fallback,
        })
        final = ("external", detail)

    outcome["duration_sec"] = round(time.time() - t0, 1)
    if final is None or final[0] == "external":
        outcome["completed"] = None
        outcome["success"] = None
    return outcome


# --------------------------------------------------------------------------
# CSV / state
# --------------------------------------------------------------------------
_CSV_HEADER = [
    "Task ID", "Completed",
    # 官方 run_benchmark 的扩展列（名称一致）
    "golden_action", "instruction", "token_total", "failure_reason",
    "actual_actions", "golden_actions_count", "actual_actions_count",
    # 本项目额外明细列
    "Environment", "Scene", "Category", "Task Type",
    "Actual Steps", "Max Steps", "Status", "Success", "Duration Sec",
    "Attempts", "Failure Type", "Prompt Tokens", "Completion Tokens",
    "API Calls",
]


def _csv_row(outcome: dict) -> list:
    completed = outcome.get("completed")
    return [
        outcome.get("task_id"),
        completed if completed in ("true", "false") else "null",
        outcome.get("golden_action"),
        outcome.get("instruction"),
        outcome.get("total_tokens"),
        outcome.get("fail_reason"),
        outcome.get("actual_actions"),
        outcome.get("golden_steps"),
        outcome.get("actual_actions_count"),
        outcome.get("env"),
        outcome.get("scene"),
        outcome.get("category"),
        outcome.get("task_type"),
        outcome.get("actual_steps"),
        outcome.get("max_steps"),
        outcome.get("status"),
        outcome.get("success"),
        outcome.get("duration_sec"),
        outcome.get("attempts"),
        outcome.get("failure_type"),
        outcome.get("prompt_tokens"),
        outcome.get("completion_tokens"),
        outcome.get("api_calls"),
    ]


def write_results_csv(run_dir: Path, rows: list[dict]) -> Path:
    path = run_dir / "results.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(_CSV_HEADER)
        for row in sorted(rows, key=lambda r: (r.get("env", ""),
                                               r.get("task_id", ""))):
            writer.writerow(_csv_row(row))
    return path


def write_state(run_dir: Path, rows: list[dict]) -> None:
    (run_dir / "state.json").write_text(
        json.dumps({
            "updated_at": datetime.now().isoformat(),
            "tasks": rows,
        }, ensure_ascii=False, indent=2), encoding="utf-8")


def load_state(run_dir: Path) -> list[dict]:
    path = run_dir / "state.json"
    if not path.exists():
        return []
    try:
        rows = json.loads(path.read_text(encoding="utf-8")).get("tasks", [])
    except Exception:
        return []
    for row in rows:
        _normalize_row(row)
    return rows


_ROW_DEFAULTS = {
    "env": "", "scene": "", "category": "", "task_type": "",
    "instruction": "", "golden_action": None, "actual_actions": None,
}


def _normalize_row(row: dict) -> None:
    for key, value in _ROW_DEFAULTS.items():
        row.setdefault(key, value)
    for key in (
        "completed", "success", "status", "duration_sec", "golden_steps",
        "actual_actions_count", "actual_steps", "max_steps", "fail_reason",
        "failure_type", "prompt_tokens", "completion_tokens", "total_tokens",
        "api_calls", "attempts", "attempts_detail",
    ):
        row.setdefault(key, None)


def _already_decided(task_id: str, rows: list[dict]) -> bool:
    if not cfg.RESUME_SKIP_DECIDED:
        return False
    for r in rows:
        if (r.get("task_id") == task_id
                and r.get("completed") in ("true", "false")):
            return True
    return False


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenes", type=str, default=None,
                    help="逗号分隔：ai2thor,procthor,virtualhome")
    ap.add_argument("--profile", choices=["llm", "wingman_llm", "rl"],
                    default=None)
    ap.add_argument("--model", type=str, default=None,
                    help="LLM 预设名，如 gpt-5 / gemini-3.1-pro / qwen3.5")
    ap.add_argument("--task", action="append", default=None,
                    help="只跑指定任务 id（可多次）")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--headless", dest="headless", action="store_true",
                    default=None)
    ap.add_argument("--no-headless", dest="headless", action="store_false")
    ap.add_argument("--force", action="store_true",
                    help="忽略已完成任务，全部重跑")
    ap.add_argument("--no-verify", action="store_true",
                    help="跳过 golden 复核")
    ap.add_argument("--dry-run", action="store_true",
                    help="只打印任务计划")
    return ap.parse_args(argv)


def _preflight() -> None:
    dry_model_sanity()
    if cfg.PROFILE in ("llm", "wingman_llm"):
        model = cfg.resolve_llm_model()
        if (str(model.get("provider", "")).lower() == "openai"
                and not model.get("api_key")
                and not os.environ.get("OPENAI_API_KEY")):
            raise SystemExit(
                "缺少 OPENAI_API_KEY 或 LLM_OVERRIDES['api_key']，"
                "官方 run_task 会在启动后直接退出。请先配置模型凭据。")


def main(argv=None):
    args = parse_args(argv)
    if args.scenes:
        cfg.SCENES = [s.strip() for s in args.scenes.split(",") if s.strip()]
    if args.profile:
        cfg.PROFILE = args.profile
    if args.model:
        cfg.LLM_PRESET = args.model
    if args.task:
        cfg.TASK_IDS = list(args.task)
    if args.workers:
        cfg.WORKERS = args.workers
    if args.headless is not None:
        cfg.HEADLESS = args.headless
    if args.force:
        cfg.RESUME_SKIP_DECIDED = False
    if args.no_verify:
        cfg.VERIFY_GOLDEN_REPLAY = False

    allowed = {"ai2thor", "procthor", "virtualhome"}
    bad = [s for s in cfg.SCENES if s not in allowed]
    if bad:
        raise SystemExit(f"不支持的场景: {bad}（可选 {sorted(allowed)}）")
    if cfg.PROFILE == "wingman_llm" and "virtualhome" in cfg.SCENES:
        raise SystemExit(
            "VirtualHome 没有米制深度，WingmanWM 的深度感知头无法运行。"
            "VH 请用 --profile llm（或在 eval_config.SCENES 中把 virtualhome "
            "移出 wingman_llm 批次）。")

    _preflight()
    specs = env_specs()
    tasks = select_tasks(cfg.SCENES, cfg.TASK_IDS or None, cfg.TASK_ID_FILTER)
    if not tasks:
        raise SystemExit("没有匹配的任务（检查 SCENES / TASK_IDS）")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = cfg.RUN_NAME or f"{stamp}_{cfg.PROFILE}_{cfg.LLM_PRESET}"
    run_dir = Path(cfg.OUTPUT_ROOT) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    config_root = run_dir / "task_configs"
    config_root.mkdir(parents=True, exist_ok=True)

    plan = {
        "created_at": datetime.now().isoformat(),
        "scenes": cfg.SCENES,
        "profile": cfg.PROFILE,
        "llm_preset": cfg.LLM_PRESET,
        "headless": cfg.HEADLESS,
        "workers": cfg.WORKERS,
        "retries": cfg.MAX_EXTERNAL_RETRIES,
        "verify_golden": cfg.VERIFY_GOLDEN_REPLAY,
        "n_tasks": len(tasks),
        "task_ids": [t.task_id for t in tasks],
    }
    (run_dir / "plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"run_dir   : {run_dir}")
    print(f"scenes    : {cfg.SCENES}")
    print(f"profile   : {cfg.PROFILE}")
    print(f"llm       : {cfg.LLM_PRESET}")
    print(f"tasks     : {len(tasks)}")
    for t in tasks[:8]:
        print(f"  - [{t.env}] {t.task_id} {t.scene} "
              f"({t.category}/{t.task_type})")
    if len(tasks) > 8:
        print(f"  ... 共 {len(tasks)} 个")
    if args.dry_run:
        print("dry-run：不启动模拟器。")
        return

    prev = load_state(run_dir)
    todo = [t for t in tasks if not _already_decided(t.task_id, prev)]
    print(f"待执行（已跳过已完成）: {len(todo)}")
    if not todo:
        print("没有待执行任务，直接汇总。")

    lock = threading.Lock()
    rows = [dict(r) for r in prev]

    def worker(task: TaskInfo) -> dict:
        out = run_single_task(specs[task.env], task, run_dir, config_root)
        with lock:
            idx = next((i for i, r in enumerate(rows)
                        if r.get("task_id") == out["task_id"]), None)
            if idx is None:
                rows.append(out)
            else:
                rows[idx] = out
            write_results_csv(run_dir, rows)
            write_state(run_dir, rows)
        return out

    done = 0
    if cfg.WORKERS <= 1:
        for task in todo:
            out = worker(task)
            done += 1
            print(f"[{done}/{len(todo)}] {out['task_id']} "
                  f"completed={out['completed']} "
                  f"status={out['status']} steps={out['actual_steps']}")
    else:
        with ThreadPoolExecutor(max_workers=cfg.WORKERS) as pool:
            futs = {pool.submit(worker, t): t for t in todo}
            for fut in as_completed(futs):
                task = futs[fut]
                try:
                    out = fut.result()
                    print(f"✔ {out['task_id']} completed={out['completed']}")
                except Exception as exc:
                    print(f"✘ {task.task_id} 异常: {exc}")
                done += 1

    from summarize import write_summary
    write_summary(run_dir, rows, plan)
    print(f"\n完成。结果: {run_dir / 'results.csv'} | "
          f"统计: {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
