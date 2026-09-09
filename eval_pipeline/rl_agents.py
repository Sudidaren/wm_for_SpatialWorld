# -*- coding: utf-8 -*-
"""DreamerV3 / DIAMOND / 任意策略的统一接入层。

设计目标：不假设你的策略内部长什么样，只要求暴露一个方法：

    class Policy:
        def __init__(self, checkpoint: str = ""): ...
        def load(self): ...                      # 可选，加载权重
        def act(self, observation, step: int, task: dict) -> dict:
            # observation: env.reset/step 返回的对象（.image_path/.metadata）
            # 返回 unified action dict，例如：
            #   {"action_type": "navigation", "action_name": "MoveAhead",
            #    "granularity": "Medium"}
            #   {"action_type": "interaction", "action_name": "PickupObject",
            #    "object_type": "Egg"}
            #   {"action_type": "task_completion", "action_name": "DONE"}
            ...

接入方式：
  1. 在 eval_config.py 里：
       PROFILE = "rl"
       RL_POLICY_IMPORT = "my_pkg.policy:Policy"   # 模块:类名
       RL_CHECKPOINT = "/path/to/ckpt"
  2. RL_POLICY_IMPORT 留空 = 内置 GoldenActionsPolicy（复现任务 golden 动作，
     用于自检整条链路，不依赖任何 RL 代码）。

本模块可被 supervisor 以子进程调用（venv 内运行，含官方 env 依赖）：
    python rl_agents.py --env ai2thor --task-json ... --config ...
                       --outdir ... --policy-import ... --checkpoint ...

评测口径与 LLM 路线一致：同一个官方 env wrapper、同一个终态 verifier
（mllm_base_agent.agent.runner.perform_final_evaluation）、同一套
log.json / episode_*.json 结果格式，因此 supervisor 的 CSV/状态判定通用。
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())


class Policy:
    """所有 RL/脚本策略的基类。接入真实模型时继承或直接鸭子类型。"""

    def __init__(self, checkpoint: str = ""):
        self.checkpoint = checkpoint

    def load(self):
        """可选：加载权重。默认无操作。"""

    def act(self, observation, step: int, task: dict) -> dict:
        raise NotImplementedError(
            "Policy.act(observation, step, task) 必须返回 unified action dict"
        )


def load_policy(import_path: str, checkpoint: str = "") -> Policy:
    """按 ``module.path:ClassName`` 导入策略实例。"""
    if not import_path:
        raise ValueError("RL_POLICY_IMPORT 为空")
    module_name, _, class_name = import_path.partition(":")
    module = importlib.import_module(module_name)
    factory = getattr(module, class_name)
    try:
        policy = factory(checkpoint=checkpoint)
    except TypeError:
        policy = factory()
    if not hasattr(policy, "act"):
        raise TypeError(f"{import_path} 没有 act(observation, step, task) 方法")
    if hasattr(policy, "load"):
        policy.load()
    return policy


class GoldenActionsPolicy(Policy):
    """内置自检策略：按顺序重放任务 golden 动作。"""

    def __init__(self, actions=None, env_type: str = "ai2thor",
                 checkpoint: str = ""):
        super().__init__(checkpoint)
        self._actions = list(actions or [])
        self._env_type = env_type

    def act(self, observation, step: int, task: dict) -> dict:
        if not self._actions:
            return {"action_type": "task_completion", "action_name": "FAIL"}
        text = self._actions.pop(0)
        return _parse_action(text, self._env_type)


def _parse_action(text: str, env_type: str) -> dict:
    if env_type == "virtualhome":
        return _parse_vh_native(text)
    from actions.parser import parse_action_string

    return parse_action_string(text, env_type=env_type)


_VH_NAV = {"WalkForward": "navigation", "TurnLeft": "navigation",
           "TurnRight": "navigation", "WalkBackward": "navigation",
           "LookUp": "navigation", "LookDown": "navigation"}
_VH_INTERACT = {"Grab", "Open", "Close", "SwitchOn", "SwitchOff",
                "PutBack", "PutIn", "Drink", "Sit", "Touch", "LookAt"}


def _parse_vh_native(text: str) -> dict:
    """VirtualHome golden 动作是原生小写/驼峰字符串（如 Grab(toothbrush)），
    官方 actions.parser 不解析它们，这里按 wrapper 的 action dict 接口转换。"""
    text = str(text).strip()
    upper = text.upper()
    if upper == "DONE":
        return {"action_type": "task_completion", "action_name": "DONE"}
    if upper == "FAIL":
        return {"action_type": "task_completion", "action_name": "FAIL"}
    if "(" not in text or not text.endswith(")"):
        name = text
        args = []
    else:
        name = text[: text.index("(")].strip()
        params = text[text.index("(") + 1: -1]
        args = [p.strip() for p in params.split(",") if p.strip()]
    if name in _VH_NAV:
        out = {"action_type": "navigation", "action_name": name}
        if name == "WalkForward" and args:
            out["granularity"] = args[0]
        return out
    if name in _VH_INTERACT:
        out = {"action_type": "interaction", "action_name": name}
        if args:
            out["object_type"] = args[0].replace(" ", "").lower()
        if len(args) > 1:
            out["object2_type"] = args[1].replace(" ", "").lower()
        return out
    raise ValueError(f"未知 VirtualHome golden 动作: {text}")


# --------------------------------------------------------------------------
# 官方 env 构建（与 scripts/<env>/work/run_task.py 同款参数）
# --------------------------------------------------------------------------
def _load_config(config_path: Path, task_config: dict) -> dict:
    import yaml

    with open(config_path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    data["task"] = task_config
    return data


def _task_config(raw: dict, env_type: str, headless: bool) -> dict:
    from actions.max_steps import resolve_max_steps_from_task

    cfg = dict(raw)
    cfg["max_steps"] = resolve_max_steps_from_task(raw, 50)
    if env_type == "procthor":
        cfg.setdefault("scene_index", 0)
    if env_type == "virtualhome":
        cfg["scene"] = int(cfg.get("scene", 0))
    return cfg


def create_env(env_type: str, full_config: dict, task: dict,
               outdir: Path, headless: bool):
    env_cfg = full_config.get("env", {}) or {}
    outdir = str(outdir)
    if env_type == "ai2thor":
        from envs.ai2thor import AI2ThorEnvWrapper

        return AI2ThorEnvWrapper(
            scene=task.get("scene", "FloorPlan1"),
            grid_size=env_cfg.get("grid_size", 0.25),
            render_depth_image=bool(env_cfg.get("render_depth", False)),
            render_instance_segmentation=bool(
                env_cfg.get("render_instance_segmentation", False)),
            width=int(env_cfg.get("width", 800)),
            height=int(env_cfg.get("height", 600)),
            output_dir=outdir,
            config=full_config,
        )
    if env_type == "procthor":
        from envs.procthor_wrapper import ProcTHOREnvWrapper

        return ProcTHOREnvWrapper(
            scene_index=int(task.get("scene_index", 0)),
            output_dir=outdir,
            config=full_config,
            headless=headless,
        )
    if env_type == "virtualhome":
        from envs.virtualhome import VirtualHomeEnvWrapper

        try:
            scene = int(task.get("scene", env_cfg.get("scene", 0)))
        except (TypeError, ValueError):
            scene = int(env_cfg.get("scene", 0))
        return VirtualHomeEnvWrapper(
            scene=scene,
            executable_path=env_cfg.get("executable_path"),
            port=env_cfg.get("port", "8080"),
            url=env_cfg.get("url", "127.0.0.1"),
            x_display=env_cfg.get("x_display"),
            width=int(env_cfg.get("width", 640)),
            height=int(env_cfg.get("height", 480)),
            output_dir=outdir,
            config=full_config,
            num_agents=int(env_cfg.get("num_agents", 1)),
            char_resource=env_cfg.get("char_resource", "Chars/Female1"),
        )
    raise ValueError(f"不支持的 RL 环境: {env_type}")


def _load_init_actions(task_json: Path, env_type: str) -> list:
    init_file = task_json.parent / "init.json"
    if not init_file.exists():
        return []
    try:
        data = json.loads(init_file.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(data, dict):
        actions = data.get("actions") or []
    elif isinstance(data, list):
        actions = data
    else:
        actions = []
    return [_parse_action(str(a), env_type) for a in actions
            if str(a).strip().upper() != "DONE"]


def _final_eval(env_type: str, env, task_config: dict, observation):
    from mllm_base_agent.agent.runner import perform_final_evaluation

    try:
        ok, score = perform_final_evaluation(
            state={"config": {"env": {"type": env_type}}},
            env=env, task_config=task_config, observation=observation)
        return bool(ok), float(score)
    except Exception as exc:
        print(f"⚠️ final evaluation failed: {exc}")
        return False, 0.0


def _action_text(action: dict) -> str:
    name = action.get("action_name", "")
    args = [str(action.get(k)) for k in
            ("object_type", "object2_type", "message", "granularity")
            if action.get(k) not in (None, "")]
    if args:
        return f"{name}({','.join(args)})"
    return str(name)


def _write_results(out_root: Path, task_id: str, task_config: dict,
                   trajectory: list, images: list, success: Optional[bool],
                   fail_reason: Optional[str], steps: int):
    out_root.mkdir(parents=True, exist_ok=True)
    max_steps = task_config.get("max_steps", 50)
    metadata = {
        "task_description": task_config.get("instruction", ""),
        "task_result": ("success" if success else "failure"),
        "fail_reason": fail_reason,
        "failure_type": None,
        "total_steps": steps,
        "max_steps": max_steps,
        "token_usage": {},
    }
    log_path = out_root / "log.json"
    messages = []
    for i, step in enumerate(trajectory):
        messages.append({"role": "user", "content": f"Step {i}",
                         "step": i,
                         "image_path": step.get("image_path")})
        messages.append({
            "role": "assistant", "content": step.get("action_string"),
            "step": i,
            "action_executed": step.get("action_string"),
            "reward": step.get("reward"),
            "error_message": step.get("error_message"),
        })
    log_path.write_text(json.dumps(
        {"metadata": metadata, "messages": messages,
         "images": images},
        ensure_ascii=False, indent=2), encoding="utf-8")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    episode = {
        "task_id": task_id,
        "success": bool(success),
        "step_count": steps,
        "max_steps": max_steps,
        "fail_reason": fail_reason,
        "failure_type": None,
        "metadata": metadata,
        "trajectory": trajectory,
    }
    (out_root / f"episode_{ts}.json").write_text(
        json.dumps(episode, ensure_ascii=False, indent=2), encoding="utf-8")


def run_rl_episode(env_type: str, task_json: Path, config_path: Path,
                   outdir: Path, policy_import: str,
                   checkpoint: str = "", headless: bool = False) -> dict:
    raw = json.loads(task_json.read_text(encoding="utf-8"))
    task_id = str(raw.get("task_id") or task_json.parent.name)
    task_config = _task_config(raw, env_type, headless)
    full_config = _load_config(config_path, task_config)

    if policy_import:
        policy = load_policy(policy_import, checkpoint)
    else:
        golden = raw.get("golden_actions") or {}
        actions = golden.get("actions") if isinstance(golden, dict) else []
        policy = GoldenActionsPolicy(
            actions=[str(a) for a in (actions or [])],
            env_type=env_type, checkpoint=checkpoint)

    out_root = outdir / task_id
    env = create_env(env_type, full_config, task_config, out_root, headless)
    trajectory = []
    images = []
    success: Optional[bool] = None
    fail_reason = None
    try:
        instruction = (task_config.get("instruction")
                       or task_config.get("target_description") or "")
        scene = (task_config.get("scene")
                 if env_type in ("ai2thor", "virtualhome") else None)
        observation = env.reset(instruction, scene=scene)
        for action in _load_init_actions(task_json, env_type):
            observation, _err = env.step_with_action_dict(action)
        steps = 0
        max_steps = int(task_config.get("max_steps", 50))
        for step in range(max_steps):
            action = policy.act(observation, step, task_config)
            steps = step + 1
            print(f"[rl:{task_id}] step {step}: {_action_text(action)}",
                  flush=True)
            observation, error_message = env.step_with_action_dict(action)
            image_path = getattr(observation, "image_path", None)
            if image_path:
                images.append(str(image_path))
            trajectory.append({
                "step": step,
                "action_string": _action_text(action),
                "action": action,
                "image_path": str(image_path) if image_path else None,
                "reward": None,
                "error_message": error_message,
            })
            if str(action.get("action_name", "")).upper() == "DONE":
                success, _score = _final_eval(
                    env_type, env, task_config, observation)
                fail_reason = None if success else \
                    "DONE but success conditions not met"
                break
            if str(action.get("action_name", "")).upper() == "FAIL":
                success = False
                fail_reason = "policy declared FAIL"
                break
        else:
            success = False
            fail_reason = f"Reached maximum step limit ({max_steps} steps)"
        if success is None:
            success = False
            fail_reason = fail_reason or "unknown termination"
        _write_results(out_root, task_id, task_config, trajectory, images,
                       success, fail_reason, steps)
        print(f"[rl:{task_id}] done success={success} steps={steps} "
              f"reason={fail_reason}", flush=True)
        return {"task_id": task_id, "success": success,
                "steps": steps, "fail_reason": fail_reason}
    finally:
        try:
            env.close()
        except Exception:
            pass


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--env", required=True,
                    choices=["ai2thor", "procthor", "virtualhome"])
    ap.add_argument("--task-json", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--policy-import", default="")
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args(argv)
    result = run_rl_episode(
        env_type=args.env,
        task_json=Path(args.task_json),
        config_path=Path(args.config),
        outdir=Path(args.outdir),
        policy_import=args.policy_import,
        checkpoint=args.checkpoint,
        headless=args.headless,
    )
    return 0 if result.get("success") else 0


if __name__ == "__main__":
    sys.exit(main())
