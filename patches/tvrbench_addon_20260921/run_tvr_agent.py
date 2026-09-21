#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TVR 任务的 agent 评测入口（addon 侧实现，不改 SpatialWorld 任何文件）。

复用的官方组件：
  config.load_config / core.llm.provider.get_vlm /
  core.agent.graph.create_agent_graph /
  envs.ai2thor.AI2ThorEnvWrapper / envs.procthor_wrapper.ProcTHOREnvWrapper

官方 runner 只在 CARLA/VH 的 image_goal 分支里支持目标图；ai2thor 分支没有。
本 addon 在评测时先把 **目标位姿** 渲染成 PNG，再把它填进
state['goal_image_path'] —— 官方 agent loop 会在第 0 步 prompt 里插入
"Goal Image (your target destination)"，正好是 TVRBench 的题面。

用法：
  TMPDIR=/tmp DISPLAY=:0 python run_tvr_agent.py --tasks stage1 --limit 1 --dry-run
  TMPDIR=/tmp DISPLAY=:0 python run_tvr_agent.py --tasks stage1 --model gpt-5 --outdir runs/smoke1
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SW_ROOT = Path(os.environ.get("SPATIALWORLD_ROOT", "/home/sudidaren/SpatialWorld"))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(SW_ROOT) not in sys.path:
    sys.path.insert(0, str(SW_ROOT))
os.chdir(SW_ROOT)                     # 官方代码按相对路径找 configs/ envs/

from config import load_config        # noqa: E402  官方配置加载
from core.llm.provider import get_vlm  # noqa: E402
from core.agent.graph import create_agent_graph  # noqa: E402

import pose_eval                      # noqa: E402

TASKS = HERE / "tasks"
PROMPT_SUFFIX = (
    "\n\nNotes:\n"
    "- The goal image shows the exact view you must reach.\n"
    "- Rotation granularity is 45 degrees: use RotateLeft(45) / RotateRight(45).\n"
    "- Look up/down in 30-degree steps; move in 0.25 m steps.\n"
    "- When your view matches the goal image, call EndTask(DONE).\n"
)


def load_task(task_id: str) -> tuple[dict, dict]:
    d = TASKS / task_id
    return json.loads((d / "task.json").read_text()), json.loads((d / "init.json").read_text())


def pick_tasks(spec: str, limit: int) -> list[str]:
    if spec == "all":
        ids = sorted(p.name for p in TASKS.iterdir() if (p / "task.json").exists())
    elif spec == "stage1":
        ids = (TASKS / "stage1_tasks.txt").read_text().split()
    elif "," not in spec and spec.endswith(".txt") and (TASKS / spec).exists():
        ids = (TASKS / spec).read_text().split()
    else:
        ids = [s.strip() for s in spec.split(",") if s.strip()]
    return ids[:limit] if limit else ids


def build_env(task: dict, cfg, outdir: Path, width: int, height: int):
    env_cfg = cfg.get("env", {}) or {}
    # 注意：SpatialWorld 的 wrapper 用 config 里的 env.width/height 覆盖构造参数，
    # 所以尺寸必须写进 config 才生效（否则永远按 config 的 800x600 渲染）。
    cfg.update("env.width", int(width))
    cfg.update("env.height", int(height))
    # 判定端要用实例分割掩码（objects_visible 判据）。
    # 这只多渲染一个模态：agent 看到的仍是 RGB 帧，first-person 文本里也只有
    # 动作反馈、不含物体可见性，所以不会往 prompt 里漏任何东西。
    cfg.update("env.render_instance_segmentation", True)
    common = dict(grid_size=env_cfg.get("grid_size", 0.25), width=width, height=height,
                  output_dir=str(outdir), config=cfg.get_all())
    if task["tvr"]["dataset"] == "procthor":
        from envs.procthor_wrapper import ProcTHOREnvWrapper
        os.environ.setdefault("PROCTHOR_DATASET_DIR",
                              "/home/sudidaren/.prior/datasets/allenai/procthor-10k/"
                              "439193522244720b86d8c81cde2e51e3a4d150cf")
        return ProcTHOREnvWrapper(scene_index=int(task["scene_index"]),
                                  dataset_name="procthor-10k", **common)
    from envs.ai2thor import AI2ThorEnvWrapper
    return AI2ThorEnvWrapper(scene=task["scene"], **common)


def teleport(env, pose: dict) -> None:
    ev = env.controller.step(action="Teleport", position=pose["position"],
                             rotation={"x": 0, "y": pose["rotation_y"], "z": 0},
                             horizon=pose["horizon"], standing=True, forceAction=True)
    if not ev.metadata["lastActionSuccess"]:
        raise RuntimeError(f"Teleport 失败: {ev.metadata.get('errorMessage')}")


TVR_CONDITION_TYPES = ("objects_visible", "agent_pose")


class _TvrEvaluator:
    """把 TVR 的判定条件接进官方 evaluate_node（运行时挂载，不改官方文件）。

    两种条件都走 pose_eval.judge：
      * objects_visible —— 正式判据（需要实例掩码，env 已在 build_env 里打开）
      * agent_pose     —— 官方位姿口径，仅作参考指标
    """

    def __init__(self, condition: dict) -> None:
        self.condition = condition

    def evaluate(self, env, metadata) -> float:
        cond = dict(self.condition)
        cond["require_stop"] = False        # evaluate_node 只在模型 DONE 时调用
        event = getattr(env.controller, "last_event", None)
        ok = pose_eval.judge(cond, metadata, event=event, stopped=True)
        return 1.0 if ok else 0.0


def install_pose_evaluator() -> None:
    from mllm_base_agent.agent import runner as _runner
    original = _runner._create_env_evaluator

    def patched(env_type, task_config):
        conds = (task_config or {}).get("success_conditions") or []
        found = [c for c in conds
                 if isinstance(c, dict) and c.get("type") in TVR_CONDITION_TYPES]
        # objects_visible 优先（正式判据）
        found.sort(key=lambda c: 0 if c.get("type") == "objects_visible" else 1)
        return _TvrEvaluator(found[0]) if found else original(env_type, task_config)

    _runner._create_env_evaluator = patched


def render_view(env, pose: dict, path: Path) -> None:
    from PIL import Image
    teleport(env, pose)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(env.controller.last_event.frame).save(path)


def enable_wm(cfg) -> dict:
    """把 WingmanWM 接进来（等价于官方 config_builder 的 _apply_wingman_block）。

    官方 agent graph 的 act_node 是靠 `config.memory_probe.enabled` +
    `config.memory_probe.world_model.enabled` 来决定建不建 MemoryProbe / WorldModel
    的（mllm_base_agent/agent/runner.py:545 起），所以 addon 只要把这一块塞进
    cfg 就行 —— 不用改官方任何文件。

    WM 只吃 agent 实际看到的那张 RGB（检测来自 RF-DETR、深度来自自己训的单目头），
    模拟器的物体可见性只用于**判定**，不会进 prompt。
    """
    root = os.environ.get("LIGHTWM_ROOT", "/home/sudidaren/lightwm_phases")
    ckpt = os.environ.get(
        "PERCEPTION_CKPT",
        f"{root}/checkpoints/small_objects_20260910/dense_depth_best.pt")
    detector = f"{root}/checkpoints/rfdetr_small_228094/checkpoint_best_total.pth"
    # 参数与官方 wm_config_patch._apply_wingman_block 对齐，保证与主表同口径
    for k, v in {
        "memory_probe.enabled": True,
        "memory_probe.world_model.enabled": True,
        "memory_probe.world_model.perception_ckpt": ckpt,
        "memory_probe.world_model.perception_runtime_root": root,
        "memory_probe.world_model.detector_backend": "rfdetr_small_depth",
        "memory_probe.world_model.detector_path": detector,
        "memory_probe.world_model.variant": os.environ.get("WM_VARIANT", "small"),
        "memory_probe.world_model.resolution": 224,
        "memory_probe.world_model.width": 256,
        "memory_probe.world_model.obj_thr": 0.40,
        "memory_probe.world_model.depth_source": os.environ.get(
            "LIGHTWM_DEPTH_SOURCE", "da2"),
        "memory_probe.world_model.pose_from_action_log": True,
        "memory_probe.world_model.pose_initial": "origin",
        # 每步注入"任务相关物体在哪"（WM_TARGET_HINT=1 那一路）
        "memory_probe.target_hint.enabled": True,
        "memory_probe.target_hint.limit": 6,
        "memory_probe.target_hint.names_limit": 5,
        "memory_probe.target_hint.max_dist": 3.0,
        "memory_probe.target_hint.max_sigma": 0.5,
        # 允许模型收尾前 CheckState() 查一次状态汇总（WM_STATE_CHECK=1）
        "memory_probe.object_query.enabled": True,
    }.items():
        cfg.update(k, v)
    # 与官方 run_ablation.sh 的默认注入通道保持一致
    os.environ.setdefault("LIGHTWM_DEPTH_SOURCE", "da2")
    os.environ.setdefault("LIGHTWM_DA2_SCALE", "1.3816")
    os.environ.setdefault("WM_TARGET_HINT", "1")
    os.environ.setdefault("WM_STATE_CHECK", "1")
    return {"perception_ckpt": ckpt, "detector_path": detector}


#: 本任务的"该找什么"——由 WM 自己的检测头看目标图得到，不是模拟器真值
_TVR_SEED_TYPES: set = set()
_PERCEP = None


def _perception():
    """WM 的感知头（RF-DETR + 单目深度），进程内只建一次。"""
    global _PERCEP
    if _PERCEP is None:
        root = os.environ.get("LIGHTWM_ROOT", "/home/sudidaren/lightwm_phases")
        if root not in sys.path:
            sys.path.insert(0, root)
        from phase_b.runtime_factory import build_runtime
        _PERCEP = build_runtime({
            "detector_backend": "rfdetr_small_depth",
            "detector_path": f"{root}/checkpoints/rfdetr_small_228094/checkpoint_best_total.pth",
            "perception_ckpt": os.environ.get(
                "PERCEPTION_CKPT",
                f"{root}/checkpoints/small_objects_20260910/dense_depth_best.pt"),
            "obj_thr": 0.35,
        })
    return _PERCEP


def seed_targets_from_goal_image(goal_png: Path) -> list:
    """让 WM 先"认领"目标图：用它自己的检测头看那张图，把检出的物体当成
    "我要找的东西"。

    为什么这么做：TVR 的指令里没有任何物体名，而 TargetHinter 是靠"指令文本
    里提到的物体"来建立目标集的 —— 不喂它，WM 的目标通道整条是哑的
    （提示永不出现，等于跑了个空 WM）。目标图本来就是**任务输入**（就摆在
    prompt 里），用 WM 自己的检测器读它不引入任何模拟器真值；
    检测器不认识的物体它自然认不出来，这也是它的真实能力边界。
    """
    import numpy as np
    from PIL import Image
    rgb = np.asarray(Image.open(goal_png).convert("RGB"))
    res = _perception()(rgb)
    types = sorted({str(d.get("type")) for d in res["detections"] if d.get("type")})
    _TVR_SEED_TYPES.clear()
    _TVR_SEED_TYPES.update(types)
    return types


def install_tvr_target_seed() -> None:
    """把上面认领到的类型并进 TargetHinter 的任务相关集合。

    官方那一层是按"指令里提到的物体"算 relevant 的，这里在它的结果上并一个
    外部种子 —— 与 install_pose_evaluator 同一种做法：只在 addon 侧做运行时
    挂载，官方文件一行不改。
    """
    from mllm_base_agent.agent import target_priority as tp
    original = tp.TargetHinter.relevant

    def relevant(self, detected):
        out = original(self, detected)
        if _TVR_SEED_TYPES:
            det = [str(t) for t in detected]
            for name in list(_TVR_SEED_TYPES):
                hit = tp.match_names([name], det)
                if hit:
                    self._relevant.add(list(hit.values())[0])
        return set(self._relevant)

    tp.TargetHinter.relevant = relevant


def run_one(task_id: str, args, cfg, vlm, outdir: Path) -> dict:
    task, _init = load_task(task_id)
    tvr = task["tvr"]
    tdir = outdir / task_id
    tdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    env = build_env(task, cfg, tdir, args.width, args.height)
    rec = {"task_id": task_id, "tier": tvr["tier"],
           "scene": task.get("scene") or f"procthor:{task['scene_index']}",
           "success": False, "stopped": False, "steps": None, "error": None}
    try:
        scene = task.get("scene")
        prompt = task["instruction"] + PROMPT_SUFFIX
        env.reset(prompt, scene=scene)
        teleport(env, tvr["start"])

        goal_png = tdir / "target_view.png"
        render_view(env, tvr["target"], goal_png)
        teleport(env, tvr["start"])
        if args.wm:
            # 让 WM 先认领目标图：用它自己的检测头读一遍那张图，
            # 检出的物体就是它接下来要盯的目标（不引入任何模拟器真值）
            try:
                seed = seed_targets_from_goal_image(goal_png)
                print(f"[wm] 从目标图认领到 {len(seed)} 类物体: {seed}", flush=True)
            except Exception as exc:                   # noqa: BLE001
                print(f"[wm] 目标图认领失败（按空处理）: {type(exc).__name__}: {exc}")
                seed = []
            rec["wm_seed_types"] = seed
        obs = env.get_observation_for_agent(0)

        cfg_dict = cfg.get_all()
        cfg_dict["task"] = {
            "name": task_id, "instruction": prompt,
            "success_conditions": task["success_conditions"],
            "success_logic": task["success_logic"],
            "target_object_types": [], "max_steps": int(tvr.get("max_steps", 40)),
        }
        state = {
            "observation": obs, "task_prompt": prompt, "env": env, "vlm": vlm,
            "step_count": 0, "max_steps": int(tvr.get("max_steps", 40)),
            "max_steps_override": args.max_steps,
            "structured_trajectory": [], "conversation_history": [],
            "short_term_history": [], "long_term_summary": "",
            "should_continue": True, "success": False, "fail_reason": None,
            "failure_type": None,
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0,
                            "total_tokens": 0, "api_calls": 0},
            "next_action": None, "think_failed": False,
            "task_done_by_model": False, "task_fail_by_model": False,
            "config": cfg_dict, "run_output_dir": str(tdir),
            "init_action_count": 0, "executor_type": None,
            "input_modality": None, "goal_image_path": str(goal_png),
        }

        if args.dry_run:
            from mllm_base_agent.agent.runner import _build_messages
            msgs = _build_messages(state, image_url="data:image/png;base64,AAAA")
            kinds = [[c.get("type") for c in (m.content if isinstance(m.content, list)
                                              else [])] for m in msgs]
            def _clean(c):
                """图片只留占位符，文本原样落盘（便于查 prompt 是否泄漏目标坐标）"""
                if not isinstance(c, dict):
                    return c
                return {k: ("<IMAGE>" if k == "image_url" else v) for k, v in c.items()}

            (tdir / "messages.json").write_text(json.dumps(
                [{"role": getattr(m, "type", "?"),
                  "content": m.content if isinstance(m.content, str)
                  else [_clean(c) for c in m.content]} for m in msgs],
                indent=1, ensure_ascii=False))
            rec.update({"dry_run": True, "n_messages": len(msgs), "message_image_layout": kinds,
                        "goal_image": str(goal_png), "prompt_head": prompt[:120]})
            print(f"  dry-run: {len(msgs)} 条消息，图片布局 {kinds}")
            print(f"  goal image: {goal_png}")
            return rec

        install_pose_evaluator()
        graph = create_agent_graph()
        final = graph.invoke(state, config={"recursion_limit": 1000})

        md = env.controller.last_event.metadata
        st = pose_eval.state_from_metadata(md)
        stopped = bool(final.get("task_done_by_model"))
        # 正式判据：目标位姿的那些可判别锚点，最终画面里是否也都在
        # （掩码已由 build_env 打开；这里只读，不额外渲染）
        fracs = pose_eval.type_fracs(env.controller.last_event, md)
        vis_cond = next((c for c in (task.get("success_conditions") or [])
                         if isinstance(c, dict) and c.get("type") == "objects_visible"), None)
        if vis_cond:
            vis_presence = pose_eval.check_objects_visible({**vis_cond, "mode": "presence"}, fracs)
            vis_area = pose_eval.check_objects_visible({**vis_cond, "mode": "area1pct"}, fracs)
            need = list(vis_cond.get("anchors") or [])
        else:
            vis_presence = vis_area = False
            need = []
        pose_ok = pose_eval.is_exact_match(st, tvr["target"])
        rec.update({
            "stopped": stopped,
            # success = 正式判据（可见性）。位姿口径作为参考指标另列，不判成败。
            "success": bool(vis_presence),
            "success_pose": bool(pose_ok and stopped),
            "vis_presence": bool(vis_presence),
            "vis_area1pct": bool(vis_area),
            "anchors_need": need,
            "anchors_seen": sorted(t for t, v in fracs.items() if v > 0),
            "steps": final.get("step_count"),
            "pos_err": round(pose_eval.position_error(st, tvr["target"]), 4),
            "rot_err": round(pose_eval.rotation_error(st, tvr["target"]), 1),
            "hor_err": round(pose_eval.horizon_error(st, tvr["target"]), 1),
            "tokens": final.get("token_usage", {}).get("total_tokens"),
            "final_pose": st,
        })
        (tdir / "final_state.json").write_text(json.dumps(
            {"result": rec, "structured_trajectory": final.get("structured_trajectory")},
            indent=1, ensure_ascii=False, default=str))
    except Exception as e:
        import traceback
        rec["error"] = f"{type(e).__name__}: {e}"
        print(traceback.format_exc())
    finally:
        try:
            env.close()
        except Exception:
            pass
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="stage1", help="任务 id 列表 / stage1 / 清单文件")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dataset", default="auto", choices=["auto", "ithor", "procthor"])
    ap.add_argument("--config", default=None, help="官方配置；不给就按数据集自动选")
    ap.add_argument("--model", default="gpt-5")
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--height", type=int, default=600)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--dry-run", action="store_true", help="只搭 prompt 不调模型")
    ap.add_argument("--wm", action="store_true",
                    help="接 WingmanWM（memory_probe + world_model），不接就是纯 VLM 基线")
    args = ap.parse_args()

    ids = pick_tasks(args.tasks, args.limit)
    # 注意：本文件 import 时已 chdir 到 SpatialWorld，相对 outdir 会落到那边；
    # 统一按 addon 目录解析，保证产物不出 addon。
    if args.outdir:
        outdir = Path(args.outdir)
        outdir = outdir if outdir.is_absolute() else (HERE / outdir)
    else:
        outdir = HERE / "runs" / f"tvr_{args.model.replace('/', '_')}_{time.strftime('%Y%m%d_%H%M%S')}"
    outdir = outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    first, _ = load_task(ids[0])
    cfg_path = args.config or ("configs/procthor/config.yaml"
                               if first["tvr"]["dataset"] == "procthor"
                               else "configs/ai2thor/config.yaml")
    cfg = load_config(cfg_path)
    cfg.update("model.vlm.model_name", args.model)
    cfg.update("model.vlm.temperature", args.temperature)
    cfg.update("model.vlm.max_tokens", args.max_tokens)
    if args.base_url:
        cfg.update("model.vlm.base_url", args.base_url)
    if args.api_key:
        cfg.update("model.vlm.api_key", args.api_key)

    # 历史窗口对齐官方主表口径：官方 run_ablation.sh 用 WM_HISTORY_TURNS=29
    # 覆盖 config_builder 的 _apply_context_block。addon 用的是另一份基底配置
    # （configs/ai2thor/config.yaml 里写的是 50），不显式钉住就和主表不可比 ——
    # 上下文长度、成本、模型能看到多少历史全都不一样。
    _hist = int(os.environ.get("TVR_HISTORY_TURNS", "29") or 29)
    cfg.update("context_management.short_term_history_window_size", _hist)
    print(f"[cfg] short_term_history_window_size={_hist}（对齐主表 WM_HISTORY_TURNS）")

    if args.wm:
        info = enable_wm(cfg)
        install_tvr_target_seed()
        print(f"[wm] 已接入: ckpt={info['perception_ckpt']} detector={info['detector_path']}")

    vlm = None
    if not args.dry_run:
        vlm = get_vlm(provider="openai", model_name=args.model,
                      temperature=args.temperature, max_tokens=args.max_tokens,
                      base_url=args.base_url, api_key=args.api_key)

    print(f"任务 {len(ids)} 条 | 配置 {cfg_path} | 模型 {'(dry-run)' if args.dry_run else args.model}")
    print(f"输出目录 {outdir}")
    rows = []
    for i, tid in enumerate(ids, 1):
        print(f"[{i}/{len(ids)}] {tid}", flush=True)
        rec = run_one(tid, args, cfg, vlm, outdir)
        print("   ->", json.dumps(rec, ensure_ascii=False, default=str)[:300], flush=True)
        rows.append(rec)
        with (outdir / "results.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
            w.writeheader()
            w.writerows([{k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                          for k, v in r.items()} for r in rows])

    n_ok = sum(1 for r in rows if r.get("success"))
    (outdir / "summary.json").write_text(json.dumps({
        "n": len(rows), "n_success": n_ok,
        "tsr": (n_ok / len(rows)) if rows else None,
        "model": "(dry-run)" if args.dry_run else args.model,
        "config": cfg_path, "tasks": rows}, indent=1, ensure_ascii=False, default=str))
    print(f"\n完成: TSR {n_ok}/{len(rows)} | 结果 {outdir/'results.csv'} | {outdir/'summary.json'}")


if __name__ == "__main__":
    main()
