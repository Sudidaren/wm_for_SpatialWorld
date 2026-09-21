#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""重放任务里**存好的** golden_actions，验证"任务文件 → 官方动作解析 → 位姿判定"这条链路。

与 verify_tasks.py 的区别：verify 是"现场生成 golden 并回放"，
这个脚本是"读文件里已存的 golden 再回放"，用来验收最终产物。

用法：
  TMPDIR=/tmp DISPLAY=:0 python replay_golden.py --tasks tvr_ithor_FloorPlan228_easy_t06_s00
  TMPDIR=/tmp DISPLAY=:0 python replay_golden.py --tasks stage1 --limit 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_tvr_agent as R      # 复用 env 构造 / teleport / 判定配置
import pose_eval
from actions.parser import parse_action_string

TASKS = HERE / "tasks"
PROCTHOR_JSONL = Path("/home/sudidaren/.prior/datasets/allenai/procthor-10k/"
                      "439193522244720b86d8c81cde2e51e3a4d150cf/train.jsonl.gz")


def available_gb() -> float:
    """MemAvailable（GB）。Unity 进程很吃内存，用它做红线和监控。"""
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024 / 1024
    return 99.0


def unity_count() -> int:
    """当前残留的 AI2-THOR Unity 进程数（正常应为 0 或 1）。"""
    n = 0
    for pid in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            if "thor-" in pid.read_bytes().decode("utf8", "ignore"):
                n += 1
        except Exception:
            pass
    return n


def replay(task_id: str, width: int, height: int, check_ref: bool = False) -> dict:
    task = json.loads((TASKS / task_id / "task.json").read_text())
    tvr = task["tvr"]
    acts = (task.get("golden_actions") or {}).get("actions") or []
    if not acts:
        return {"task_id": task_id, "error": "该任务没有存 golden_actions"}

    cfg = R.load_config("configs/procthor/config.yaml" if tvr["dataset"] == "procthor"
                        else "configs/ai2thor/config.yaml")
    outdir = HERE / "runs" / "replay_golden" / task_id
    outdir.mkdir(parents=True, exist_ok=True)
    env = R.build_env(task, cfg, outdir, width, height)
    steps, fails = [], []
    try:
        env.reset(task["instruction"], scene=task.get("scene"))
        R.teleport(env, tvr["start"])
        session_goal = None
        if check_ref:
            # 会话内目标图：与 agent 评测时喂给模型的 goal image 完全同路径
            from PIL import Image
            R.teleport(env, tvr["target"])
            session_goal = Image.fromarray(env.controller.last_event.frame).convert("RGB")
            R.teleport(env, tvr["start"])
        for a in acts:
            if str(a).upper() in ("DONE", "FAIL"):
                steps.append({"action": a, "skipped": True})
                break
            parsed = parse_action_string(a, env_type="ai2thor")
            try:
                env.step_with_action_dict(parsed)
                steps.append({"action": a, "ok": True})
            except Exception as e:                       # noqa: BLE001
                steps.append({"action": a, "ok": False, "error": str(e)[:120]})
                fails.append(a)
        md = env.controller.last_event.metadata
        st = pose_eval.state_from_metadata(md)
        st["_stopped"] = True            # golden 末项即 Done/Stop
        cond = task["success_conditions"][0]
        # 判定走 pose_eval.judge，两种口径都支持：
        #   objects_visible（现在的正式判据）需要实例掩码
        #   agent_pose（官方口径，留存的老任务/参考指标）
        ok = pose_eval.judge(cond, md, event=env.controller.last_event, stopped=True)
        # 顺带把两档可见性口径都记下来，便于逐步收严
        fracs = pose_eval.type_fracs(env.controller.last_event, md)
        vis = next((c for c in (task.get("success_conditions") or [])
                    if isinstance(c, dict) and c.get("type") == "objects_visible"), None)
        vis_presence = bool(vis) and pose_eval.check_objects_visible({**vis, "mode": "presence"}, fracs)
        vis_area = bool(vis) and pose_eval.check_objects_visible({**vis, "mode": "area1pct"}, fracs)
        rec = {
            "task_id": task_id, "n_actions": len(acts),
            "failed_actions": fails,
            "final_pos_err": round(pose_eval.position_error(st, tvr["target"]), 4),
            "final_rot_err": round(pose_eval.rotation_error(st, tvr["target"]), 1),
            "final_hor_err": round(pose_eval.horizon_error(st, tvr["target"]), 1),
            "success": bool(ok),
            "vis_presence": vis_presence,
            "vis_area1pct": vis_area,
            "steps": steps,
        }
        if check_ref:
            # (1) 会话内一致性：reset 回初始态、回到目标位姿再渲一帧，
            #     与"开场渲的目标图"比较 —— 这才是 agent 实际面对的一致性要求。
            # (2) 与落盘参考图比对：跨会话（不同进程）比较，ProcTHOR 房子可能
            #     存在渲染不确定性，仅作参考，不作为验收标准。
            try:
                from PIL import Image, ImageChops, ImageStat
                # 用 wrapper 自己的 reset 重建场景（与评测开场完全同一条加载路径）
                env.reset(task["instruction"], scene=task.get("scene"))
                R.teleport(env, tvr["target"])
                cur = Image.fromarray(env.controller.last_event.frame).convert("RGB")
                ref = Image.open(TASKS / task_id / "target_view.png").convert("RGB")
                if cur.size != ref.size:
                    ref = ref.resize(cur.size)
                stored = sum(ImageStat.Stat(ImageChops.difference(cur, ref)).mean) / 3
                rec["ref_stored_diff"] = round(stored, 2)
                if session_goal is not None:
                    same = sum(ImageStat.Stat(
                        ImageChops.difference(cur, session_goal)).mean) / 3
                    rec["ref_session_diff"] = round(same, 2)
                    rec["ref_session_match"] = same < 2.0
            except Exception as e:                    # noqa: BLE001
                rec["ref_check_error"] = str(e)[:120]
        return rec
    finally:
        try:
            env.close()
        except Exception:
            pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="stage1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--width", type=int, default=640, help="官方 eval 观测尺寸 640x360")
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--check-ref", action="store_true",
                    help="同时校验 target_view.png 是否等于目标位姿的渲染帧（建议配 --width 1280 --height 720）")
    ap.add_argument("--results-file", default=str(HERE / "tasks" / "replay_results.jsonl"),
                    help="每条结果逐行追加，便于断点续跑与合并")
    ap.add_argument("--resume", action="store_true",
                    help="跳过 results-file 里已完成的任务")
    ap.add_argument("--shard", default="0/1",
                    help="分片，如 1/2（第 2 片，共 2 片），用于并行加速")
    ap.add_argument("--mem-floor-gb", type=float, default=3.0,
                    help="可用内存低于该值立即停下（可 --resume 续跑）")
    args = ap.parse_args()

    ids = R.pick_tasks(args.tasks, args.limit)
    si, sn = (int(x) for x in args.shard.split("/"))
    if sn > 1:
        ids = [t for k, t in enumerate(ids) if k % sn == si]
    if args.resume and Path(args.results_file).exists():
        done = {json.loads(l)["task_id"] for l in Path(args.results_file).read_text().splitlines() if l.strip()}
        before = len(ids)
        ids = [t for t in ids if t not in done]
        print(f"续跑：跳过已完成 {before - len(ids)} 条，本片剩 {len(ids)} 条", flush=True)
    results = []
    for i, tid in enumerate(ids, 1):
        avail = available_gb()
        if avail < args.mem_floor_gb:
            print(f"⚠️ 可用内存 {avail:.1f} GB < {args.mem_floor_gb} GB，立即暂停；"
                  f"内存恢复后加 --resume 继续", flush=True)
            break
        print(f"[{i}/{len(ids)}] {tid} | mem {avail:.1f}GB | unity {unity_count()}", flush=True)
        r = replay(tid, args.width, args.height, check_ref=args.check_ref)
        results.append(r)
        with Path(args.results_file).open("a") as f:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        if args.verbose:
            for s in r.get("steps", []):
                print("   ", s)
        print("   ->", json.dumps({k: v for k, v in r.items() if k != "steps"},
                                  ensure_ascii=False)[:260], flush=True)
    n_ok = sum(1 for r in results if r.get("success"))
    # 汇总：读全部已落盘结果（含其它分片/上一轮）
    all_recs = []
    if Path(args.results_file).exists():
        for line in Path(args.results_file).read_text().splitlines():
            if line.strip():
                try:
                    all_recs.append(json.loads(line))
                except Exception:
                    pass
    merged = {r["task_id"]: r for r in all_recs}
    (HERE / "tasks" / "replay_report.json").write_text(json.dumps(
        {"n": len(merged), "n_success": sum(1 for r in merged.values() if r.get("success")),
         "tasks": sorted(merged.values(), key=lambda r: r["task_id"])},
        indent=1, ensure_ascii=False))
    print(f"\n本片 {n_ok}/{len(results)} 命中判定 | 累计 {len(merged)} 条 → tasks/replay_report.json")


if __name__ == "__main__":
    main()
