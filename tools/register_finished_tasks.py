#!/usr/bin/env python3
"""把"结果已经落盘、但 supervisor 没来得及登记"的任务补写进 state.json / results.csv。

场景：supervisor 被 SIGSTOP 冻结后，已经跑完的 worker 仍会把 episode JSON 写到盘上，
但只有 supervisor 活着才会把它登记成一行。冻结/中断后手动补登记，
这样下次续跑时 `load_state` 会认为这些任务"已判定"，不会重复跑。

默认 dry-run；加 --write 才真正写入（写入前自动备份 .bak_before_register_<时间>）。

用法：python3 register_finished_tasks.py <run_dir> [--write]
"""

from __future__ import annotations

import glob
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, '/home/sudidaren/spatialworld_eval')
import supervisor as sup  # noqa: E402

TASKS_ROOT = '/home/sudidaren/SpatialWorld/data/ai2thor/tasks'


def duration_sec(task_dir: str) -> float | None:
    """用 reset/step 的 PNG 时间戳估任务耗时。"""
    stamps = []
    for p in glob.glob(os.path.join(task_dir, '*.png')):
        m = re.search(r'(\d{8}_\d{6})\.png$', p)
        if not m:
            continue
        try:
            stamps.append(time.mktime(time.strptime(m.group(1), '%Y%m%d_%H%M%S')))
        except ValueError:
            continue
    return round(max(stamps) - min(stamps), 1) if len(stamps) >= 2 else None


def main() -> None:
    run_dir = Path(sys.argv[1])
    write = '--write' in sys.argv
    rows = sup.load_state(run_dir)
    have = {r.get('task_id') for r in rows}

    # 每个任务可能有多份 episode（重试过）：只取最后一次尝试的
    latest: dict[str, str] = {}
    for ep in sorted(glob.glob(os.path.join(str(run_dir), 'ai2thor', '*',
                                            'worker_*', '*', 'episode_*.json')),
                     key=os.path.getmtime):
        tid = ep.split(os.sep)[-2]          # .../ai2thor/<tid>/worker_N/<tid>/ep.json
        latest[tid] = ep                    # 按 mtime 递增，最后写入的即最新

    added = []
    for tid, ep in sorted(latest.items()):
        if tid in have:
            continue
        # 只登记本批 plan.json 里列出的任务（别的批次残留的目录不算）
        plan_path = run_dir / 'plan.json'
        if plan_path.exists():
            wanted = set(json.loads(plan_path.read_text(encoding='utf-8'))
                         .get('task_ids') or [])
            if wanted and tid not in wanted:
                print(f'  - 跳过 {tid}（不在本批任务清单里）')
                continue
        with open(ep, encoding='utf-8') as handle:
            epi = json.load(handle)
        # 被中途打断留下的残局（1~2 步 + 环境异常）不算结果
        reason = str(epi.get('fail_reason') or '')
        if (epi.get('step_count') or 0) <= 2 and 'Environment exception' in reason:
            print(f'  - 跳过 {tid}（中途被打断的残局：{reason[:38]}）')
            continue
        traj = epi.get('trajectory') or []
        acts = [e.get('action_string') for e in traj if e.get('action_string')]
        usage = (epi.get('metadata') or {}).get('token_usage') or {}
        if not usage:
            usage = {
                'prompt_tokens': sum((e.get('llm_token_usage') or {}).get('prompt_tokens', 0) for e in traj),
                'completion_tokens': sum((e.get('llm_token_usage') or {}).get('completion_tokens', 0) for e in traj),
                'total_tokens': sum((e.get('llm_token_usage') or {}).get('total_tokens', 0) for e in traj),
                'api_calls': len(traj),
            }
        meta = {}
        tp = os.path.join(TASKS_ROOT, tid, 'task.json')
        if os.path.exists(tp):
            meta = json.load(open(tp, encoding='utf-8'))
        ok = bool(epi.get('success'))
        row = {
            'task_id': tid,
            'env': 'ai2thor',
            'scene': epi.get('scene') or '',
            'category': '',
            'task_type': '',
            'completed': 'true' if ok else 'false',
            'success': ok,
            'status': 'success' if ok else 'failed_model',
            'attempts': 1,
            'duration_sec': duration_sec(os.path.dirname(ep)),
            'golden_steps': (meta.get('golden_actions') or {}).get('steps'),
            'golden_action': ' | '.join((meta.get('golden_actions') or {}).get('actions') or []) or None,
            'instruction': meta.get('instruction') or '',
            'actual_actions': ' | '.join(acts) or None,
            'actual_actions_count': len(acts) or None,
            'actual_steps': epi.get('step_count'),
            'max_steps': epi.get('max_steps'),
            'fail_reason': epi.get('fail_reason'),
            'failure_type': epi.get('failure_type'),
            'prompt_tokens': usage.get('prompt_tokens'),
            'completion_tokens': usage.get('completion_tokens'),
            'total_tokens': usage.get('total_tokens'),
            'api_calls': usage.get('api_calls'),
            'attempts_detail': [],
        }
        sup._normalize_row(row) if hasattr(sup, '_normalize_row') else None
        added.append(row)
        print(f'  + {tid}  status={row["status"]}  steps={row["actual_steps"]}  '
              f'fail={str(row["fail_reason"])[:44]}  tokens={row["total_tokens"]}')

    print(f'\n待补登记 {len(added)} 条（现有 {len(rows)} 条）')
    if not added:
        return
    if not write:
        print('dry-run：加 --write 才真正写入')
        return
    stamp = time.strftime('%Y%m%d_%H%M%S')
    for name in ('state.json', 'results.csv'):
        src = os.path.join(str(run_dir), name)
        if os.path.exists(src):
            shutil.copy2(src, f'{src}.bak_before_register_{stamp}')
    rows.extend(added)
    sup.write_state(run_dir, rows)
    sup.write_results_csv(run_dir, rows)
    print(f'已写入（备份后缀 .bak_before_register_{stamp}）；'
          f'现在共 {len(rows)} 条')


if __name__ == '__main__':
    main()
