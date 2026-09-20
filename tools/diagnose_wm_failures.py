#!/usr/bin/env python3
"""把"失败到底卡在哪"拆成可统计的几类，用各任务自己的 steps.jsonl。

对每条失败任务回答四个问题：
  1) 目标物体**有没有被感知头看见过**？（WM 有没有素材帮忙）
  2) 看见过的话，模型**离它最近到过多少米**？卡在什么距离上？
  3) WM 说"你被挡住了"之后，模型**下一步听没听**（转个身 vs 继续硬走）？
  4) 动作预算花在哪：导航 vs 交互。

用法：
    python3 tools/diagnose_wm_failures.py --run wm_gemini31pro_fail210
    python3 tools/diagnose_wm_failures.py --run wm_gemini31pro_simple40 --only-failed
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import statistics
from collections import Counter

RUNS = '/home/sudidaren/spatialworld_eval/runs'
TASKS = '/home/sudidaren/SpatialWorld/data/ai2thor/tasks'

NAV = {'MoveAhead', 'MoveBack', 'MoveLeft', 'MoveRight', 'RotateLeft',
       'RotateRight', 'LookUp', 'LookDown', 'Teleport'}
ROTATE = {'RotateLeft', 'RotateRight'}
INTERACT = {'PickupObject', 'PutObject', 'OpenObject', 'CloseObject',
            'ToggleObjectOn', 'ToggleObjectOff', 'SliceObject', 'DropHandObject',
            'CleanObject', 'FillObjectWithLiquid', 'DirtyObject', 'UseUpObject',
            'BreakObject', 'CookObject', 'EmptyLiquidFromObject'}

DIST_RE = re.compile(r'约\s*([0-9.]+)\s*m')


def action_name(action: str) -> str:
    return action.split('(')[0].strip()


def load_steps(run: str, task: str) -> list[dict]:
    """取该任务最后一次尝试的 steps.jsonl。"""
    paths = sorted(glob.glob(os.path.join(RUNS, run, 'ai2thor', task,
                                          'worker_*', '*', 'steps.jsonl')),
                   key=os.path.getmtime)
    if not paths:
        return []
    with open(paths[-1], encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--only-failed', action='store_true')
    args = ap.parse_args()

    with open(os.path.join(RUNS, args.run, 'results.csv'),
              encoding='utf-8-sig') as handle:
        rows = list(csv.DictReader(handle))

    stats = Counter()
    min_dists, budgets, nav_after_block = [], [], Counter()
    examples = {'never_seen': [], 'seen_never_reached': [],
                'blocked_ignored': [], 'reached_no_interact': []}

    for row in rows:
        task, status = row['Task ID'], row['Status']
        if status not in ('success', 'failed_model'):
            continue
        if args.only_failed and status == 'success':
            continue
        task_json = os.path.join(TASKS, task, 'task.json')
        if not os.path.exists(task_json):
            continue
        with open(task_json, encoding='utf-8') as handle:
            meta = json.load(handle)
        targets = [t.lower() for t in meta.get('target_object_types', [])]
        golden = [action_name(a) for a in meta['golden_actions']['actions']]
        golden_int = [a for a in golden if a in INTERACT]
        golden_nav = sum(1 for a in golden if a in NAV)

        steps = load_steps(args.run, task)
        if not steps:
            stats['无 steps.jsonl'] += 1
            continue
        stats['样本'] += 1
        if status == 'success':
            stats['成功'] += 1
        else:
            stats['失败'] += 1

        acts = [action_name(s.get('action_string') or '') for s in steps]
        hints = [s.get('mem_hint') or '' for s in steps]
        n = len(acts)
        nav = sum(1 for a in acts if a in NAV)
        inter = sum(1 for a in acts if a in INTERACT)
        budgets.append((nav / n if n else 0, inter / n if n else 0,
                        golden_nav, len(golden_int)))

        # 目标有没有在提示里出现过（= 感知头看见过）
        seen_steps = [i for i, h in enumerate(hints)
                      if any(t in h.lower() for t in targets)
                      and ('视野内' in h or '任务相关记忆' in h)]
        if seen_steps:
            stats['目标被看见过'] += 1
            dists = []
            for i in seen_steps:
                dists += [float(m) for m in DIST_RE.findall(hints[i])]
            if dists:
                min_dists.append(min(dists))
        else:
            stats['目标从未被看见'] += 1
            if status != 'success' and len(examples['never_seen']) < 5:
                examples['never_seen'].append((task, meta['instruction'][:60]))

        # 被挡住之后模型听不听
        for i, h in enumerate(hints):
            if '移动没有让画面发生变化' not in h or i + 1 >= n:
                continue
            nav_after_block[acts[i + 1] in ROTATE] += 1
            if acts[i + 1] not in ROTATE and status != 'success':
                if len(examples['blocked_ignored']) < 5:
                    examples['blocked_ignored'].append(
                        (task, acts[i], acts[i + 1], hints[i][-90:]))

        # 看见了但一次交互都没做
        if status != 'success' and seen_steps and inter == 0:
            stats['看见了但一次交互都没做'] += 1
            if len(examples['reached_no_interact']) < 5:
                examples['reached_no_interact'].append(
                    (task, meta['instruction'][:55], n))
        # 做了交互但和目标无关
        if status != 'success' and seen_steps and inter > 0:
            stats['看见了也交互了但仍失败'] += 1

        if status != 'success' and seen_steps and not stats.get('_x'):
            pass

    print(f'=== {args.run} ===')
    for k, v in stats.items():
        if not k.startswith('_'):
            print(f'  {k}: {v}')
    if min_dists:
        print(f'  目标最小距离中位: {statistics.median(min_dists):.2f} m '
              f'(n={len(min_dists)}；1.0m 是交互门槛)')
        close = sum(1 for d in min_dists if d <= 1.2)
        print(f'  其中曾经走到 ≤1.2m 的: {close}/{len(min_dists)}')
    if nav_after_block:
        obeyed = nav_after_block.get(True, 0)
        total = sum(nav_after_block.values())
        print(f'  "被挡住"提示后下一步转身的比例: {obeyed}/{total} '
              f'({obeyed / total:.0%})')
    if budgets:
        navs = [b[0] for b in budgets]
        ints = [b[1] for b in budgets]
        print(f'  导航占比中位 {statistics.median(navs):.0%}，'
              f'交互占比中位 {statistics.median(ints):.0%}')
    for name, items in examples.items():
        if items:
            print(f'\n  样例[{name}]:')
            for it in items:
                print('   ', it)


if __name__ == '__main__':
    main()
