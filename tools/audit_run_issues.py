#!/usr/bin/env python3
"""把一批跑完的结果从头到尾复检一遍：到底还剩什么问题。

只看盘上已有的东西，不发请求、不碰模拟器：
  * 每步注入的提示（steps.jsonl 的 mem_hint）里，各通道各发了多少次
  * 每条提示发出去之后，模型下一步到底照做没有（提示的"有效率"）
  * 模拟器报错的分布（episode 的 trajectory.error_message）
  * 失败模式：步数耗尽 / 提前 DONE / 模型自认做不到 / 网络
  * 动作预算：导航 vs 交互、原地重复

    python3 tools/audit_run_issues.py --run wm_gemini31pro_fail210
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import statistics
from collections import Counter, defaultdict

RUNS = '/home/sudidaren/spatialworld_eval/runs'
TASKS = '/home/sudidaren/SpatialWorld/data/ai2thor/tasks'

NAV = {'MoveAhead', 'MoveBack', 'MoveLeft', 'MoveRight', 'RotateLeft',
       'RotateRight', 'LookUp', 'LookDown'}
ROT = {'RotateLeft', 'RotateRight'}
INTERACT = {'PickupObject', 'PutObject', 'OpenObject', 'CloseObject',
            'ToggleObjectOn', 'ToggleObjectOff', 'SliceObject', 'DropHandObject',
            'CleanObject', 'FillObjectWithLiquid', 'DirtyObject', 'UseUpObject',
            'BreakObject', 'CookObject', 'EmptyLiquidFromObject'}
DIST = re.compile(r'约\s*([0-9.]+)\s*m')
ERRD = re.compile(r'distance:\s*([0-9.]+)m')

nm = lambda a: (a or '').split('(')[0].strip()


def classify(m: str) -> str | None:
    if 'is not in view' in m:
        return '① 目标不在视野（没对准/太远）'
    if 'is blocking Agent' in m or 'Path is blocked' in m:
        return '② 被家具挡住（导航）'
    if 'Hand already has an object' in m:
        return '③ 手上已有物还想拿'
    if 'No valid positions' in m or "isn't holding" in m or 'not holding' in m:
        return '④ 放置失败'
    if 'does not exist in the scene' in m:
        return '⑤ 名字幻觉（加工态）'
    if 'beyond 60 degrees' in m:
        return '⑥ 低头超引擎上限'
    if 'already' in m:
        return '⑦ 状态已满足还重复操作'
    return None


def load(run: str, tid: str):
    sp = sorted(glob.glob(f'{RUNS}/{run}/ai2thor/{tid}/worker_*/*/steps.jsonl'),
                key=os.path.getmtime)
    ep = sorted(glob.glob(f'{RUNS}/{run}/ai2thor/{tid}/worker_*/*/episode_*.json'),
                key=os.path.getmtime)
    steps = [json.loads(l) for l in open(sp[-1], encoding='utf-8')
             if l.strip()] if sp else []
    traj = json.load(open(ep[-1])).get('trajectory', []) if ep else []
    return steps, traj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='wm_gemini31pro_fail210')
    ap.add_argument('--only-failed', action='store_true')
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(
        open(f'{RUNS}/{args.run}/results.csv', encoding='utf-8-sig'))
        if r['Status'] in ('success', 'failed_model')]
    ok = [r for r in rows if r['Status'] == 'success']
    bad = [r for r in rows if r['Status'] != 'success']
    print(f'=== {args.run} ===')
    print(f'已判定 {len(rows)} 条：成功 {len(ok)} ({len(ok)/len(rows):.0%})，'
          f'失败 {len(bad)}')

    chan = Counter()            # 各提示通道发了多少次
    follow = defaultdict(Counter)   # 提示发出后下一步动作类别
    errs, err_dist = Counter(), []
    err_task_ids = defaultdict(set)
    modes = Counter()
    nav_ratio, int_ratio, repeats = [], [], Counter()
    seen_any, min_dists = 0, []
    hold_tasks = 0

    todo = bad if args.only_failed else rows
    for r in todo:
        tid = r['Task ID']
        steps, traj = load(args.run, tid)
        if not steps:
            continue
        acts = [nm(s.get('action_string')) for s in steps]
        n = len(acts)
        nav_ratio.append(sum(1 for a in acts if a in NAV) / n)
        int_ratio.append(sum(1 for a in acts if a in INTERACT) / n)
        for i in range(1, n):
            if acts[i] == acts[i - 1]:
                repeats[acts[i]] += 1

        if r['Status'] != 'success':
            reason = r['failure_reason'] or ''
            modes['步数耗尽' if 'maximum step' in reason
                  else '提前 DONE' if 'DONE but success' in reason
                  else '模型自认做不到' if 'cannot be complete' in reason
                  else reason[:34]] += 1

        tj_p = f'{TASKS}/{tid}/task.json'
        targets = [t.lower() for t in json.load(open(tj_p))['target_object_types']] \
            if os.path.exists(tj_p) else []
        ever_seen = False
        has_hold = False
        for i, s in enumerate(steps):
            h = s.get('mem_hint') or ''
            if '手持：' in h and '空手' not in h:
                has_hold = True
            if any(t in h.lower() for t in targets) and ('视野内' in h or '任务相关记忆' in h):
                ever_seen = True
                for m in DIST.findall(h):
                    min_dists.append(float(m))
            tags = []
            if '移动提示：' in h:
                tags.append('移动提示(被挡)')
            if '重复提示' in h:
                tags.append('重复提示')
            if '建议先转向它再靠近' in h:
                tags.append('建议靠近')
            if '要先走到' in h:
                tags.append('距离提示')
            for t in tags:
                chan[t] += 1
                if i + 1 < n:
                    nxt = acts[i + 1]
                    follow[t]['转身' if nxt in ROT else
                               '交互' if nxt in INTERACT else
                               '平移' if nxt in NAV else '其他'] += 1
        if ever_seen:
            seen_any += 1
        if has_hold:
            hold_tasks += 1
        for e in traj:
            c = classify(e.get('error_message') or '')
            if not c:
                continue
            errs[c] += 1
            err_task_ids[c].add(tid)
            mm = ERRD.search(e.get('error_message') or '')
            if mm and '不在视野' in c:
                err_dist.append(float(mm.group(1)))

    print(f'\n--- 失败模式（{len(bad)} 条失败）---')
    for k, v in modes.most_common():
        print(f'  {v:>4}  {k}')

    print(f'\n--- 模拟器报错分布 ---')
    for k, v in errs.most_common():
        print(f'  {v:>4} 次  涉及 {len(err_task_ids[k]):>3} 个任务   {k}')
    if err_dist:
        err_dist.sort()
        print(f'  "不在视野"报错时的距离: n={len(err_dist)} 中位 '
              f'{statistics.median(err_dist):.2f}m  '
              f'（≤1.5m 占 {sum(1 for d in err_dist if d<=1.5)/len(err_dist):.0%}）')

    print(f'\n--- 提示通道：发了多少 / 模型下一步照做没有 ---')
    for t, c in chan.most_common():
        tot = sum(follow[t].values())
        top = '、'.join(f'{k} {v/tot:.0%}' for k, v in follow[t].most_common(3))
        print(f'  {t:<14} 发了 {c:>4} 次   下一步：{top}')

    print(f'\n--- 动作与证据 ---')
    print(f'  导航占比中位 {statistics.median(nav_ratio):.0%}，'
          f'交互占比中位 {statistics.median(int_ratio):.0%}')
    print(f'  连续重复最多的动作: ' +
          '、'.join(f'{k}×{v}' for k, v in repeats.most_common(5)))
    print(f'  目标至少被看见过一次的任务: {seen_any}/{len(todo)}')
    print(f'  报过"手持"的任务: {hold_tasks}/{len(todo)}')
    if min_dists:
        print(f'  目标最近距离中位 {statistics.median(min_dists):.2f}m')


if __name__ == '__main__':
    main()
