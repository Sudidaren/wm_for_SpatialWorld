#!/usr/bin/env python3
"""被挡住之后，"转身"和"横挪"哪个更管用？用真实跑批的数据回答。

索引语义（容易错，先写清楚）：
  steps.jsonl 第 k 条 = {动作 a_k, 注入提示 h_k}，而 h_k 描述的是**上一步
  a_{k-1} 的结果**（"上一个动作：…"、"移动提示：上一步…"）。所以
  * h_k 含"移动提示"  ⇒  a_{k-1} 撞墙了；
  * 模型看到 h_k 之后做出的**就是 a_k**（不是 a_{k+1}）。

对每个"撞墙事件"记录模型选的脱困动作，再看它有没有真的脱困：
脱困 = 之后 2 步内出现一次**画面确实变了**的移动动作（mse > 1）。

    python3 tools/eval_escape_actions.py --runs wm_gemini31pro_fail210,wm_gemini31pro_simple40
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics
import sys
from collections import Counter, defaultdict

sys.path.insert(0, '/home/sudidaren/SpatialWorld')
from mllm_base_agent.agent.self_observation import frame_mse  # noqa: E402

RUNS = '/home/sudidaren/spatialworld_eval/runs'
MSE_THRESHOLD = 1.0
MOVE = {'MoveAhead', 'MoveBack', 'MoveLeft', 'MoveRight'}
ROT = {'RotateLeft', 'RotateRight'}
STRAFE = {'MoveLeft', 'MoveRight'}
LOOK = {'LookUp', 'LookDown'}


def nm(a: str) -> str:
    return (a or '').split('(')[0].strip()


def kind(a: str) -> str:
    n = nm(a)
    if n in ROT:
        return '转身'
    if n in STRAFE:
        return '横挪'
    if n in {'MoveAhead', 'MoveBack'}:
        return '前后走'
    if n in LOOK:
        return '抬头低头'
    if not n:
        return '无动作'
    return '交互/其他'


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', default='wm_gemini31pro_fail210,'
                                     'wm_gemini31pro_simple40')
    ap.add_argument('--window', type=int, default=2)
    args = ap.parse_args()

    outcome = defaultdict(Counter)   # 脱困动作 -> {脱困, 未脱困}
    esc_mse = defaultdict(list)
    examples = []

    for run in [r for r in args.runs.split(',') if r]:
        res = os.path.join(RUNS, run, 'results.csv')
        if not os.path.exists(res):
            continue
        for row in csv.DictReader(open(res, encoding='utf-8-sig')):
            tid = row['Task ID']
            for d in glob.glob(os.path.join(RUNS, run, 'ai2thor', tid,
                                            'worker_*', tid)):
                try:
                    imgs = json.load(open(os.path.join(d, 'log.json')))['images']
                    traj = json.load(open(glob.glob(
                        os.path.join(d, 'episode_*.json'))[0]))['trajectory']
                    steps = [json.loads(l) for l in open(
                        os.path.join(d, 'steps.jsonl'), encoding='utf-8')
                        if l.strip()]
                except Exception:
                    continue
                n = min(len(imgs), len(traj), len(steps))
                for k in range(1, n):
                    if '移动提示：' not in (steps[k].get('mem_hint') or ''):
                        continue
                    esc = nm(steps[k].get('action_string'))
                    if not esc:
                        continue
                    # 脱困动作本身改变了画面吗
                    m = frame_mse(imgs[k], imgs[k + 1]) if k + 1 < len(imgs) else None
                    if m is not None:
                        esc_mse[kind(esc)].append(m)
                    # 之后 window 步内有没有"真的动起来"
                    freed = False
                    for j in range(k, min(k + args.window + 1, n - 1)):
                        if nm(steps[j].get('action_string')) in MOVE:
                            mm = frame_mse(imgs[j], imgs[j + 1])
                            if mm is not None and mm > MSE_THRESHOLD:
                                freed = True
                                break
                    outcome[kind(esc)]['脱困' if freed else '仍被困'] += 1
                    if len(examples) < 4 and kind(esc) == '横挪':
                        examples.append((tid, k, esc, freed,
                                         steps[k].get('mem_hint', '')[-80:]))

    print('=== 撞墙之后，模型选了什么 / 有没有脱困 ===')
    print(f'{"脱困动作":<10}{"次数":>6}{"脱困":>7}{"仍被困":>8}{"脱困率":>9}')
    for k, c in sorted(outcome.items(), key=lambda kv: -sum(kv[1].values())):
        n = sum(c.values())
        print(f'{k:<10}{n:>6}{c["脱困"]:>7}{c["仍被困"]:>8}'
              f'{c["脱困"]/n:>9.0%}')
    print()
    print('=== 脱困动作本身有没有让画面变化（mse 中位）===')
    for k, v in sorted(esc_mse.items(), key=lambda kv: -len(kv[1])):
        if v:
            print(f'  {k:<10} n={len(v):>4}  中位 {statistics.median(v):8.1f}'
                  f'   >1 的比例 {sum(1 for x in v if x > 1)/len(v):.0%}')
    if examples:
        print('\n横挪的例子：')
        for tid, k, esc, freed, tail in examples:
            print(f'  [{tid} step{k}] {esc}  → {"脱困" if freed else "仍被困"}')


if __name__ == '__main__':
    main()
