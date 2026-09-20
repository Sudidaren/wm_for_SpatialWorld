#!/usr/bin/env python3
"""WM 臂（wm_gemini31pro_simple40）与 Gemini 基线的配对对比。

基线 = runs/replay_legacy311_v1/results.csv（Gemini-3.1-Pro，修复后判定链回放重判）。
产出：
  * paired_vs_baseline.csv（逐题：基线/WM 的状态、步数、失败原因）
  * 控制台表格 + McNemar 精确检验
"""

from __future__ import annotations

import csv
import json
import math
import os

RUNS = '/home/sudidaren/spatialworld_eval/runs'
BASE = os.path.join(RUNS, 'replay_legacy311_v1/results.csv')
WM = os.path.join(RUNS, 'wm_gemini31pro_simple40/results.csv')
SEL = os.path.join(RUNS, '_selection_wm40/selection.json')
OUT = os.path.join(RUNS, 'wm_gemini31pro_simple40/paired_vs_baseline.csv')


def mcnemar_exact(b: int, c: int) -> float:
    """双侧精确 McNemar（二项检验 p=0.5）。"""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main() -> None:
    base = {r['Task ID']: r
            for r in csv.DictReader(open(BASE, encoding='utf-8-sig'))}
    rows = list(csv.DictReader(open(WM, encoding='utf-8-sig')))
    sel = {t['task_id']: t for t in json.load(open(SEL))['tasks']}
    ok = lambda r: r['Status'] == 'success'

    b = sum(1 for r in rows if not ok(base[r['Task ID']]) and ok(r))
    c = sum(1 for r in rows if ok(base[r['Task ID']]) and not ok(r))
    same_ok = sum(1 for r in rows if ok(base[r['Task ID']]) and ok(r))
    same_bad = sum(1 for r in rows if not ok(base[r['Task ID']]) and not ok(r))

    print(f'n={len(rows)}  基线成功={same_ok + c}  WM成功={same_ok + b}')
    print(f'2x2: WM独有成功(b)={b}  基线独有成功(c)={c}  都成功={same_ok}  都失败={same_bad}')
    print(f'McNemar 精确双侧 p = {mcnemar_exact(b, c):.3e}')

    modes = {}
    for r in rows:
        if ok(r):
            continue
        reason = r['failure_reason']
        key = ('提前DONE' if 'DONE but success' in reason
               else '步数耗尽' if 'maximum step' in reason else reason[:40])
        modes[key] = modes.get(key, 0) + 1
    print('WM 失败模式:', modes)

    zero = [r for r in rows if sel.get(r['Task ID'], {}).get('zero_injection')]
    if zero:
        print(f'零注入 {len(zero)} 条：WM 成功 {sum(1 for r in zero if ok(r))}')

    with open(OUT, 'w', newline='', encoding='utf-8') as handle:
        w = csv.writer(handle)
        w.writerow(['task_id', 'scene', 'category', 'golden_steps',
                    'n_targets', 'zero_injection', 'baseline_status',
                    'wm_status', 'wm_steps', 'wm_failure_reason',
                    'baseline_failure_reason', 'instruction'])
        for r in sorted(rows, key=lambda r: (r['Status'] == 'success',
                                             r['Task ID'])):
            tid = r['Task ID']
            meta = sel.get(tid, {})
            w.writerow([tid, r['Scene'], r['Category'], meta.get('golden'),
                        meta.get('n_targets'), int(bool(meta.get('zero_injection'))),
                        base[tid]['Status'], r['Status'], r['Actual Steps'],
                        r['failure_reason'], base[tid]['failure_reason'],
                        meta.get('instruction')])
    print('写入', OUT)


if __name__ == '__main__':
    main()
