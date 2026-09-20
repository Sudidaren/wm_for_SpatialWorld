#!/usr/bin/env python3
"""排查注入提示里有没有互相矛盾的建议。

WM 每一步只应该给**一条移动指令**（这是交接文档自己定的规则），但提示是由
好几个通道拼出来的，任何一个通道都可能各自带上一条"你去哪儿"的指令：

  移动类: 移动提示(被挡) / 建议先转向它再靠近 / 探索建议 / 距离提示(走到 1m 内)
  姿态类: 手持 / 已在手中
  事实类: 视野内 / 记住的位置 / 上一个动作 / 重复提示 / 常识先验

这个脚本按步统计：哪些通道同时出现、**同时给了几条移动指令**、
以及那些指令指向的目标是不是同一个。

    python3 tools/audit_hint_conflicts.py --run wm_gemini31pro_fail210
"""

from __future__ import annotations

import argparse
import csv
import glob
import itertools
import json
import os
import re
from collections import Counter, defaultdict

RUNS = '/home/sudidaren/spatialworld_eval/runs'

#: 通道名 -> 判定用的子串。顺序按"这条通道会不会下指令"排。
MOVE_CHANNELS = {
    '被挡脱困': '移动提示：',
    '建议靠近目标': '建议先转向它再靠近',
    '探索建议': '探索建议',
    '距离提示(走近)': '要先走到',
}
FACT_CHANNELS = {
    '手持': '手持：',
    '视野内': '视野内',
    '记住的位置': '记住的位置在',
    '上一个动作': '上一个动作：',
    '重复提示': '重复提示',
    '常识先验': '常出现在',
    '已在手中': '已在手中',
    'CheckState': 'CheckState',
}
ALL = {**MOVE_CHANNELS, **FACT_CHANNELS}


def present(hint: str) -> set:
    got = set()
    for name, needle in ALL.items():
        if needle in hint:
            # "手持：" 要排除"手持：空手"这种否定式
            if name == '手持' and '手持：空手' in hint and hint.count('手持：') == hint.count('手持：空手'):
                continue
            got.add(name)
    return got


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='wm_gemini31pro_fail210')
    ap.add_argument('--also', default='wm_gemini31pro_simple40')
    args = ap.parse_args()

    pair = Counter()
    move_counts = Counter()
    steps_total = 0
    move_examples = []
    chan_hits = Counter()

    for run in [args.run] + [r for r in args.also.split(',') if r]:
        res = os.path.join(RUNS, run, 'results.csv')
        if not os.path.exists(res):
            continue
        for r in csv.DictReader(open(res, encoding='utf-8-sig')):
            for sp in glob.glob(os.path.join(RUNS, run, 'ai2thor',
                                             r['Task ID'], 'worker_*', '*',
                                             'steps.jsonl')):
                for line in open(sp, encoding='utf-8'):
                    if not line.strip():
                        continue
                    hint = json.loads(line).get('mem_hint') or ''
                    if not hint:
                        continue
                    steps_total += 1
                    got = present(hint)
                    for c in got:
                        chan_hits[c] += 1
                    for a, b in itertools.combinations(sorted(got), 2):
                        pair[(a, b)] += 1
                    nmove = sum(1 for c in MOVE_CHANNELS if c in got)
                    move_counts[nmove] += 1
                    if nmove >= 2 and len(move_examples) < 6:
                        move_examples.append((r['Task ID'], sorted(
                            c for c in got if c in MOVE_CHANNELS), hint))

    print(f'总步数（有提示的）: {steps_total}')
    print('\n--- 各通道出现次数 ---')
    for c, n in chan_hits.most_common():
        print(f'  {c:<16} {n:>5}  ({n/steps_total:.2%})')

    print('\n--- 一段提示里有几条"移动指令" ---')
    for k in sorted(move_counts):
        print(f'  {k} 条: {move_counts[k]:>5} 步 ({move_counts[k]/steps_total:.2%})')

    print('\n--- 最常同时出现的通道对（前 14）---')
    for (a, b), n in pair.most_common(14):
        flag = ''
        if a in MOVE_CHANNELS and b in MOVE_CHANNELS:
            flag = '  ← 两条移动指令打架'
        elif (a in MOVE_CHANNELS) != (b in MOVE_CHANNELS):
            flag = ''
        print(f'  {n:>5}  {a} + {b}{flag}')

    if move_examples:
        print('\n--- 同时给了两条移动指令的样例 ---')
        for tid, mv, hint in move_examples[:3]:
            print(f'  [{tid}] {mv}')
            print(f'    {hint.replace(chr(10), " | ")[:230]}')


if __name__ == '__main__':
    main()
