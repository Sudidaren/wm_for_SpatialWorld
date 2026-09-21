#!/usr/bin/env python3
"""挑"最可能被今天的 bug 修复救回来"的失败任务。

候选 = Gemini+WM 跑过并且**失败**的任务（fail210 的 174 条 + simple40 的 22 条）。
不看基线，只看我们自己那次是怎么失败的 —— 因为要验证的是"修复有没有效"。

打分依据（每条都对应今天修掉的一个 bug）：
  hand      episode 里出现 "Hand already has an object" 的次数
            → 修①（手持判定）。这是最直接的靶子。
  held_dup  一段提示里同时出现「手持：X」和「X 在视野内（带位置）」的步数
            → 修②（手持物体不再报位置）
  conflict  一段提示里同时出现「移动提示：」和「建议先转向它再靠近」的步数
            → 修③（矛盾指令压制）
  repeat    「重复提示：」出现的步数 → 修⑥（整条删除）
  far_tail  「你在远离它」出现次数 → 修⑤（同物体同锚点才比）
  blocked   "is blocking Agent" 报错次数 → 修③的另一种触发
简单度：golden 步数（越少越简单）、目标数（越少越简单）。

    python3 tools/select_fix_targets.py --n 40 --out /tmp/fix40.json
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from collections import Counter

RUNS = '/home/sudidaren/spatialworld_eval/runs'
TASKS = '/home/sudidaren/SpatialWorld/data/ai2thor/tasks'
SRC = ['wm_gemini31pro_simple40', 'wm_gemini31pro_fail210']


def hint_features(hint: str) -> dict:
    flat = hint.replace('\n', ' | ')
    vis = re.search(r'视野内：([^；|\n]*)', flat)
    vis_names = set(re.findall(r'([A-Z][A-Za-z]+)', vis.group(1))) if vis else set()
    held = re.search(r'手持：([A-Za-z]+)', flat)
    return {
        'held_dup': bool(held and held.group(1) in vis_names),
        'conflict': ('移动提示：' in flat and '建议先转向它再靠近' in flat),
        'repeat': '重复提示：' in flat,
        'far_tail': '你在远离它' in flat,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=40)
    ap.add_argument('--out', default='')
    ap.add_argument('--per-scene', type=int, default=1)
    args = ap.parse_args()

    rows = []
    for run in SRC:
        path = os.path.join(RUNS, run, 'results.csv')
        if not os.path.exists(path):
            continue
        for r in csv.DictReader(open(path, encoding='utf-8-sig')):
            if r['Status'] != 'failed_model':
                continue
            tid = r['Task ID']
            feats = Counter()
            for d in glob.glob(os.path.join(RUNS, run, 'ai2thor', tid,
                                            'worker_*', tid)):
                ep = glob.glob(os.path.join(d, 'episode_*.json'))
                sp = glob.glob(os.path.join(d, 'steps.jsonl'))
                if ep:
                    try:
                        traj = json.load(open(ep[0])).get('trajectory', [])
                    except Exception:
                        traj = []
                    for e in traj:
                        m = e.get('error_message') or ''
                        if 'Hand already has an object' in m:
                            feats['hand'] += 1
                        if 'is blocking Agent' in m:
                            feats['blocked'] += 1
                if sp:
                    for line in open(sp[0], encoding='utf-8'):
                        if not line.strip():
                            continue
                        h = json.loads(line).get('mem_hint') or ''
                        if not h:
                            continue
                        f = hint_features(h)
                        for k, v in f.items():
                            if v:
                                feats[k] += 1
            tj = os.path.join(TASKS, tid, 'task.json')
            if not os.path.exists(tj):
                continue
            meta = json.load(open(tj, encoding='utf-8'))
            golden = meta['golden_actions']['steps']
            ntar = len(meta.get('target_object_types') or [])
            # 修复相关性：直接对应修①的 hand 权重最高
            fix = (3 * feats['hand'] + 2 * feats['held_dup']
                   + 2 * feats['conflict'] + 1 * feats['repeat']
                   + 1 * feats['far_tail'] + 0.3 * min(feats['blocked'], 10))
            rows.append({
                'task_id': tid, 'scene': r['Scene'], 'run': run,
                'golden': golden, 'n_targets': ntar,
                'fix': round(fix, 1), **{k: feats[k] for k in
                                         ('hand', 'held_dup', 'conflict',
                                          'repeat', 'far_tail', 'blocked')},
                'instruction': meta.get('instruction', ''),
                # 排序键：先要"修复相关"（fix>0），再简单（golden 少），再 fix 大小
                'key': (fix > 0, -golden, fix),
            })

    rows.sort(key=lambda r: r['key'], reverse=True)
    picked, seen = [], Counter()
    for r in rows:
        if seen[r['scene']] >= args.per_scene:
            continue
        seen[r['scene']] += 1
        picked.append(r)
        if len(picked) == args.n:
            break

    print(f'候选 {len(rows)} 条失败任务；选出 {len(picked)} 条')
    print(f'  其中"直接命中今天修复"的（fix>0）: '
          f'{sum(1 for r in picked if r["fix"] > 0)}')
    print(f'\n{"task_id":<15}{"g":>3}{"tg":>3}{"fix":>6}{"hand":>5}{"hdup":>5}'
          f'{"confl":>6}{"rep":>4}{"far":>4}{"blk":>4}  scene')
    for r in picked:
        print(f'{r["task_id"]:<15}{r["golden"]:>3}{r["n_targets"]:>3}'
              f'{r["fix"]:>6}{r["hand"]:>5}{r["held_dup"]:>5}{r["conflict"]:>6}'
              f'{r["repeat"]:>4}{r["far_tail"]:>4}{r["blocked"]:>4}  {r["scene"]}')
    ids = ','.join(r['task_id'] for r in picked)
    print('\nSMOKE_TASKS=' + ids)
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as fh:
            json.dump({'n': len(picked), 'tasks': picked}, fh,
                      ensure_ascii=False, indent=2)
        print(f'\n写入 {args.out}')


if __name__ == '__main__':
    main()
