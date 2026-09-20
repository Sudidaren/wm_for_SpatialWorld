#!/usr/bin/env python3
"""从 Gemini 基线**失败**的任务里挑"最简单"的 N 个（只用于选题）。

口径（2026-09-19 深夜版交接文档 §6）：
  * 基线 = spatialworld_eval/runs/replay_legacy311_v1/results.csv（311 行，61 成功）；
  * "简单" = golden 步数少 + 单目标 + 指令里能被检测头命中的类名多（更好落地）；
  * 一个场景最多一条（避免同一间房子反复出现）；
  * 标出"零注入任务"（指令里一个检测类名都没有）——A′ 加工态归并对它们收益最大，
    按文档要求优先纳入。

**只读 task.json 用于选题**；这些字段不进运行时（红线 §0.1）。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re

BASELINE = ('/home/sudidaren/spatialworld_eval/runs/'
            'replay_legacy311_v1/results.csv')
TASKS_ROOT = '/home/sudidaren/SpatialWorld/data/ai2thor/tasks'
INVENTORY = '/home/sudidaren/lightwm_phases/data/inventory.json'


def detector_classes() -> list[str]:
    """训练时用的 117 类词表（与感知头逐字一致）。"""
    with open(INVENTORY, encoding='utf-8') as handle:
        pools = json.load(handle).get('pools', {})
    for pool in pools.values():
        vocab = pool.get('class_coverage', {})
        if len(vocab) == 117:
            return sorted(vocab)
    raise SystemExit('找不到 117 类词表')


def class_hits(instruction: str, classes: list[str]) -> list[str]:
    """指令里出现了哪些检测类名（整词、忽略大小写）。"""
    hits = []
    for name in classes:
        # 驼峰类名在自然语言里通常写成分开的词（CoffeeMachine -> coffee machine），
        # 所以两种拼法都试。
        spaced = re.sub(r'(?<!^)(?=[A-Z])', ' ', name)
        pattern = r'\b(%s)\b' % '|'.join(
            re.escape(p).replace(r'\ ', r'[\s-]?') for p in {name, spaced})
        if re.search(pattern, instruction, flags=re.IGNORECASE):
            hits.append(name)
    return hits


def load_rows() -> list[dict]:
    with open(BASELINE, encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=40)
    parser.add_argument('--max-golden', type=int, default=0,
                        help='0 = 自动放开到够 n 条为止')
    parser.add_argument('--per-scene', type=int, default=1)
    parser.add_argument('--exclude', default='',
                        help='逗号分隔的 task_id，排除')
    parser.add_argument('--out', default='')
    args = parser.parse_args()

    classes = detector_classes()
    exclude = {t for t in args.exclude.split(',') if t}

    failures = [r for r in load_rows()
                if r['Status'] != 'success' and r['Task ID'] not in exclude]
    print(f'基线失败 {len(failures)} 条（总计 {len(load_rows())}）')

    scored = []
    for row in failures:
        tid = row['Task ID']
        path = os.path.join(TASKS_ROOT, tid, 'task.json')
        if not os.path.isfile(path):
            continue
        with open(path, encoding='utf-8') as handle:
            task = json.load(handle)
        hits = class_hits(task.get('instruction', ''), classes)
        scored.append({
            'task_id': tid,
            'scene': row['Scene'],
            'category': row['Category'],
            'golden': int(row['golden_actions_count']),
            'n_targets': len(task.get('target_object_types', [])),
            'hits': hits,
            'zero_injection': not hits,
            'instruction': task.get('instruction', ''),
        })

    # 排序键：golden 步数 → 目标数 → 命中类名数（多者为佳）→ task_id
    scored.sort(key=lambda r: (r['golden'], r['n_targets'],
                               -len(r['hits']), r['task_id']))

    picked, seen = [], {}
    if args.max_golden:
        pool = [r for r in scored if r['golden'] <= args.max_golden]
    else:
        pool = scored
    for row in pool:
        used = seen.get(row['scene'], 0)
        if used >= args.per_scene:
            continue
        picked.append(row)
        seen[row['scene']] = used + 1
        if len(picked) == args.n:
            break

    n_zero = sum(1 for r in picked if r['zero_injection'])
    n_scene = len(set(r['scene'] for r in picked))
    print(f'选中 {len(picked)} 条；golden 范围 '
          f'{picked[0]["golden"]}–{picked[-1]["golden"]}；'
          f'零注入 {n_zero} 条；场景 {n_scene} 个')
    print()
    print(f'{"task_id":<16} {"g":>3} {"tg":>3} {"hits":>4}  scene')
    for row in picked:
        flag = ' (零注入)' if row['zero_injection'] else ''
        print(f'{row["task_id"]:<16} {row["golden"]:>3} {row["n_targets"]:>3} '
              f'{len(row["hits"]):>4}  {row["scene"]}{flag}')

    ids = ','.join(r['task_id'] for r in picked)
    print()
    print('SMOKE_TASKS=' + ids)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as handle:
            json.dump({'baseline': BASELINE, 'n': len(picked),
                       'tasks': picked}, handle, ensure_ascii=False, indent=2)
        print(f'写入 {args.out}')


if __name__ == '__main__':
    main()
