#!/usr/bin/env python3
"""把"帧差判成败"判错的两类案例做成可看的对照图。

唯一允许的动作成败信号是帧差（``self_observation``：``success = mse > 1.0``）。
这份脚本用**真实跑批的帧**复算它准不准，并把两类错判各挑若干条落成图：

  * 判成功但真失败  (false positive)  —— 画面变了，可模拟器说这一步没生效
  * 判失败但真成功  (false negative)  —— 画面没变，可模拟器说这一步成功了

帧的对齐是从 ``log.json`` 的 ``images`` 拿的（它按 step 排好，长度等于步数），
第 k 步的帧差 = mse(images[k], images[k+1])；模拟器真值取
``episode_*.json`` 里 ``trajectory[k].error_message``（空 = 成功）。
对齐已用"被挡的动作 mse 必须 ≈ 0"验证过。

    python3 tools/make_mse_evidence.py --out /mnt/c/Users/29742/Desktop/MSE判定证据
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import shutil
import sys
from collections import Counter

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, '/home/sudidaren/SpatialWorld')
from mllm_base_agent.agent.self_observation import frame_mse  # noqa: E402

RUNS = '/home/sudidaren/spatialworld_eval/runs'
DEFAULT_RUNS = ['wm_gemini31pro_fail210', 'wm_gemini31pro_simple40']
MSE_THRESHOLD = 1.0
INTERACT = {'PickupObject', 'PutObject', 'OpenObject', 'CloseObject',
            'ToggleObjectOn', 'ToggleObjectOff', 'SliceObject', 'DropHandObject',
            'CleanObject', 'FillObjectWithLiquid', 'DirtyObject', 'UseUpObject',
            'BreakObject', 'CookObject', 'EmptyLiquidFromObject'}
NAV = {'MoveAhead', 'MoveBack', 'MoveLeft', 'MoveRight', 'RotateLeft',
       'RotateRight', 'LookUp', 'LookDown'}
FONT_CANDIDATES = ['/mnt/c/Windows/Fonts/msyh.ttc',
                   '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']


def _font(size: int):
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def nm(a: str) -> str:
    return (a or '').split('(')[0].strip()


def collect(runs: list[str]) -> list[dict]:
    out = []
    for run in runs:
        res = os.path.join(RUNS, run, 'results.csv')
        if not os.path.exists(res):
            continue
        for r in csv.DictReader(open(res, encoding='utf-8-sig')):
            tid = r['Task ID']
            for d in glob.glob(os.path.join(RUNS, run, 'ai2thor', tid,
                                            'worker_*', tid)):
                log_p = os.path.join(d, 'log.json')
                ep_p = glob.glob(os.path.join(d, 'episode_*.json'))
                sp_p = glob.glob(os.path.join(d, 'steps.jsonl'))
                if not (os.path.exists(log_p) and ep_p):
                    continue
                try:
                    imgs = json.load(open(log_p))['images']
                    traj = json.load(open(ep_p[0]))['trajectory']
                except Exception:
                    continue
                hints = []
                if sp_p:
                    hints = [json.loads(l) for l in open(sp_p[0], encoding='utf-8')
                             if l.strip()]
                for k in range(min(len(imgs), len(traj)) - 1):
                    act = traj[k].get('action_string') or ''
                    name = nm(act)
                    if name in INTERACT:
                        kind = '交互'
                    elif name in NAV:
                        kind = '移动'
                    else:
                        continue
                    err = traj[k].get('error_message') or ''
                    truth = err in ('', 'None')
                    mse = frame_mse(imgs[k], imgs[k + 1])
                    if mse is None:
                        continue
                    pred = mse > MSE_THRESHOLD
                    cat = ('fp' if (pred and not truth)
                           else 'fn' if ((not pred) and truth) else None)
                    if not cat:
                        continue
                    nxt = hints[k + 1].get('mem_hint', '') if k + 1 < len(hints) else ''
                    out.append({'cat': cat, 'run': run, 'task': tid, 'step': k,
                                'action': act, 'kind': kind,
                                'truth': truth, 'mse': mse,
                                'err': err, 'before': imgs[k], 'after': imgs[k + 1],
                                'next_hint': nxt, 'hint_step': k + 1})
    return out


def render(item: dict, out_path: str) -> None:
    H = 360
    a = Image.open(item['before']).convert('RGB')
    b = Image.open(item['after']).convert('RGB')
    a = a.resize((int(a.width * H / a.height), H))
    b = b.resize((int(b.width * H / b.height), H))
    head, foot = 108, 74
    W = a.width + b.width + 24
    canvas = Image.new('RGB', (W, H + head + foot), (250, 250, 250))
    canvas.paste(a, (8, head))
    canvas.paste(b, (a.width + 16, head))
    d = ImageDraw.Draw(canvas)
    f_big, f_mid, f_small = _font(30), _font(23), _font(21)

    if item['cat'] == 'fp':
        title, color = '判“成功” 但真值失败（画面变了，动作其实没生效）', (190, 30, 30)
    else:
        title, color = '判“失败” 但真值成功（画面没变，动作其实生效了）', (20, 110, 190)
    d.text((12, 8), title, font=f_big, fill=color)
    d.text((12, 48), f"{item['task']}  step {item['step']}  "
                     f"{item['kind']}动作 {item['action']}", font=f_mid,
           fill=(20, 20, 20))
    d.text((12, 76), f"帧差 mse = {item['mse']:.1f}   阈值 = {MSE_THRESHOLD}   "
                     f"WM 判定 = {'成功' if item['mse'] > MSE_THRESHOLD else '失败'}   "
                     f"模拟器 = {'成功' if item['truth'] else '失败'}"
                     + (f"   报错: {item['err'][:52]}" if item['err'] else ''),
           font=f_small, fill=(70, 70, 70))
    d.rectangle([0, head - 2, a.width + 8, head], fill=(180, 180, 180))
    d.text((12, head + H + 6), '动作前', font=f_small, fill=(90, 90, 90))
    d.text((a.width + 20, head + H + 6), '动作后', font=f_small, fill=(90, 90, 90))

    said = ''
    for line in (item['next_hint'] or '').split('\n'):
        if '上一个动作' in line:
            said = line.strip()
    if said:
        d.text((12, head + H + 32), f"WM 下一步告诉模型：{said[:118]}",
               font=f_small, fill=(150, 40, 40))
    canvas.save(out_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/mnt/c/Users/29742/Desktop/MSE判定证据')
    ap.add_argument('--per-cat', type=int, default=8)
    ap.add_argument('--runs', default=','.join(DEFAULT_RUNS))
    args = ap.parse_args()

    items = collect([r for r in args.runs.split(',') if r])
    for cat, label in (('fp', '判成功但真失败'), ('fn', '判失败但真成功')):
        sub = [i for i in items if i['cat'] == cat]
        by_kind = Counter(i['kind'] for i in sub)
        print(f'{label}: {len(sub)} 条  {dict(by_kind)}')

    if os.path.isdir(args.out):
        for name in os.listdir(args.out):
            p = os.path.join(args.out, name)
            if os.path.isfile(p):
                os.remove(p)
    os.makedirs(args.out, exist_ok=True)

    picked = []
    for cat, label in (('fp', '判成功但真失败'), ('fn', '判失败但真成功')):
        sub = [i for i in items if i['cat'] == cat]
        # 先把同一任务里的分散开，再按"错得最明显"排序
        seen = set()
        sub.sort(key=lambda i: (-abs(i['mse'] - MSE_THRESHOLD)))
        choose = []
        for i in sub:
            if i['task'] in seen:
                continue
            seen.add(i['task'])
            choose.append(i)
            if len(choose) >= args.per_cat:
                break
        for n, i in enumerate(choose, 1):
            path = os.path.join(args.out, f'{label}_{n:02d}_{i["task"]}_step{i["step"]}.png')
            render(i, path)
            picked.append((label, i, path))
            print(f'  {path}')

    with open(os.path.join(args.out, 'manifest.csv'), 'w', newline='',
              encoding='utf-8-sig') as handle:
        w = csv.writer(handle)
        w.writerow(['类别', '任务', '步', '动作', '帧差mse', '阈值',
                    'WM判定', '模拟器真值', '报错', '图'])
        for label, i, path in picked:
            w.writerow([label, i['task'], i['step'], i['action'], round(i['mse'], 2),
                        MSE_THRESHOLD, '成功' if i['mse'] > MSE_THRESHOLD else '失败',
                        '成功' if i['truth'] else '失败', i['err'],
                        os.path.basename(path)])
    print(f'\n共 {len(picked)} 张，写入 {args.out}')


if __name__ == '__main__':
    main()
