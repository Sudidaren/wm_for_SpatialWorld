#!/usr/bin/env python3
"""盯着 Gemini+WM(210) → GPT-5+WM(250) 这条链，保证"不断"。

职责（单一写者：只保证链条活着，不自己起评测，避免和 chain 抢着拉进程）：
  1. 每 5 分钟记一行进度（条数/成功/速率/内存/在跑的批次）；
  2. 速率掉到 0 且仍有未完成任务时，报警（不自动重启评测，交给 chain）；
  3. chain 进程不在 → 重新拉起 chain（chain 自己会判断该等、该补跑还是该交班）；
  4. 两个批次都跑完后，输出汇总并退出。

用法：setsid nohup python3 watch_wm_chain.py >/dev/null 2>&1 </dev/null &
"""

from __future__ import annotations

import csv
import os
import subprocess
import time
from datetime import datetime

RUNS = '/home/sudidaren/spatialworld_eval/runs'
BASE = os.path.join(RUNS, 'replay_legacy311_v1/results.csv')
GEM = os.path.join(RUNS, 'wm_gemini31pro_fail210')
GPT = os.path.join(RUNS, 'wm_gpt5_ai2thor_fail250')
CHAIN = '/home/sudidaren/lightwm_phases/tools/chain_gpt5_after_gemini.sh'
LOG = '/mnt/d/lightwm_out/watch_chain.log'
INTERVAL = 300
SWE = '/home/sudidaren/spatialworld_eval'
NEXT_RUN = 'wm_gpt5_ai2thor_fail250'


def rows(path: str) -> int:
    f = os.path.join(path, 'results.csv')
    if not os.path.exists(f):
        return 0
    with open(f, encoding='utf-8-sig') as handle:
        return sum(1 for r in csv.DictReader(handle)) 


def decided(path: str) -> int:
    """真正判定过的条数（success + failed_model）。

    重跑 failed_external 时是**按 task_id 覆盖写**，总行数不变，
    所以进度必须按"判定过的"来数，不能按行数。
    """
    f = os.path.join(path, 'results.csv')
    if not os.path.exists(f):
        return 0
    with open(f, encoding='utf-8-sig') as handle:
        return sum(1 for r in csv.DictReader(handle)
                   if r['Status'] in ('success', 'failed_model'))


def ok(path: str) -> int:
    f = os.path.join(path, 'results.csv')
    if not os.path.exists(f):
        return 0
    with open(f, encoding='utf-8-sig') as handle:
        return sum(1 for r in csv.DictReader(handle) if r['Status'] == 'success')


def alive(pattern: str) -> bool:
    return subprocess.run(['pgrep', '-f', pattern],
                          stdout=subprocess.DEVNULL).returncode == 0


def sup_pat(run: str) -> str:
    """supervisor 的 argv 特征，避免把监控命令/日志路径误判成在跑。"""
    return rf'[.]venv/bin/python -u - .* {run} '


def pending_tasks(run: str) -> list[str]:
    want = [r['Task ID']
            for r in csv.DictReader(open(BASE, encoding='utf-8-sig'))
            if r['Status'] != 'success']
    f = os.path.join(RUNS, run, 'results.csv')
    done = set()
    if os.path.exists(f):
        done = {r['Task ID'] for r in csv.DictReader(open(f, encoding='utf-8-sig'))}
    return [t for t in want if t not in done]


def launch(run: str, tasks: list[str], workers: int) -> None:
    env = dict(os.environ)
    # 不要代理：Clash 一挂就走 SSL EOF 全军覆没（2026-09-20 那 92 条就是这么来的）
    for key in ('HTTPS_PROXY', 'HTTP_PROXY', 'https_proxy', 'http_proxy'):
        env.pop(key, None)
    env.update({
        'TMPDIR_OVERRIDE': '/tmp/lightwm_tmp',
        'VLM_FORCE_IPV4': '1',
        'VLM_API_TIMEOUT': '120',
        'SMOKE_TASKS': ','.join(tasks),
        'RUN_NAME': run,
        'WORKERS': str(workers),
        'MODEL_NAME': 'gpt-5',
        'LLM_API_KEY': os.environ.get('OPENAI_API_KEY', ''),
    })
    log = open(f'/mnt/d/lightwm_out/{run}.log', 'a')
    subprocess.Popen(['setsid', 'nohup', 'bash',
                      os.path.join(SWE, 'run_wm_gemini_ai2thor.sh')],
                     cwd=SWE, env=env, stdout=log, stderr=log,
                     stdin=subprocess.DEVNULL, start_new_session=True)


def say(msg: str) -> None:
    line = f"[{datetime.now():%F %T}] {msg}"
    print(line, flush=True)
    with open(LOG, 'a', encoding='utf-8') as handle:
        handle.write(line + '\n')


def mem_available_gb() -> float:
    for line in open('/proc/meminfo'):
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) / 1024 / 1024
    return float('nan')


def external_count(run: str) -> int:
    """网络/环境类失败（failed_external）的条数。"""
    f = os.path.join(RUNS, run, 'results.csv')
    if not os.path.exists(f):
        return 0
    with open(f, encoding='utf-8-sig') as handle:
        return sum(1 for r in csv.DictReader(handle)
                   if r['Status'] == 'failed_external')


def proc_stats() -> dict:
    """僵尸 / 孤儿 / 模拟器进程盘点（用户反复提醒：别留僵尸）。"""
    zombie = orphan_thor = thor = worker = 0
    for entry in os.listdir('/proc'):
        if not entry.isdigit():
            continue
        try:
            with open(f'/proc/{entry}/stat') as handle:
                parts = handle.read().rsplit(')', 1)[1].split()
            state, ppid = parts[0], int(parts[1])
            cmdline = open(f'/proc/{entry}/cmdline', 'rb').read().decode(
                'utf-8', 'replace').replace('\x00', ' ')
        except OSError:
            continue
        if state == 'Z':
            zombie += 1
        if 'thor-Linux64' in cmdline:
            thor += 1
            if ppid == 1:
                orphan_thor += 1
        elif 'work.run_task' in cmdline:
            worker += 1
    return {'zombie': zombie, 'orphan_thor': orphan_thor,
            'thor': thor, 'worker': worker}


def main() -> None:
    need = 250          # Gemini 失败任务总数（40 已跑 + 210 本轮）
    gem_prev, t_prev, stall = decided(GEM), time.time(), 0
    ext_prev = external_count(GEM)
    say('watcher 启动')
    while True:
        g, go = decided(GEM), ok(GEM)
        p, po = decided(GPT), ok(GPT)
        ext = external_count(GEM) + external_count(GPT)
        gem_done = g >= 210          # 本轮就该跑 210 条
        gpt_done = p >= need

        rate = (g - gem_prev) / max((time.time() - t_prev) / 60, 1e-9)
        say(f'gemini {g}/210(ok {go})  gpt5 {p}/{need}(ok {po})  '
            f'速率 {rate:.2f} 条/分  可用内存 {mem_available_gb():.1f}G  '
            f'chain={"up" if alive("chain_gpt5_after_gemini") else "DOWN"}  '
            f'failed_external={ext}')

        ps_ = proc_stats()
        say(f'  进程：worker={ps_["worker"]} thor={ps_["thor"]} '
            f'孤儿thor={ps_["orphan_thor"]} 僵尸={ps_["zombie"]}')
        if ps_['orphan_thor'] or ps_['zombie']:
            say(f'🚨 进程卫生告警：孤儿 thor={ps_["orphan_thor"]} '
                f'僵尸={ps_["zombie"]} → 建议 reap_sim.sh --workers')
        if ps_['worker'] > 6:
            say(f'⚠ worker 数 {ps_["worker"]} > 6，可能有前一批次的残留')
        mem = mem_available_gb()
        if mem < 2.5:
            say(f'🚨 可用内存 {mem:.1f}G < 2.5G：'
                f'（本机 2026-08-24 有过 OOM 整机崩）考虑把 WORKERS 降回 4')

        # 网络类失败一涨就报警（AGENTS.md：连续 5 个 failed_external 要停下汇报）
        if ext > ext_prev:
            say(f'🚨 failed_external {ext_prev} → {ext}（涨 {ext - ext_prev}）'
                f'；立刻验网关（不要走代理）：'
                f'curl -s -o /dev/null -w "%{{http_code}}" '
                f'https://apic1.ohmycdn.com/v1/models')

        if gem_done and not gpt_done and not alive('chain_gpt5_after_gemini'):
            say('⚠ gemini 已完成但 chain 不在 → 重新拉起 chain')
            subprocess.Popen(['setsid', 'nohup', 'bash', CHAIN],
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL,
                             stdin=subprocess.DEVNULL, start_new_session=True)
        elif (os.path.exists(os.path.join(GPT, 'plan.json')) and not gpt_done
              and not alive('chain_gpt5_after_gemini')
              and not alive(sup_pat(NEXT_RUN))):
            pend = pending_tasks(NEXT_RUN)
            if pend:
                say(f'⚠ gpt5 批次中断且无人在管 → 续跑 {len(pend)} 条')
                launch(NEXT_RUN, pend, 6)
                time.sleep(60)
        elif not gem_done and rate == 0:
            stall += 1
            say(f'⚠ 本轮无新完成（第 {stall} 次）；'
                f'gemini supervisor={"up" if alive(r"venv/bin/python -u - .* wm_gemini31pro_fail210 ") else "DOWN"}')

        if gem_done and gpt_done:
            say(f'全部结束：gemini {go}/210，gpt5 {po}/{need} → '
                f'Gemini 失败集合计 {go + 18}/250 成功')
            return

        gem_prev, t_prev = g, time.time()
        ext_prev = ext
        time.sleep(INTERVAL)


if __name__ == '__main__':
    main()
