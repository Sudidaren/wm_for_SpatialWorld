#!/usr/bin/env python3
"""卡上看护：自动清理今天踩过的每一类"僵尸 / 卡死"，并记账。

在卡上后台跑，每 5 分钟一次。处理的四类问题（全部是 2026-09-20 实际踩到的）：

  1. **孤儿 Unity**（ppid==1 的 thor-Linux64）—— worker 被杀后留下的，白烧
     CPU，会拖慢在跑的任务。
  2. **卡死的 ai2thor 构建锁持有者** —— 最坑的一类：早先的 smoke / provision
     进程卡在 `Build.download()`（55 KB/s 下 2GB 的构建），一直占着
     `~/.ai2thor/{tmp,releases}/*.lock`；之后**每个 worker 都会阻塞在同一把
     fcntl 锁上**，表现为"任务目录建了、0 帧、0 个 thor 进程"。判据：一个
     内联 `python -` 进程跑了超过 30 分钟，且不是 worker / supervisor / vLLM。
     杀掉之后顺手清锁（构建文件已经在盘上，下一个人会跳过下载）。
  3. **同一个 run 的重复 supervisor** —— 重复启动导致两批 worker 抢同一个
     results.csv。每 5 分钟只保留**启动最晚**的那一个。
  4. **进程卫生记账** —— 每轮写一行：worker / thor / 孤儿 / supervisor 数，
     以及当前 run 的完成数，方便回头查。

用法（卡上）：
    setsid nohup /root/miniconda3/bin/python /root/guard.py > /root/autodl-tmp/guard.log 2>&1 &
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import time

INTERVAL = 300
LOG = "/root/autodl-tmp/guard.log"
STALE_SECONDS = 1800            # 30 分钟
#: 这些不算"卡死的内联脚本"，不许杀
KEEP = ("work.run_task", "v1 wm wm_", "vllm", "guard.py", "jupyter",
        "tensorboard", "multiprocessing.resource_tracker",
        #: 2026-09-21 晚：闭源主批次的 supervisor 也是内联 python，
        #: 它的 args 里没有 "v1 wm wm_" 这个老标记，30 分钟后会被当"卡死
        #: 的内联脚本"杀掉。加上 run 名前缀白名单。
        "main_gpt5")


def sh(cmd: str) -> str:
    try:
        return subprocess.run(["bash", "-c", cmd], capture_output=True,
                              text=True, timeout=60).stdout
    except Exception:
        return ""


def ps_snapshot():
    out = sh("ps -eo pid,ppid,etimes,args --no-headers")
    rows = []
    for line in out.splitlines():
        parts = line.split(None, 3)
        if len(parts) < 4:
            continue
        try:
            rows.append((int(parts[0]), int(parts[1]), int(parts[2]), parts[3]))
        except ValueError:
            continue
    return rows


def say(msg: str) -> None:
    line = f"[guard {time.strftime('%F %T')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def kill(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass


def cycle() -> None:
    rows = ps_snapshot()
    # 2026-09-21：把 TVR addon 的 runner 也算作 worker —— 它同样会拉起
    # ai2thor/ProcTHOR 的 Unity 进程，不算进来的话看护会报 "worker=0 thor=N"，
    # 排障时看着像一堆孤儿，其实是正常的评测进程。
    workers = [r for r in rows
               if "work.run_task" in r[3] or "run_tvr_agent" in r[3]]
    sups = [r for r in rows if "v1 wm wm_" in r[3]]
    thor = [r for r in rows if "thor-Linux64" in r[3]]
    killed = []
    worker_pids = {r[0] for r in workers}
    live_pids = {r[0] for r in rows}
    #: 2026-09-21 补：supervisor 被 TERM/KILL 之后，worker 会被 init 收养
    #（ppid 变 1），但它**不会自己退出** —— 继续跑任务、继续写同一个
    # results.csv，和新的 supervisor 抢文件。旧判据只看 thor 的 ppid==1，
    # 漏掉了这种"孤儿 worker + 它名下的 thor"（thor 的 ppid 是那个孤儿
    # worker 的 pid，既不等于 1 也不是活着的 worker，于是两边都躲开了）。
    # 注意：TVR addon 的 shard 是 run_tvr.sh 用 `setsid nohup` 起的，启动脚本
    # 一退出它们的 ppid 就变成 1 —— 这是**设计如此**，不是孤儿。2026-09-21
    # 14:25 就因为把 run_tvr_agent 也认成 worker，一口气把 4 个正在跑的 wm
    # shard 连它们名下的 Unity 一起杀了。所以：只让它们参与计数（下面 thor 的
    # 判据要用），**绝不把它们当孤儿 worker 杀**。
    orphan_workers = [r for r in workers
                      if r[1] == 1 and "run_tvr_agent" not in r[3]]
    ow_pids = {r[0] for r in orphan_workers}
    for pid, _, etimes, _ in orphan_workers:
        kill(pid)
        killed.append(f"orphan-worker:{pid}({etimes}s)")
    #: thor 算孤儿的三种情况：父进程是 init；父进程是刚被我们杀掉的孤儿
    # worker；父进程既不是活着的 worker、也已经不存在。
    orphans = [r for r in thor
               if r[1] == 1
               or r[1] in ow_pids
               or (r[1] not in worker_pids and r[1] not in live_pids)]

    # 1) 孤儿 Unity
    for pid, _, _, _ in orphans:
        kill(pid)
        killed.append(f"orphan-thor:{pid}")

    # 2) 卡死的构建锁持有者（内联 python，跑太久，而且不是我们在跑的东西）
    stale = []
    for pid, ppid, etimes, args in rows:
        if etimes < STALE_SECONDS:
            continue
        if not re.search(r"(\.venv/bin/python|miniconda3/bin/python)\b", args):
            continue
        if any(k in args for k in KEEP):
            continue
        # 只杀"内联脚本"：argv 里没有真正脚本名（形如 `python -`）
        if re.search(r"python3? -(\s|$)", args):
            stale.append((pid, etimes, args[:70]))
    for pid, etimes, args in stale:
        kill(pid)
        killed.append(f"stale-python:{pid}({etimes}s)")
    if stale:
        sh("rm -f /root/.ai2thor/tmp/*.lock /root/.ai2thor/releases/*.lock")
        killed.append("build-locks-cleared")

    # 3) 同一 run 的重复 supervisor —— 只留启动最晚的那个
    if len(sups) > 1:
        # etimes 越小 = 起得越晚
        keep = min(sups, key=lambda r: r[2])
        for pid, _, _, args in sups:
            if pid == keep[0]:
                continue
            kill(pid)
            killed.append(f"dup-supervisor:{pid}")

    # 4) 记账
    run = sh("ls -dt /home/sudidaren/spatialworld_eval/runs/wm_* 2>/dev/null | head -1").strip()
    done = succ = "?"
    if run:
        try:
            import csv
            with open(os.path.join(run, "results.csv"), encoding="utf-8-sig") as fh:
                recs = list(csv.DictReader(fh))
            done, succ = len(recs), sum(1 for r in recs if r["Status"] == "success")
        except Exception:
            pass
    say(f"{os.path.basename(run) if run else 'no-run'}  判定={done} 成功={succ}  "
        f"worker={len(workers)} thor={len(thor) - len(orphans)} "
        f"孤儿={len(orphans)} supervisor={len(sups)}"
        + (f"  | 清理: {', '.join(killed)}" if killed else ""))

    # 5) 资源记账 + 告警（用户明确要求：别爆 CPU / 内存）
    load = os.getloadavg()[0]
    cores = os.cpu_count() or 1
    mem = {}
    for line in open("/proc/meminfo"):
        k = line.split(":")[0]
        if k in ("MemTotal", "MemAvailable"):
            mem[k] = int(line.split()[1]) / 1024 / 1024
    avail_pct = mem.get("MemAvailable", 0) / max(mem.get("MemTotal", 1), 1) * 100
    say(f"  资源: load={load:.1f}/{cores}核({load / cores:.0%})  "
        f"内存可用={mem.get('MemAvailable', 0):.0f}G/"
        f"{mem.get('MemTotal', 0):.0f}G({avail_pct:.0f}%)")
    if load > cores * 0.9:
        say(f"🚨 CPU 告警: load {load:.1f} > 90% of {cores} 核 —— 考虑减 worker")
    if avail_pct < 12:
        say(f"🚨 内存告警: 可用只剩 {mem.get('MemAvailable', 0):.0f}G"
            f"({avail_pct:.0f}%) —— 考虑减 worker 或清缓存")


def main() -> None:
    say("guard 启动")
    while True:
        try:
            cycle()
        except Exception as exc:                       # noqa: BLE001
            say(f"cycle 出错: {type(exc).__name__}: {exc}")
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
