#!/usr/bin/env python3
"""本机资源看护：每 5 分钟记一行 CPU/内存 + 评测进度，过线就告警。

本机只有 16 核 / 14G，跑 6 个 worker 时负载已经到 ~80%、可用内存 ~5G ——
2026-08-24 有过一次"内存打满 → WSL 整机崩"的事故，所以这里盯死两条线：

    load > 90% 核数      → 告警（该减 worker）
    可用内存 < 12%       → 告警（该减 worker / 清缓存）

    setsid nohup python3 tools/local_guard.py > /mnt/d/lightwm_out/local_guard.log 2>&1 &
"""

from __future__ import annotations

import csv
import os
import subprocess
import time

INTERVAL = 300
LOG = "/mnt/d/lightwm_out/local_guard.log"
RUN = "/home/sudidaren/spatialworld_eval/runs/wm_gemini31pro_fix_rest156"


def sh(cmd: str) -> str:
    try:
        return subprocess.run(["bash", "-c", cmd], capture_output=True,
                              text=True, timeout=30).stdout.strip()
    except Exception:
        return ""


def say(msg: str) -> None:
    line = f"[local {time.strftime('%F %T')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def main() -> None:
    say("local guard 启动")
    warned_load = warned_mem = False
    while True:
        load = os.getloadavg()[0]
        cores = os.cpu_count() or 1
        mem = {}
        for line in open("/proc/meminfo"):
            k = line.split(":")[0]
            if k in ("MemTotal", "MemAvailable"):
                mem[k] = int(line.split()[1]) / 1024 / 1024
        avail = mem.get("MemAvailable", 0)
        total = max(mem.get("MemTotal", 1), 1)

        done = succ = "?"
        try:
            with open(os.path.join(RUN, "results.csv"), encoding="utf-8-sig") as fh:
                recs = list(csv.DictReader(fh))
            done, succ = len(recs), sum(1 for r in recs if r["Status"] == "success")
        except Exception:
            pass
        workers = sh("pgrep -cf '[w]ork.run_task'") or "0"
        thor = sh("pgrep -cf '[t]hor-Linux64'") or "0"
        say(f"判定={done}/156 成功={succ}  worker={workers} thor={thor}  "
            f"load={load:.1f}/{cores}({load / cores:.0%})  "
            f"内存可用={avail:.1f}G({avail / total:.0%})")

        if load > cores * 0.9 and not warned_load:
            say(f"🚨 CPU 告警: load {load:.1f} > 90% —— 建议把 worker 从 6 降到 4")
            warned_load = True
        if avail / total < 0.12 and not warned_mem:
            say(f"🚨 内存告警: 可用 {avail:.1f}G({avail / total:.0%}) —— "
                f"建议减 worker（2026-08-24 就是这么崩的）")
            warned_mem = True
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
