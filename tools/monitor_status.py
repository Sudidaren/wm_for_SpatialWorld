#!/usr/bin/env python3
"""一轮巡检：本机 + 四张新卡（30b/8b/kimi/gpt5）的状态，追加到一行日志。

盯的就是 2026-09-21 夜里实际踩过的几类坑：
  * supervisor 掉到 0（评测没人管了）；
  * 孤儿 worker / 孤儿 thor（父进程死了还在跑，和新的抢 results.csv）；
  * vLLM 挂了或被杀（8b 卡就是因为显存互抢让 EngineCore 自己 OOM 崩的）；
  * 卡上显存剩余为 0（WM 感知头分配不到就会成批 env_error）；
  * 判定数十分钟不动（stalled）。

    python3 tools/monitor_status.py --loop 300

日志：/mnt/d/lightwm_out/status_all.log（告警行以 🚨 开头）
"""

from __future__ import annotations

import argparse
import os
import time

import paramiko

from newcard import CARDS

LOG = "/mnt/d/lightwm_out/status_all.log"

#: 卡 -> (run 名, 要不要看 vLLM)
CARD_RUNS = {
    "gpt5": ("wm_gpt5_ai2thor_fail250", False),
    "30b": ("wm_q30b_438_v2", True),
    "8b": ("wm_q8b_438_v2", True),
    "kimi": ("wm_kimi_438_v2", True),
}
LOCAL_RUN = "/home/sudidaren/spatialworld_eval/runs/wm_gemini31pro_fix_rest156"

PROBE = r"""
RUN="$1"; NEED_VLLM="$2"
echo -n "sup=";  pgrep -cf "[.]venv/bin/python -u - .* $RUN " || true
echo -n "worker=";  pgrep -cf "work.run_tas[k]" || true
echo -n "thor=";  pgrep -cf "thor-Linux6[4]" || true
echo -n "orphan="
n=0
for p in $(pgrep -f "work.run_tas[k]"); do
  pp=$(ps -o ppid= -p $p 2>/dev/null | tr -d ' ')
  [ "$pp" = "1" ] && n=$((n+1))
done
for p in $(pgrep -f "thor-Linux6[4]"); do
  pp=$(ps -o ppid= -p $p 2>/dev/null | tr -d ' ')
  if [ "$pp" = "1" ]; then n=$((n+1)); fi
done
echo "$n"
if [ "$NEED_VLLM" = "1" ]; then
  echo -n "vllm="; (curl -sf -m 6 http://127.0.0.1:8000/v1/models >/dev/null && echo up || echo DOWN)
else
  echo "vllm=n/a"
fi
echo -n "gpu_free_mib="; nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1
# instruction 里带逗号，必须用 csv 解析，不能 awk -F,
/root/miniconda3/bin/python - "/home/sudidaren/spatialworld_eval/runs/$RUN/results.csv" <<'PY'
import csv, sys
try:
    rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8-sig")))
except Exception:
    rows = []
print("rows=%d" % len(rows))
print("decided=%d" % sum(1 for r in rows if (r.get("Status") or "") not in ("", "pending")))
print("success=%d" % sum(1 for r in rows if r.get("Status") == "success"))
PY
echo -n "recent_err="; find "/home/sudidaren/spatialworld_eval/runs/$RUN" -name "run_error.txt" -newermt "-10 min" 2>/dev/null | wc -l
"""


def say(msg: str) -> None:
    line = f"[{time.strftime('%F %T')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def probe_card(tag: str) -> str:
    run, need_vllm = CARD_RUNS[tag]
    host, port, pwd = CARDS[tag]
    try:
        c = paramiko.SSHClient()
        c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        c.connect(host, port=port, username="root", password=pwd,
                  timeout=20, banner_timeout=40, auth_timeout=40)
        _in, out, _err = c.exec_command(
            f"bash -s -- {run} {1 if need_vllm else 0}", timeout=60)
        _in.write(PROBE)
        _in.channel.shutdown_write()
        txt = out.read().decode("utf-8", "replace")
        c.close()
    except Exception as exc:                       # noqa: BLE001
        return f"{tag}: UNREACHABLE {type(exc).__name__}"
    parts = {}
    for line in txt.splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            parts[k.strip()] = v.strip()
    warn = []
    if parts.get("sup") == "0":
        warn.append("supervisor 没了")
    if parts.get("orphan", "0") not in ("0", ""):
        warn.append(f"孤儿={parts['orphan']}")
    if need_vllm and parts.get("vllm") != "up":
        warn.append("vLLM 挂了")
    try:
        if int(parts.get("gpu_free_mib", "99999")) < 1500:
            warn.append(f"显存只剩{parts.get('gpu_free_mib')}MiB")
    except ValueError:
        pass
    try:
        if int(parts.get("recent_err", "0")) >= 8:
            warn.append(f"10分钟内{parts['recent_err']}个错误")
    except ValueError:
        pass
    flag = "🚨 " if warn else ""
    return (f"{tag}: {flag}sup={parts.get('sup')} worker={parts.get('worker')} "
            f"thor={parts.get('thor')} vllm={parts.get('vllm')} "
            f"gpu_free={parts.get('gpu_free_mib')}MiB "
            f"判定={parts.get('decided')}/{parts.get('rows')} 成功={parts.get('success')}"
            + (f" ⚠{'/'.join(warn)}" if warn else ""))


def probe_local() -> str:
    def sh(cmd: str) -> str:
        try:
            import subprocess
            return subprocess.run(["bash", "-c", cmd], capture_output=True,
                                  text=True, timeout=30).stdout.strip()
        except Exception:                          # noqa: BLE001
            return ""

    import csv
    f = os.path.join(LOCAL_RUN, "results.csv")
    try:
        with open(f, encoding="utf-8-sig") as fh:
            rs = list(csv.DictReader(fh))
    except Exception:                              # noqa: BLE001
        rs = []
    rows = str(len(rs))
    decided = str(sum(1 for r in rs
                      if (r.get("Status") or "") not in ("", "pending")))
    success = str(sum(1 for r in rs if r.get("Status") == "success"))
    worker = sh("pgrep -cf 'work.run_tas[k]'")
    load = os.getloadavg()[0]
    mem = {}
    for line in open("/proc/meminfo"):
        k = line.split(":")[0]
        if k in ("MemTotal", "MemAvailable"):
            mem[k] = int(line.split()[1]) / 1024 / 1024
    avail_pct = mem.get("MemAvailable", 0) / max(mem.get("MemTotal", 1), 1) * 100
    warn = []
    if worker == "0":
        warn.append("worker 全停")
    if load > os.cpu_count() * 0.95:
        warn.append(f"load {load:.1f} 过高")
    if avail_pct < 12:
        warn.append("内存不足")
    flag = "🚨 " if warn else ""
    return (f"local: {flag}worker={worker} 判定={decided}/{rows} 成功={success} "
            f"load={load:.1f}/{os.cpu_count()} 内存可用={mem.get('MemAvailable', 0):.1f}G"
            f"({avail_pct:.0f}%)" + (f" ⚠{'/'.join(warn)}" if warn else ""))


def once() -> None:
    say(" | ".join([probe_local()] + [probe_card(t) for t in CARD_RUNS]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", type=int, default=0, help="秒；0=只跑一轮")
    args = ap.parse_args()
    if not args.loop:
        once()
        return
    while True:
        try:
            once()
        except Exception as exc:                    # noqa: BLE001
            say(f"巡检出错: {type(exc).__name__}: {exc}")
        time.sleep(args.loop)


if __name__ == "__main__":
    main()
