#!/usr/bin/env python3
"""Poll every evaluation card and append one health line per card.

Run it in a loop on the workstation; it is the thing that notices a dead
orchestrator, a stalled arm, a growing zombie count or an out-of-memory box
*before* the batch is lost.  Everything it prints comes from the card itself
(no cached state), so a card that stops answering shows up as "unreachable".

    python3 tools/monitor_cards.py            # one pass
    python3 tools/monitor_cards.py --loop 300 # every 5 min, append to the log
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path

import paramiko

sys_cards = {
    "A": ("connect.weste.seetacloud.com", 15868, "Yl3WnjEDllk+"),
    "B": ("connect.westb.seetacloud.com", 52296, "hxeahvC88uMM"),
    "C": ("connect.cqa1.seetacloud.com", 47531, "0yy1HhF4KfYV"),
    "D": ("connect.weste.seetacloud.com", 32808, "kQLLhVht3BEB"),
    "E": ("connect.westb.seetacloud.com", 22416, "cZmriU8VEh+F"),
}
LOG = Path("/mnt/d/lightwm_out/cards_monitor.log")

PROBE = r"""
echo -n "vllm=";    (curl -sf -m 6 http://127.0.0.1:8000/v1/models >/dev/null && echo up || echo down)
echo -n "orch=";    (pgrep -f "[c]loud_orchestrator" >/dev/null && echo up || echo down)
echo -n "wd=";      (pgrep -f "[c]ard_watchdog" >/dev/null && echo up || echo down)
echo -n "ka=";      (if [ -f /root/keepalive.pid ] && kill -0 "$(cat /root/keepalive.pid)" 2>/dev/null; then echo up; else echo down; fi)
echo -n "xvfb=";    (pgrep -f "[X]vfb :99" >/dev/null && echo up || echo down)
echo -n "sup=";     python3 - <<'PYEOF' 2>/dev/null || echo none
import glob, os
# Classify by argv, not by substring: this probe's own command line contains
# every string it looks for, so a substring match finds itself.  The harness
# throws the interpreter "-u -" (stdin program); a worker adds "-m ...run_task".
arm, workers = None, 0
for d in glob.glob('/proc/[0-9]*'):
    try:
        argv = open(os.path.join(d, 'cmdline'), 'rb').read().decode(errors='replace').split('\0')
    except Exception:
        continue
    argv = [a for a in argv if a]
    if not argv or 'python' not in os.path.basename(argv[0]):
        continue
    if '-m' in argv[1:3] and any('run_task' in a for a in argv):
        workers += 1
        continue
    if argv[1:3] != ['-u', '-']:
        continue
    try:
        env = open(os.path.join(d, 'environ'), 'rb').read().decode(errors='replace')
    except Exception:
        continue
    for kv in env.split('\0'):
        if kv.startswith('RUN_NAME='):
            arm = kv.split('=', 1)[1]
print(f"{arm or 'none'}/w{workers}")
PYEOF
echo -n "tasks=";   python3 - <<'PYEOF' 2>/dev/null || echo 0
import glob, json, os, time
# only runs touched in the last 12 h are "active": the boxes also hold the
# historical batches the paper cites, and mixing them into one counter hid
# both a stalled arm and a finished one.
now = time.time()
try:
    TOTAL = int(open('/root/keepalive.total').read().strip())
except Exception:
    TOTAL = 0
parts = []
for f in glob.glob('/root/autodl-tmp/runs_local/*/state.json'):
    try:
        if now - os.path.getmtime(f) > 12 * 3600:
            continue
        ts = json.load(open(f)).get('tasks', [])
    except Exception:
        continue
    if not ts:
        continue
    dec = sum(1 for t in ts
              if str(t.get('completed', '')).lower() in ('true', 'false')
              or (t.get('status') == 'failed_external' and int(t.get('attempts') or 0) >= 3))
    suc = sum(1 for t in ts if str(t.get('success', '')).lower() == 'true')
    parts.append(f"{os.path.basename(os.path.dirname(f))}={dec}/{TOTAL or len(ts)}(ok{suc})")
newest = 0
for d in glob.glob('/root/autodl-tmp/runs_local/*'):
    try:
        if now - os.path.getmtime(d) > 12 * 3600:
            continue
    except OSError:
        continue
    for root, dirs, files in os.walk(d):
        if len(dirs) > 40:
            dirs[:] = dirs[:40]
        for fn in files:
            try:
                newest = max(newest, os.path.getmtime(os.path.join(root, fn)))
            except OSError:
                pass
stall = int(now - newest) if newest else 999999
print(("stall=%ds " % stall if parts else "idle ") + " ".join(parts))
PYEOF
echo -n "zombie="; ps -eo stat= | grep -c '^Z' || true
echo -n "orphan="; python3 - <<'PYEOF' 2>/dev/null || echo 0
import glob, os
n = 0
for d in glob.glob('/proc/[0-9]*'):
    try:
        argv = [a for a in open(os.path.join(d, 'cmdline'), 'rb').read().decode(errors='replace').split('\0') if a]
        stat = open(os.path.join(d, 'stat')).read()
    except Exception:
        continue
    if not argv or 'python' not in os.path.basename(argv[0]):
        continue
    if '-m' not in argv[1:3] or not any('run_task' in a for a in argv):
        continue
    ppid = stat.rsplit(')', 1)[1].split()[1]
    if ppid == '1':
        n += 1
print(n)
PYEOF
echo -n "mem=";    free -g | awk '/^Mem:/{print $7}'
echo -n "disk=";   df -h /root/autodl-tmp | awk 'NR==2{print $4" free ("$5" used)"}'
echo -n "gpu=";    nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | head -1 | tr -d ' '
echo -n "load=";   cut -d' ' -f1 /proc/loadavg
"""


def probe(tag: str) -> str:
    host, port, pw = sys_cards[tag]
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        c.connect(host, port=port, username="root", password=pw, timeout=20)
        _, out, _ = c.exec_command(PROBE, timeout=90)
        res = out.read().decode(errors="replace")
        return " ".join(res.split())
    except Exception as exc:
        return f"UNREACHABLE {type(exc).__name__}"
    finally:
        c.close()


def one_pass() -> str:
    stamp = dt.datetime.now().strftime("%F %T")
    lines = [f"[{stamp}]"]
    for tag in sys_cards:
        lines.append(f"  {tag}: {probe(tag)}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", type=int, default=0, help="seconds between passes")
    args = ap.parse_args()
    LOG.parent.mkdir(parents=True, exist_ok=True)
    while True:
        block = one_pass()
        print(block, flush=True)
        with LOG.open("a", encoding="utf-8") as fh:
            fh.write(block + "\n")
        if not args.loop:
            return 0
        time.sleep(args.loop)


if __name__ == "__main__":
    raise SystemExit(main())
