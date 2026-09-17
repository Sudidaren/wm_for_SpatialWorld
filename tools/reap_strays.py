#!/usr/bin/env python3
"""Reap orphaned AI2-THOR worker / Unity processes so they cannot accumulate.

Why this exists: when a batch is stopped (or a supervisor dies), the child
``run_task`` python process and the Unity process it spawned are re-parented to
PID 1 and keep burning CPU for hours.  Killing only the batch chain is not
enough -- they have to be reaped.

Rules (conservative, never touches a live task):

  * a process whose command line mentions ``thor-`` (the Unity player) or
    ``scripts.*.work.run_task`` (the per-task worker) is a candidate;
  * it is killed if its parent is PID 1 (its supervisor is gone), or if it has
    been alive longer than ``--max-age`` hours (a single task never takes that
    long);
  * the reaper never kills itself or its own process group.

Usage:
  python3 tools/reap_strays.py --once           # one pass, print what it does
  python3 tools/reap_strays.py --loop --interval 60 --log /root/reaper.log
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time

MATCHES = ("thor-", "work.run_task")

#: Commands that legitimately live for hours (a whole collection run or a batch)
#: -- their children are per-task/per-house processes and must be judged by age.
PARENT_ENTRYPOINTS = ("work.run_task", "collect_", "run_ablation", "train_")


def uptime_seconds() -> float:
    with open("/proc/uptime") as fh:
        return float(fh.read().split()[0])


def iter_processes():
    """Yield (pid, ppid, state, age_seconds, cmdline)."""
    hz = os.sysconf("SC_CLK_TCK")
    up = uptime_seconds()
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        try:
            with open(f"/proc/{pid}/stat", "rb") as fh:
                stat = fh.read().decode(errors="ignore")
            with open(f"/proc/{pid}/cmdline", "rb") as fh:
                cmd = fh.read().decode(errors="ignore").replace("\0", " ").strip()
        except OSError:
            continue
        rp = stat.rfind(")")
        if rp < 0:
            continue
        fields = stat[rp + 2:].split()
        if len(fields) < 20:
            continue
        state, ppid, starttime = fields[0], int(fields[1]), int(fields[19])
        age = max(0.0, up - starttime / hz)
        yield pid, ppid, state, age, cmd


def reap_once(max_age_h, dry_run: bool, log, child_age_h=None):
    """child_age_h: age limit for a child of a live long-running parent.

    A data-collection script keeps one *per-house* Unity process alive at a
    time; a house never takes ``child_age_h`` hours, so a child older than that
    is hung even though its parent is still running.
    """
    child_limit = (child_age_h if child_age_h is not None else max_age_h) * 3600
    killed = []
    me = os.getpid()
    my_group = os.getpgid(0)
    # parent pid -> (cmdline, age)
    parents = {}
    for pid, ppid, state, age, cmd in iter_processes():
        parents[pid] = (cmd, age)
    for pid, ppid, state, age, cmd in iter_processes():
        if pid == me or not cmd:
            continue
        if not any(m in cmd for m in MATCHES):
            continue
        try:
            if os.getpgid(pid) == my_group:
                continue                      # never kill our own group
        except OSError:
            continue
        pcmd, _page = parents.get(ppid, ("", 0.0))
        # A healthy Unity/worker always has one of OUR python scripts as its
        # parent.  When that script dies the child is adopted by init -- which
        # is PID 1 *or* /init (PID != 1) inside a container, so testing
        # ``ppid == 1`` alone misses exactly the cases we care about.
        parent_is_ours = ("python" in pcmd
                          and any(m in pcmd for m in PARENT_ENTRYPOINTS))
        reason = None
        if not parent_is_ours:
            reason = f"orphan (parent is {pcmd.split()[0][:32] if pcmd else 'gone'})"
        elif age > max_age_h * 3600:
            reason = f"stale ({age/3600:.1f}h > {max_age_h}h)"
        elif age > child_limit:
            reason = (f"child of a live {pcmd.split()[-1][:24]} older than "
                      f"{child_limit/3600:.1f}h")
        if reason is None:
            continue
        short = " ".join(cmd.split())[:90]
        msg = f"[reap] pid={pid} {reason} age={age/3600:.1f}h  {short}"
        print(msg, flush=True)
        if log:
            log.write(msg + "\n")
            log.flush()
        if not dry_run:
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass
        killed.append(pid)
    if not dry_run:
        time.sleep(5)
        for pid in killed:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
    return killed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--max-age", type=float, default=3.0,
                    help="hours; a single task never runs longer than this")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--log", default="")
    args = ap.parse_args()

    log = open(args.log, "a", buffering=1) if args.log else None
    if not args.loop:
        killed = reap_once(args.max_age, args.dry_run, log)
        print(f"[reap] candidates killed: {len(killed)}")
        return 0
    print(f"[reap] watching every {args.interval}s, max-age {args.max_age}h",
          flush=True)
    while True:
        try:
            reap_once(args.max_age, args.dry_run, log)
        except Exception as exc:                        # keep the loop alive
            print(f"[reap] error: {exc}", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
