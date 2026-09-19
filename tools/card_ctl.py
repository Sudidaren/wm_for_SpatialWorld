#!/usr/bin/env python3
"""Launch or stop one evaluation arm on a card, through one audited entry point.

Why this exists: every arm we lost today was lost to a hand-typed remote
command -- a missing DISPLAY, a missing watchdog, a stale RUN_NAME.  Launching
is therefore one command here, and it always does the same four things:

  1. confirm Xvfb (and start the keep-alive watchdog if it is not running)
  2. confirm vLLM is serving the model the arm asks for
  3. archive any run directory it is about to reuse
  4. hand the queue to cloud_orchestrator_v4.sh via tools/card_run_queue.sh

    python3 tools/card_ctl.py probe D
    python3 tools/card_ctl.py stop  D --run wm_q30_da2_a311
    python3 tools/card_ctl.py launch D \
        --depth-source da2 --da2-scale 1.3816 --scenes ai2thor --total 311 \
        --arm "qwen3vl-30b|http://127.0.0.1:8000/v1|off|wm|wm_q30_da2_a311|10" \
        --arm ";;" \
        --arm "qwen3vl-30b|http://127.0.0.1:8000/v1|off|llm|base_q30_a311|10"
"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

from card_ssh import CARDS, connect, run  # noqa: PLC2701 - sibling tool

KEEPALIVE = "/home/sudidaren/lightwm_phases/tools/card_keepalive.sh"
ARM_LAUNCH = "/root/arm_launch.sh"


def sh(parts: list[str]) -> str:
    return " ".join(shlex.quote(p) for p in parts)


def stop_run(tag: str, run_name: str, kill_orchestrator: bool) -> str:
    """Stop one arm by RUN_NAME (from /proc/<pid>/environ), never by pattern."""
    lines = [
        "set -u",
        f'RUN={shlex.quote(run_name)}',
        'for p in $(pgrep -f "[p]ython -u -"); do',
        '  if tr "\\0" "\\n" < /proc/$p/environ 2>/dev/null | grep -qx "RUN_NAME=$RUN"; then',
        '    echo "  supervisor pid=$p -> kill tree"',
        '    for c in $(pgrep -P "$p"); do',
        '      for g in $(pgrep -P "$c"); do kill -9 "$g" 2>/dev/null; done',
        '      kill -9 "$c" 2>/dev/null',
        '    done',
        '    kill -9 "$p" 2>/dev/null',
        '  fi',
        'done',
        'for p in $(pgrep -f "[r]un_ablation.sh"); do kill -TERM "$p" 2>/dev/null; done',
    ]
    if kill_orchestrator:
        lines += ['for p in $(pgrep -f "[c]loud_orchestrator_v4"); do kill -TERM "$p" 2>/dev/null; done']
    lines += [
        "sleep 5",
        'for p in $(pgrep -f "[t]hor-Linux64"); do kill -9 "$p" 2>/dev/null; done',
        "sleep 2",
        'echo "  leftover workers: $(pgrep -fc \'[p]ython -u -\' || true)"',
        'echo "  leftover unity:   $(pgrep -fc \'[t]hor-Linux64\' || true)"',
        'echo "  zombies:          $(ps -eo stat= | grep -c \'^Z\')"',
    ]
    return "\n".join(lines)


def stop_helpers(tag: str, kill_orchestrator: bool) -> str:
    """Take down the control plane (watchdog + orchestrator) and dead work.

    Nothing here matches by name pattern for the *harness*: the watchdog and
    the orchestrator are found by the path of their own script, and the
    leftover workers are found by the fact that their supervisor is gone.
    """
    lines = [
        "set -u",
        'echo "--- stop watchdog ---"',
        'for p in $(pgrep -f "[c]ard_watchdog.sh"); do echo "  watchdog pid=$p"; '
        'kill -TERM "$p" 2>/dev/null; done',
    ]
    if kill_orchestrator:
        lines += [
            'echo "--- stop orchestrator ---"',
            'for p in $(pgrep -f "[c]loud_orchestrator_v4"); do echo "  orchestrator pid=$p"; '
            'kill -TERM "$p" 2>/dev/null; done',
        ]
    lines += [
        "sleep 3",
        'for p in $(pgrep -f "[c]loud_orchestrator_v4"); do kill -9 "$p" 2>/dev/null; done',
        'for p in $(pgrep -f "[c]ard_watchdog.sh"); do kill -9 "$p" 2>/dev/null; done',
        'echo "--- reap orphaned workers/unity ---"',
        'for p in $(pgrep -f "[r]un_task" 2>/dev/null); do',
        '  ppid=$(awk \'{print $4}\' "/proc/$p/stat" 2>/dev/null)',
        '  if [ -z "$ppid" ] || [ "$ppid" = "1" ]; then echo "  orphan run_task pid=$p"; kill -9 "$p" 2>/dev/null; fi',
        'done',
        'sleep 2',
        'for p in $(pgrep -f "[t]hor-Linux64" 2>/dev/null); do kill -9 "$p" 2>/dev/null; done',
        'sleep 1',
        'echo "  run_task left: $(pgrep -fc \'[r]un_task\' || true)"',
        'echo "  unity left:    $(pgrep -fc \'[t]hor-Linux64\' || true)"',
        'echo "  zombies:       $(ps -eo stat= | grep -c \'^Z\')"',
    ]
    return "\n".join(lines)


def probes(tag: str) -> str:
    return "\n".join([
        'echo "--- clock ---"; date',
        'echo "--- xvfb ---"; (pgrep -af "[X]vfb" || echo none)',
        'echo "--- display usable ---"; '
        '(DISPLAY=${DISPLAY:-:99} timeout 10 xdpyinfo >/dev/null 2>&1 && echo ok || echo BAD)',
        'echo "--- vllm ---"; (curl -sf -m 6 http://127.0.0.1:8000/v1/models || echo down)',
        'echo "--- watchdog ---"; (pgrep -af "[c]ard_watchdog" || echo down)',
        'echo "--- orchestrator ---"; (pgrep -af "[c]loud_orchestrator_v4" || echo down)',
        'echo "--- arms running ---"; for p in $(pgrep -f "[p]ython -u -"); do '
        'tr "\\0" "\\n" < /proc/$p/environ 2>/dev/null | sed -n "s/^RUN_NAME=/   arm=/p"; done',
        'echo "--- zombies ---"; ps -eo stat= | grep -c "^Z"',
        'echo "--- mem ---"; free -g | awk "/^Mem:/{print \\$7\\"G available\\"}"',
        'echo "--- gpu ---"; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader',
        'echo "--- disk ---"; df -h /root/autodl-tmp | tail -1',
        'echo "--- models ---"; du -sh /root/autodl-tmp/models/* 2>/dev/null || echo none',
        'echo "--- watchdog log ---"; tail -3 /root/autodl-tmp/logs/watchdog.log 2>/dev/null || echo none',
        'echo "--- orchestrator log ---"; tail -6 /root/orchestrator_v4.log 2>/dev/null || echo none',
    ])


def ensure_keepalive(tag: str) -> str:
    return "\n".join([
        "set -u",
        'echo "--- arm list ---"; cat /root/keepalive.arms /root/keepalive.total 2>/dev/null',
        # pidfile, not pgrep: any script that mentions the keepalive's own path
        # makes pgrep match its own command line (this bit us twice today).
        'if [ -f /root/keepalive.pid ] && kill -0 "$(cat /root/keepalive.pid)" 2>/dev/null; then',
        '  echo "keepalive already running (pid $(cat /root/keepalive.pid))"',
        "else",
        "  setsid nohup env LAUNCH=" + ARM_LAUNCH + " INTERVAL=90 MAX_RESTARTS=12 \\",
        "    bash " + KEEPALIVE + " > /root/autodl-tmp/logs/keepalive.boot.log 2>&1 < /dev/null &",
        "  sleep 2",
        '  pgrep -f "[c]ard_keepalive.sh" >/dev/null && echo "keepalive started" || echo "keepalive FAILED"',
        "fi",
        'echo "--- keepalive log ---"; tail -3 /root/autodl-tmp/logs/keepalive.log 2>/dev/null || echo none',
    ])


def launch(tag: str, arms: list[str], env_pairs: dict[str, str],
           archive: str | None, watchdog_interval: int) -> str:
    lines = ["set -u"]
    lines.append('export DISPLAY="${DISPLAY:-:99}"')
    for k, v in env_pairs.items():
        if v is not None:
            lines.append(f"export {k}={shlex.quote(v)}")

    if archive:
        # never delete a run: the errored batch is evidence the paper cites
        lines += [
            f'AR=/root/autodl-tmp/runs_local/{shlex.quote(archive)}',
            'if [ -d "$AR" ]; then',
            '  n=1; while [ -e "${AR}_errored_$n" ]; do n=$((n+1)); done',
            '  mv "$AR" "${AR}_errored_$n" && echo "  archived $AR -> ${AR}_errored_$n"',
            'fi',
        ]

    lines += [
        'echo "--- arm list for keepalive ---"',
        "cat > /root/keepalive.arms <<'ARMEOF'",
        *[a.split("|")[-2] for a in arms if a != ";;"],
        "ARMEOF",
        f"echo {env_pairs.get('TOTAL', '438')} > /root/keepalive.total",
        "cat /root/keepalive.arms /root/keepalive.total",
        'echo "--- watchdog ---"',
        'if ! pgrep -f "[c]ard_watchdog.sh" >/dev/null; then',
        '  mkdir -p /root/autodl-tmp/logs',
        f'  setsid nohup env RUN_NAME="" INTERVAL={watchdog_interval} DISPLAY="$DISPLAY" \\',
        '    bash /home/sudidaren/lightwm_phases/tools/card_watchdog.sh \\',
        '    > /root/autodl-tmp/logs/watchdog.boot.log 2>&1 < /dev/null &',
        '  sleep 3',
        'fi',
        'pgrep -f "[c]ard_watchdog.sh" >/dev/null && echo "  watchdog ok" || echo "  watchdog FAILED"',
        'echo "--- keepalive ---"',
        'if ! pgrep -f "[c]ard_keepalive.sh" >/dev/null; then',
        f'  setsid nohup env LAUNCH={ARM_LAUNCH} INTERVAL=90 MAX_RESTARTS=12 \\',
        f'    bash {KEEPALIVE} > /root/autodl-tmp/logs/keepalive.boot.log 2>&1 < /dev/null &',
        '  sleep 2',
        'fi',
        'pgrep -f "[c]ard_keepalive.sh" >/dev/null && echo "  keepalive ok" || echo "  keepalive FAILED"',
        'echo "--- queue ---"',
        f'bash {ARM_LAUNCH}',
    ]
    return "\n".join(lines)


def arm_launch_text(arms: list[str], env_pairs: dict[str, str]) -> str:
    """The relaunch script: keepalive re-runs exactly this, nothing else."""
    body = ["#!/usr/bin/env bash",
            "# Written by tools/card_ctl.py.  Re-run by tools/card_keepalive.sh",
            "# when the orchestrator is gone; safe to run by hand too.",
            "set -u",
            'export DISPLAY="${DISPLAY:-:99}"']
    body += [f"export {k}={shlex.quote(v)}" for k, v in env_pairs.items()]
    body.append("cd /home/sudidaren/lightwm_phases || exit 1")
    body.append("bash tools/card_run_queue.sh " + " ".join(shlex.quote(a) for a in arms))
    return "\n".join(body) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action",
                    choices=["probe", "stop", "launch", "stop-helpers", "ensure-keepalive"])
    ap.add_argument("card")
    ap.add_argument("--run", help="run name (stop)")
    ap.add_argument("--arm", action="append", default=[],
                    help='arm spec "model|url|plan|profile|run_name|workers" (repeatable)')
    ap.add_argument("--depth-source", default=None)
    ap.add_argument("--da2-scale", default=None)
    ap.add_argument("--scenes", default=None)
    ap.add_argument("--total", default=None)
    ap.add_argument("--archive", default=None, help="run dir name to move aside first")
    ap.add_argument("--watchdog-interval", type=int, default=30)
    ap.add_argument("--keep-orchestrator", action="store_true",
                    help="stop: leave the orchestrator alive (it will restart the arm)")
    args = ap.parse_args()

    if args.card not in CARDS:
        ap.error(f"card must be one of {list(CARDS)}")

    if args.action == "probe":
        script = probes(args.card)
    elif args.action == "ensure-keepalive":
        if not args.arm:
            ap.error("ensure-keepalive needs the --arm list the keepalive should re-run")
        env_pairs = {"RUNS_DIR": "/root/autodl-tmp/runs_local"}
        if args.depth_source:
            env_pairs["LIGHTWM_DEPTH_SOURCE"] = args.depth_source
        if args.da2_scale:
            env_pairs["LIGHTWM_DA2_SCALE"] = args.da2_scale
        if args.scenes:
            env_pairs["SCENES"] = args.scenes
        if args.total:
            env_pairs["TOTAL"] = args.total
        c = connect(args.card)
        try:
            sftp = c.open_sftp()
            sftp.put(str(Path(__file__).with_name("card_keepalive.sh")), KEEPALIVE)
            with sftp.open(ARM_LAUNCH, "w") as fh:
                fh.write(arm_launch_text(args.arm, env_pairs))
            with sftp.open("/root/keepalive.arms", "w") as fh:
                fh.write("\n".join(a.split("|")[-2] for a in args.arm if a != ";;") + "\n")
            with sftp.open("/root/keepalive.total", "w") as fh:
                fh.write(env_pairs.get("TOTAL", "438") + "\n")
            sftp.close()
        finally:
            c.close()
        script = ensure_keepalive(args.card)
    elif args.action == "stop-helpers":
        script = stop_helpers(args.card, kill_orchestrator=not args.keep_orchestrator)
    elif args.action == "stop":
        if not args.run:
            ap.error("stop needs --run")
        script = stop_run(args.card, args.run, kill_orchestrator=not args.keep_orchestrator)
    else:
        if not args.arm:
            ap.error("launch needs at least one --arm")
        env_pairs = {}
        if args.depth_source:
            env_pairs["LIGHTWM_DEPTH_SOURCE"] = args.depth_source
        if args.da2_scale:
            env_pairs["LIGHTWM_DA2_SCALE"] = args.da2_scale
        if args.scenes:
            env_pairs["SCENES"] = args.scenes
        if args.total:
            env_pairs["TOTAL"] = args.total
        env_pairs["RUNS_DIR"] = "/root/autodl-tmp/runs_local"
        # the keepalive needs the relaunch script and the keepalive itself on
        # the card *before* the boot script starts them
        c = connect(args.card)
        try:
            sftp = c.open_sftp()
            sftp.put(str(Path(__file__).with_name("card_keepalive.sh")), KEEPALIVE)
            with sftp.open(ARM_LAUNCH, "w") as fh:
                fh.write(arm_launch_text(args.arm, env_pairs))
            sftp.close()
        finally:
            c.close()
        script = launch(args.card, args.arm, env_pairs, args.archive, args.watchdog_interval)

    print(f"===== {args.card} : {args.action} =====", flush=True)
    rc, out, err = run(args.card, script, timeout=1800)
    sys.stdout.write(out)
    if err.strip():
        print(f"--- stderr ---\n{err}", end="")
    print(f"--- rc={rc} ---")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
