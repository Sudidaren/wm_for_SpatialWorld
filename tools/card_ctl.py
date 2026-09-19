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
ORCH_SRC = Path(__file__).resolve().parents[2] / "spatialworld_eval/cloud_orchestrator_v4.sh"
ORCH_DST = "/home/sudidaren/spatialworld_eval/cloud_orchestrator_v4.sh"
RUN_ABL_SRC = Path(__file__).resolve().parents[2] / "spatialworld_eval/run_ablation.sh"
WM_PATCH_SRC = Path(__file__).resolve().parents[2] / "spatialworld_eval/wm_config_patch.py"
SW_ROOT = Path(__file__).resolve().parents[2] / "SpatialWorld"
REGISTRY_SRC = SW_ROOT / "mllm_base_agent/prompts/registry.py"
REGISTRY_DST = "/home/sudidaren/SpatialWorld/mllm_base_agent/prompts/registry.py"


def push_tools(sftp) -> None:
    """Ship the current control scripts to the card.

    These live outside the tarball payload, so a card provisioned yesterday
    would otherwise keep running yesterday's watchdog -- and today's bugs
    (a watchdog that kills supervisors, an orchestrator that cannot tell a
    worker from a supervisor) were exactly that.
    """
    here = Path(__file__).resolve().parent
    for tool in ("card_watchdog.sh", "card_run_queue.sh"):
        sftp.put(str(here / tool), f"/home/sudidaren/lightwm_phases/tools/{tool}")
    sftp.put(str(here / "card_keepalive.sh"), KEEPALIVE)
    if ORCH_SRC.exists():
        sftp.put(str(ORCH_SRC), ORCH_DST)
    if RUN_ABL_SRC.exists():
        sftp.put(str(RUN_ABL_SRC), "/home/sudidaren/spatialworld_eval/run_ablation.sh")
    if WM_PATCH_SRC.exists():
        sftp.put(str(WM_PATCH_SRC), "/home/sudidaren/spatialworld_eval/wm_config_patch.py")
    if REGISTRY_SRC.exists():
        # the same-information baseline (T5) needs the WM_PROMPT_EXTRA hook
        sftp.put(str(REGISTRY_SRC), REGISTRY_DST)


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
        # The keepalive must go first: it restarts the orchestrator within its
        # poll interval, and if it fires between "stop" and "write the new
        # queue" it resurrects the old one (that is how card D kept running
        # yesterday's arm list after a relaunch).
        'echo "--- stop keepalive ---"',
        'if [ -f /root/keepalive.pid ] && kill -0 "$(cat /root/keepalive.pid)" 2>/dev/null; then',
        '  echo "  keepalive pid=$(cat /root/keepalive.pid)"; '
        'kill -TERM "$(cat /root/keepalive.pid)" 2>/dev/null; fi',
        'echo "--- stop watchdog ---"',
        'for p in $(pgrep -f "[c]ard_watchdog.sh"); do echo "  watchdog pid=$p"; '
        'kill -TERM "$p" 2>/dev/null; done',
    ]
    if kill_orchestrator:
        lines += [
            'echo "--- stop orchestrator ---"',
            "ORCH=$(ps -eo pid=,args= | awk '/cloud_orchestrator_v4\\.sh [a-z0-9]/{print $1}')",
            'for p in $ORCH; do echo "  orchestrator pid=$p"; kill -TERM "$p" 2>/dev/null; done',
        ]
    lines += [
        "sleep 3",
        'for p in ${ORCH:-}; do kill -9 "$p" 2>/dev/null; done',
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
        '  if [ -f /root/keepalive.pid ] && kill -0 "$(cat /root/keepalive.pid)" 2>/dev/null; then',
        '    echo "keepalive started (pid $(cat /root/keepalive.pid))"',
        "  else",
        '    echo "keepalive FAILED"',
        "  fi",
        "fi",
        'echo "--- keepalive log ---"; tail -3 /root/autodl-tmp/logs/keepalive.log 2>/dev/null || echo none',
    ])


def smoke(tag: str) -> str:
    """The one check that would have caught the outage: run a real scene.

    Importing ai2thor is not enough -- the Linux64 build needs a live X11
    display with GLX, and a card without one fails every task in ~2 s while
    looking perfectly healthy from the outside.
    """
    lines = [
        "set -u",
        'export DISPLAY="${DISPLAY:-:99}"',
        "mkdir -p /root/autodl-tmp/logs",
        "if ! xdpyinfo >/dev/null 2>&1; then",
        '  setsid nohup Xvfb "$DISPLAY" -screen 0 1280x1024x24 '
        "> /root/autodl-tmp/logs/xvfb.log 2>&1 < /dev/null &",
        "  for _ in $(seq 1 20); do xdpyinfo >/dev/null 2>&1 && break; sleep 1; done",
        "fi",
        'xdpyinfo >/dev/null 2>&1 && echo "display ok" || echo "display BAD"',
        'VENV=/home/sudidaren/SpatialWorld/envs/ai2thor/.venv',
        '[ -x "$VENV/bin/python" ] || /root/miniconda3/bin/python -m venv '
        '--system-site-packages "$VENV"',
        '"$VENV/bin/python" - <<\'PY\'',
        "import ai2thor, time",
        "from ai2thor.controller import Controller",
        'print("ai2thor", ai2thor.__version__)',
        "t0 = time.time()",
        'c = Controller(scene="FloorPlan1", platform="Linux64", width=300, height=300,',
        "               server_timeout=300.0, start_unity_process=True)",
        'print("controller up in %.1fs" % (time.time() - t0))',
        'ev = c.step("Pass")',
        'print("frame ok:", ev.frame.shape)',
        "c.stop()",
        "PY",
    ]
    return "\n".join(lines)


def restart_watchdog(tag: str, interval: int) -> str:
    return "\n".join([
        "set -u",
        'export DISPLAY="${DISPLAY:-:99}"',
        "mkdir -p /root/autodl-tmp/logs",
        'OLD=$(ps -eo pid,args | awk \'/[c]ard_watchdog\\.sh/{print $1}\' | head -1)',
        'if [ -n "${OLD:-}" ]; then echo "stopping old watchdog pid=$OLD"; '
        'kill -TERM "$OLD" 2>/dev/null; sleep 3; kill -9 "$OLD" 2>/dev/null; fi',
        f'setsid nohup env RUN_NAME="" INTERVAL={interval} bash '
        f"{KEEPALIVE.replace('card_keepalive.sh', 'card_watchdog.sh')} "
        "> /root/autodl-tmp/logs/watchdog.boot.log 2>&1 < /dev/null &",
        "sleep 3",
        'if [ -f /root/watchdog.pid ] && kill -0 "$(cat /root/watchdog.pid)" 2>/dev/null; then',
        '  echo "watchdog ok (pid $(cat /root/watchdog.pid))"',
        "else",
        '  echo "watchdog FAILED"; tail -5 /root/autodl-tmp/logs/watchdog.log',
        "fi",
        'tail -2 /root/autodl-tmp/logs/watchdog.log',
    ])


def refresh_orchestrator(tag: str) -> str:
    """Replace the orchestrator script and restart just that process.

    The arm keeps running: ensure_batch() leaves a live supervisor alone, so
    this is a control-plane restart and costs nothing.  It cannot be done by
    overwriting the file under a running orchestrator -- bash reads a script
    incrementally, so the process has to be restarted to pick it up.
    """
    return "\n".join([
        "set -u",
        # awk with an escaped dot and a required trailing letter: this script
        # itself names the orchestrator, so a pgrep pattern would kill the
        # shell running this very command (it did, four times).
        "PIDS=$(ps -eo pid=,args= | awk '/cloud_orchestrator_v4\\.sh [a-z0-9]/{print $1}')",
        'for p in $PIDS; do echo "stop orchestrator $p"; kill -TERM "$p" 2>/dev/null; done',
        "sleep 3",
        'for p in $PIDS; do kill -9 "$p" 2>/dev/null; done',
        'bash -n ' + ORCH_DST + ' && echo "orchestrator syntax ok"',
        'if [ ! -f ' + ARM_LAUNCH + ' ]; then echo "no ' + ARM_LAUNCH + '" ; exit 1; fi',
        "setsid nohup bash " + ARM_LAUNCH + " > /root/orch_refresh.out 2>&1 < /dev/null &",
        "sleep 25",
        'tail -4 /root/orchestrator_v4.log',
        'echo "supervisor: $(pgrep -fc \'[p]ython -u - \' || true) processes"',
    ])


def restart_keepalive(tag: str) -> str:
    """Restart the keepalive process in place, keeping arm_launch.sh as is."""
    return "\n".join([
        "set -u",
        'if [ -f /root/keepalive.pid ] && kill -0 "$(cat /root/keepalive.pid)" 2>/dev/null; then',
        '  echo "stopping keepalive $(cat /root/keepalive.pid)"; '
        'kill -TERM "$(cat /root/keepalive.pid)" 2>/dev/null; sleep 3',
        "fi",
        'if [ ! -f ' + ARM_LAUNCH + ' ]; then echo "no ' + ARM_LAUNCH + ' -> not starting"; exit 1; fi',
        f"setsid nohup env LAUNCH={ARM_LAUNCH} INTERVAL=90 MAX_RESTARTS=12 bash {KEEPALIVE} "
        "> /root/autodl-tmp/logs/keepalive.boot.log 2>&1 < /dev/null &",
        "sleep 3",
        'if [ -f /root/keepalive.pid ] && kill -0 "$(cat /root/keepalive.pid)" 2>/dev/null; then',
        '  echo "keepalive ok (pid $(cat /root/keepalive.pid))"',
        "else",
        '  echo "keepalive FAILED"',
        "fi",
        'tail -2 /root/autodl-tmp/logs/keepalive.log 2>/dev/null',
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
        'if [ -f /root/watchdog.pid ] && kill -0 "$(cat /root/watchdog.pid)" 2>/dev/null; then',
        '  echo "  already running (pid $(cat /root/watchdog.pid))"',
        "else",
        '  mkdir -p /root/autodl-tmp/logs',
        f'  setsid nohup env RUN_NAME="" INTERVAL={watchdog_interval} DISPLAY="$DISPLAY" \\',
        '    bash /home/sudidaren/lightwm_phases/tools/card_watchdog.sh \\',
        '    > /root/autodl-tmp/logs/watchdog.boot.log 2>&1 < /dev/null &',
        '  sleep 3',
        'fi',
        'if [ -f /root/watchdog.pid ] && kill -0 "$(cat /root/watchdog.pid)" 2>/dev/null; then',
        '  echo "  watchdog ok (pid $(cat /root/watchdog.pid))"',
        "else",
        '  echo "  watchdog FAILED"',
        "fi",
        'echo "--- keepalive ---"',
        # pidfile again: this script's own text contains the keepalive's path,
        # so a pgrep here always finds itself and the keepalive never starts.
        'if [ -f /root/keepalive.pid ] && kill -0 "$(cat /root/keepalive.pid)" 2>/dev/null; then',
        '  echo "  already running (pid $(cat /root/keepalive.pid))"',
        "else",
        f'  setsid nohup env LAUNCH={ARM_LAUNCH} INTERVAL=90 MAX_RESTARTS=12 \\',
        f'    bash {KEEPALIVE} > /root/autodl-tmp/logs/keepalive.boot.log 2>&1 < /dev/null &',
        '  sleep 2',
        'fi',
        'if [ -f /root/keepalive.pid ] && kill -0 "$(cat /root/keepalive.pid)" 2>/dev/null; then',
        '  echo "  keepalive ok (pid $(cat /root/keepalive.pid))"',
        "else",
        '  echo "  keepalive FAILED"',
        "fi",
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
                    choices=["probe", "stop", "launch", "stop-helpers", "ensure-keepalive",
                             "smoke", "restart-watchdog", "refresh-orchestrator",
                             "restart-keepalive"])
    ap.add_argument("card")
    ap.add_argument("--run", help="run name (stop)")
    ap.add_argument("--arm", action="append", default=[],
                    help='arm spec "model|url|plan|profile|run_name|workers" (repeatable)')
    ap.add_argument("--depth-source", default=None)
    ap.add_argument("--da2-scale", default=None)
    ap.add_argument("--scenes", default=None)
    ap.add_argument("--total", default=None)
    ap.add_argument("--archive", default=None, help="run dir name to move aside first")
    ap.add_argument("--env", action="append", default=[],
                    help="extra KEY=VALUE in the arm's environment (repeatable)")
    ap.add_argument("--watchdog-interval", type=int, default=30)
    ap.add_argument("--keep-orchestrator", action="store_true",
                    help="stop: leave the orchestrator alive (it will restart the arm)")
    args = ap.parse_args()

    if args.card not in CARDS:
        ap.error(f"card must be one of {list(CARDS)}")

    if args.action == "probe":
        script = probes(args.card)
    elif args.action == "smoke":
        script = smoke(args.card)
    elif args.action == "restart-watchdog":
        c = connect(args.card)
        try:
            sftp = c.open_sftp()
            sftp.put(str(Path(__file__).with_name("card_watchdog.sh")),
                     "/home/sudidaren/lightwm_phases/tools/card_watchdog.sh")
            sftp.close()
        finally:
            c.close()
        script = restart_watchdog(args.card, args.watchdog_interval)
    elif args.action == "refresh-orchestrator":
        c = connect(args.card)
        try:
            sftp = c.open_sftp()
            push_tools(sftp)
            sftp.close()
        finally:
            c.close()
        script = refresh_orchestrator(args.card)
    elif args.action == "restart-keepalive":
        c = connect(args.card)
        try:
            sftp = c.open_sftp()
            sftp.put(str(Path(__file__).with_name("card_keepalive.sh")), KEEPALIVE)
            sftp.close()
        finally:
            c.close()
        script = restart_keepalive(args.card)
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
        for kv in args.env:
            k, _, v = kv.partition("=")
            if k:
                env_pairs[k] = v
        if "VLLM_NAME" in env_pairs and "WAIT_MODEL" not in env_pairs:
            # the orchestrator refuses to start a stage until this model is
            # actually being served, so a slow vLLM cannot turn a batch into
            # 311 model failures
            env_pairs["WAIT_MODEL"] = env_pairs["VLLM_NAME"]
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
        for kv in args.env:
            k, _, v = kv.partition("=")
            if k:
                env_pairs[k] = v
        # the keepalive needs the relaunch script and the keepalive itself on
        # the card *before* the boot script starts them
        c = connect(args.card)
        try:
            sftp = c.open_sftp()
            push_tools(sftp)
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
