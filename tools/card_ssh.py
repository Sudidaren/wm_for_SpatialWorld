#!/usr/bin/env python3
"""Run a shell command on one named evaluation card, with connection retries.

    python3 tools/card_ssh.py D "nvidia-smi"
    python3 tools/card_ssh.py D --file probe.sh
    python3 tools/card_ssh.py --all "uptime"

Direct paramiko connects to these boxes time out often enough that a single
attempt is not a usable building block for monitoring, so every call retries.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import paramiko

CARDS = {
    "A": ("connect.weste.seetacloud.com", 15868, "Yl3WnjEDllk+"),
    "B": ("connect.westb.seetacloud.com", 52296, "hxeahvC88uMM"),
    "C": ("connect.cqa1.seetacloud.com", 47531, "0yy1HhF4KfYV"),
    "D": ("connect.weste.seetacloud.com", 32808, "kQLLhVht3BEB"),
    "E": ("connect.westb.seetacloud.com", 22416, "cZmriU8VEh+F"),
}


def connect(tag: str, retries: int = 6):
    host, port, pw = CARDS[tag]
    last: Exception | None = None
    for i in range(retries):
        try:
            c = paramiko.SSHClient()
            c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            c.connect(host, port=port, username="root", password=pw,
                      timeout=30, banner_timeout=60, auth_timeout=60)
            return c
        except Exception as exc:  # noqa: BLE001 - retry on anything
            last = exc
            print(f"  connect {tag} failed {i + 1}/{retries}: {type(exc).__name__}",
                  file=sys.stderr, flush=True)
            time.sleep(6)
    raise last  # type: ignore[misc]


def run(tag: str, script: str, timeout: float = 300.0) -> tuple[int, str, str]:
    c = connect(tag)
    try:
        _, out, err = c.exec_command(script, timeout=timeout, get_pty=False)
        o = out.read().decode(errors="replace")
        e = err.read().decode(errors="replace")
        rc = out.channel.recv_exit_status()
        return rc, o, e
    finally:
        c.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("card", nargs="?", help="A|B|C|D|E")
    ap.add_argument("command", nargs="?", help="shell command")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--file", help="read the command from a local file")
    ap.add_argument("--upload", action="append", default=[],
                    metavar="LOCAL:REMOTE", help="sftp a file up before running")
    ap.add_argument("--timeout", type=float, default=300.0)
    args = ap.parse_args()

    script = Path(args.file).read_text() if args.file else args.command
    if args.upload and not script:
        script = "true"
    if not script:
        ap.error("give a command, --file, or --all with a command")

    tags = list(CARDS) if args.all else [args.card]
    rc_all = 0
    for tag in tags:
        print(f"===== {tag} =====", flush=True)
        try:
            for spec in args.upload:
                local, _, remote = spec.partition(":")
                c = connect(tag)
                try:
                    sftp = c.open_sftp()
                    sftp.put(local, remote)
                    sftp.close()
                    print(f"  uploaded {local} -> {remote}", flush=True)
                finally:
                    c.close()
            rc, o, e = run(tag, script, args.timeout)
        except Exception as exc:  # noqa: BLE001
            print(f"  UNREACHABLE {type(exc).__name__}: {exc}")
            rc_all = 1
            continue
        sys.stdout.write(o)
        if e.strip():
            print(f"--- stderr ---\n{e}", end="")
        print(f"--- rc={rc} ---")
        rc_all |= rc
    return rc_all


if __name__ == "__main__":
    raise SystemExit(main())
