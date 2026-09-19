#!/usr/bin/env python3
"""Card-to-card provisioning: use one ready card to prepare the others.

Uploads ``tools/c2c_provision_remote.py`` to a card that already has the
evaluation layout, then runs it there.  The remote side streams its own
progress, which this driver prints line by line, so a stalled transfer is
visible instead of looking like a hang.

    python3 tools/c2c_provision.py --source D --targets A E --with-ai2thor
    python3 tools/c2c_provision.py --source D --targets A --build-only

Why not just upload from here: the workstation reaches AutoDL at ~60 KB/s, so
the 330 MB payload alone costs 90 minutes per card.  The cards reach each
other's gateways at ~30 ms.
"""

from __future__ import annotations

import argparse
import queue
import shlex
import sys
import threading
import time
from pathlib import Path

from card_ssh import CARDS, connect  # noqa: PLC2701 - sibling tool

HELPER = Path(__file__).with_name("c2c_provision_remote.py")
REMOTE_HELPER = "/root/autodl-tmp/c2c_provision_remote.py"


def stream(c, cmd: str) -> int:
    """Run cmd on the source card, printing its output as it arrives."""
    chan = c.get_transport().open_session()
    chan.exec_command(cmd)
    q: queue.Queue[bytes | None] = queue.Queue()

    def pump() -> None:
        while True:
            if chan.recv_ready():
                q.put(chan.recv(65536))
            elif chan.exit_status_ready() and not chan.recv_ready():
                q.put(None)
                return
            else:
                time.sleep(0.2)

    t = threading.Thread(target=pump, daemon=True)
    t.start()
    buf = b""
    while True:
        item = q.get()
        if item is None:
            break
        buf += item
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            sys.stdout.write(line.decode(errors="replace") + "\n")
            sys.stdout.flush()
    if buf:
        sys.stdout.write(buf.decode(errors="replace"))
    return chan.recv_exit_status()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="D")
    ap.add_argument("--targets", nargs="+", required=True)
    ap.add_argument("--with-ai2thor", action="store_true")
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--payload", default="/root/autodl-tmp/c2c")
    args = ap.parse_args()

    if args.source not in CARDS:
        ap.error(f"source must be one of {list(CARDS)}")
    for t in args.targets:
        if t not in CARDS:
            ap.error(f"unknown target {t}")

    c = connect(args.source)
    try:
        print(f"uploading helper to {args.source} ...", flush=True)
        sftp = c.open_sftp()
        sftp.put(str(HELPER), REMOTE_HELPER)
        sftp.close()

        cmd = (f"/root/miniconda3/bin/python -u {REMOTE_HELPER} "
               f"--self {shlex.quote(args.source)} "
               f"--targets {' '.join(shlex.quote(t) for t in args.targets)} "
               f"--payload {shlex.quote(args.payload)}"
               + (" --with-ai2thor" if args.with_ai2thor else "")
               + (" --build-only" if args.build_only else ""))
        print(f"running on {args.source}: {cmd}\n", flush=True)
        return stream(c, cmd)
    finally:
        c.close()


if __name__ == "__main__":
    raise SystemExit(main())
