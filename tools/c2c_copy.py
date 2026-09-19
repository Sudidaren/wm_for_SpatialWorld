#!/usr/bin/env python3
"""Card-to-card directory copy, driven from the workstation.

    python3 tools/c2c_copy.py --source B --target A \
        --item /root/autodl-tmp/models/kimivl-a3b-bf16:/root/autodl-tmp/models

Bytes move card -> card; only the command travels through the workstation.
"""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path

from card_ssh import CARDS, connect  # noqa: PLC2701 - sibling tool
from c2c_provision import stream  # noqa: PLC2701 - sibling tool

HELPER = Path(__file__).with_name("c2c_copy_remote.py")
REMOTE_HELPER = "/root/autodl-tmp/c2c_copy_remote.py"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--item", action="append", required=True)
    args = ap.parse_args()
    for tag in (args.source, args.target):
        if tag not in CARDS:
            ap.error(f"unknown card {tag}")
    if args.source == args.target:
        ap.error("source and target are the same card")

    c = connect(args.source)
    try:
        sftp = c.open_sftp()
        sftp.put(str(HELPER), REMOTE_HELPER)
        sftp.close()
        cmd = (f"/root/miniconda3/bin/python -u {REMOTE_HELPER} "
               f"--target {shlex.quote(args.target)} "
               + " ".join(f"--item {shlex.quote(i)}" for i in args.item))
        print(f"{args.source} -> {args.target}: {cmd}\n", flush=True)
        return stream(c, cmd)
    finally:
        c.close()


if __name__ == "__main__":
    raise SystemExit(main())
