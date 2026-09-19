#!/usr/bin/env python3
"""Runs ON a card: copy directories to a sibling card over the fast path.

Same reason as c2c_provision_remote.py -- the workstation's upstream is
~60 KB/s, the cards are ~17 MB/s to each other -- but for model weights, which
are the other multi-gigabyte thing a card needs and does not have.

Only ever adds files: it writes to ``<dest>/<basename>`` and refuses to
overwrite a directory that already has the expected number of shards.

    python3 c2c_copy_remote.py --target E \
        --item /root/autodl-tmp/models/qwen3vl-8b-bf16:/root/autodl-tmp/models
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path
from shlex import quote as shlex_quote

import paramiko

CARDS = {
    "A": ("connect.weste.seetacloud.com", 15868, "Yl3WnjEDllk+"),
    "B": ("connect.westb.seetacloud.com", 52296, "hxeahvC88uMM"),
    "C": ("connect.cqa1.seetacloud.com", 47531, "0yy1HhF4KfYV"),
    "D": ("connect.weste.seetacloud.com", 32808, "kQLLhVht3BEB"),
    "E": ("connect.westb.seetacloud.com", 22416, "cZmriU8VEh+F"),
}


def human(n: float) -> str:
    return f"{n / 1e6:.0f} MB" if n < 1e9 else f"{n / 1e9:.2f} GB"


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
        except Exception as exc:  # noqa: BLE001
            last = exc
            print(f"    connect {tag} {i + 1}/{retries}: {type(exc).__name__}", flush=True)
            time.sleep(8)
    raise last  # type: ignore[misc]


def sh(c, cmd: str, timeout: int = 7200) -> str:
    _, out, err = c.exec_command(cmd, timeout=timeout)
    text = out.read().decode(errors="replace")
    etext = err.read().decode(errors="replace").strip()
    if etext:
        # stderr is noise for a command like `df <missing path>`, but it must
        # never end up inside a value we int() -- that turned a missing target
        # directory into a ValueError nobody could see (the driver only read
        # stdout).  Keep the streams apart and show the noise.
        print(f"      (stderr) {etext[:300]}", flush=True)
    return text


def as_int(text: str) -> int:
    text = (text or "").strip().splitlines()
    for line in reversed(text):
        line = line.strip()
        if line.isdigit():
            return int(line)
    return 0


def local(cmd: str, timeout: int = 7200) -> str:
    p = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        print(f"      !! local rc={p.returncode}: {p.stderr.strip()[:200]}", flush=True)
    return p.stdout.strip()


def copy_item(target: str, src: str, dest_base: str) -> bool:
    """Copy file by file, skipping files already complete on the target.

    Deliberately not tar: a tar has to land somewhere before it is unpacked,
    so a 31 GB model needs 62 GB of target disk.  A card with a 50 GB disk
    filled up mid-extract that way, at which point tar reported ENOSPC on
    files it could no longer create.  File-by-file peaks at the size of the
    content, and a re-run only moves what is missing.
    """
    name = os.path.basename(src.rstrip("/"))
    dest = f"{dest_base.rstrip('/')}/{name}"
    print(f"    {src} -> {target}:{dest}", flush=True)

    listing = local(f"find {src} -type f -printf '%s\\t%p\\n'")
    files: list[tuple[int, str]] = []
    for line in listing.splitlines():
        size_s, _, path = line.partition("\t")
        if path and size_s.isdigit():
            files.append((int(size_s), path))
    if not files:
        print("      source has no files -> skip", flush=True)
        return False
    total = sum(s for s, _ in files)

    c = connect(target)
    try:
        free = as_int(sh(c, f"mkdir -p {dest_base} && df -B1 --output=avail {dest_base} | tail -1"))
        have_bytes = 0
        need_files: list[tuple[int, str]] = []
        for size, path in files:
            rel = os.path.relpath(path, src)
            remote = f"{dest}/{rel}"
            if as_int(sh(c, f"stat -c %s {shlex_quote(remote)} 2>/dev/null")) == size:
                have_bytes += size
                continue
            need_files.append((size, path))
        missing = sum(s for s, _ in need_files)
        print(f"      {human(total)} total, {human(have_bytes)} already there, "
              f"{human(missing)} to move, {human(free)} free on {target}", flush=True)
        if missing > free * 0.98:
            print("      !! not enough space on the target -> skip", flush=True)
            return False
        if not need_files:
            print("      nothing to do", flush=True)
            return True

        # directories first (one round trip each, and there are few)
        dirs = sorted({os.path.dirname(f"{dest}/{os.path.relpath(p, src)}") for _, p in files})
        sh(c, " && ".join(f"mkdir -p {shlex_quote(d)}" for d in dirs))

        sftp = c.open_sftp()
        t0, sent, mark = time.time(), 0, time.time()
        for size, path in need_files:
            remote = f"{dest}/{os.path.relpath(path, src)}"
            with sftp.open(remote, "wb") as rf:
                rf.set_pipelined(True)
                with open(path, "rb") as lf:
                    while True:
                        b = lf.read(8 << 20)
                        if not b:
                            break
                        rf.write(b)
                        sent += len(b)
                        now = time.time()
                        if now - mark >= 10:
                            pct = 100.0 * (have_bytes + sent) / max(total, 1)
                            print(f"        {pct:5.1f}%  {sent / (now - t0) / 1e6:5.2f} MB/s"
                                  f"  ({os.path.basename(path)})", flush=True)
                            mark = now
            got = sftp.stat(remote).st_size
            if got != size:
                print(f"      !! {remote}: got {got} != {size} -> abandon", flush=True)
                sftp.close()
                return False
        sftp.close()
        dt = max(time.time() - t0, 1e-6)
        print(f"      moved {human(sent)} in {dt / 60:.1f} min "
              f"({sent / dt / 1e6:.2f} MB/s)", flush=True)
        print("      " + sh(c, f"du -sh {dest}").strip(), flush=True)
        return True
    finally:
        c.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--item", action="append", required=True,
                    help="SRC_DIR:DEST_BASE (repeatable)")
    args = ap.parse_args()

    rc = 0
    for item in args.item:
        src, _, dest_base = item.rpartition(":")
        ok = copy_item(args.target, src, dest_base)
        rc |= 0 if ok else 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
