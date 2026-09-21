#!/usr/bin/env python3
"""盯本机的 GPT-5+WM 剩余批次（closed_gpt5_wm，72 条）。

三张卡 22:0x 被关停后剩下的 72 条搬到本机跑；已拉回的 68 条在
/mnt/d/lightwm_out/closed_pull/*_wm_*/results.csv，合并时两边拼。

日志：/mnt/d/lightwm_out/gpt5wm_watch.log
"""
from __future__ import annotations

import csv
import glob
import time

LOG = "/mnt/d/lightwm_out/gpt5wm_watch.log"
RUN = "/home/sudidaren/spatialworld_eval/runs/closed_gpt5_wm"
PULLED = "/mnt/d/lightwm_out/closed_pull"
TARGET = 72


def say(msg: str) -> None:
    line = f"[g5wmw {time.strftime('%F %T')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def main() -> None:
    say("开始看护（本机 GPT-5+WM 剩余批次）")
    last = -1
    while True:
        try:
            rows = list(csv.DictReader(open(f"{RUN}/results.csv", encoding="utf-8-sig")))
        except Exception:
            rows = []
        n = len(rows)
        s = sum(1 for r in rows if str(r.get("Success")).lower() == "true")
        cloud = 0
        for f in glob.glob(f"{PULLED}/*_wm_*/results.csv"):
            cloud += sum(1 for _ in csv.DictReader(open(f, encoding="utf-8-sig")))
        if n != last:
            say(f"本机 {n}/{TARGET} 成功{s}（+卡上已拉回 {cloud} 条 → 合计 {n + cloud}/140）")
            last = n
        if n >= TARGET:
            say(f"🎉 收尾：本机 {n} 条 + 卡上 {cloud} 条 = {n + cloud}/140")
            return
        time.sleep(60)


if __name__ == "__main__":
    main()
