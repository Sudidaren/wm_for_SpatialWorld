#!/usr/bin/env python3
"""闭源主批次的看护：每分钟记一行进度，两条线都收尾就退出。

本地：/home/sudidaren/spatialworld_eval/runs/closed_gemini_wm（71 条）
云上：三张卡的 main_gpt5_base_s0/1/2（共 134 条）

日志：/mnt/d/lightwm_out/closed_watch.log
用法：setsid nohup python3 watch_closed.sh >/dev/null 2>&1 &
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, "/home/sudidaren/lightwm_phases/tools")
import paramiko  # noqa: E402

from newcard import CARDS  # noqa: E402

LOG = "/mnt/d/lightwm_out/closed_watch.log"
LOCAL_RUN = "/home/sudidaren/spatialworld_eval/runs/closed_gemini_wm"
SHARDS = {"30b": 0, "8b": 1, "kimi": 2}
LOCAL_TARGET = 71
#: 分片是按"全部 140 条"切的三份（47/47/46），不是只有 134 条 —— 那 6 条
#: 是伙伴先导批次里已有的，照跑一遍拿到同一套 harness 的新数据。
CLOUD_TARGET = 140


def say(msg: str) -> None:
    line = f"[watch {time.strftime('%F %T')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def local_count() -> tuple[int, int]:
    try:
        rows = list(csv.DictReader(open(f"{LOCAL_RUN}/results.csv", encoding="utf-8-sig")))
        return len(rows), sum(1 for r in rows if str(r.get("Success")).lower() == "true")
    except Exception:
        return 0, 0


REMOTE = (
    "P=/home/sudidaren/SpatialWorld/envs/ai2thor/.venv/bin/python;"
    "$P -c \"import csv,json;"
    "pick=lambda f: list(csv.DictReader(open(f,encoding='utf-8-sig')));"
    "b=pick('/home/sudidaren/spatialworld_eval/runs/main_gpt5_base_s{i}/results.csv');"
    "w=pick('/home/sudidaren/spatialworld_eval/runs/main_gpt5_wm_s{i}/results.csv');"
    "f=lambda rows:sum(1 for r in rows if str(r.get('Success')).lower()=='true');"
    "print(json.dumps([len(b),f(b),len(w),f(w)]))\" "
    "2>/dev/null || echo '[0,0,0,0]'"
)


def cloud_count() -> dict[str, tuple[int, int, int, int]]:
    out = {}
    for tag, i in SHARDS.items():
        host, port, pw = CARDS[tag]
        try:
            c = paramiko.SSHClient()
            c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            c.connect(host, port=port, username="root", password=pw,
                      timeout=20, banner_timeout=40, auth_timeout=40)
            _in, o, _e = c.exec_command(REMOTE.format(i=i), timeout=60)
            txt = o.read().decode().strip().splitlines()[-1]
            n, s, wn, ws = json.loads(txt)
            out[tag] = (n, s, wn, ws)
            c.close()
        except Exception as exc:  # noqa: BLE001
            out[tag] = (-1, -1, 0, 0)
            say(f"  {tag} 读取失败: {exc}")
    return out


def main() -> None:
    say("开始看护")
    while True:
        ln, ls = local_count()
        cc = cloud_count()
        cn = sum(v[0] for v in cc.values() if v[0] > 0)
        cs = sum(v[1] for v in cc.values() if v[1] >= 0)
        wn = sum(v[2] for v in cc.values())
        ws = sum(v[3] for v in cc.values())
        detail = " ".join(f"{t}=b{v[0]}/w{v[2]}" for t, v in cc.items())
        say(f"本机 Gemini+WM {ln}/{LOCAL_TARGET} 成功{ls} | 云 GPT-5 基线 {cn}/{CLOUD_TARGET} 成功{cs} "
            f"| 云 GPT-5+WM {wn}/{CLOUD_TARGET} 成功{ws} ({detail})")
        if ln >= LOCAL_TARGET and cn >= CLOUD_TARGET and wn >= CLOUD_TARGET:
            say("🎉 两条线都跑完")
            return
        time.sleep(60)


if __name__ == "__main__":
    main()
