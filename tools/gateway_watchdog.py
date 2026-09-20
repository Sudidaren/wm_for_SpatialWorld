#!/usr/bin/env python3
"""网关额度用尽时的看门狗：暂停两条吃外部 API 的线，等网关恢复后自动续跑。

2026-09-21 01:3x 的实际情况：
  * `apic1.ohmycdn.com` 对项目 key 返回
    401 `This API key's maximum fee exceeded.`；
  * 之后因为失败风暴，chat/completions 被临时封：429
    `You have been temporarily blocked due to too many failed attempts`；
  * 同期 `/v1/models` 还是 200 —— 所以"key 有效"不等于"能推理"，
    探针必须打 chat/completions。

策略（对应用户 harness 第 4 条"401/额度：暂停该批，尝试备用 key；无备用则
停机汇报"）：
  1. 先把两条线停掉，别继续烧 CPU、也别继续加深封禁；
  2. 每 INTERVAL 秒用最小请求探一次（gemini 与 gpt-5 各一发，模型参数按各自
     的硬约束给）；
  3. 两个模型都 200 才恢复，恢复动作只做一次（写 state 文件防重复）。

    setsid nohup python3 tools/gateway_watchdog.py >> /mnt/d/lightwm_out/gateway_watchdog.log 2>&1 &
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.request

URL = "https://apic1.ohmycdn.com/v1/chat/completions"
KEY = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
INTERVAL = 600
LOG = "/mnt/d/lightwm_out/gateway_watchdog.log"
STATE = "/mnt/d/lightwm_out/gateway_state.txt"
LOCAL_ENV = "/mnt/d/lightwm_out/local_env_wm_gemini31pro_fix_rest156.sh"
SWE = "/home/sudidaren/spatialworld_eval"
LOCAL_RUN = "wm_gemini31pro_fix_rest156"
GPT5_RUN = "wm_gpt5_ai2thor_fail250"
#: 连续多少次坏响应才判定"又断了"（防抖：别被单次抖动来回切）
BAD_STREAK = 2


def say(msg: str) -> None:
    line = f"[gwwd {time.strftime('%F %T')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def probe(model: str, extra: dict) -> tuple[int, str]:
    """用 curl 探，不用 urllib。

    踩过：python urllib 的默认 UA 会被网关前面的 Cloudflare 直接挡回
    `403 error code: 1010`，看起来像"key 还是坏的"，其实 curl 同参数是 429/200。
    """
    body = json.dumps({"model": model,
                       "messages": [{"role": "user", "content": "say ok"}],
                       "max_tokens": 64, **extra})
    cmd = ["bash", "-c",
           f"curl -s -m 45 -o /tmp/gwwd_body.json -w '%{{http_code}}' "
           f"{URL} -H 'Authorization: Bearer {KEY}' "
           f"-H 'Content-Type: application/json' -d '{body}'"]
    try:
        code = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=60).stdout.strip()
        with open("/tmp/gwwd_body.json", encoding="utf-8",
                  errors="replace") as fh:
            return int(code or 0), fh.read(200)
    except Exception as exc:                       # noqa: BLE001
        return 0, f"{type(exc).__name__}: {exc}"


def sh(cmd: str) -> str:
    try:
        return subprocess.run(["bash", "-c", cmd], capture_output=True,
                              text=True, timeout=60).stdout.strip()
    except Exception as exc:                       # noqa: BLE001
        return f"ERR {exc}"


def resume() -> None:
    say("网关恢复，续跑两条线")
    # 本机 Gemini + WM
    cmd = (f"cd {SWE} && set -a && . {LOCAL_ENV} && set +a && "
           "setsid nohup bash run_wm_gemini_ai2thor.sh "
           ">>/mnt/d/lightwm_out/wm_fix_rest156_resume.log 2>&1 </dev/null &")
    sh(cmd)
    say(f"本机已提交: {sh(f'pgrep -cf \"[.]venv/bin/python -u - .* wm_gemini31pro_fix_rest156 \"')}")
    # gpt5 卡
    from newcard import CARDS
    import paramiko
    host, port, pwd = CARDS["gpt5"]
    try:
        c = paramiko.SSHClient()
        c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        c.connect(host, port=port, username="root", password=pwd, timeout=30)
        _in, out, _err = c.exec_command("bash /root/restart_gpt5.sh", timeout=300)
        say("gpt5 卡: " + out.read().decode("utf-8", "replace")[-400:].replace("\n", " | "))
        c.close()
    except Exception as exc:                       # noqa: BLE001
        say(f"gpt5 卡恢复失败: {type(exc).__name__}: {exc}")
    set_state("running")


def set_state(s: str) -> None:
    with open(STATE, "w", encoding="utf-8") as fh:
        fh.write(f"{s} {time.strftime('%F %T')}\n")


def get_state() -> str:
    try:
        with open(STATE, encoding="utf-8") as fh:
            return fh.read().split()[0]
    except Exception:                              # noqa: BLE001
        return "paused"


def pause_all() -> None:
    """网关又不可用：把两条线停掉（结果不删，续跑会重排未成功的）。"""
    say("网关又不可用，暂停两条线")
    here = os.path.dirname(os.path.abspath(__file__))
    say(sh(f"bash {here}/pause_local_run.sh {LOCAL_RUN}")[-300:])
    from newcard import CARDS
    import paramiko
    host, port, pwd = CARDS["gpt5"]
    try:
        c = paramiko.SSHClient()
        c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        c.connect(host, port=port, username="root", password=pwd, timeout=30)
        _in, out, _err = c.exec_command(f"bash /root/stop_run.sh {GPT5_RUN}",
                                        timeout=300)
        say("gpt5 卡: " + out.read().decode("utf-8", "replace")[-200:])
        c.close()
    except Exception as exc:                       # noqa: BLE001
        say(f"gpt5 卡停止失败: {type(exc).__name__}: {exc}")
    set_state("paused")


def main() -> None:
    say(f"看门狗启动（每 {INTERVAL}s 探一次，当前状态={get_state()}）")
    bad = 0
    while True:
        cg, bg = probe("gemini-3.1-pro-preview",
                       {"temperature": 1.0, "top_p": 0.9})
        c5, b5 = probe("gpt-5", {"temperature": 1.0})
        state = get_state()
        say(f"[{state}] gemini http={cg} {bg[:80]!r} | gpt-5 http={c5} {b5[:80]!r}")
        ok = cg == 200 and c5 == 200
        if state == "paused" and ok:
            bad = 0
            resume()
        elif state == "running" and not ok:
            bad += 1
            say(f"坏响应连续 {bad}/{BAD_STREAK}")
            if bad >= BAD_STREAK:
                bad = 0
                pause_all()
        else:
            bad = 0
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
