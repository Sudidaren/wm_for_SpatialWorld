#!/usr/bin/env python3
"""Continuous monitor for the three AutoDL SpatialWorld eval boxes.

Read-only: only runs inspection commands over SSH, never touches runs.

Usage:
  python3 monitor_cloud_eval.py                 # one snapshot to stdout
  python3 monitor_cloud_eval.py --write         # one snapshot + files
  python3 monitor_cloud_eval.py --watch 120     # loop every 120s, write files
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

import paramiko

BASE = Path(__file__).resolve().parent
SNAP_DIR = BASE / "snapshots"
HISTORY = BASE / "history.tsv"
ALERTS = BASE / "alerts.log"
LATEST = BASE / "LATEST.md"
PULL_DIR = BASE / "pulled"
MUTED = BASE / "muted.txt"

TOTAL_TASKS = 438
MAX_TRIES = 3          # 1 + eval_config.MAX_EXTERNAL_RETRIES

#: 某些批次只跑一个场景（名字里带场景 + 数量），分母不是 438
PLAN_OVERRIDES = {"ai2thor311": 311, "procthor127": 127}


def plan_of(run_name: str) -> int:
    for key, size in PLAN_OVERRIDES.items():
        if key in run_name:
            return size
    return TOTAL_TASKS

HOSTS = [
    {
        "name": "weste",
        "host": "connect.weste.seetacloud.com",
        "port": 32808,
        "password": "PUT_AUTODL_PASSWORD_HERE",
        "role": "30B WM+old / WM+new",
        "model": "qwen3vl-30b",
    },
    {
        "name": "westd",
        "host": "connect.westd.seetacloud.com",
        "port": 28564,
        "password": "PUT_AUTODL_PASSWORD_HERE",
        "role": "Kimi pure -> WM+new -> WM+old",
        "model": "kimivl-a3b",
    },
    {
        "name": "cqa1",
        "host": "connect.cqa1.seetacloud.com",
        "port": 47531,
        "password": "PUT_AUTODL_PASSWORD_HERE",
        "role": "8B pure -> WM+new -> WM+old",
        "model": "qwen3vl-8b",
    },
]

REMOTE_CMD = r"""
echo "###HOSTNAME"; hostname
echo "###DATE"; date '+%Y-%m-%d %H:%M:%S'
echo "###UPTIME"; uptime
echo "###GPU"; nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null
echo "###DISK"; df -h /root/autodl-tmp | tail -1
echo "###VLLM"; pgrep -af '[v]llm serve' | head -3
echo "###ORCH"; pgrep -af '[c]loud_orchestrator_v[0-9]*[.]sh' | head -5
echo "###SUP"; pgrep -af 'python -u - ' | grep -v run_task | wc -l
echo "###WORKERS"; pgrep -f '[r]un_task' | wc -l
echo "###WORKER_AGE"; ps -eo etimes,args | grep '[r]un_task' | awk '{print $1}' | sort -n | tail -1
echo "###WORKER_TASKS"; ps -eo args | grep -oE '[r]un_task --config [^ ]+ --tasks [^ ]+' | awk '{printf "%s ", $NF}'; echo
echo "###ORCHLOG"; tail -6 /root/orchestrator_v2.log 2>/dev/null
echo "###RUNS"
/root/miniconda3/bin/python - <<'PYEOF'
import json, os, glob, time
root = "/root/autodl-tmp/runs_local"
MAX_TRIES = int(os.environ.get("MAX_TRIES", "3"))   # 1 + MAX_EXTERNAL_RETRIES
out = []
for d in sorted(glob.glob(os.path.join(root, "*"))):
    if not os.path.isdir(d):
        continue
    name = os.path.basename(d)
    sp = os.path.join(d, "state.json")
    rec = {"name": name, "state_age_sec": None, "tasks": 0, "decided": 0,
           "terminal": 0,
           "success": 0, "failed_model": 0, "failed_external": 0,
           "failed_external_exhausted": 0,
           "pending": 0, "other_status": {}, "csv_rows": 0, "size_mb": 0,
           "aux": name.startswith(("cloud_smoke", "verify_freeze", "_old"))}
    try:
        rec["size_mb"] = round(sum(
            os.path.getsize(os.path.join(r, f))
            for r, _, fs in os.walk(d) for f in fs) / 1e6, 1)
    except Exception:
        pass
    cp = os.path.join(d, "results.csv")
    if os.path.exists(cp):
        try:
            with open(cp, "r", encoding="utf-8-sig", errors="replace") as fh:
                rec["csv_rows"] = max(0, sum(1 for _ in fh) - 1)
        except Exception:
            pass
    if os.path.exists(sp):
        rec["state_age_sec"] = int(time.time() - os.path.getmtime(sp))
        s = None
        # state.json is rewritten after every finished task, so a single read
        # can catch it mid-write.  Retry until it holds the whole task list.
        for _try in range(3):
            try:
                with open(sp, "r", encoding="utf-8") as fh:
                    s = json.load(fh)
                tasks = s.get("tasks", s if isinstance(s, list) else [])
                if len(tasks) >= int(os.environ.get("EXPECTED_TASKS", "438")):
                    break
                time.sleep(2)
            except Exception as ex:
                rec["error"] = repr(ex)
                time.sleep(2)
        try:
            tasks = s.get("tasks", s if isinstance(s, list) else [])
            rec["tasks"] = len(tasks)
            for t in tasks:
                comp = str(t.get("completed", "")).lower()
                st = str(t.get("status") or "")
                if comp in ("true", "false"):
                    rec["decided"] += 1
                    rec["terminal"] += 1
                    if t.get("success"):
                        rec["success"] += 1
                    if st == "failed_model":
                        rec["failed_model"] += 1
                    elif st in ("failed_external", "failed_infra"):
                        rec["failed_external"] += 1
                    else:
                        rec["other_status"][st or "unknown"] = rec["other_status"].get(st or "unknown", 0) + 1
                elif st in ("failed_external", "failed_infra"):
                    # terminal even though completed stays None, once the
                    # retry budget is used up (matches cloud_orchestrator_v3)
                    rec["failed_external"] += 1
                    if int(t.get("attempts") or 0) >= MAX_TRIES:
                        rec["terminal"] += 1
                        rec["failed_external_exhausted"] += 1
                    else:
                        rec["pending"] += 1
                else:
                    rec["pending"] += 1
        except Exception as ex:
            rec["error"] = repr(ex)
    out.append(rec)
print("###RUNS_JSON")
print(json.dumps(out, ensure_ascii=False))
PYEOF
"""


def parse_remote(text: str) -> dict:
    sections: dict[str, str] = {}
    cur = None
    buf: list[str] = []
    for line in text.splitlines():
        if line.startswith("###") and line.strip().upper() == line:
            if cur is not None:
                sections[cur] = "\n".join(buf).strip()
            cur = line[3:].strip()
            buf = []
        else:
            buf.append(line)
    if cur is not None:
        sections[cur] = "\n".join(buf).strip()
    data = {k: v for k, v in sections.items()}
    runs_json = data.pop("RUNS_JSON", "")
    try:
        data["runs"] = json.loads(runs_json) if runs_json else []
    except Exception:
        data["runs"] = []
    return data


def collect(host: dict) -> dict:
    out = {
        "name": host["name"], "host": host["host"], "port": host["port"],
        "role": host["role"], "model": host["model"],
        "ok": False, "error": None,
        "collected_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(
            host["host"], port=host["port"], username="root",
            password=host["password"], timeout=20,
            banner_timeout=25, auth_timeout=25,
        )
        _, stdout, stderr = client.exec_command(REMOTE_CMD, timeout=120)
        text = stdout.read().decode("utf-8", "replace")
        err = stderr.read().decode("utf-8", "replace")
        data = parse_remote(text)
        data["ok"] = True
        if err.strip():
            data["stderr"] = err.strip()[:500]
        out.update(data)
    except Exception as ex:  # noqa: BLE001
        out["error"] = f"{type(ex).__name__}: {ex}"
    finally:
        client.close()
    return out


def rates(window_min: int = 45) -> dict[tuple[str, str], float]:
    """decided-per-minute over the trailing window, from history.tsv."""
    if not HISTORY.exists():
        return {}
    now = dt.datetime.now().timestamp()
    series: dict[tuple[str, str], list[tuple[float, int]]] = {}
    with HISTORY.open("r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            try:
                ts = dt.datetime.strptime(row["ts"], "%Y-%m-%d %H:%M:%S").timestamp()
                series.setdefault((row["host"], row["run"]), []).append((ts, int(row["decided"])))
            except Exception:
                continue
    out: dict[tuple[str, str], float] = {}
    for key, pts in series.items():
        pts = [p for p in pts if now - p[0] <= window_min * 60]
        pts.sort()
        if len(pts) < 3:
            continue
        if pts[0][1] > pts[-1][1]:
            continue  # restarted, ignore
        span = (pts[-1][0] - pts[0][0]) / 60.0
        if span < 5:
            continue
        out[key] = (pts[-1][1] - pts[0][1]) / span
    return out


def summarize(snap: dict, rate_map: dict | None = None) -> str:
    rate_map = rate_map or {}
    lines = [f"# 云上评测巡检 {snap['collected_at']}", ""]
    for h in snap["hosts"]:
        if not h.get("ok"):
            lines.append(f"## {h['name']} ({h['role']}) — 采集失败: {h.get('error')}")
            lines.append("")
            continue
        gpu = (h.get("GPU") or "?").replace(",", " /")
        disk = (h.get("DISK") or "?")
        disk_bits = disk.split()
        disk_txt = f"{disk_bits[2]} used, {disk_bits[3]} free ({disk_bits[4]})" if len(disk_bits) >= 5 else disk
        vllm = "up" if h.get("VLLM", "").strip() else "DOWN"
        orch = "up" if h.get("ORCH", "").strip() else "DOWN"
        workers = (h.get("WORKERS") or "0").strip()
        wa = (h.get("WORKER_AGE") or "").strip()
        wt = (h.get("WORKER_TASKS") or "").strip()
        lines.append(f"## {h['name']} ({h['role']})")
        lines.append(
            f"- GPU {gpu} | 磁盘 {disk_txt} | vLLM {vllm} | 编排 {orch} | "
            f"在跑任务 {workers} 个（最久 {int(wa)//60 if wa.isdigit() else '?'} 分钟）| 当前: {wt or '-'}"
        )
        for r in h.get("runs", []):
            d = r["decided"]
            plan = plan_of(r["name"])
            pct = 100.0 * d / plan
            age = r.get("state_age_sec")
            age_txt = f"{age//60}m{age%60}s" if isinstance(age, int) else "?"
            rate = rate_map.get((h["name"], r["name"]))
            rate_txt = ""
            if rate and rate > 0.02:
                remain = plan - d
                eta_min = remain / rate
                eta_at = dt.datetime.now() + dt.timedelta(minutes=eta_min)
                rate_txt = (
                    f", {rate:.2f} 任务/分, ETA {eta_min/60:.1f}h (~{eta_at.strftime('%H:%M')})"
                )
            lines.append(
                f"- `{r['name']}`: decided {d}/{plan} ({pct:.1f}%), "
                f"已终结 {r.get('terminal', d)}/{plan}, "
                f"success {r['success']}, failed_model {r['failed_model']}, "
                f"failed_external {r['failed_external']}"
                + (f"（已耗尽 {r['failed_external_exhausted']}）"
                   if r.get('failed_external_exhausted') else "")
                + f", 状态更新于 {age_txt} 前, "
                f"csv {r['csv_rows']} 行, {r['size_mb']}MB{rate_txt}"
            )
        lines.append("")
    return "\n".join(lines)


def detect_alerts(prev: dict | None, snap: dict) -> list[str]:
    alerts: list[str] = []
    stalled = stalled_runs(snap)
    muted = set()
    if MUTED.exists():
        muted = {ln.split("#")[0].strip() for ln in MUTED.read_text().splitlines()
                 if ln.split("#")[0].strip()}
    for h in snap["hosts"]:
        n = h["name"]
        if not h.get("ok"):
            alerts.append(f"[{n}] 无法连接: {h.get('error')}")
            continue
        if not (h.get("VLLM") or "").strip():
            alerts.append(f"[{n}] vLLM 进程不在")
        if not (h.get("ORCH") or "").strip():
            alerts.append(f"[{n}] 编排进程不在（卡住的批次不会被拉起）")
        disk = (h.get("DISK") or "").split()
        if len(disk) >= 5 and disk[4].endswith("%"):
            if int(disk[4].rstrip("%")) >= 90:
                alerts.append(f"[{n}] 磁盘用量 {disk[4]}，仅剩 {disk[3]}")
        for r in h.get("runs", []):
            if r.get("aux"):
                continue
            if (r["failed_external"] >= 5
                    and r.get("terminal", r["decided"]) < plan_of(r["name"])):
                alerts.append(
                    f"[{n}/{r['name']}] failed_external={r['failed_external']} 且批次仍在跑，"
                    f"可能是环境故障（耗尽重试后会被 v3 终结并推进下一阶段）")
            if r["pending"] > 0 and isinstance(r.get("state_age_sec"), int) and r["state_age_sec"] > 1800:
                alerts.append(f"[{n}/{r['name']}] 有 {r['pending']} 个 pending 且 state 超过 30 分钟未更新")
        for r in h.get("runs", []):
            if r["name"] in stalled and r["name"] not in muted:
                mins = stalled[r["name"]]
                alerts.append(
                    f"[{n}/{r['name']}] decided 停在 {r['decided']} 已 {mins} 分钟，"
                    f"有 {r['pending']} 个任务在跑但无新判定，疑似卡住"
                )
        wa = (h.get("WORKER_AGE") or "").strip()
        if wa.isdigit() and int(wa) > 1800:
            alerts.append(f"[{n}] 有单个任务已在跑 {int(wa)//60} 分钟未结束（阈值 30 分钟，疑似卡死）")
    if prev:
        prev_map = {h["name"]: h for h in prev.get("hosts", [])}
        for h in snap["hosts"]:
            p = prev_map.get(h["name"])
            if not p or not h.get("ok") or not p.get("ok"):
                continue
            pm = {r["name"]: r for r in p.get("runs", [])}
            for r in h.get("runs", []):
                if r.get("aux") or r.get("terminal", r["decided"]) >= plan_of(r["name"]):
                    continue
                pr = pm.get(r["name"])
                if not pr:
                    continue
                if r["decided"] < pr["decided"]:
                    alerts.append(
                        f"[{h['name']}/{r['name']}] decided 回退 {pr['decided']} -> {r['decided']}"
                    )
    return alerts


def stalled_runs(snap: dict, threshold_min: int = 12) -> dict[str, int]:
    """Runs scheduled by the current orchestrator whose decided count is flat.

    Returns {run_name: minutes_stalled} for batches the running orchestrator
    actually owns (so leftovers from earlier attempts don't raise noise).
    """
    if not HISTORY.exists():
        return {}
    samples: dict[tuple[str, str], list[tuple[float, int]]] = {}
    with HISTORY.open("r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            try:
                ts = dt.datetime.strptime(row["ts"], "%Y-%m-%d %H:%M:%S").timestamp()
                samples.setdefault((row["host"], row["run"]), []).append(
                    (ts, int(row["decided"]))
                )
            except Exception:
                continue
    now = dt.datetime.now().timestamp()
    out: dict[str, int] = {}
    for h in snap["hosts"]:
        if not h.get("ok"):
            continue
        orch_text = h.get("ORCH") or ""
        workers = int((h.get("WORKERS") or "0").strip() or 0)
        if workers <= 0:
            continue
        for r in h.get("runs", []):
            if r.get("aux") or r["name"] not in orch_text:
                continue
            if r.get("terminal", r["decided"]) >= plan_of(r["name"]):
                continue
            hist = samples.get((h["name"], r["name"]))
            if not hist:
                continue
            hist.sort()
            last = hist[-1][1]
            if last != r["decided"]:
                continue
            # earliest timestamp of the trailing flat segment
            flat_start = hist[-1][0]
            for ts, val in reversed(hist):
                if val == last:
                    flat_start = ts
                else:
                    break
            mins = int((now - flat_start) / 60)
            if mins >= threshold_min:
                out[r["name"]] = mins
    return out


def append_history(snap: dict) -> None:
    new = not HISTORY.exists()
    with HISTORY.open("a", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t")
        if new:
            w.writerow(["ts", "host", "run", "decided", "terminal", "success",
                        "failed_model", "failed_external", "state_age_sec",
                        "gpu_used", "disk_free", "vllm", "orch"])
        for h in snap["hosts"]:
            if not h.get("ok"):
                continue
            disk = (h.get("DISK") or "").split()
            for r in h.get("runs", []):
                if r.get("aux"):
                    continue
                w.writerow([
                    snap["collected_at"], h["name"], r["name"], r["decided"],
                    r.get("terminal", r["decided"]),
                    r["success"], r["failed_model"], r["failed_external"],
                    r.get("state_age_sec"), (h.get("GPU") or "").split(",")[0],
                    disk[3] if len(disk) > 3 else "", bool(h.get("VLLM")),
                    bool(h.get("ORCH")),
                ])


def pull_artifacts(snap: dict) -> list[str]:
    """Copy small result artifacts (csv/state/plan) locally as a safety net."""
    pulled: list[str] = []
    for h in snap["hosts"]:
        if not h.get("ok"):
            continue
        orch_text = h.get("ORCH") or ""
        targets = [
            r["name"] for r in h.get("runs", [])
            if not r.get("aux") and r["name"] in orch_text and r["decided"] > 0
        ]
        if not targets:
            continue
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            client.connect(
                h["host"], port=h["port"], username="root",
                password=next(x["password"] for x in HOSTS if x["name"] == h["name"]),
                timeout=20, banner_timeout=25, auth_timeout=25,
            )
            sftp = client.open_sftp()
            for run in targets:
                dest = PULL_DIR / h["name"] / run
                dest.mkdir(parents=True, exist_ok=True)
                for fname in ("results.csv", "state.json", "plan.json"):
                    src = f"/root/autodl-tmp/runs_local/{run}/{fname}"
                    try:
                        sftp.get(src, str(dest / fname))
                        pulled.append(f"{h['name']}/{run}/{fname}")
                    except IOError:
                        pass
            sftp.close()
        except Exception as ex:  # noqa: BLE001
            pulled.append(f"PULL_ERROR {h['name']}: {type(ex).__name__}: {ex}")
        finally:
            client.close()
    return pulled


def run_once(write: bool) -> dict:
    hosts = [collect(h) for h in HOSTS]
    snap = {
        "collected_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "hosts": hosts,
    }
    prev = None
    if LATEST.exists():
        try:
            prev = json.loads((BASE / "last_snapshot.json").read_text())
        except Exception:
            prev = None
    alerts = detect_alerts(prev, snap)
    snap["alerts"] = alerts
    if write:
        SNAP_DIR.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        (SNAP_DIR / f"{stamp}.json").write_text(json.dumps(snap, ensure_ascii=False, indent=2))
        (BASE / "last_snapshot.json").write_text(json.dumps(snap, ensure_ascii=False, indent=2))
        LATEST.write_text(summarize(snap, rates()) + "\n" + alerts_section(alerts))
        append_history(snap)
        pulled = pull_artifacts(snap)
        snap["pulled"] = pulled
        if alerts:
            with ALERTS.open("a", encoding="utf-8") as fh:
                for a in alerts:
                    fh.write(f"{snap['collected_at']}\t{a}\n")
    return snap


def alerts_section(alerts: list[str]) -> str:
    if not alerts:
        return "\n## 异常\n\n无\n"
    return "\n## 异常\n\n" + "\n".join(f"- {a}" for a in alerts) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", type=int, default=0, help="loop interval seconds")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    if args.watch:
        while True:
            try:
                snap = run_once(True)
                print(summarize(snap))
                if snap["alerts"]:
                    print("ALERTS:", *snap["alerts"], sep="\n  ")
                sys.stdout.flush()
            except Exception as ex:  # noqa: BLE001
                print(f"[{dt.datetime.now()}] monitor error: {type(ex).__name__}: {ex}", flush=True)
            time.sleep(args.watch)
    else:
        snap = run_once(args.write)
        print(summarize(snap))
        if snap["alerts"]:
            print(alerts_section(snap["alerts"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
