#!/usr/bin/env python3
"""Runs ON an already-provisioned card: hand the same layout to sibling cards.

The workstation's upstream to AutoDL measures ~60 KB/s, so provisioning a
fresh card from here takes hours.  The cards sit ~30 ms from each other's
gateways (verified from inside D: every gateway answers), so a card that is
already provisioned can push the same bytes in minutes.

Two rules keep this honest:
  * every file on the target is size-checked against the source after the
    transfer, because a truncated tar extracts into a half-working box;
  * ``code_only.tar`` is pushed last and its presence is what "done" means.

Uploaded to the source card and executed there by ``tools/c2c_provision.py``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
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

REMOTE_TMP = "/root/autodl-tmp/eval_payload"
LAYOUT = "/home/sudidaren"
TARS = ("lightwm_phases.tar", "spatialworld.tar", "spatialworld_eval.tar", "code_only.tar")

PIP = ("ai2thor==5.0.0", "rfdetr", "supervision", "opencv-python-headless",
       "transformers", "openai", "python-dotenv", "pyyaml", "paramiko")

#: A card is only "ready" when it can actually run a scene.  Importing ai2thor
#: fetches the 2.1 GB Unity build on first use; the Controller call is what
#: proves the display, the build and the package all agree.  This is the check
#: that would have caught the Xvfb outage before it burned 229 tasks.
SMOKE = """
set -e
export DISPLAY=:99
pgrep -f "[X]vfb :99" >/dev/null || (setsid nohup Xvfb :99 -screen 0 1280x1024x24 \
  > /root/autodl-tmp/logs/xvfb.log 2>&1 < /dev/null &)
sleep 3
VENV=/home/sudidaren/SpatialWorld/envs/ai2thor/.venv
[ -x "$VENV/bin/python" ] || /root/miniconda3/bin/python -m venv --system-site-packages "$VENV"
$VENV/bin/python - <<'PY'
import ai2thor, time
from ai2thor.controller import Controller
print("ai2thor", ai2thor.__version__)
t0 = time.time()
c = Controller(scene="FloorPlan1", platform="Linux64",
               width=300, height=300, server_timeout=300.0,
               start_unity_process=True)
print("controller up in %.1fs" % (time.time() - t0))
ev = c.step("Pass")
print("frame ok:", ev.frame.shape)
c.stop()
PY
"""


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
    etext = err.read().decode(errors="replace")
    return (text + ("\n[stderr] " + etext if etext.strip() else "")).strip()


class Progress:
    """Print throughput every few seconds so a stalled link is obvious."""

    def __init__(self, label: str, total: int, every: float = 5.0):
        self.label, self.total, self.every = label, total, every
        self.done = 0
        self.t0 = self.t_last = time.time()
        self.mark = 0

    def tick(self, sent: int) -> None:
        self.done = sent
        now = time.time()
        if now - self.t_last >= self.every:
            inst = (sent - self.mark) / max(now - self.t_last, 1e-6) / 1e6
            pct = 100.0 * sent / max(self.total, 1)
            print(f"      {self.label}: {pct:5.1f}% {human(sent)}/{human(self.total)}"
                  f"  {inst:5.2f} MB/s", flush=True)
            self.t_last, self.mark = now, sent


def local(cmd: str, timeout: int = 7200) -> str:
    p = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        print(f"      !! local rc={p.returncode}: {cmd[:60]}... {p.stderr.strip()[:200]}",
              flush=True)
    return p.stdout.strip()


def build_tars(payload: Path, rebuild: bool = False) -> None:
    """Pack the layout into the four tars, skipping only byte-complete ones.

    A tar that failed halfway is the dangerous case: it looks built and then
    extracts into a box that half-works.  So the "already built" test is a
    sidecar listing of the exact paths that went in, not a size threshold.
    """
    payload.mkdir(parents=True, exist_ok=True)
    specs = [
        ("lightwm_phases.tar",
         ["lightwm_phases/phase_b", "lightwm_phases/shared",
          "lightwm_phases/runtime_overlay", "lightwm_phases/tools",
          "lightwm_phases/configs", "lightwm_phases/scripts",
          "lightwm_phases/data", "lightwm_phases/eval_pipeline",
          "lightwm_phases/tests",
          "lightwm_phases/checkpoints/rfdetr_small_228094",
          "lightwm_phases/checkpoints/small_objects_20260910",
          "lightwm_phases/checkpoints/da2_metric_indoor_small"]),
        ("spatialworld.tar",
         ["SpatialWorld/data", "SpatialWorld/mllm_base_agent",
          "SpatialWorld/evaluation", "SpatialWorld/experiments",
          "SpatialWorld/tests"]),
        ("spatialworld_eval.tar",
         ["spatialworld_eval"]),
        ("code_only.tar",
         ["lightwm_phases/phase_b", "lightwm_phases/shared",
          "lightwm_phases/runtime_overlay", "lightwm_phases/tools",
          "lightwm_phases/configs", "lightwm_phases/scripts",
          "lightwm_phases/eval_pipeline", "spatialworld_eval"]),
    ]
    for name, paths in specs:
        out = payload / name
        present = [p for p in paths if (Path(LAYOUT) / p).exists()]
        missing = sorted(set(paths) - set(present))
        stamp = payload / f"{name}.contents"
        want = "\n".join(present) + "\n"
        if not rebuild and out.exists() and stamp.exists() and stamp.read_text() == want:
            print(f"    {name}: already built ({human(out.stat().st_size)})", flush=True)
            continue
        if missing:
            print(f"    ({name}: absent on this card, skipped: {' '.join(missing)})", flush=True)
        excl = "--exclude='__pycache__' --exclude='*.pyc'"
        if name == "spatialworld_eval.tar":
            excl += " --exclude='runs'"
        print(f"    building {name} ...", flush=True)
        local(f"cd {LAYOUT} && tar cf {tmp_of(out)} {excl} {' '.join(present)}")
        if not Path(tmp_of(out)).exists():
            print(f"    !! {name} failed to build", flush=True)
            continue
        os.replace(tmp_of(out), out)
        stamp.write_text(want)
        print(f"      -> {human(out.stat().st_size)}", flush=True)


def tmp_of(out: Path) -> str:
    return str(out) + ".part"


def provision(target: str, payload: Path, with_ai2thor: bool, me: str) -> bool:
    print(f"===== {me} -> {target} =====", flush=True)
    c = connect(target)
    try:
        sh(c, f"mkdir -p {REMOTE_TMP} {LAYOUT}")
        sh(c, f"rm -f {REMOTE_TMP}/.done")
        sftp = c.open_sftp()

        for name in TARS:
            path = payload / name
            if not path.exists():
                print(f"    skip {name} (not built)", flush=True)
                continue
            size = path.stat().st_size
            try:
                if sftp.stat(f"{REMOTE_TMP}/{name}").st_size == size:
                    print(f"    {name}: already complete ({human(size)})", flush=True)
                    continue
            except OSError:
                pass
            prog = Progress(name, size)
            t0 = time.time()
            sftp.put(str(path), f"{REMOTE_TMP}/{name}",
                     callback=lambda sent, _total: prog.tick(sent))
            dt = max(time.time() - t0, 1e-6)
            print(f"    {name}: {human(size)} in {dt / 60:.1f} min "
                  f"({size / dt / 1e6:.2f} MB/s)", flush=True)
            got = sftp.stat(f"{REMOTE_TMP}/{name}").st_size
            if got != size:
                print(f"    !! {name} size mismatch {got} != {size} -> abort", flush=True)
                return False

        sftp.close()
        print("    extracting ...", flush=True)
        print(sh(c, f"cd {LAYOUT} && " +
                 " && ".join(f"tar xf {REMOTE_TMP}/{t}" for t in TARS
                             if (payload / t).exists()) +
                 f" && rm -f {REMOTE_TMP}/*.tar && ls {LAYOUT}"), flush=True)

        # DA2 is public: let the card pull it from hf-mirror, which is fast there
        # DA2 rides along in the tar when the source has it (95 MB at LAN speed);
        # hf-mirror is the fallback, and it is what a bare image reaches anyway.
        print("    DA2 weights ...", flush=True)
        da2 = f"{LAYOUT}/lightwm_phases/checkpoints/da2_metric_indoor_small"
        print(sh(c, f"if [ -f {da2}/model.safetensors ]; then "
                    f"  echo 'already present:' $(du -h {da2}/model.safetensors | cut -f1); "
                    f"else "
                    f"  /root/miniconda3/bin/pip install -q -i "
                    f"https://pypi.tuna.tsinghua.edu.cn/simple huggingface_hub 2>&1 | tail -1;"
                    f"  mkdir -p {da2}; "
                    f"  HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 "
                    f"/root/miniconda3/bin/python -c \""
                    f"from huggingface_hub import snapshot_download as d;import shutil,os;"
                    f"p=d('depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf');"
                    f"t='{da2}';"
                    f"[shutil.copy(os.path.join(p,f), os.path.join(t,f)) for f in "
                    f"('config.json','model.safetensors','preprocessor_config.json')]\" "
                    f"2>&1 | tail -2; "
                    f"fi; ls {da2} | tr '\\n' ' '", timeout=3600), flush=True)

        print("    pip install (base + the ai2thor venv) ...", flush=True)
        print(sh(c, "/root/miniconda3/bin/pip install -q -i "
                    "https://pypi.tuna.tsinghua.edu.cn/simple " + " ".join(PIP) +
                    " 2>&1 | tail -2", timeout=3600), flush=True)
        print("    ai2thor build + Controller smoke (first run downloads ~2 GB) ...",
              flush=True)
        smoke = sh(c, "mkdir -p /root/autodl-tmp/logs && " + SMOKE, timeout=7200)
        print(smoke[-2000:], flush=True)
        ok = "frame ok:" in smoke
        print(sh(c, "ls -d /home/sudidaren/* ; du -sh /root/.ai2thor 2>/dev/null; "
                    "df -h /root/autodl-tmp | tail -1"), flush=True)
        if not ok:
            print(f"    !! {target}: Controller smoke FAILED -> not marking ready",
                  flush=True)
            return False
        sh(c, f"touch {REMOTE_TMP}/.done")
        print(f"    {target}: DONE (ready to run an arm)", flush=True)
        return True
    finally:
        c.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self", dest="me", default="D")
    ap.add_argument("--targets", nargs="+", required=True)
    ap.add_argument("--payload", default="/root/autodl-tmp/c2c")
    ap.add_argument("--with-ai2thor", action="store_true")
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--rebuild", action="store_true",
                    help="repack the tars even if a matching sidecar says they are current")
    args = ap.parse_args()

    payload = Path(args.payload)
    build_tars(payload, rebuild=args.rebuild)
    if args.build_only:
        return 0

    rc = 0
    for t in args.targets:
        if t == args.me:
            continue
        try:
            rc |= 0 if provision(t, payload, args.with_ai2thor, args.me) else 1
        except Exception as exc:  # noqa: BLE001
            print(f"  card {t}: FAILED {type(exc).__name__}: {exc}", flush=True)
            rc |= 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
