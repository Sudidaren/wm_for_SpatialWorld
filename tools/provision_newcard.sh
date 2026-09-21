#!/usr/bin/env bash
# 把本机环境铺到新卡（2026-09-21 晚新开的 westb 卡，跑 ProCTHOR 用）。
#
# 老的三张卡全部下线，没有源卡可以 c2c 复制，只能从本机上传。
# 实测本机→AutoDL 约 1.06 MB/s。要传的（排除了用不上的）：
#   SpatialWorld 代码 + data + envs/{ai2thor,procthor}/.venv   ≈ 2.8G
#   spatialworld_eval（排除 runs/，40G 全是结果）               ≈ 20M
#   lightwm_phases（只带 checkpoints/ 与代码，排除两个大 venv）  ≈ 1.0G
#   procthor-10k 数据集                                        ≈ 122M
# AI-THOR 的 Unity 二进制（本机 2.1G）**先让卡自己下**，下不动再传。
#
# 用法：bash provision_newcard.sh [tars|upload|all]
set -uo pipefail
cd /home/sudidaren
MODE="${1:-all}"
STAGE=/home/sudidaren/_newcard_stage
LOG=/mnt/d/lightwm_out/provision_newcard.log
HOST=connect.westb.seetacloud.com
PORT=56266
PW='/dgVLk3vXhc1'

say() { echo "[prov $(date '+%F %T')] $*" | tee -a "$LOG"; }

mkdir -p "$STAGE"

make_tars() {
    say "打 4 个包（先传运的，不压缩 venv 省 CPU）"
    cd /home/sudidaren
    say "  1/4 SpatialWorld 代码 + data"
    tar -cf "$STAGE/spatialworld_code.tar" \
        --exclude='SpatialWorld/envs' --exclude='SpatialWorld/.git' SpatialWorld 2>/dev/null
    say "  2/4 SpatialWorld 的两个 venv（ai2thor + procthor）"
    tar -cf "$STAGE/spatialworld_envs.tar" \
        SpatialWorld/envs/ai2thor SpatialWorld/envs/procthor 2>/dev/null
    say "  3/4 spatialworld_eval（排除 runs/）"
    tar -cf "$STAGE/spatialworld_eval.tar" --exclude='spatialworld_eval/runs' spatialworld_eval 2>/dev/null
    say "  4/4 lightwm_phases（只留代码 + checkpoints/）"
    tar -cf "$STAGE/lightwm_phases.tar" \
        --exclude='lightwm_phases/.venv' --exclude='lightwm_phases/.venv-rfdetr' \
        --exclude='lightwm_phases/data' --exclude='lightwm_phases/.git' \
        --exclude='lightwm_phases/checkpoints_cloud' --exclude='lightwm_phases/checkpoints_local' \
        --exclude='lightwm_phases/checkpoints_dq' --exclude='lightwm_phases/checkpoints_v2' \
        --exclude='lightwm_phases/checkpoints_smoke' \
        lightwm_phases 2>/dev/null
    tar -cf "$STAGE/procthor10k.tar" -C /home/sudidaren/.prior/datasets/allenai procthor-10k 2>/dev/null
    ls -lh "$STAGE"
}

upload() {
    say "上传到 $HOST:$PORT（本机→AutoDL 实测约 1MB/s）"
    python3 - "$HOST" "$PORT" "$PW" "$STAGE" <<'PY'
import os, sys, time, paramiko
host, port, pw, stage = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(host, port=port, username='root', password=pw, timeout=30, banner_timeout=60)
sftp = c.open_sftp()
for name in ['spatialworld_code.tar', 'spatialworld_eval.tar', 'lightwm_phases.tar',
             'procthor10k.tar', 'spatialworld_envs.tar']:
    local = os.path.join(stage, name)
    if not os.path.exists(local):
        print(f'  跳过（不存在）{name}', flush=True); continue
    size = os.path.getsize(local) / 1e6
    t = time.time()
    print(f'  上传 {name} ({size:.0f} MB) …', flush=True)
    sftp.put(local, f'/root/autodl-tmp/{name}')
    dt = time.time() - t
    print(f'    完成 {dt/60:.1f} 分钟，{size/dt*1e3:.0f} KB/s', flush=True)
sftp.close(); c.close()
PY
}

unpack() {
    say "在卡上解包到 /home/sudidaren"
    python3 - "$HOST" "$PORT" "$PW" <<'PY'
import sys, paramiko
host, port, pw = sys.argv[1], int(sys.argv[2]), sys.argv[3]
c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(host, port=port, username='root', password=pw, timeout=30, banner_timeout=60)
CMDS = [
    "mkdir -p /home/sudidaren /root/.prior/datasets/allenai /root/.ai2thor",
    "cd /home/sudidaren && for f in spatialworld_code spatialworld_eval lightwm_phases spatialworld_envs; do "
    "  echo \"解 $f\"; tar -xf /root/autodl-tmp/$f.tar || echo \"$f 失败\"; done",
    "tar -xf /root/autodl-tmp/procthor10k.tar -C /root/.prior/datasets/allenai && echo '数据集 OK'",
    "ls /home/sudidaren; du -sh /home/sudidaren/* 2>/dev/null",
]
for cmd in CMDS:
    _i, o, e = c.exec_command(cmd, timeout=1800)
    print(o.read().decode()[-1500:], flush=True)
    err = e.read().decode()
    if err.strip(): print('  stderr:', err[-300:], flush=True)
c.close()
PY
}

case "$MODE" in
    tars)   make_tars ;;
    upload) upload ;;
    unpack) unpack ;;
    all)    make_tars && upload && unpack ;;
esac
say "完成（mode=$MODE）"
