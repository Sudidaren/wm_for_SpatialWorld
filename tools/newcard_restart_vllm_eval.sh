#!/usr/bin/env bash
# 用法: bash restart_vllm_eval.sh <RUN_NAME> <UTIL>
#
# 目的：卡上是"vLLM 服务 + 6 个评测 worker（每个自带 WM 感知头，约 0.9 GiB
# 显存）"共用一张卡。vLLM 按 0.92 起，几乎把 48 GB 吃满，感知头分配不到显存
# -> "Environment exception: CUDA out of memory"，任务成批掉成 env_error。
#
# 顺序（不能颠倒，否则 vLLM 重启窗口里评测的 LLM 调用会全部失败）：
#   1. 从旧 supervisor 抓启动环境（用于原样重启）；
#   2. 停评测 supervisor + worker + thor；
#   3. 用更低的 --gpu-memory-utilization 重启 vLLM，等到 /v1/models 就绪；
#   4. 用抓到的环境原样重启评测（supervisor 会重排未成功/外部失败的任务）。
set -uo pipefail

RUN="${1:?用法: restart_vllm_eval.sh <RUN_NAME> <UTIL> [MODEL_DIR] [SERVED_NAME] [WORKERS] [API_KEY]}"
UTIL="${2:?用法: restart_vllm_eval.sh <RUN_NAME> <UTIL> [MODEL_DIR] [SERVED_NAME] [WORKERS] [API_KEY]}"
MODEL_ARG="${3:-}"
NAME_ARG="${4:-}"
WORKERS_ARG="${5:-}"
KEY_ARG="${6:-EMPTY}"        # 本地 vLLM 不校验 key
SWE=/home/sudidaren/spatialworld_eval
ENVF="/root/autodl-tmp/eval_env_${RUN}.sh"
LOGD=/root/autodl-tmp/logs
MAINLOG="/root/autodl-tmp/${RUN}_main.log"

# AI2THOR_SERVER_TIMEOUT 必须显式给：软件渲染下大场景用默认的 100s 会成批
# 超时（2026-09-21 踩过——重启时漏了这个变量，8b 卡上立刻多出一批
# "Reading from AI2-THOR backend timed out (using 100.0s)"）。
export DISPLAY=:99 TMPDIR_OVERRIDE=/tmp/lightwm_tmp AI2THOR_SERVER_TIMEOUT=300
say() { echo "[vrestart $(date '+%F %T')] $*"; }

PAT="[.]venv/bin/python -u - .* ${RUN} "
P=$(pgrep -f "$PAT" | head -1 || true)
if [ -z "${P:-}" ]; then say "找不到 ${RUN} 的 supervisor，退出"; exit 1; fi

# 1) 抓环境
# 注意：卡上 /proc/<pid>/environ 有时读出来是 0 字节（实测），抓不到就退回
# 从 supervisor 的 argv 重建（argv = <model> <url> wm <RUN> <SCENES> <WORKERS>）。
tr '\0' '\n' < "/proc/$P/environ" 2>/dev/null \
    | grep -E '^(MODEL_NAME|BASE_URL|PROFILE|RUN_NAME|SCENES|WORKERS|LLM_API_KEY|OPENAI_API_KEY)=' \
    > "$ENVF" || true
if ! grep -q '^MODEL_NAME=' "$ENVF"; then
    say "/proc environ 读不到，改从 argv 重建"
    ARGV=$(ps -o args= -p "$P")
    M=$(printf '%s\n' "$ARGV" | awk '{for(i=1;i<=NF;i++) if($i=="-"){print $(i+1); exit}}')
    U=$(printf '%s\n' "$ARGV" | awk '{for(i=1;i<=NF;i++) if($i=="-"){print $(i+2); exit}}')
    R2=$(printf '%s\n' "$ARGV" | awk '{for(i=1;i<=NF;i++) if($i=="-"){print $(i+4); exit}}')
    S=$(printf '%s\n' "$ARGV" | awk '{for(i=1;i<=NF;i++) if($i=="-"){print $(i+5); exit}}')
    W=$(printf '%s\n' "$ARGV" | awk '{for(i=1;i<=NF;i++) if($i=="-"){print $(i+6); exit}}')
    printf 'MODEL_NAME=%s\nBASE_URL=%s\nPROFILE=wm\nRUN_NAME=%s\nSCENES=%s\nWORKERS=%s\nLLM_API_KEY=%s\n' \
        "$M" "$U" "$R2" "$S" "$W" "${KEY_ARG:-EMPTY}" > "$ENVF"
fi
export AI2THOR_SERVER_TIMEOUT=300
chmod 600 "$ENVF"
say "已抓环境 -> $ENVF"
# vLLM 的模型目录 / served name 优先用命令行给的（vLLM 可能已经崩了，
# 这时 pgrep 抓不到），否则从活着的进程命令行取，保证重启后一字不差。
VL_MODEL="$MODEL_ARG"
VL_NAME="$NAME_ARG"
if [ -z "$VL_MODEL" ] || [ -z "$VL_NAME" ]; then
    VLLM_ARGS=$(ps -eo args --no-headers | grep "[v]llm serve" | head -1 || true)
    VL_MODEL=$(printf '%s\n' "$VLLM_ARGS" | awk '{for(i=1;i<=NF;i++) if($i=="serve") print $(i+1)}')
    VL_NAME=$(printf '%s\n' "$VLLM_ARGS" | awk '{for(i=1;i<=NF;i++) if($i=="--served-model-name") print $(i+1)}')
fi
[ -n "$VL_MODEL" ] && [ -n "$VL_NAME" ] || { say "没解析到 vLLM 参数，退出"; exit 1; }
if [ -n "$WORKERS_ARG" ]; then
    sed -i "s#^WORKERS=.*#WORKERS=${WORKERS_ARG}#" "$ENVF"
    say "worker 数改为 ${WORKERS_ARG}"
fi
say "vLLM model=$VL_MODEL name=$VL_NAME -> util=$UTIL"

# 2) 停评测
say "停 supervisor pid=$P"
kill -TERM "$P" 2>/dev/null || true
for _ in $(seq 1 20); do kill -0 "$P" 2>/dev/null || break; sleep 2; done
kill -0 "$P" 2>/dev/null && kill -KILL "$P" 2>/dev/null || true
sleep 3
for q in $(pgrep -f "[.]venv/bin/python -u -m scripts.ai2thor.work.run_task"); do
    args=$(ps -o args= -p "$q" 2>/dev/null || true)
    case "$args" in *"$RUN"*) kill -TERM "$q" 2>/dev/null || true ;; esac
done
sleep 6
for q in $(pgrep -f "[.]venv/bin/python -u -m scripts.ai2thor.work.run_task"); do
    args=$(ps -o args= -p "$q" 2>/dev/null || true)
    case "$args" in *"$RUN"*) kill -9 "$q" 2>/dev/null || true ;; esac
done
sleep 2
for t in $(pgrep -f "thor-Linux6[4]"); do kill -9 "$t" 2>/dev/null || true; done
sleep 2
say "评测已停 worker=$(pgrep -cf 'work.run_tas[k]' || true) thor=$(pgrep -cf 'thor-Linux6[4]' || true)"

# 3) 重启 vLLM
export VLLM_USE_FLASHINFER_SAMPLER=0
for v in $(pgrep -f "vllm serv[e]"); do kill "$v" 2>/dev/null || true; done
sleep 8
# EngineCore 等子进程可能还挂在显存上，等它真的放掉
for _ in $(seq 1 12); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "${used:-99999}" -lt 8000 ] && break
    sleep 5
done
say "vLLM 停后显存占用=${used:-?} MiB"
mkdir -p "$LOGD"
say "重启 vLLM（$UTIL）"
setsid nohup /root/miniconda3/bin/vllm serve "$VL_MODEL" \
    --served-model-name "$VL_NAME" --host 127.0.0.1 --port 8000 \
    --max-model-len 32768 --gpu-memory-utilization "$UTIL" \
    --limit-mm-per-prompt '{"image":32}' --trust-remote-code \
    >> "$LOGD/vllm_${VL_NAME}.log" 2>&1 </dev/null &
UP=0
for i in $(seq 1 120); do
    sleep 5
    if curl -sf -m 5 http://127.0.0.1:8000/v1/models 2>/dev/null | grep -q "$VL_NAME"; then
        say "vLLM UP after $((i * 5))s"; UP=1; break
    fi
done
if [ "$UP" != "1" ]; then
    say "vLLM 450s 内没起来，最后日志："
    tail -20 "$LOGD/vllm_${VL_NAME}.log"
    exit 2
fi
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader

# 4) 重启评测
set -a; . "$ENVF"; set +a
LAUNCH="$SWE/run_wm_gemini_ai2thor.sh"
[ -f "$LAUNCH" ] || LAUNCH="$SWE/run_ablation.sh"
say "重启评测：$RUN（$LAUNCH）"
cd "$SWE"
setsid nohup bash "$LAUNCH" >>"$MAINLOG" 2>&1 </dev/null &
sleep 25
say "提交后 supervisor=$(pgrep -cf "$PAT" || true)"
tail -4 "$MAINLOG"
