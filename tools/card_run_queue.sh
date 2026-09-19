#!/usr/bin/env bash
# Start one card's arm queue: Xvfb + vLLM + watchdog + orchestrator.
#
#   VLLM_MODEL=/root/autodl-tmp/models/qwen3vl-30b-bf16 \
#   VLLM_NAME=qwen3vl-30b \
#   bash tools/card_run_queue.sh "qwen3vl-30b|http://127.0.0.1:8000/v1|off|wm|wm_q30_da2_438_v2|8"
#
# Everything long-running is started with setsid+nohup so an ssh drop cannot
# kill it, and everything is stopped by PID (never by name pattern: the
# handover doc records `pkill -f x` killing the shell that ran it).
set -u

QUEUE=("$@")
[ "${#QUEUE[@]}" -eq 0 ] && { echo "usage: $0 \"model|url|plan|profile|run_name|workers\" ..."; exit 2; }

VLLM_MODEL="${VLLM_MODEL:-/root/autodl-tmp/models/qwen3vl-30b-bf16}"
VLLM_NAME="${VLLM_NAME:-qwen3vl-30b}"
VLLM_PORT="${VLLM_PORT:-8000}"
# 显存余量：Kimi-VL 的视觉塔在图片多的请求上会额外申请激活显存，0.92 让引擎在
# 卡 A 上 OOM（2026-09-19 14:18）。留出余量比跑满更划算。
VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.92}"
# 显式赋值：`${V:-{"image":32}}` 的括号在 bash 里有歧义，实测会多出一个 `}`
if [ -z "${VLLM_MM_LIMIT:-}" ]; then VLLM_MM_LIMIT='{"image":32}'; fi
LOGDIR=/root/autodl-tmp/logs
DISPLAY_NUM="${ORCH_DISPLAY:-:99}"
mkdir -p "$LOGDIR"

echo "== [1/4] Xvfb on $DISPLAY_NUM =="
if ! pgrep -f "[X]vfb $DISPLAY_NUM" >/dev/null; then
  setsid nohup Xvfb "$DISPLAY_NUM" -screen 0 1280x1024x24 \
    > "$LOGDIR/xvfb.log" 2>&1 < /dev/null &
  sleep 3
fi
pgrep -f "[X]vfb $DISPLAY_NUM" >/dev/null && echo "   xvfb ok" || echo "   xvfb FAILED (see $LOGDIR/xvfb.log)"

echo "== [2/4] vLLM ($VLLM_NAME from $VLLM_MODEL) =="
if curl -sf -m 6 "http://127.0.0.1:$VLLM_PORT/v1/models" >/dev/null 2>&1; then
  echo "   already serving"
else
  export VLLM_USE_FLASHINFER_SAMPLER=0
  setsid nohup /root/miniconda3/bin/vllm serve "$VLLM_MODEL" \
    --served-model-name "$VLLM_NAME" --host 127.0.0.1 --port "$VLLM_PORT" \
    --max-model-len 32768 --gpu-memory-utilization "$VLLM_GPU_UTIL" \
    --limit-mm-per-prompt "$VLLM_MM_LIMIT" --trust-remote-code \
    > "$LOGDIR/vllm.log" 2>&1 < /dev/null &
  echo "   starting; waiting for /v1/models (up to 20 min)"
  for i in $(seq 1 120); do
    curl -sf -m 6 "http://127.0.0.1:$VLLM_PORT/v1/models" >/dev/null 2>&1 && break
    sleep 10
  done
  if curl -sf -m 6 "http://127.0.0.1:$VLLM_PORT/v1/models" >/dev/null 2>&1; then
    echo "   vLLM ready"
  else
    # Do not start an arm here: every task would call an endpoint that is not
    # there, and the harness records that as a model failure -- a full batch
    # of failed_model with a perfectly healthy-looking card.
    echo "   vLLM NOT ready -> not starting the arm (tail $LOGDIR/vllm.log)"
    tail -5 "$LOGDIR/vllm.log" 2>/dev/null | sed 's/^/     /'
    exit 1
  fi
fi

echo "== [3/4] watchdog =="
if [ -f /root/watchdog.pid ] && kill -0 "$(cat /root/watchdog.pid)" 2>/dev/null; then
  echo "   already running (pid $(cat /root/watchdog.pid))"
else
  setsid nohup env RUN_NAME="" INTERVAL=60 bash \
    /home/sudidaren/lightwm_phases/tools/card_watchdog.sh \
    > "$LOGDIR/watchdog.boot.log" 2>&1 < /dev/null &
  sleep 2
fi
if [ -f /root/watchdog.pid ] && kill -0 "$(cat /root/watchdog.pid)" 2>/dev/null; then
  echo "   watchdog ok (pid $(cat /root/watchdog.pid))"
else
  echo "   watchdog FAILED"
fi

echo "== [4/4] arm queue =="
if pgrep -f "[c]loud_orchestrator_v4" >/dev/null 2>&1; then
  # Two orchestrators on one box would both start the same arm and race on
  # results.csv, so the second caller is a no-op (the keepalive re-runs this
  # script after a crash, and a stale pair would be worse than a stall).
  echo "   orchestrator already running -> not starting a second one"
  pgrep -af "[c]loud_orchestrator_v4" | head -2
  exit 0
fi
export RUNS_DIR=/root/autodl-tmp/runs_local
export ORCH_LOG=/root/orchestrator_v4.log
export ORCH_DISPLAY="$DISPLAY_NUM"
export SCENES="${SCENES:-ai2thor,procthor}"
export TOTAL="${TOTAL:-438}"
cd /home/sudidaren/spatialworld_eval || exit 1
setsid nohup bash /home/sudidaren/spatialworld_eval/cloud_orchestrator_v4.sh "${QUEUE[@]}" \
  > /root/orch_boot.out 2>&1 < /dev/null &
sleep 20
echo "   --- orchestrator log ---"
tail -5 "$ORCH_LOG" 2>/dev/null
echo "   --- running ---"
pgrep -af "[c]loud_orchestrator_v4" | head -2
