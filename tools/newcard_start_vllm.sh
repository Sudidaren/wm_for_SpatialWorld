#!/bin/bash
# Start vLLM on a card.  Args: <model_dir> <served_name>
# Same recipe the earlier cloud arms used (spatialworld_eval/remote_start_vllm.sh):
#   * VLLM_USE_FLASHINFER_SAMPLER=0  -- required on Blackwell (sm120), otherwise
#     vLLM dies with "requires GPUs with sm75 or higher";
#   * max-model-len 32768 and up to 32 images per prompt, which is what the
#     eval's 29-turn history needs;
#   * gpu-memory-utilization 0.92.
# ASCII only: this travels through paramiko.
set -u
MODEL="$1"; NAME="$2"
LOG=/root/autodl-tmp/logs/vllm_$NAME.log
mkdir -p /root/autodl-tmp/logs
export VLLM_USE_FLASHINFER_SAMPLER=0
pkill -f 'vllm serv[e]' 2>/dev/null
sleep 4
echo "[vllm] serving $MODEL as $NAME -> $LOG"
setsid nohup /root/miniconda3/bin/vllm serve "$MODEL" \
    --served-model-name "$NAME" --host 127.0.0.1 --port 8000 \
    --max-model-len 32768 --gpu-memory-utilization 0.92 \
    --limit-mm-per-prompt '{"image":32}' --trust-remote-code \
    > "$LOG" 2>&1 < /dev/null &
for i in $(seq 1 90); do
    sleep 5
    if curl -sf -m 5 http://127.0.0.1:8000/v1/models 2>/dev/null | grep -q "$NAME"; then
        echo "[vllm] UP after $((i*5))s"
        curl -s http://127.0.0.1:8000/v1/models | head -c 200
        exit 0
    fi
done
echo "[vllm] NOT UP after 450s; last log lines:"
tail -15 "$LOG"
exit 1
