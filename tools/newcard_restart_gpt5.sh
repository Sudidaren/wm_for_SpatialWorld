#!/usr/bin/env bash
# 重启 GPT-5 + WM 这条臂（修掉 top_p / temperature 之后）。
#
# 背景：GPT-5 是 reasoning 模型，网关对它的要求比 OpenAI 兼容预设更严：
#   * 请求带 top_p            -> 400 unsupported_parameter
#   * temperature != 1        -> 400 unsupported_value
# preset 从 qwen3.5 继承 top_p=0.9，导致整条臂每个任务都在第一个 LLM 调用
# 上 400；worker 无输出被 supervisor 的 stall 看门狗（600s）SIGKILL，表现为
# 任务目录建了、跑了 init action、然后 returncode=-9、0 成功。
# run_ablation.sh 已按模型名把这些字段剥掉（见该文件 gpt-5 门控）。
#
# 步骤：停 supervisor -> 收 worker/thor -> 清掉旧 task_configs（YAML 里带
#       top_p）-> 用同一批 250 条任务重新拉起。
set -uo pipefail

SWE=/home/sudidaren/spatialworld_eval
RUN=wm_gpt5_ai2thor_fail250
LOG=/root/autodl-tmp/gpt5_main.log
TASKS_FILE=/root/autodl-tmp/gpt5_tasks.txt
KEY_FILE=/root/autodl-tmp/gpt5_key.txt

export DISPLAY=:99 TMPDIR_OVERRIDE=/tmp/lightwm_tmp
say() { echo "[restart $(date '+%F %T')] $*"; }

PAT="[.]venv/bin/python -u - .* ${RUN} "
PID=$(pgrep -f "$PAT" | head -1 || true)
if [ -n "${PID:-}" ]; then
    say "停 supervisor pid=$PID"
    kill -TERM "$PID" 2>/dev/null || true
    for _ in $(seq 1 20); do kill -0 "$PID" 2>/dev/null || break; sleep 2; done
    if kill -0 "$PID" 2>/dev/null; then kill -KILL "$PID" 2>/dev/null || true; fi
fi
sleep 3
pkill -TERM -f "run_task.py --config" 2>/dev/null || true
sleep 5
pkill -f thor-Linux64 2>/dev/null || true
sleep 3
say "残留 run_task=$(pgrep -cf 'run_task.py --config' || true) thor=$(pgrep -cf thor-Linux64 || true)"

# 旧 YAML 是带 top_p 的版本：清掉，交给 supervisor 重新生成。
rm -rf "$SWE/runs/$RUN/task_configs"
say "已清 task_configs"

N=$(tr ',' '\n' < "$TASKS_FILE" | grep -c . || true)
say "重新拉起：$N 条 workers=6"
cd "$SWE"
MODEL_NAME=gpt-5 BASE_URL=https://apic1.ohmycdn.com/v1 PROFILE=wm \
RUN_NAME="$RUN" SCENES=ai2thor WORKERS=6 \
LLM_API_KEY="$(cat "$KEY_FILE")" \
SMOKE_TASKS="$(cat "$TASKS_FILE")" \
setsid nohup bash run_wm_gemini_ai2thor.sh >>"$LOG" 2>&1 </dev/null &
sleep 25
say "提交后 supervisor=$(pgrep -cf "$PAT" || true)"
tail -6 "$LOG"
