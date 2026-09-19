#!/usr/bin/env bash
# 云端编排 v4：在 v3 基础上增加"停测名单"（stopped_arms.txt）。
#   某个臂被判定无收益（见 gain_guard.sh）后，会被写进名单；ensure_batch 看到
#   名单里的 run 直接当作"已完成"，不再拉起、也不再继续后面的场景。
#
# v2 的完成判据只看 completed∈{true,false}，而外部故障（no result json /
# 超时 / 渲染崩溃）的任务 completed=None、status=failed_external，于是：
#   supervisor 每轮都会重新排队这 46 个任务（RESUME_SKIP_DECIDED 不认它们）
#   → 编排看到 decided 永远到不了 TOTAL → 整批重启 → 无限循环，后续阶段排不上。
# v3 把"任务已终结"定义为：
#   completed∈{true,false}  **或者**  status=failed_external 且 attempts 已用尽
# （attempts 上限 = 1 + eval_config.MAX_EXTERNAL_RETRIES，默认 3）。
# 只改编排何时收工，不碰冻结的判定逻辑：state.json / results.csv 逐字节不动，
# 外部失败仍然记为 null，仍然不算进 TSR 分母。
#
# 用法：
#   bash cloud_orchestrator_v4.sh "spec" ["spec"...] [ ";;" "spec"... ]
#   spec = model|base_url|plan|profile|run_name|workers
#   ";;" 分隔各阶段，阶段内并行、阶段间串行。
#
# 可用环境变量：RUNS_DIR / ORCH_LOG / IDLE_LIMIT / CHECK_EVERY / TOTAL / WAIT_MODEL / DISPLAY
set -u
cd /home/sudidaren/spatialworld_eval || exit 1

# A pidfile: every process-finding command in this harness has to avoid
# matching its own command line, and a script that mentions
# "cloud_orchestrator_v4.sh" matches a pgrep for it.  A pid has no such
# problem, and the refresh path (tools/card_ctl.py) needs to know which
# process to restart.
PIDFILE="${PIDFILE:-/root/orchestrator.pid}"
echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT

# 找到"真的在跑编排器"的进程：命令行里 `cloud_orchestrator_v4.sh ` 之后紧跟
# 队列表述。用 awk 正则而不是 pgrep，任何提到脚本名的命令都不会匹配自己。
orchestrator_pids() {
  ps -eo pid=,args= | awk '/cloud_orchestrator_v4\.sh [a-z0-9]/{print $1}'
}

RUNS_DIR="${RUNS_DIR:-/root/autodl-tmp/runs_local}"
LOG="${ORCH_LOG:-/root/orchestrator_v2.log}"
IDLE_LIMIT="${IDLE_LIMIT:-900}"
MAX_TRIES="${MAX_TRIES:-3}"          # 与 supervisor 的 1+MAX_EXTERNAL_RETRIES 对齐
STOP_FILE="${STOP_FILE:-/root/stopped_arms.txt}"   # 每行一个 run 名：停测
CHECK_EVERY="${CHECK_EVERY:-120}"
TOTAL="${TOTAL:-438}"
SCENES="${SCENES:-ai2thor,procthor}"     # 本次可只跑 ai2thor（311 条）
WAIT_MODEL="${WAIT_MODEL:-}"
if [ -n "${ORCH_DISPLAY:-}" ]; then export DISPLAY="$ORCH_DISPLAY"; fi

# python3 不一定在 PATH（部分云镜像只有 conda 的 python）
PY="${PY:-}"
if [ -z "$PY" ]; then
  for cand in python3 /root/miniconda3/bin/python /usr/bin/python3 python; do
    if command -v "$cand" >/dev/null 2>&1; then PY="$cand"; break; fi
  done
fi

log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

decided() {
  local f="$RUNS_DIR/$1/state.json"
  if [ ! -f "$f" ]; then echo 0; return; fi
  "$PY" -c '
import json,sys
try: d=json.load(open(sys.argv[1]))
except Exception: print(0); raise SystemExit
print(sum(1 for t in d.get("tasks",[]) if str(t.get("completed","")).lower() in ("true","false")))
' "$f" 2>/dev/null || echo 0
}

# 已终结 = 计分完成（true/false）或外部故障且重试次数已用尽
terminal() {
  local f="$RUNS_DIR/$1/state.json"
  if [ ! -f "$f" ]; then echo 0; return; fi
  "$PY" -c '
import json,sys
path, maxtries = sys.argv[1], int(sys.argv[2])
try: d=json.load(open(path))
except Exception: print(0); raise SystemExit
n=0
for t in d.get("tasks",[]):
    if str(t.get("completed","")).lower() in ("true","false"):
        n+=1; continue
    if str(t.get("status","")) == "failed_external" and int(t.get("attempts") or 0) >= maxtries:
        n+=1
print(n)
' "$f" "$MAX_TRIES" 2>/dev/null || echo 0
}

# 哪一个是 supervisor。RUN_NAME 会被 worker 继承，所以只看环境变量会把
# worker 认成 supervisor：2026-09-19 卡 D 上 supervisor 被误杀后，编排器每轮
# 都看见"运行中"，整批就停在那里不再补进程。worker 的命令行里有 run_task，
# supervisor 没有，用这一点区分。
sup_pid() {
  local p cmd
  for p in $(pgrep -f "[p]ython -u -" 2>/dev/null); do
    cmd=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null)
    case "$cmd" in *run_task*) continue;; esac
    if tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | grep -q "^RUN_NAME=$1$"; then
      echo "$p"; return 0
    fi
  done
  return 1
}

sup_elapsed() { ps -o etimes= -p "$1" 2>/dev/null | tr -d ' ' | head -1; }

newest_age() {
  local f t now
  f=$(find "$RUNS_DIR/$1" -type f -printf '%T@\n' 2>/dev/null | sort -rn | head -1)
  if [ -z "$f" ]; then echo 999999; return; fi
  t=${f%%.*}; now=$(date +%s); echo $(( now - t ))
}

kill_batch() {
  local rn="$1" p c g
  for p in $(pgrep -f "[p]ython -u -" 2>/dev/null); do
    if tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | grep -q "^RUN_NAME=$rn$"; then
      for c in $(pgrep -P "$p" 2>/dev/null); do
        for g in $(pgrep -P "$c" 2>/dev/null); do kill -9 "$g" 2>/dev/null; done
        kill -9 "$c" 2>/dev/null
      done
      kill -9 "$p" 2>/dev/null
    fi
  done
  sleep 3
}

start_batch() {   # model url plan profile run workers [extra_env]
  local model="$1" url="$2" plan="$3" profile="$4" rn="$5" workers="$6" extra="${7:-}"
  log "启动 $rn (model=$model plan=$plan profile=$profile workers=$workers env=${extra:-<none>})"
  # AI2THOR_* timeout: 软件渲染(Xvfb)下场景装载只要被抢 CPU 就可能超过 ai2thor
  # 默认的 100s server_timeout，进而被误判成环境故障。放宽到 300s/600s。
  # 每个臂自带环境变量（spec 的第 7 段，`K=V;K=V`）。一条队列里要塞
  # 不同配置（换深度源、换头、换标定常数）就必须有这个，否则整条队列共用
  # 一套 env，"换个头再跑"只能一条条手动开。
  (
    if [ -n "$extra" ]; then
      IFS=';' read -r -a _pairs <<< "$extra"
      for _kv in "${_pairs[@]}"; do
        [ -n "$_kv" ] && export "$_kv"
      done
    fi
    WM_HISTORY_TURNS="${WM_HISTORY_TURNS:-29}" WM_HISTORY_MAX_SIDE="${WM_HISTORY_MAX_SIDE:-0}" \
      AI2THOR_SERVER_TIMEOUT="${AI2THOR_SERVER_TIMEOUT:-300}" \
      AI2THOR_START_TIMEOUT="${AI2THOR_START_TIMEOUT:-600}" \
      MODEL_NAME="$model" BASE_URL="$url" PLAN="$plan" PROFILE="$profile" \
      RUN_NAME="$rn" SCENES="$SCENES" WORKERS="$workers" \
      setsid nohup bash /home/sudidaren/spatialworld_eval/run_ablation.sh \
      > "/root/run_${rn}.log" 2>&1 < /dev/null &
  )
  sleep 20
}

# 返回 0 = 仍需继续（未完成），1 = 该批次已完成
ensure_batch() {  # model url plan profile run workers [extra_env]
  local model="$1" url="$2" plan="$3" profile="$4" rn="$5" workers="$6" extra="${7:-}"
  local n t pid age el
  if [ -f "$STOP_FILE" ] && grep -qx "$rn" "$STOP_FILE" 2>/dev/null; then
    log "$rn 在停测名单中 -> 视为已完成，不再拉起"
    pid=$(sup_pid "$rn" || true)
    [ -n "$pid" ] && { log "$rn 停测名单命中 -> 停掉在跑进程 (pid=$pid)"; kill_batch "$rn"; }
    return 1
  fi
  n=$(decided "$rn")
  t=$(terminal "$rn")
  if [ "${t:-0}" -ge "$TOTAL" ]; then
    pid=$(sup_pid "$rn" || true)
    if [ -n "$pid" ]; then
      log "$rn 已终结 $t/$TOTAL -> 停掉仍在重跑已耗尽任务的进程 (pid=$pid)"
      kill_batch "$rn"
    fi
    log "$rn 已完成 $t/$TOTAL（计分 $n，外部失败 $((t - n))）"
    return 1
  fi
  pid=$(sup_pid "$rn" || true)
  if [ -z "$pid" ]; then
    log "$rn 无进程 (已终结 $t/$TOTAL, 计分 $n) -> 启动"
    kill_batch "$rn"
    start_batch "$model" "$url" "$plan" "$profile" "$rn" "$workers" "$extra"
    return 0
  fi
  age=$(newest_age "$rn"); el=$(sup_elapsed "$pid")
  if [ "${age:-0}" -gt "$IDLE_LIMIT" ] && [ "${el:-0}" -gt "$IDLE_LIMIT" ]; then
    log "$rn 卡住 ${age}s (已终结 $t/$TOTAL, 计分 $n, pid=$pid) -> 重启"
    kill_batch "$rn"
    start_batch "$model" "$url" "$plan" "$profile" "$rn" "$workers" "$extra"
    return 0
  fi
  log "$rn 运行中 计分 $n/已终结 $t/$TOTAL (最新产出 ${age}s 前)"
  return 0
}

wait_stage() {   # 参数为若干 spec；全部完成后返回
  local pending spec
  while true; do
    pending=0
    for spec in "$@"; do
      IFS='|' read -r model url plan profile rn workers envspec <<< "$spec"
      ensure_batch "$model" "$url" "$plan" "$profile" "$rn" "$workers" "${envspec:-}" || continue
      pending=1
    done
    [ "$pending" -eq 0 ] && { log "阶段完成"; return 0; }
    sleep "$CHECK_EVERY"
  done
}

log "=== 编排 v4 启动 (runs=$RUNS_DIR, display=${DISPLAY:-<unset>}) ==="

if [ -n "$WAIT_MODEL" ]; then
  ok=0
  for _ in $(seq 1 90); do
    if curl -s -m 6 http://127.0.0.1:8000/v1/models | grep -q "$WAIT_MODEL"; then
      log "vLLM $WAIT_MODEL 就绪"; ok=1; break
    fi
    sleep 20
  done
  [ "$ok" = "1" ] || { log "vLLM $WAIT_MODEL 超时"; exit 1; }
fi

stage=()
for arg in "$@"; do
  if [ "$arg" = ";;" ]; then
    [ "${#stage[@]}" -gt 0 ] && { wait_stage "${stage[@]}"; stage=(); }
  else
    stage+=("$arg")
  fi
done
[ "${#stage[@]}" -gt 0 ] && wait_stage "${stage[@]}"

log "=== 全部阶段完成 ==="
