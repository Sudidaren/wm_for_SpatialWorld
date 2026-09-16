#!/bin/bash
# 停掉卡上的评测相关进程：编排 / 批次 supervisor / 各类守护 / 残留 Unity。
# 保留 vLLM 与 Xvfb（重加载模型要几分钟，等你确认是否也停）。
echo "=== 停之前 ==="
echo "  orchestrator=$(pgrep -fc '[c]loud_orchestrator' || echo 0)" \
     "batch=$(pgrep -fc '[p]ython -u - ' || echo 0)" \
     "run_task=$(pgrep -fc '[s]cripts.ai2thor.work.run_task' || echo 0)" \
     "unity=$(pgrep -f '[t]hor-Linux64-.* -screen' | wc -l)" \
     "guards=$(pgrep -fc '[g]ain_guard|[z]ero_success_guard|[r]ender_watchdog' || echo 0)" \
     "vllm=$(pgrep -fc '[v]llm serve' || echo 0)"

for p in $(pgrep -f '[c]loud_orchestrator'); do kill "$p" 2>/dev/null; done
for p in $(pgrep -f '[g]ain_guard'); do kill "$p" 2>/dev/null; done
for p in $(pgrep -f '[z]ero_success_guard'); do kill "$p" 2>/dev/null; done
for p in $(pgrep -f '[r]ender_watchdog'); do kill "$p" 2>/dev/null; done
sleep 2

# 批次 supervisor（vLLM 的进程名是 "vllm serve"，不会匹配到）
for p in $(pgrep -f '[p]ython -u - '); do kill -9 "$p" 2>/dev/null; done
sleep 1
for p in $(pgrep -f '[s]cripts.ai2thor.work.run_task'); do kill -9 "$p" 2>/dev/null; done
for u in $(pgrep -f '[t]hor-Linux64-.* -screen'); do kill -9 "$u" 2>/dev/null; done
for u in $(pgrep -f '[t]hor-CloudRendering'); do kill -9 "$u" 2>/dev/null; done
sleep 3

echo "=== 停之后 ==="
echo "  orchestrator=$(pgrep -fc '[c]loud_orchestrator' || echo 0)" \
     "batch=$(pgrep -fc '[p]ython -u - ' || echo 0)" \
     "run_task=$(pgrep -fc '[s]cripts.ai2thor.work.run_task' || echo 0)" \
     "unity=$(pgrep -f '[t]hor-Linux64-.* -screen' | wc -l)" \
     "guards=$(pgrep -fc '[g]ain_guard|[z]ero_success_guard|[r]ender_watchdog' || echo 0)" \
     "vllm=$(pgrep -fc '[v]llm serve' || echo 0)"
