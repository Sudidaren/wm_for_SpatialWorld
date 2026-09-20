# 2026-09-21 夜里的两个云上故障与修复

两份都发生在"卡上同时跑 vLLM + WingmanWM 评测"这个组合里。都不是 WM 的
算法问题，是**部署/网关约定**问题；但都会把整条臂打成一堆假失败，所以记在
这里，避免下次重开卡再踩。

## 一、GPT-5 通过网关调用必须去掉 top_p / 固定 temperature=1

**症状**：gpt5 卡上 250 条任务，任务目录都建了、init action 也跑了，然后在
`Starting task execution...` 之后 `returncode=-9`、`no result json produced`，
判定 0 成功。

**根因**：`run_ablation.sh` 里 `cfg.LLM_PRESET = 'qwen3.5'`，而 qwen3.5 这个
preset 自带 `top_p: 0.9`；`LLM_OVERRIDES` 只覆盖 model_name/base_url/key，
于是 `top_p` 漏进了 GPT-5 的请求。网关（apic1.ohmycdn.com）对 reasoning 模型
直接 400：

```
Unsupported parameter: 'top_p' is not supported with this model.
Unsupported value: 'temperature' does not support 0.2 with this model.
```

第一条会在**每个任务的第一个 LLM 调用**上炸；worker 没有 stdout 输出，被
supervisor 的 stall 看门狗（600s）SIGKILL，于是表现为上面那串 -9。
第二条是 planner 块：`config_builder` 把 planner 的 temperature 写死 0.2，
一旦打开子目标分解同样会被拒。

**修复**：在 `run_ablation.sh` 里按模型名门控（`patches/run_ablation_gpt5_safe.sh`
就是修好的整份文件，与本机 `spatialworld_eval/run_ablation.sh` 逐字节一致）：

* preset 里剔掉 `top_p`；
* monkey-patch `config_builder._apply_model_block`，把 vlm / planner 两个块的
  `temperature` 归一成 1.0、再兜底 `pop('top_p')`。

只认 `model_name.startswith('gpt-5')`，本地 vLLM 的 qwen / kimi 不受影响。
`config_builder.py` / `eval_config.py` 在 `FROZEN.sha256` 里，不能改，所以
补丁只能落在 `run_ablation.sh`（它内部本来就是靠 monkey-patch 覆盖冻结配置的）。

**冒烟验证**：单任务 `ai2thor03030` 在 Step 2..6 正常输出
`RotateRight / MoveRight / ToggleObjectOff(LightSwitch)`；生成的
`task_configs/...yaml` 里 `grep -c top_p` = 0，vlm/planner 都是 `temperature: 1.0`。

## 二、vLLM 的 gpu-memory-utilization=0.92 会把 WM 感知头饿死（8B/Kimi 卡）

**症状**：8b / kimi 卡上成批
`Environment exception: CUDA out of memory. Tried to allocate 2.00 MiB`，
以及大量 `no result json produced`；8b 卡上 vLLM 自己在 01:01 崩掉
（`EngineCore` OOM，`Process 81793 has 894.00 MiB memory in use` × 5）。

**根因**：一张 48 GB 卡上，vLLM 按 0.92 起就把 43.6 GiB 全预定了，而每个评测
worker 自带一个 WM 感知头（RF-DETR + DA2），实测 **894 MiB/worker**，6 个就是
5.4 GiB。两边互相抢：WM 分不到显存 → env_error；vLLM 需要临时分配 48 MiB 时
也没有 → EngineCore 直接 OOM 退出。

各卡实测（vLLM 启动日志）：

| 卡 | 权重+非torch | 激活 | CUDAGraph | 0.92 下的 KV | 结论 |
|---|---|---|---|---|---|
| 8b (48G) | 16.95 GiB | 2.71 | 0.24 | 23.93 GiB | KV 富余，压低即可 |
| kimi (48G) | 31.16 GiB | 1.20 | 0.38 | 11.22 GiB | 权重占大头，要留够 |
| 30b (96G) | — | — | — | — | 96G 卡余量够，不动 |

**修复**：`tools/newcard_restart_vllm_eval.sh <RUN> <UTIL> [MODEL] [NAME] [WORKERS]`

* 先抓旧 supervisor 的启动环境（`/proc/<pid>/environ`），再停评测；
* 用更低的 `--gpu-memory-utilization` 重启 vLLM，等 `/v1/models` 就绪；
* 再原样重启评测（supervisor 会重排 `failed_external` 的任务）。

实际取值：**8b → 0.76（6 worker）**、**kimi → 0.80（worker 降到 5）**。
改完 8b 空闲 13.1 G、kimi 空闲 5.5 G，不再互相抢。

## 三、看护漏掉的"孤儿 worker"

`guard.py` 原来只把 `ppid==1` 的 thor 当孤儿。但 supervisor 被 TERM/KILL 之后，
worker 被 init 收养（ppid==1）**并不会自己退出**，还会继续写同一个
`results.csv`；而它名下的 thor 的 ppid 是那个孤儿 worker，既不是 1、也不是
活着的 worker —— 旧判据两边都躲开了。

现在 `tools/newcard_guard.py` 三种都杀：孤儿 worker、`ppid==1` 的 thor、
父进程既不是活 worker 也已不存在的 thor。
