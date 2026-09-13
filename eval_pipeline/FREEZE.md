# eval_pipeline v1.0 — FROZEN (2026-09-13)

本目录的评测管线自本次冻结起视为**只读**：不得修改任何判定逻辑、指标口径、
状态分类、CSV/统计结构。任何修改都必须重新运行 `verify_freeze.sh`、更新本
文件的校验和并升版本号。

## 1. 冻结基线

- SpatialWorld 官方仓库：`Hongcheng-Gao/SpatialWorld`，commit
  `f47b1e0`（2026-06-19）。已用全新克隆确认该 commit 与本地
  `origin/main` 完全一致。
- 评测集：`eval_sets/eval_set_v1.json`（家庭单 agent 476 个任务；
  AI2-THOR 311 + ProcTHOR 127 + VirtualHome 38），
  SHA256 `54626ae4ae859c7178a6e1481da201a8f1a2573c2963e05b94433dc4f3037227`。

## 2. 必需的环境补丁（原因与证据）

文件：`patches/ai2thor_run_task_success_conditions.patch`
（作用对象：官方 `scripts/ai2thor/work/run_task.py`）

原因：官方 evaluator（`evaluation/ai2thor/base.py::create_evaluator_from_config`）
读取 `success_conditions`（复数键）+ `success_logic`；而官方 ai2thor 的
`load_task_from_folder` 只写单数键并把 list 塞进去，导致 evaluator 走 legacy
分支、对 list 调用 `.get()` 抛异常、`perform_final_evaluation` 静默返回 0
—— 所有任务恒判失败。官方 VirtualHome 的加载器（同仓库
`scripts/virtualhome/work/run_task.py` 294–317 行）**本来就做了同样的归一化**，
ProcTHOR 走完整 task.json；只有 ai2thor 缺失。该补丁让 ai2thor 与其官方
evaluator 的接口契约一致，**不改变任何任务的成功条件本身**。

## 3. 冻结前验证记录（2026-09-13）

| 验证 | 内容 | 结果 |
|---|---|---|
| golden 重放（不用模型） | ai2thor05523（2 步）、ai2thor02436（6 步）、procthor304（5 步） | **3/3 Completed=true，TSR=1.0**（`runs/freeze_check_golden`） |
| 官方 LLM 路径 | Gemini-3.1-Pro-Preview 跑 ai2thor05523（官方 run_task） | **Completed=true，steps=2**（`runs/check2_gemini_ai2thor05523`） |
| 判定口径 | 13 组样例对比官方 `decide_csv_status_from_result` | 全部一致 |
| null 规则 | API/env 故障 → null，不计入 TSR 分母，自动重试 2 次 | 已实现 |

复现命令：`bash verify_freeze.sh`（运行上述 golden 三任务并要求全部为 true）。

## 4. 已知边界（未纳入本次冻结声明）

- **VirtualHome 38 个任务**：评测代码路径存在，但本地 WSLg 下官方
  Unity 启动器返回 HTTP 502（`-batchmode`/窗口模式两种都试过），本次未
  完成运行验证。在修复并单独验证之前，VH 结果不得进入论文表格。
- WingmanWM 仅适用于 AI2-THOR + ProcTHOR（438），VH 无米制深度。
- 全局 760 条记录中，multi-agent（46）、CARLA（80）、EmbodiedCity（53）、
  Game（105）不在本冻结范围。

## 5. 文件校验和

见 `FROZEN.sha256`（覆盖本目录所有 `.py`、README、本文件与补丁文件）。

