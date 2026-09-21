# 手工补的一行：procthor601（2026-09-21 21:0x）

## 为什么会有这一行

`closed_gemini_wm` 的最后一条 `procthor601` 跑了两次：

| 尝试 | 进度 | 记录的结果 |
|---|---|---|
| 第 1 次（`worker_0`） | 44/56 步 | `task_result=failure`，`fail_reason="Model indicated FAIL"`，模型自己声明放弃 |
| 第 2 次（`worker_2`，重试） | 41/56 步 | —— 用户决定不再重试，人工叫停 |

supervisor 只在任务**最终判定后**才写 `results.csv`，所以叫停时这条不在表里。
用户裁定：**模型自己放弃即算失败**，于是补一行计入分母。

## 这一行的来源（全部取自第 1 次的原始记录，未臆造）

* 原始文件：`runs/closed_gemini_wm/procthor/procthor601/worker_0/log.json`
* `instruction` / `failure_reason` / `Actual Steps` / `Max Steps` / token 用量 / API 调用数
  全部来自该文件的 `metadata`
* `actual_actions`：从该文件 assistant 消息里的 `<ACTION>…</ACTION>` 逐条抽出（44 条，末条为 `FAIL`）
* `golden_action` / `Scene` / `Category` / `Task Type` / `golden_actions_count`：
  取自冻结基线 `runs/gemini31pro_procthor127_frozen_v1/results.csv` 的同一任务
* `Duration Sec=1158.0`：`reset_20260921_200504.png`(20:05:05) → `log.json`(20:24:23) 的墙钟差
* `Status=failed_model`、`Success=False`：按"模型放弃=失败"的裁定

第 2 次尝试的 token 没有计入（它没有产出判定，`results.csv` 里也只保留一次尝试的用量），
所以这条的账面花费是**低估**的：实际两次合计约 $5。

## 影响

补上这条之后 ProCTHOR 20 条齐了：**基线 1/20 → WM+Gemini 2/20**。
配对上是 1 条两边都成、1 条基线败而 WM 救回、18 条两边都败。
换句话说，`procthor601`（"去客厅桌上拿笔"）这条 WM 也没救回来，和基线一样算失败。
