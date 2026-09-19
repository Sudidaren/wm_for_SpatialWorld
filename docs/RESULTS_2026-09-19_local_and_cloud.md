# 2026-09-19 全天结果汇总（本地 + 云端）

生成时间：2026-09-20 00:50 ｜ 执行：Codex

---

## 0. 一句话

**云上今天确实跑了东西**，而且有记录：5 张卡从 11:27 起陆续恢复、12:50 起被
`monitor_cards.py` 每 300 秒采一次，**最后可达快照是 2026-09-19 17:57:35**；
之后 5 张卡全部 `UNREACHABLE`（和"今天开不了"对得上）。
**卡上有完成的批次（含 311 全量跑完的臂），但 `results.csv` 一个都没拉回本地**
（`cloud_monitor/pulled/` 最后一次更新停在 09-17 16:31）。

---

## 1. 云端：今天跑过哪些臂（17:57:35 最后可达快照）

记录来源：`/mnt/d/lightwm_out/monitor_loop.out`（141 个快照，12:50:40 → 00:43:17）
＋ `cards_monitor.log`。格式 `批次=已完成/总数(ok成功数)`。

| 卡 | 批次 | 进度 | 成功 | 备注 |
|---|---|---|---|---|
| **C** | **`base_q30_a311`** | **311/311 完成** | **18** | Qwen3-VL-30B **纯基线**，13:04 建好权重后启动 |
| C | `base_q30_sameinfo_a311` | 200/311 | 9 | 仍在跑时卡掉了 |
| **D** | **`wm_q30_da2_a311`** | **311/311 完成** | **13** | WM+30B（DA2 深度） |
| D | `wm_q30_da2_a2_noinject` | 199/311 | 8 | A2 消融（不注入） |
| D | `wm_q30_da2_a311_xvfb_failed` | 216/311 | 0 | 环境故障作废 |
| D | `wm_q30_da2_a311_errored_xvfb_20260919` | 261/311 | 3 | 环境故障作废 |
| B | `wm_q30_da2_final_a311` | 152/311 | 7 | 未完成 |
| B | `wm_q30_da2_a1_nomem` | 189/311 | 7 | 未完成 |
| A | `wm_kimi_da2_a311` | 219/311 | 3 | WM + Kimi-VL-A3B |
| A | `wm_kimi_da2_a311_errored_1` / `_2` | 311/311 · 54/311 | 0 · 1 | 环境故障作废 |
| E | `wm_q8b_da2_a311` | 192/311 | 10 | WM + Qwen3-VL-8B |
| E | `wm_q8b_da2_a311_errored_1` / `_2` | 311/311 · 70/311 | 0 · 8 | 环境故障作废 |

**两个跑完的全量数字（AI2-THOR 311）**：
Qwen3-VL-30B 纯基线 **18/311（5.8%）**；WM(DA2)+30B **13/311（4.2%）**。
—— 注意这是**云上口径**（10 worker、卡上 vLLM），和本地 Gemimi 口径不能直接相加。

**没拉的**：以上每个批次的 `results.csv` / `summary.json` 都还在对应卡上。
卡恢复后第一件事应该是各卡 `card_ctl.py stop` 前先把 run 目录拉回来。

---

## 2. 本地：今天跑过的全部批次

### 2.1 AI2-THOR（`spatialworld_eval/runs/`）

| 批次 | 任务数 | 成功 | 时间 | 说明 |
|---|---:|---:|---|---|
| `gemini31pro_ai2thor_procthor_438_frozen_v1` | 105 | **29** | 16:09 | Gemini 冻结臂（含 6 个 failed_external）＝ 主表 T6 基线 |
| `wm_gemini31pro_da2_nohint_a311` | 10 | 6 | 16:38 | WM + Gemini，**不注入** |
| `wm_gemini31pro_da2_a311` | 24 | 5 | 18:44 | WM + Gemini，**注入** |
| `wm_gemini31pro_simple20` | 20 | **15** | 21:33 | 简单题 pilot（改动前口径） |
| `wm30a_03001_retest` | 1 | 1 | 23:31 | A′ 修复后单任务复测：failure → success |
| `dbg03001` / `dbg03001b` / `dbg_payload` | 1 / 1 / 1 | 0 / 0 / 0 | 21:46–22:22 | 排障 |
| `dbg_payload2` | 1 | 1 | 22:24 | 排障 |
| **`wm_gemini31pro_simple40`** | **40** | **18** | 00:28 | 本轮：40 个"基线失败里最简单"的配对 |
| `wm_gemini31pro_fail210` | 4/210 | 1 | 进行中 | 本轮：剩下的**全部**基线失败任务 |

### 2.2 TVRBench 加题（`tvrbench_addon/runs/`）

| 批次 | 结果 |
|---|---|
| `stage1_ithor_gpt5` | 0/5（GPT-5，iTHOR 目标视角复现） |
| `smoke_gpt5_2` | 0/2（GPT-5） |
| `dryrun1` / `dryrun_check` / `dryrun_check2` | 0/1 各（dry-run，非模型成绩） |
| `replay_golden` | 255 条 golden 轨迹回放，验收用，非模型成绩 |

---

## 3. Gemini 基线到底多少（回答"一共成功了多少"）

| 口径 | 总数 | 成功 | 成功率 |
|---|---:|---:|---:|
| **Gemini-3.1-Pro 基线（AI2-THOR 311，回放重判）** | **311** | **61** | **19.6%** |
| Qwen3-VL-30B 纯基线（本地） | 311 | 20 | 6.4% |
| Qwen3-VL-30B 纯基线（云上 `base_q30_a311`） | 311 | 18 | 5.8% |

分类：Daily 42/177→成功 42；Work 10；Entertain 7；Travel 2。
`Completed=true` 与 `Status=success` 逐行一致，没有对不上的行。

**所以"基线失败"共 250 条**：其中 40 条已在 `wm_gemini31pro_simple40` 跑完
（WM 成功 18），**剩下 210 条正在跑**（`wm_gemini31pro_fail210`）。

---

## 4. 现在的机器状态与接下来

**在跑的**

1. `wm_gemini31pro_fail210`：210 条 Gemini 基线失败任务 × WM+Gemini，4 worker；
2. 后接链条已挂好（`lightwm_phases/tools/chain_gpt5_after_gemini.sh`）：
   等 210 跑完 → `reap_sim.sh` 清场 → **自动启动 GPT-5 + WM**（同一批 **250** 条失败任务，
   `wm_gpt5_ai2thor_fail250`）。中途挂了会自动补跑一次再交班。

**别处的进程（不是我这轮起的，之前的会话留的）**

* `monitor_cards.py --loop 300`（09-19 12:38 起）—— 还在每 5 分钟探一次卡，卡一回来就会记上；
* `watch_gemini_pair.py`（16:25 起）、`watch_main_table.py`（16:32 起）—— 盯本地 Gemini 三臂；
* `c2c_copy.py`（09-19 14:10 起，D→B 拷 30B 权重）——**已经卡死**：最后一条日志 14:23、
  停在 `futex_do_wait`，卡不通所以永远不会完成。卡恢复后要重新起。

**没动的**：`spatialworld_eval` 冻结文件 19/19 指纹全对；本轮只新增了 `runs/` 下的目录
和 `lightwm_phases/tools/` 里的三个只读工具。
