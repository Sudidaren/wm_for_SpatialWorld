# WingmanWM 消融实验设计（方案 B：5 臂）

> **状态**：设计于 2026-09-22 经用户批准并执行；三个模型 × 5 臂（Qwen3-VL-8B /
> Qwen3-VL-30B-A3B / Kimi-VL-A3B）已跑完。
> **数字不在这份文档里维护**——一律以 [`../eval_tables/ablation_table.md`](../eval_tables/ablation_table.md)
> / [`../eval_tables/ablation_table.csv`](../eval_tables/ablation_table.csv) 为准，
> 由 [`../eval_tables/build_ablation_table.py`](../eval_tables/build_ablation_table.py) 从原始 run 重算。
> 本文只写**设计**：要证明什么、怎么隔离、控制哪些变量、判据是什么、怎么复现。
>
> 相关：[主表](../eval_tables/main_table.md) ·
> 运行脚本 [`../tools/card_run_ablation_planb.sh`](../tools/card_run_ablation_planb.sh) ·
> 审批与执行记录 [`HANDOVER_2026-09-22_ablation.md`](HANDOVER_2026-09-22_ablation.md) §4

---

## 1. 要回答的问题

主表（统一 120+20 样本）显示 WingmanWM 对每个模型都有增益，但幅度差别很大：
Gemini +16.1pp、GPT-5 +11.0pp、8B +2.7pp、Kimi +1.9pp、30B +0.5pp。
主表只能说明**整体有用**，不能说明**增益从哪里来**。消融专门回答归因问题：

1. 增益真的来自**注入到 prompt 的那段文本**，还是 WM 跑起来本身的副作用
   （多花算力、改变 agent 与环境的时序、改变历史长度）？
2. 增益主要来自**目标物位置提示**（历史批次统计占每步注入文本的 67%，见
   §4.1 of `HANDOVER_2026-09-22_ablation.md`）还是别的通道？
3. 增益来自**记忆**（跨帧累积的锚点/账本），还是只来自**当前帧的感知**
   （RF-DETR 检测 + DA2 单目深度）？

## 2. 五臂设计

每个臂只改一个开关，其余全部相同（同一份 BF16 权重、同一服务、同一清单、
同一判分口径）。

| 臂 | `PROFILE` | 额外环境变量 | WM 是否在跑 | prompt 里有 WM 文本 | 隔离出什么 |
|---|---|---|---|---|---|
| `base` | `llm` | — | ❌ 完全没有 | ❌ | 对照（纯 VLM） |
| `wm` | `wm` | — | ✅ | ✅ | 上限（两道注入通道全开） |
| `noinject` | `wm` | `WM_NO_INJECT=1` | ✅（感知+记忆+渲染都在算） | ❌ 一行都不注入 | **注入文本**的净贡献 |
| `notarget` | `wm` | `WM_TARGET_HINT=0` | ✅ | ✅ 但没有目标物位置块 | 目标物位置提示这一条通道 |
| `nomem` | `wm` | `LIGHTWM_MEMORY_FRAMES=0` | ✅（每帧照常检测+测深） | ✅ 但只报当前帧 | **感知 vs 记忆** |

默认值来源：`run_wm_gemini_ai2thor.sh` 里 `WM_TARGET_HINT="${WM_TARGET_HINT:-1}"`、
`WM_STATE_CHECK="${WM_STATE_CHECK:-1}"`——即 `wm` 臂是 `WM_TARGET_HINT=1` +
`WM_STATE_CHECK=1` 的完整版本（与主表一致）。`notarget` 只关前者，
完成前查状态 `CheckState()` 仍然开着。

### 2.1 为什么是这五个臂

* `base` / `wm` 给出上下界，是所有对比的分母。
* `noinject` 是**最关键的对照**：WM 照跑（吃一样的 GPU/CPU、一样的代码路径），
  只是一行都不写进 prompt。它的意义是把"文本通道"从"WM 运行的开销/时序副作用"里
  剥出来：
  * `noinject ≈ base` → 增益确实来自注入文本；
  * `noinject > base` → 还有别的通道在起作用，主张要改。
* `notarget` 针对占提示量最大的一块通道（目标物位置/距离/所在容器）。
* `nomem` 是**最可能推翻我们主张的一条**：若 `nomem ≈ wm`，说明增益来自
  当前帧的感知栈（检测+测深）而不是记忆，那么"空间记忆"这个卖点就必须重新表述。
  注意它仍然付全部感知开销，只是不留跨帧历史。

### 2.2 刻意不做的臂

* 不做"WM 关掉但把提示写死/由人手写"的伪臂——那等于把人的先验塞进去，
  不是 WM 的贡献。
* 不动渲染参数（`WM_TARGET_HINT_LIMIT` / `WM_TARGET_HINT_NAMES` /
  `WM_TARGET_HINT_MAX_DIST` / `WM_TARGET_HINT_MAX_SIGMA`）——改这些等于换方法
  变体，不是消融。
* 不跑 TVR（用户 2026-09-22 决定，卡上 runner 里的 TVR 段已停用）。

## 3. 任务集：诊断层 ∪ 官方 39 条（方案 B）

| 模型 | 官方清单 | 诊断层（任一臂成功过） | 并集 | AI2-THOR / ProcTHOR | 清单文件 |
|---|---:|---:|---:|---|---|
| Qwen3-VL-8B | 39 | 10 | **49** | 37 / 12 | `plans/ablation/q8b_tasklist.txt` |
| Qwen3-VL-30B-A3B | 39 | 15 | **54**（实跑 53，见下） | 42 / 12 | `plans/ablation/q30b_tasklist.txt` |
| Kimi-VL-A3B | 39 | 7 | **46** | 34 / 12 | `plans/ablation/kimi_tasklist.txt` |

* **官方清单 = 39 条**（27 AI2-THOR + 12 ProcTHOR，`tvrbench_addon/tasks/ablation_official.txt`）：
  保证与官方口径可比，并覆盖 ProcTHOR（WM 的家庭泛化）。
* **诊断层**：在已跑批次里**任一臂成功过**的任务。加它的原因是我们的模型偏弱，
  311 条里绝大多数任务对所有臂都失败，**没有区分度**；诊断层把样本挪到"至少有人
  做得到"的区间，让 5 个臂之间的差异可测。
* ⚠️ **诊断层是按结果挑的，论文里必须写成 informative subset**，只能用于**归因**
  （哪条通道在起作用），**不能当整体结论**。主结论仍然用 120+20 主表。
* **毒任务 `ai2thor05519`**：云端环境 init 成功但第一个 `step` 永不返回
  （重试 3 次都一样），会把整臂卡死。30B 清单里已剔除（54 → 53），
  `state.json` 记 `failure_type=env_error`，按主表口径**不进分母**；
  8B / Kimi 的清单本来就不含它。

## 4. 控制变量与判分口径

* **模型**：每张卡上同一份 **BF16 原版权重**（禁止量化），同一次 vLLM 服务、
  同一 `--max-model-len 32768`；5 个臂**串行**跑（`base → wm → noinject →
  notarget → nomem`），避免并发抢显存带来时序噪声。
* **服务**：本地自建 vLLM，`BASE_URL=http://127.0.0.1:8000/v1`，`LLM_API_KEY=EMPTY`
  → 只花卡时，**$0 API 费用**。
* **注入参数**：除被消融的那一条开关外，其余与主表一致——
  `LIGHTWM_DEPTH_SOURCE=da2`、`LIGHTWM_DA2_SCALE=1.3816`、
  `WM_TARGET_HINT=1`、`WM_STATE_CHECK=1`，感知后端 = RF-DETR Small 检测 +
  已训练单目深度头。
* **判分**（与主表一致）：只有 `status ∈ {success, failed_model}` 进分母；
  `failed_external` / `pending` / `env_error` 不计入分母，只体现在 Coverage 列。
* **指标**：TSR、Avg steps、Avg invalid actions、Avg tokens/task、Coverage。
* **可复现性**：任务清单随仓库版本化（`plans/ablation/*.txt`），
  同一 commit 下可复现同一套样本。

## 5. 判据（跑之前先写下来）

* `nomem ≈ wm` → 增益来自**感知栈**而不是记忆，主张要重新表述。
* `noinject ≈ wm` → 增益不来自提示文本。
* 每次比较给：**同任务配对**的 McNemar + bootstrap 95% CI + 步数分布。
* 样本量小（n = 49 / 46 / 53），检验力有限：结论按**方向 + 置信区间**报，
  不靠"p<0.05 就算赢"。

## 6. 怎么复现（卡上）

```bash
# 1) 起 vLLM（模型目录按卡上实际路径替换）
VLLM_USE_FLASHINFER_SAMPLER=0 /root/miniconda3/bin/vllm serve <模型目录> \
  --served-model-name <名字> --host 127.0.0.1 --port 8000 \
  --max-model-len 32768 --gpu-memory-utilization 0.90 \
  --limit-mm-per-prompt '{"image":32}' --trust-remote-code

# 2) 把对应模型的清单放到 /root/ablation_tasks.txt，然后
bash tools/card_run_ablation_planb.sh <模型名>            # 默认跑 5 个臂
bash tools/card_run_ablation_planb.sh <模型名> base wm    # 也可以只跑指定臂
```

产出：`spatialworld_eval/runs/abl_<arm>_<模型>/`（`results.csv` +
每任务的 `episode_*.json`，后者是 Avg invalid actions 的唯一原料）。
拉回本地后 `python3 eval_tables/build_ablation_table.py` 重算表。

实测吞吐 **144 episode/小时（6 worker）** → 每卡 5 臂 ≈ 330–345 episode ≈
**2.5 小时**，token **$0**。

## 7. 结果快照（2026-09-23，合计 = AI2-THOR + ProcTHOR）

| 模型 | 臂 | N（进分母） | TSR | Avg invalid actions | 相对 `base` |
|---|---|---:|---:|---:|---:|
| Qwen3-VL-8B | base | 49 | 10.2% | 8.7 | — |
| | wm | 49 | **20.4%** | 6.2 | **+10.2pp** |
| | noinject | 49 | 10.2% | 9.6 | +0.0pp |
| | notarget | 49 | 12.2% | 5.6 | +2.0pp |
| | nomem | 49 | **28.6%** | 5.2 | +18.4pp |
| Qwen3-VL-30B-A3B | base | 53（未判定 1） | 24.5% | 5.2 | — |
| | wm | 53（未判定 1） | **43.4%** | 3.1 | **+18.9pp** |
| | noinject | 53 | 30.2% | 6.1 | +5.6pp |
| | notarget | 53 | 30.2% | 5.5 | +5.6pp |
| | nomem | 53 | 26.4% | 5.5 | +1.9pp |
| Kimi-VL-A3B | base | 46 | 4.3% | 4.6 | — |
| | wm | 46 | 13.0% | 4.5 | +8.7pp |
| | noinject | 46 | 8.7% | 4.4 | +4.3pp |
| | notarget | 46 | **17.4%** | 2.3 | +13.0pp |
| | nomem | 46 | 8.7% | 3.7 | +4.3pp |

逐环境、逐指标（含 Avg steps / tokens / Coverage）见
[`../eval_tables/ablation_table.md`](../eval_tables/ablation_table.md)。

三条可以直接读出来的结论（同样只对这套 informative subset 成立）：

1. **8B 的增益全部来自注入文本**：设计上 `noinject` 应当与 `base` 完全一致
   （`memory_probe.py` 里 `WM_NO_INJECT=1` 直接把提示块置空并返回空串），
   8B 实测就是同一个数字（10.2%），说明那 +10.2pp 与 WM 的运行开销/时序无关。
   但 30B（`noinject` 30.2% vs `base` 24.5%）与 Kimi（8.7% vs 4.3%）的
   `noinject` 仍高于 `base`，即还有**非文本效应**（时序/历史长度/算力）没被定位，
   报告时要如实写出来。
2. **记忆的作用与模型相关**：30B 上 `nomem`（26.4%）明显低于 `wm`（43.4%）、
   接近 `base`（24.5%）→ 记忆是 30B 增益的主要来源；但 8B 上 `nomem`
   （28.6%）**反超** `wm`（20.4%），Kimi 上两者持平（8.7% vs 13.0% 低于 wm）
   → 对弱模型，长记忆反而可能是噪声。这条必须写进 limitation。
3. **目标物位置提示不总是正贡献**：Kimi 上 `notarget`（17.4%）高于 `wm`
   （13.0%）；30B 上持平（30.2% vs 43.4% 低于 wm）。也就是说"给位置"这条
   通道对强模型有帮助、对弱模型可能有害（和 §2.1 的 VoI 动机一致）。

## 8. 已知限制

* **informative subset**：任务集含按结果挑出的诊断层，不能外推成整体成功率。
* **ProcTHOR 无区分度**：12 条 ProcTHOR 在三个模型的所有臂上都是 0.0%
  （`wm` 与 `nomem` 的步数/token 反而暴涨到 700k–870k，说明 WM 让它多走但走不出来），
  这一段只能作为"未退化/成本"的证据，不能作为增益证据。
* **Coverage 不是满的**：云端毒任务（`ai2thor05519` 等）导致个别臂有未判定条目，
  已在表里标注并 `env_error` 处理。
* **每臂只用一张卡、串行跑**：n 小、单次运行，没有跨卡重复；置信区间偏宽。

## 9. 相关文件

| 用途 | 路径 |
|---|---|
| 设计（本文） | `docs/ABLATION_DESIGN.md` |
| 结果表（md/csv）+ 生成器 | `eval_tables/ablation_table.{md,csv}`、`eval_tables/build_ablation_table.py` |
| 5 臂 runner（方案 B 清单） | `tools/card_run_ablation_planb.sh` |
| 5 臂 runner（官方 39 + TVR 版，TVR 已停用） | `tools/newcard_run_ablation.sh` |
| 任务清单 | `plans/ablation/{q8b,kimi,q30b}_tasklist.txt` |
| 原始 run | `spatialworld_eval/runs/abl_<arm>_{qwen3vl-8b,qwen3vl-30b,kimivl-a3b}/` |
| 审批与执行记录 | `docs/HANDOVER_2026-09-22_ablation.md` §4 |

## 10. 变更记录

* **2026-09-22**：方案 B（5 臂 + 诊断层并集）经用户批准；8B 卡、Kimi 卡开始跑。
* **2026-09-23**：三个模型 × 5 臂全部跑完，结果落到 `eval_tables/ablation_table.md`；
  本文补成独立设计文档。
