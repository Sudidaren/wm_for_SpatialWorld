# 附表 — 配对翻盘：base vs +WingmanWM（同一批任务，逐任务配对）

> 生成脚本 `eval_tables/build_pairwise_table.py`，口径与主表完全一致（直接复用 `build_main_table.py` 的 run 列表与合并规则）。
> **救回** = base 失败且 +WM 成功；**弄坏** = base 成功且 +WM 失败；**净** = 救回 − 弄坏。**p** = McNemar 精确双侧检验。
>
> 为什么用配对而不是直接比两个 TSR：配对消掉了任务难度差异，同样样本量下灵敏度更高；两个 TSR 的差在 n=120 时标准误约 4.3 个百分点。

| 模型 | 环境 | N | base TSR | +WM TSR | Δ | 救回 | 弄坏 | 净 | McNemar p |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Gemini 3.1 Pro | AI2-THOR | 120 | 31.7% | **39.2%** | **+7.5pp** | 14 | 5 | +9 | 0.064 |
| Gemini 3.1 Pro (frozen v1) | ProcTHOR | 20 | 5.0% | **10.0%** | **+5.0pp** | 1 | 0 | +1 | 1.000 |
| GPT-5 | AI2-THOR | 120 | 19.2% | **30.0%** | **+10.8pp** | 22 | 9 | +13 | 0.029 |
| GPT-5 | ProcTHOR | 20 | 0.0% | **0.0%** | **+0.0pp** | 0 | 0 | +0 | 1.000 |
| Qwen3-VL-30B-A3B | AI2-THOR | 119 | 6.7% | **8.4%** | **+1.7pp** | 5 | 3 | +2 | 0.727 |
| Qwen3-VL-30B-A3B | ProcTHOR | 20 | 0.0% | **0.0%** | **+0.0pp** | 0 | 0 | +0 | 1.000 |
| Qwen3-VL-8B | AI2-THOR | 120 | 0.8% | **4.2%** | **+3.3pp** | 5 | 1 | +4 | 0.219 |
| Qwen3-VL-8B | ProcTHOR | 20 | 0.0% | **0.0%** | **+0.0pp** | 0 | 0 | +0 | 1.000 |
| Kimi-VL-A3B | AI2-THOR | 120 | 2.5% | **3.3%** | **+0.8pp** | 3 | 2 | +1 | 1.000 |
| Kimi-VL-A3B | ProcTHOR | 20 | 0.0% | **0.0%** | **+0.0pp** | 0 | 0 | +0 | 1.000 |

> 说明：ProcTHOR 只有 20 条样本，配对检验几乎不可能显著，仅作参考。
> Gemini 的 +WM 行已包含 2026-09-23 的 9 条反例重跑（主表脚注 $\W$）；把随机波动挤掉后它的净翻盘从 +5 变为 +9。

## 明细：救回 / 弄坏的任务 ID

**Gemini 3.1 Pro / AI2-THOR**

- 救回（14）：ai2thor03019 ai2thor03036 ai2thor04001 ai2thor04062 ai2thor04072 ai2thor04240 ai2thor04500 ai2thor04502 ai2thor05014 ai2thor05508 ai2thor05528 ai2thor05555 ai2thor05558 ai2thor05573
- 弄坏（5）：ai2thor03035 ai2thor03061 ai2thor05045 ai2thor05518 ai2thor05562

**Gemini 3.1 Pro (frozen v1) / ProcTHOR**

- 救回（1）：procthor615
- 弄坏（0）：（无）

**GPT-5 / AI2-THOR**

- 救回（22）：ai2thor03012 ai2thor03065 ai2thor04006 ai2thor04055 ai2thor04064 ai2thor04072 ai2thor04118 ai2thor04225 ai2thor04226 ai2thor04236 ai2thor04238 ai2thor04500 ai2thor05029 ai2thor05041 ai2thor05060 ai2thor05065 ai2thor05069 ai2thor05077 ai2thor05512 ai2thor05533 ai2thor05554 ai2thor05558
- 弄坏（9）：ai2thor03035 ai2thor03040 ai2thor04062 ai2thor04065 ai2thor04228 ai2thor05004 ai2thor05027 ai2thor05508 ai2thor05515

**GPT-5 / ProcTHOR**

- 救回（0）：（无）
- 弄坏（0）：（无）

**Qwen3-VL-30B-A3B / AI2-THOR**

- 救回（5）：ai2thor03017 ai2thor04001 ai2thor04228 ai2thor05065 ai2thor05558
- 弄坏（3）：ai2thor04067 ai2thor05519 ai2thor05562

**Qwen3-VL-30B-A3B / ProcTHOR**

- 救回（0）：（无）
- 弄坏（0）：（无）

**Qwen3-VL-8B / AI2-THOR**

- 救回（5）：ai2thor03065 ai2thor04240 ai2thor05065 ai2thor05519 ai2thor05562
- 弄坏（1）：ai2thor04106

**Qwen3-VL-8B / ProcTHOR**

- 救回（0）：（无）
- 弄坏（0）：（无）

**Kimi-VL-A3B / AI2-THOR**

- 救回（3）：ai2thor04238 ai2thor04242 ai2thor05056
- 弄坏（2）：ai2thor05519 ai2thor05558

**Kimi-VL-A3B / ProcTHOR**

- 救回（0）：（无）
- 弄坏（0）：（无）
