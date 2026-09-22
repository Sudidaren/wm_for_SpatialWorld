# 主表 — 统一样本（AI2-THOR 120 + ProcTHOR 20）

> **无效动作** = 该步环境返回非空 `error_message`（如 `X is not in view`、`Y is blocking Agent 0`、`does not exist in scene`）——花了步数但什么也没改变。
> 取自 `episode_*.json → trajectory[i].error_message`，不是 `run_stream.log` 里的 `Action failed`（那里含重试，会重复计数）。
> **成功率（TSR）** = success / (success+failure)。不进分母的有三类：`Status=failed_external`（API/模拟器故障）、`Status=pending`（批次没跑完）、以及 `Failure Type ∈ {api_error, env_error, external_error, external}` —— 最后一类是历史口径（少数 `failed_model` 的失败原因其实是环境异常，例如模型传了非法参数导致模拟器报错），一律按未判定处理。三者都只体现在 Coverage 列里。
> **Avg tokens/task** = 该行进入分母的那些任务的平均总 token（`token_total`）——即跑一条任务平均要花多少 token 的直接口径。

| Method | Env | N | **TSR** | **Avg steps** | **Avg invalid actions** | **Avg tokens/task** | Coverage |
|---|---|---:|---:|---:|---:|---:|---|
| Qwen3-VL-30B-A3B (BF16, vLLM)  | AI2-THOR | 120 | **6.7%** | **18.4** | **12.3** | **209k** | 120/120 |
| Qwen3-VL-30B-A3B (BF16, vLLM)  | ProcTHOR | 20 | **0.0%** | **16.0** | **5.0** | **-** | 20/20 |
| Qwen3-VL-30B-A3B + WingmanWM  | AI2-THOR | 120 | **7.2%** | **23.2** | **-** | **288k** | 111/120 · ep0 |
| Qwen3-VL-30B-A3B + WingmanWM $\aleph$ | ProcTHOR | 20 | **0.0%** | **16.4** | **-** | **194k** | 20/20 · ep0 |
| Qwen3-VL-8B + WingmanWM  | AI2-THOR | 120 | **3.6%** | **24.7** | **-** | **303k** | 112/120 · ep0 |
| Qwen3-VL-8B + WingmanWM $\aleph$ | ProcTHOR | 20 | **0.0%** | **22.2** | **-** | **277k** | 19/20 · ep0 |
| Kimi-VL-A3B + WingmanWM  | AI2-THOR | 120 | **2.8%** | **15.7** | **-** | **222k** | 109/120 · ep0 |
| Kimi-VL-A3B + WingmanWM $\aleph$ | ProcTHOR | 20 | **0.0%** | **23.5** | **-** | **396k** | 20/20 · ep0 |
| Qwen3-VL-8B (BF16, vLLM)  | AI2-THOR | 120 | **0.9%** | **26.4** | **-** | **320k** | 108/120 · ep0 |
| Qwen3-VL-8B (BF16, vLLM)  | ProcTHOR | 20 | **0.0%** | **41.5** | **-** | **668k** | 19/20 · ep0 |
| Kimi-VL-A3B (BF16, vLLM)  | AI2-THOR | 120 | **0.9%** | **12.9** | **-** | **173k** | 108/120 · ep0 |
| Kimi-VL-A3B (BF16, vLLM)  | ProcTHOR | 20 | **0.0%** | **15.4** | **-** | **181k** | 20/20 · ep0 |
| Qwen3-VL-30B-A3B + WingmanWM v1 $\S$ | AI2-THOR | 120 | **5.1%** | **21.5** | **-** | **258k** | 118/120 · ep0 |
| Qwen3-VL-8B + WingmanWM v1 $\S$ | AI2-THOR | 120 | **3.7%** | **24.9** | **-** | **302k** | 109/120 · ep0 |
| Kimi-VL-A3B + WingmanWM v1 $\S$ | AI2-THOR | 120 | **1.9%** | **13.3** | **-** | **163k** | 104/120 · ep0 |
| Gemini 3.1 Pro  | AI2-THOR | 120 | **20.0%** | **22.9** | **3.1** | **-** | 120/120 |
| Gemini 3.1 Pro (frozen v1) $\dagger$ | ProcTHOR | 20 | **5.0%** | **40.5** | **3.1** | **945k** | 20/20 ⚠️partial |
| Gemini 3.1 Pro + WingmanWM  | AI2-THOR | 120 | **36.1%** | **23.0** | **4.0** | **472k** | 119/120 |
| Gemini 3.1 Pro + WingmanWM  | ProcTHOR | 20 | **10.0%** | **42.6** | **3.2** | **1222k** | 20/20 |
| GPT-5 $\star$ | AI2-THOR | 120 | **19.5%** | **23.7** | **-** | **344k** | 113/120 · ep0 |
| GPT-5 $\star$ | ProcTHOR | 20 | **0.0%** | **53.5** | **4.6** | **1080k** | 20/20 |
| GPT-5 + WingmanWM $\ddagger$ | AI2-THOR | 120 | **30.5%** | **22.8** | **6.5** | **333k** | 118/120 · ep51 |
| GPT-5 + WingmanWM $\ddagger$ | ProcTHOR | 20 | **0.0%** | **40.5** | **2.5** | **1058k** | 20/20 |

> 脚注：`⚠️partial` = 批次没跑完或该环境还没跑；`· epN` = 该行只有 N 条能拿到 `episode_*.json`（旧卡下线，卡上两条 GPT-5 臂的 episode 取不回来），**无效动作**列只在有 episode 的样本上算。

> `Gemini 3.1 Pro + WingmanWM` / `GPT-5` / `GPT-5 + WingmanWM` 这三组的**原批次**跑的是 120 + 20 的分层抽样（与 311/127 同分布）；本表里所有模型都被限制到这同一套样本上，N 就是样本规模，Coverage 是实际判定数。

> $\aleph$ = **该批 ProcTHOR 没有接入 WM**（2026-09-21 之前 procthor 的 agent 循环里根本没有 MemoryProbe，当天修的正是这个）——这几行的数值等同纯基线，**不得当作 WM 结果引用**，重跑未做。
>
> $\S$ = **WingmanWM v1**（2026-09-16/17 那批，`WM_TARGET_HINT` 与 `WM_STATE_CHECK` 全关）——测的是「只给记忆、不给提示」的底数，与上面不带标记的 v2（两道通道全开）是**不同方法变体**，不能混着比。
>
> **本表所有模型都只取同一套共同样本**（AI2-THOR 120 / ProcTHOR 20，分层抽样、与 311/127 同分布），这样跨模型可以直接比。各模型自己跑过的完整批次见 `main_table_full.md`。
