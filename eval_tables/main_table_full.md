# 附表 — 各模型实际跑过的完整批次

> **无效动作** = 该步环境返回非空 `error_message`（如 `X is not in view`、`Y is blocking Agent 0`、`does not exist in scene`）——花了步数但什么也没改变。
> 取自 `episode_*.json → trajectory[i].error_message`，不是 `run_stream.log` 里的 `Action failed`（那里含重试，会重复计数）。
> **成功率（TSR）** = success / (success+failure)；`failed_external`（API/模拟器故障）不进分母，只体现在 Coverage 列里，避免把半批次当最终结果。
> **Avg tokens/task** = 该行进入分母的那些任务的平均总 token（`token_total`）——即跑一条任务平均要花多少 token 的直接口径。

| Method | Env | N | **TSR** | **Avg steps** | **Avg invalid actions** | **Avg tokens/task** | Coverage |
|---|---|---:|---:|---:|---:|---:|---|
| Qwen3-VL-30B-A3B (BF16, vLLM)  | AI2-THOR | 311 | **6.4%** | **18.0** | **13.0** | **202k** | 311/311 |
| Qwen3-VL-30B-A3B (BF16, vLLM)  | ProcTHOR | 127 | **0.0%** | **17.6** | **4.9** | **-** | 127/127 |
| Qwen3-VL-30B-A3B + WingmanWM  | AI2-THOR | 311 | **7.9%** | **22.7** | **-** | **277k** | 280/311 · ep0 |
| Qwen3-VL-30B-A3B + WingmanWM $\aleph$ | ProcTHOR | 127 | **0.0%** | **15.0** | **-** | **171k** | 126/127 · ep0 |
| Qwen3-VL-8B + WingmanWM  | AI2-THOR | 311 | **3.9%** | **24.0** | **-** | **292k** | 279/311 · ep0 |
| Qwen3-VL-8B + WingmanWM $\aleph$ | ProcTHOR | 127 | **0.0%** | **29.1** | **-** | **422k** | 121/127 · ep0 |
| Kimi-VL-A3B + WingmanWM  | AI2-THOR | 311 | **2.2%** | **15.0** | **-** | **208k** | 271/311 · ep0 |
| Kimi-VL-A3B + WingmanWM $\aleph$ | ProcTHOR | 127 | **0.0%** | **18.3** | **-** | **273k** | 124/127 · ep0 |
| Qwen3-VL-8B (BF16, vLLM)  | AI2-THOR | 311 | **2.2%** | **25.1** | **-** | **300k** | 267/311 · ep0 |
| Qwen3-VL-8B (BF16, vLLM)  | ProcTHOR | 127 | **0.0%** | **37.6** | **-** | **596k** | 113/127 · ep0 |
| Kimi-VL-A3B (BF16, vLLM)  | AI2-THOR | 311 | **1.9%** | **13.4** | **-** | **178k** | 267/311 · ep0 |
| Kimi-VL-A3B (BF16, vLLM)  | ProcTHOR | 127 | **0.0%** | **18.8** | **-** | **272k** | 125/127 · ep0 |
| Qwen3-VL-30B-A3B + WingmanWM v1 $\S$ | AI2-THOR | 311 | **5.5%** | **21.2** | **-** | **253k** | 308/311 · ep0 |
| Qwen3-VL-8B + WingmanWM v1 $\S$ | AI2-THOR | 311 | **5.1%** | **24.1** | **-** | **292k** | 272/311 · ep0 |
| Kimi-VL-A3B + WingmanWM v1 $\S$ | AI2-THOR | 311 | **2.3%** | **13.3** | **-** | **162k** | 262/311 · ep0 |
| Gemini 3.1 Pro  | AI2-THOR | 311 | **19.6%** | **22.6** | **3.0** | **-** | 311/311 |
| Gemini 3.1 Pro (frozen v1) $\dagger$ | ProcTHOR | 127 | **0.8%** | **47.3** | **4.5** | **1441k** | 127/127 ⚠️partial |
| Gemini 3.1 Pro + WingmanWM  | AI2-THOR | 311 | **27.1%** | **24.4** | **4.6** | **509k** | 218/311 |
| Gemini 3.1 Pro + WingmanWM  | ProcTHOR | 127 | **10.0%** | **42.6** | **3.2** | **1222k** | 20/127 |
| GPT-5 $\star$ | AI2-THOR | 311 | **18.3%** | **23.7** | **-** | **344k** | 120/311 · ep0 |
| GPT-5 $\star$ | ProcTHOR | 127 | **0.0%** | **53.5** | **4.6** | **1080k** | 20/127 |
| GPT-5 + WingmanWM $\ddagger$ | AI2-THOR | 311 | **30.3%** | **22.8** | **6.5** | **333k** | 119/311 ⚠️partial · ep51 |
| GPT-5 + WingmanWM $\ddagger$ | ProcTHOR | 127 | **-** | **-** | **-** | **-** | 7/127 ⚠️partial |

> 脚注：`⚠️partial` = 批次没跑完或该环境还没跑；`· epN` = 该行只有 N 条能拿到 `episode_*.json`（旧卡下线，卡上两条 GPT-5 臂的 episode 取不回来），**无效动作**列只在有 episode 的样本上算。


> $\aleph$ = **该批 ProcTHOR 没有接入 WM**（2026-09-21 之前 procthor 的 agent 循环里根本没有 MemoryProbe，当天修的正是这个）——这几行的数值等同纯基线，**不得当作 WM 结果引用**，重跑未做。
>
> $\S$ = **WingmanWM v1**（2026-09-16/17 那批，`WM_TARGET_HINT` 与 `WM_STATE_CHECK` 全关）——测的是「只给记忆、不给提示」的底数，与上面不带标记的 v2（两道通道全开）是**不同方法变体**，不能混着比。
>
> 这一张是**各模型真实跑过的全量**（311 / 127 / 438），不同模型的 N 和覆盖率都不一样——跨模型比之前先看 Coverage 列。统一样本的主表见 `main_table.md`。
>
> Gemini + WM 那一行的 Coverage 只算「**最终版配置**」（`fix40` ∪ `fix_rest156`）。把 9/19 起所有 WM 版本并起来是 **266/311**，但混了 `nohint` 等不同配置，**没有合并进这一行**；procthor 侧只有 3 条（`probe6`），等于没测。
