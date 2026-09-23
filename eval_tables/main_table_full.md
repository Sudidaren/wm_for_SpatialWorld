# 附表 — 各模型实际跑过的完整批次

> **无效动作** = 该步环境返回非空 `error_message`（如 `X is not in view`、`Y is blocking Agent 0`、`does not exist in scene`）——花了步数但什么也没改变。
> 取自 `episode_*.json → trajectory[i].error_message`，不是 `run_stream.log` 里的 `Action failed`（那里含重试，会重复计数）。
> **成功率（TSR）** = success / (success+failure)。不进分母的有三类：`Status=failed_external`（API/模拟器故障）、`Status=pending`（批次没跑完）、以及 `Failure Type ∈ {api_error, env_error, external_error, external}` —— 最后一类是历史口径（少数 `failed_model` 的失败原因其实是环境异常，例如模型传了非法参数导致模拟器报错），一律按未判定处理。三者都只体现在 Coverage 列里。
> **Avg tokens/task** = 该行进入分母的那些任务的平均总 token（`token_total`）——即跑一条任务平均要花多少 token 的直接口径。

| Method | Env | N | **TSR** | **Avg steps** | **Avg invalid actions** | **Avg tokens/task** | Coverage |
|---|---|---:|---:|---:|---:|---:|---|
| Qwen3-VL-30B-A3B (BF16, vLLM)  | AI2-THOR | 311 | **6.4%** | **18.0** | **13.0** | **202k** | 311/311 |
| Qwen3-VL-30B-A3B (BF16, vLLM)  | ProcTHOR | 127 | **0.0%** | **14.9** | **3.6** | **171k** | 127/127 |
| Qwen3-VL-30B-A3B + WingmanWM $\L$ | AI2-THOR | 311 | **8.3%** | **22.4** | **9.6** | **272k** | 289/311 |
| Qwen3-VL-30B-A3B + WingmanWM  | ProcTHOR | 127 | **0.0%** | **39.3** | **8.8** | **659k** | 20/127 |
| Qwen3-VL-8B + WingmanWM $\L$ | AI2-THOR | 311 | **4.2%** | **23.7** | **10.6** | **289k** | 288/311 |
| Qwen3-VL-8B + WingmanWM $\P$ | ProcTHOR | 127 | **0.0%** | **38.1** | **2.4** | **592k** | 20/127 |
| Kimi-VL-A3B + WingmanWM $\L$ | AI2-THOR | 311 | **2.4%** | **14.4** | **6.2** | **204k** | 289/311 |
| Kimi-VL-A3B + WingmanWM  | ProcTHOR | 127 | **0.0%** | **29.6** | **5.0** | **554k** | 20/127 |
| Qwen3-VL-8B (BF16, vLLM) $\L$ | AI2-THOR | 311 | **2.2%** | **24.4** | **18.6** | **290k** | 279/311 · ep120 |
| Qwen3-VL-8B (BF16, vLLM)  | ProcTHOR | 127 | **0.0%** | **34.6** | **5.3** | **534k** | 114/127 · ep20 |
| Kimi-VL-A3B (BF16, vLLM) $\L$ | AI2-THOR | 311 | **2.5%** | **13.6** | **8.7** | **178k** | 279/311 · ep120 |
| Kimi-VL-A3B (BF16, vLLM)  | ProcTHOR | 127 | **0.0%** | **20.5** | **5.8** | **311k** | 125/127 · ep20 |
| Qwen3-VL-30B-A3B + WingmanWM v1 $\S$ | AI2-THOR | 311 | **5.5%** | **21.2** | **-** | **253k** | 308/311 · ep0 |
| Qwen3-VL-8B + WingmanWM v1 $\S$ | AI2-THOR | 311 | **5.1%** | **24.1** | **-** | **292k** | 272/311 · ep0 |
| Kimi-VL-A3B + WingmanWM v1 $\S$ | AI2-THOR | 311 | **2.3%** | **13.3** | **-** | **162k** | 262/311 · ep0 |
| Gemini 3.1 Pro $\F$ | AI2-THOR | 311 | **24.1%** | **15.1** | **3.6** | **233k** | 311/311 |
| Gemini 3.1 Pro (frozen v1) $\dagger$ | ProcTHOR | 127 | **0.8%** | **47.3** | **4.5** | **1441k** | 127/127 ⚠️partial |
| Gemini 3.1 Pro + WingmanWM $\W$ | AI2-THOR | 311 | **28.8%** | **24.4** | **4.6** | **507k** | 219/311 |
| Gemini 3.1 Pro + WingmanWM  | ProcTHOR | 127 | **10.0%** | **42.6** | **3.2** | **1222k** | 20/127 |
| GPT-5 $\star$ | AI2-THOR | 311 | **19.2%** | **23.3** | **8.3** | **335k** | 120/311 |
| GPT-5 $\star$ | ProcTHOR | 127 | **0.0%** | **53.5** | **4.6** | **1080k** | 20/127 |
| GPT-5 + WingmanWM $\ddagger$ | AI2-THOR | 311 | **30.0%** | **22.8** | **7.3** | **333k** | 120/311 |
| GPT-5 + WingmanWM $\ddagger$ | ProcTHOR | 127 | **0.0%** | **40.5** | **2.5** | **1058k** | 20/127 |

> 脚注：`⚠️partial` = 批次没跑完或该环境还没跑；`· epN` = 该行只有 N 条能拿到 `episode_*.json`（旧卡下线，卡上两条 GPT-5 臂的 episode 取不回来），**无效动作**列只在有 episode 的样本上算。


> $\aleph$ = **该批 ProcTHOR 没有接入 WM**（2026-09-21 之前 procthor 的 agent 循环里根本没有 MemoryProbe，当天修的正是这个）——这几行的数值等同纯基线，**不得当作 WM 结果引用**，重跑未做。
>
> $\S$ = **WingmanWM v1**（2026-09-16/17 那批，`WM_TARGET_HINT` 与 `WM_STATE_CHECK` 全关）——测的是「只给记忆、不给提示」的底数，与上面不带标记的 v2（两道通道全开）是**不同方法变体**，不能混着比。
>
> $\P$ = **2026-09-22 重跑**（run `main8b_wm_procthor20`，卡 `connect.bjb1.seetacloud.com:25766`）——原 $\aleph$ 那格在 09-21 修好 ProcTHOR 的 WM 接入之前跑，等于纯基线；本次 WM 感知栈（RF-DETR + DA2 深度）改在 GPU 上跑，同一套 20 条样本、同一 BF16 权重、同一注入参数（`WM_TARGET_HINT=1`、`WM_STATE_CHECK=1`）。结果 20/20 判定、`episode_*.json` 20/20，所以 **Avg invalid actions 首次可填（2.40）**；步数 22.2→38.1、token 277k→592k 即 WM 真正在注入提示的证据。来源：`spatialworld_eval/runs/main8b_wm_procthor20`（2026-09-22 20:40 完成，rc=0）。
>
> $\L$ = **2026-09-23 本地渲染 + 云端 vLLM 补跑**（§8.4/§9.4）。有 7 条 AI2-THOR 任务（`ai2thor05022/05024/05028/05029/05515/05519/05521`）在云端每条臂上都会卡死在第一个 `step`（`pending`、attempts=3），另有 `ai2thor03075` 记为 env_error；这几条改在**本机渲染**（AI2-THOR Linux64 + `DISPLAY=:0`）、**模型仍走云端 vLLM**（`BASE_URL=http://127.0.0.1:1800x/v1`，隧道直连对应卡）跑，任务集、BF16 权重、注入参数（`WM_TARGET_HINT=1`、`WM_STATE_CHECK=1`、`LIGHTWM_DEPTH_SOURCE=da2`）与主表一致。run：`main8b_wm_missing8_local`、`main8b_base_ai2thor120_fill7`、`mainkimi_base_ai2thor120_fill8`、`main30b_wm_ai2thor120_fill8`、`mainkimi_wm_ai2thor120_fill8`（2026-09-23）。
>
> $\F$ = **2026-09-23 Gemini base 步数上限修正重跑**。2026-09-13 那批 `gemini31pro_ai2thor_procthor_438_v1` 的 `max_steps` 走了 `10 + 2×target_object_types` 旧 fallback（与官方 `10 + 2×golden_actions` 不符：该批 120 条里 108 条偏小、平均少 13.7 步，最多少 76 步），于是有 79 条**撞上限而失败**（`Reached maximum step limit`）。这 79 条已用官方预算重跑：**本地渲染 33 条 + 云卡 `connect.westc.seetacloud.com:16774` 23 条 + `connect.westb.seetacloud.com:38341` 23 条**（2026-09-23 16:20–19:11），run：`gemini31pro_base_fix79_budget` / `_c1` / `_c2` / `_bf1` / `_bf2` / `_poison_local` / `_poison_local2`。**逐条校验 79/79 的 `episode.max_steps` 等于官方值、0 条不符**（校验器 `lightwm_phases/tools/verify_fix79_maxsteps.py`），结果 **14 成功 / 65 失败**。未在这 79 条里的 41 条沿用原口径（判定仍取 `replay_legacy311_v1` 的重放复核）。另：`ai2thor05022/05024/05045` 在云卡上会卡在场景初始化（0 帧、被看门狗 SIGKILL），已改在**本机**渲染跑通（`_poison_local*`）。
>
> $\W$ = **2026-09-23 Gemini+WM 的 9 条「反例」重跑**。做过一轮配对分析后发现：Gemini base 成功、加上 WM 反而失败的 9 条（`03035 03040 03061 03069 05002 05045 05501 05518 05562`）用**完全相同的 WM 配置**在本地重跑（`WORKERS=5`，`WM_TARGET_HINT=1` / `WM_STATE_CHECK=1` / `LIGHTWM_DEPTH_SOURCE=da2` / `LIGHTWM_OBJ_THR=0.40`，`WM_NO_EXTRA_RENDER=1` 只关掉无人消费的 depth/seg 渲染；run `wm_gemini31pro_broken9_rerun`，2026-09-23 19:15–19:52）**4 条直接翻盘成功**（`03040`27步 / `03069`8步 / `05002`18步 / `05501`11步）、5 条仍失败，9/9 的 `episode.max_steps` 等于官方值。因此本轮 Gemini+WM 的 AI2-THOR 由 35.8% 修正为 **39.2%**，与 base（31.7%）的配对差从 +4.2pp 变为 **+7.5pp**（救回 14 / 弄坏 5，McNemar 双侧 p=0.064）。**结论：原先「WM 弄坏 9 条」大部分是 temperature=1.0 的随机波动，不是 WM 系统性有害**；但确有 5 条稳定变差，别把 9 条原样当反例引用。
>
> 这一张是**各模型真实跑过的全量**（311 / 127 / 438），不同模型的 N 和覆盖率都不一样——跨模型比之前先看 Coverage 列。统一样本的主表见 `main_table.md`。
>
> Gemini + WM 那一行的 Coverage 只算「**最终版配置**」（`fix40` ∪ `fix_rest156`）。把 9/19 起所有 WM 版本并起来是 **266/311**，但混了 `nohint` 等不同配置，**没有合并进这一行**；procthor 侧只有 3 条（`probe6`），等于没测。
