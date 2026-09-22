# 交接 —— WingmanWM / SpatialWorld（2026-09-22 傍晚）

给下一个 Codex。本文只写**现在已经完成什么、正在跑什么、卡在哪、以及今天踩过的坑**。

---

## 0. 一句话

**主表两张已经完成并推到 GitHub**（统一 120+20 样本 + 各模型全量）。
现在在两张新卡上铺**消融实验**，卡在 vLLM 安装上（see §3）。

---

## 1. 今天完成了什么

### 1.1 闭源主批次收尾
* **GPT-5 的 ProcTHOR 基线**：20/20 跑完，**0 成功**（旧卡那批 20 条全是 `failed_external/api_error`，
  踩的是 top_p bug，在本地重跑修好）。
* **GPT-5+WM 的 ProcTHOR**：卡上 7 条 + 本地 13 条 = **20/20，0 成功**。
  这一格跑完，主表就没有空格了。

### 1.2 主表两张（已推 GitHub）

| 文件 | 内容 |
|---|---|
| `eval_tables/main_table.md` | **统一样本表**：所有模型都只取 AI2-THOR 120 / ProcTHOR 20（分层抽样，与 311/127 同分布） |
| `eval_tables/main_table_full.md` | **全量表**：各模型真实跑过的 311/127/438，带 Coverage |
| `eval_tables/main_table.tex` / `_snippet.tex` / `_zh.tex` | 论文用的 LaTeX 版 |
| `eval_tables/build_main_table.py` / `build_main_table_tex.py` | 生成脚本（`--mode sample\|full`） |

列口径（用户定的）：`Method | Env | N | TSR | Avg steps | Avg invalid actions | Avg tokens/task | Coverage`
—— 去掉了 Succ/Fail/Null/Median/Invalid-share。

### 1.3 统一 120+20 样本的结果

| Method | AI2-THOR 120 | ProcTHOR 20 |
|---|---:|---:|
| Qwen3-VL-30B-A3B | 6.7% | 0.0% |
| Qwen3-VL-30B-A3B **+WM** | 7.2% | 0.0% |
| Qwen3-VL-8B | 0.9% | 0.0% |
| Qwen3-VL-8B **+WM** | 3.6% | 0.0% |
| Kimi-VL-A3B | 0.9% | 0.0% |
| Kimi-VL-A3B **+WM** | 2.8% | 0.0% |
| Gemini 3.1 Pro | 20.0% | 5.0% |
| **Gemini 3.1 Pro +WM** | **36.1%** | **10.0%** |
| GPT-5 | 19.5% | 0.0% |
| **GPT-5 +WM** | **30.5%** | 0.0% |

结论：**WM 在所有模型上都有效，但幅度差很多**（Gemini +16.1、GPT-5 +11.0、8B +2.7、Kimi +1.9、30B +0.5）。

### 1.4 核查时发现并修掉的真 bug
`build_main_table.py` 原来把 **`Status=pending` 的行当成失败**计入分母
（卡下线时留下的 7 条），把 GPT-5 基线从 19.5% 压成 18.3%。已改成"pending = 未判定"。
同时把三条排除规则写进表头：`failed_external` / `pending` / `Failure Type ∈ {api_error, env_error, external_error, external}`。

### 1.5 补齐了 8B / Kimi 的基线
它们**不在** `spatialworld_eval/runs/` 里，而在 **`wm_dev/pull_latest/*.state.json`**
（当时只落了 state.json）。已转成 run 目录：`q8b_base_438_v1` / `kimi_base_438_v1`。

---

## 2. 正在跑什么

| 卡 | 用途 | 状态 |
|---|---|---|
| **Kimi 卡** | Kimi-VL-A3B 的 5 臂消融 | 正在装**最新版 vLLM**（0.11.0 不支持 Kimi-VL 架构） |
| **8B 卡** | Qwen3-VL-8B 的 5 臂消融 | 正在装 vLLM 0.11.0（**这张卡出网很差**，wheel 下得很慢） |

两张卡**除了 vLLM 以外全部就绪**（见 §3）。

---

## 3. 两张新卡的凭据与已铺好的内容

| | Kimi 卡 | 8B 卡 |
|---|---|---|
| SSH | `ssh -p 25403 root@connect.westd.seetacloud.com` | `ssh -p 43375 root@connect.cqa1.seetacloud.com` |
| 密码 | `QzhXv0E1Hp2P` | `u7gvq0FgBUiz` |
| GPU | RTX 4090 **48G** | RTX 4090 **D 24G** |
| CPU/内存 | 208 核 / 1TB | 192 核 / 1TB |
| 模型 | `/root/autodl-tmp/models/kimivl-a3b-bf16`（31G ✅） | `/root/autodl-tmp/models/qwen3vl-8b-bf16`（17G ✅） |
| 环境 | ✅ 代码 + venv + AI-THOR + 数据集 | ✅ 同左 |
| Xvfb :99 | ✅ | ✅ |
| 消融清单 | `/root/ablation_tasks.txt`（**46 条**） | `/root/ablation_tasks.txt`（**49 条**） |
| 消融脚本 | `/root/run_ablation.sh` | 同左 |
| **vLLM** | ⬜ 装中 | ⬜ 装中 |

**venv 软链已修**：卡上没有 `/usr/bin/python3`，把两个 venv 的 `bin/python` 指到
`/root/miniconda3/bin/python3.12` 并改了 `pyvenv.cfg` 的 `home`。

---

## 4. 消融设计（方案 B，用户批准）

### 4.1 五个臂（隔离"哪条通道真在被读"）

| 臂 | 环境变量 | 隔离出什么 |
|---|---|---|
| `base` | `PROFILE=llm` | 对照 |
| `wm` | `PROFILE=wm` | 上限（两道通道全开） |
| `noinject` | `+ WM_NO_INJECT=1` | **注入文本**的贡献（WM 照跑但不说话） |
| `notarget` | `+ WM_TARGET_HINT=0` | 目标位置提示（占总提示量 67%） |
| `nomem` | `+ LIGHTWM_MEMORY_FRAMES=0` | **感知 vs 记忆**（只报当前帧）← 最可能推翻主张的一条 |

### 4.2 任务集 = **方案 B**：诊断层 + 官方 39 条的并集

理由：我们模型弱，311 条里绝大多数对所有臂都失败，**没有区分度**。

| 模型 | 诊断层（任一臂成功过） | 并集 |
|---|---:|---:|
| 8B | 21 | **49** |
| Kimi | 13 | **46** |
| 30B | 32 | **54** |

清单在 `lightwm_phases/plans/ablation/{q8b,kimi,q30b}_tasklist.txt`，
**诊断层是按结果挑的**，论文里必须写成 informative subset，不能当整体结论。

### 4.3 判据（先写下来）

* `nomem ≈ wm` → 增益来自**感知栈**而不是记忆，主张要重新表述
* `noinject ≈ wm` → 增益不来自提示文本
* 每次比较给：配对 McNemar + bootstrap 95% CI + 步数分布

### 4.4 跑法

```bash
# 卡上
VLLM_USE_FLASHINFER_SAMPLER=0 /root/miniconda3/bin/vllm serve <模型目录> \
  --served-model-name <名字> --host 127.0.0.1 --port 8000 \
  --max-model-len 32768 --gpu-memory-utilization 0.90 \
  --limit-mm-per-prompt '{"image":32}' --trust-remote-code
# 然后
bash /root/run_ablation.sh <名字>
```

实测吞吐 **144 episode/小时（6 worker）** → 每卡 5 臂 ≈ 330-345 episode ≈ **2.5 小时**，**$0 API**。

---

## 5. 今天踩的坑（血泪，务必避免重犯）

1. **老卡是"卡到卡复制"铺的，从没在新卡上装过 vLLM。**
   今天老卡全部下线、没有源卡，才第一次从零装——暴露了下面 2/3/4 三个问题。
   **以后铺新卡：先找一台准备好的卡，用 `c2c_provision` 拷，别从 pip 装。**
2. **卡上预装 transformers 5.17.0，和 vLLM 0.11.0 不兼容**：
   `TikTokenTokenizer has no attribute all_special_tokens_extended`。降到 `4.57.1` 才行。
3. **但降到 4.57 后，vLLM 0.11.0 又不认 Kimi-VL 的架构**：
   `ValueError: Model architectures ['KimiVLForConditionalGeneration'] failed to be inspected`。
   → **Kimi-VL 必须用最新版 vLLM**（会拉 torch+CUDA13 那一大串）。
4. **pip 源要按卡挑**（今天实测同一个包，不同卡差 100 倍）：
   hf-mirror 30MB/s、清华（Kimi 卡 12.6MB/s，8B 卡 27KB/s）、阿里云 0~1MB/s。
   **装之前先在卡上 `curl` 测一下哪个源快。**
5. **别在同一个环境上并发跑 pip**：我留了个 `--dry-run` 进程没杀，把 8B 卡拖死了半小时。
6. **`pgrep -f` 会匹配到自己的 shell**（我今天又犯了三次，误判"pip 正在装"白等半小时）。
   一律写成脚本文件，或用 `ps | grep '[p]attern'`。
7. **`tar` 里的路径是相对仓库根的**（`SpatialWorld/envs/...`），在 `/root` 下解会解错地方，
   必须在 `/home/sudidaren` 下解。
8. **本机→AutoDL 上传 ≈ 1-3.5 MB/s，8 路分片能到 8 MB/s**（`tools/upload_parallel.py`，
   已支持 `UPLOAD_HOST/PORT/PW` 环境变量）。**能下载的绝不上传**，本机只传卡上下不到的。

---

## 6. 下一步

1. **等两张卡的 vLLM 装完** → 起服务（参数见 §4.4）→ `bash /root/run_ablation.sh <模型名>`
2. **30B 的消融还没开始**：需要一张 ≥80G 的卡（30B BF16 ≈ 60G 权重），清单已备好（54 条）
3. **卡回来/数据找回**：老的三张卡（30b/8b/kimi）上还有**上一轮跑过的 5 臂消融数据**，
   本地没有副本；卡能开机的话第一件事是把 run 目录拉回来
4. **分析**：跑完按 §4.3 出配对表 + McNemar + 分层分析（分层不用重跑，离线做）

---

## 7. 关键文件与仓库

* **GitHub**：`git@github.com:Sudidaren/wm_for_SpatialWorld.git`（master）
  最近三个 commit：`3aec383`（主表完成）、`638eb41`（GPT-5 ProcTHOR 基线）、`7411686`（统一样本表）
* **主表**：`eval_tables/`（两张 md + 三份 tex + 两个脚本）
* **抽样方案**：`plans/README_sampling_20260921.md`、`plans/ai2thor_main120.txt`、`plans/procthor_main20.txt`
* **消融清单**：`plans/ablation/{q8b,kimi,q30b}_tasklist.txt`
* **卡上工具**：`tools/newcard.py`（卡上执行，凭据在这个文件里）、`tools/upload_parallel.py`（并行上传）、
  `tools/card_run_ablation_planb.sh`（5 臂 runner）
* **本机数据**：`spatialworld_eval/runs/`（所有批次）、`/mnt/d/lightwm_out/`（日志与拉回的数据）
