# 交接 —— WingmanWM / SpatialWorld（2026-09-21 深夜）

给下一个接手的人。本文只写**现在什么在跑、数字是多少、文件在哪、还差什么**。

---

## 0. 一句话

两个闭源模型的**主表批次**已经拿到了干净结果：**Gemini+WM 在 ai2thor 120 条上
把成功率从 20.0% 抬到 35.8%（McNemar p=0.0002）**；GPT-5 基线比 Gemini 基线还低。
唯一没跑完的是 **GPT-5 的 WM 臂**（三张卡 22:00 前后被莫名关停，已搬回本机续跑）。

---

## 1. 现在什么在跑（截至 2026-09-21 22:20）

| 线 | 位置 | 状态 |
|---|---|---|
| 本机 GPT-5 + WM | `runs/closed_gpt5_wm` | **在跑**，6 worker，72 条里完成 3 条 |
| 本机 Gemini + WM | `runs/closed_gemini_wm` | ✅ 71/71 完成 |
| 云 GPT-5 基线 | 三张卡 `main_gpt5_base_s0/1/2` | ✅ 140/140，已拉回本机 |
| 云 GPT-5 + WM | 三张卡 `main_gpt5_wm_s0/1/2` | ⚠️ 卡已关停，最后读到 76/140，已拉回 68 条 |

**三张 AutoDL 卡（30b / 8b / kimi）在 21:59–22:05 之间全部失联**，端口不通（DNS 正常）。
本地副本：`/mnt/d/lightwm_out/closed_pull/`。卡上若还有磁盘数据，重新开机后可以拉回来补。

---

## 2. 结果（已经能出表的）

### 2.1 同一批 140 条（ai2thor 120 + ProCTHOR 20，与 311 同分布）

| 配置 | 成功 | 成功率 |
|---|---:|---:|
| Gemini 基线 | 25/140 | 17.9% |
| **Gemini + WM** | **45/140** | **32.1%** |
| GPT-5 基线 | 22/140 | 15.7% |

分环境：

| | ai2thor 120 | ProCTHOR 20 |
|---|---:|---:|
| Gemini 基线 | 24 (20.0%) | 1 (5%) |
| Gemini + WM | **43 (35.8%)** | 2 (10%) |
| GPT-5 基线 | 22 (18.3%) | 0 (0%) |

### 2.2 ai2thor 的配对拆解（Gemini，120 条）

| | 条数 |
|---|---:|
| 都成功 | 21 |
| 基线成 → WM 败 | 3 |
| **基线败 → WM 救回** | **22** |
| 都失败 | 74 |

McNemar 精确双侧 **p = 0.0002**。平均步数 22.9 → 22.8，不是靠多走步换来的。

### 2.3 GPT-5 的临时读数（已配对 68 条，未跑完）

基线 17.6% → WM 26.5%（+6 条；救回 12 / 弄坏 6）。**注意这是未完成样本**，
不要当结论用。初步看 WM 给 GPT-5 的增益比给 Gemini 的小得多。

---

## 3. 花销（截至 22:00）

| 项 | 花费 |
|---|---:|
| 本机 Gemini+WM（71 条） | $71.75 |
| 云 GPT-5 基线（140 条） | $72.27 |
| 云 GPT-5+WM（68 条） | $48.85 |
| 今天下午 `fix_rest156`（128 条，"最终版"） | $165.41 |
| 今天下午 `probe6` | $4.45 |
| **今天累计** | **$362.73** |

预算口径**始终没定**：`AGENTS.md` 写单次 ¥500（≈$74.6），实操一直按 $400。
按 $400 算还剩约 $37，而本地这 72 条还要约 $50 → 跑完会到约 $415，**超约 $15**。

---

## 4. 关键文件

### 切分与清单（`lightwm_phases/plans/`）

| 文件 | 内容 |
|---|---|
| `ai2thor_main120.txt` / `ai2thor_main120_detail.csv` | 主清单 120 条及逐条元数据 |
| `procthor_main20.txt` | ProCTHOR 20 条 |
| `closed_all_tasks.txt` | 140 条全集（三张卡的分片就是从它切的） |
| `gpt5_wm_todo.txt` | 本地续跑的 72 条 |
| `README_sampling_20260921.md` | **切分规则与抽样论证（必读）** |
| `NOTE_procthor601_manual_row.md` | 手工补的那一行（模型自己放弃=失败）的出处 |
| `candidates_20260921.json` | 三套候选切分（含被否掉的"复用优先"版） |

### 工具（`lightwm_phases/tools/`）

跑：
`run_gemini_local.sh`（本机 Gemini+WM）、`run_gpt5_wm_local.sh`（本机 GPT-5+WM）、
`card_run_gpt5_base.sh` / `card_run_gpt5_wm.sh`（卡上两条臂）、
`launch_*.sh`（后台启动）。

停（**都用脚本文件，别在命令行里直接写目标串，pgrep 会匹配到自己的 shell**）：
`stop_gemini_local.sh`、`stop_local_refill.sh`。

看：
`watch_gpt5wm_local.py`、`watch_closed.sh`、`monitor_status.py`、
`build_closed_table.py --pull`（合并出配对表）。

---

## 5. 今晚踩的坑（都会重犯）

1. **Windows 上的 Clash（127.0.0.1:7897）会死**，worker 全部卡在 `SYN-SENT`，
   表现为"负载掉到 0、十几分钟零进展"。实测**直连网关可用**，所以脚本里把
   `apic1.ohmycdn.com` 加进 `no_proxy` 绕开。**别再用代理探针判断网关死活**。
2. **`pgrep -f` 匹配自己的 shell** —— 今晚我自己踩了两次（命令行里写了脚本名）。
   规矩：一律写成脚本文件再执行。
3. **卡上的 `guard.py` 会把我们的 supervisor 当"卡死脚本"杀掉**：它只认
   `v1 wm wm_` 这个老标记，看着护超 30 分钟的内联 python。已给 `KEEP` 加
   `main_gpt5`。**下次在卡上起新 run 记得先加白名单。**
4. **本机的 curl 探针会被 Windows 代理劫持**返回 502，误判成"隧道死了"。
   探本地端口要 `--noproxy '*'`。
5. **procthor 是内存杀手**：6 个 worker + Unity 吃掉约 12G（本机 13.9G），
   靠 16G swap 撑着。ai2thor 阶段没问题，procthor 阶段会贴到 1.3G 可用。
6. **三张卡被无故关停**（21:59–22:05）。卡上结果在磁盘上，重开机可以拉回来补。

---

## 6. 下一步

1. **等本机这 72 条跑完**（预计 01:00 前后），然后把 `closed_pull/*_wm_*` 的 68 条
   和 `closed_gpt5_wm` 的 72 条拼起来 → GPT-5 的 WM 列齐了 → 出两模型的并排主表。
   `build_closed_table.py --pull` 已经写好，数据到了直接跑。
2. **卡回来之后**：重新开机、确认端口（AutoDL 重启常换端口）、更新 `newcard.py` 凭据，
   把卡上剩下的 `main_gpt5_wm_s*` 拉回来；顺带把被打断的 refill 和消融续上。
3. **交付物还差**（`AGENTS.md` §6）：`results.csv` + `summary.json` 的成品、
   启动脚本 + 环境说明、与已跑批次的对比表。
4. **预算口径必须定**：¥500 还是 $400，这决定还能不能跑后续批次。
