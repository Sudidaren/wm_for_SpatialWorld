# 交接：修 Gemini base 的步数上限口径（2026-09-23）

给下一个 codex。**唯一目标**：把 Gemini base 那一行（AI2-THOR 120 样本）里
**79 条**「步数上限写错、且确实是撞上限才失败」的 episode 重跑掉，
让这一行变成可引用的官方口径数字。用户已拍板范围，不要再扩大。

---

## 1. 问题是什么（已查实，不用重查）

官方规则（`SpatialWorld/actions/max_steps.py:110 resolve_max_steps_from_task`）：

```
max_steps = 10 + 2 * n       # n = task.json 的 golden_actions 里可执行动作数（去掉 DONE/FAIL）
```

但 **Gemini base 那批（run `gemini31pro_ai2thor_procthor_438_v1`，2026-09-13）**
走的是另一条 fallback：

```
10 + 2 * len(task.json 的 target_object_types)      # 实测 120/120、311/311 全中
```

逐条 episode 比对 `max_steps` 的结果：

| | 样本 120 | 全量 311 |
|---|---:|---:|
| 与官方一致 | 12 | 24 |
| 不一致（一律偏小） | 108 | 287 |

base 的上限被压在 12/14/16/18/20 步（分布 23/52/36/7/2），官方应是 18–94 步，
平均少 13.7 步，最多少 76 步。极端例子：`ai2thor03020` 官方 94 步而 base 只给 18；
`04062` 官方 50 / base 14；`04223` 官方 50 / base 16。

### 没有问题的部分（已核，别再动）

| 对象 | 结论 |
|---|---|
| 8B / 30B / Kimi / GPT-5 四对 base ↔ WM | 120/120 逐任务预算完全一致，且等于官方值 |
| Gemini+WM 四个 run（AI2-THOR 135 条 + ProcTHOR 22 条） | 0 条违规 |
| Gemini base 的 ProcTHOR 行（`gemini31pro_procthor127_frozen_v1`） | 35 条 episode 0 条违规 |
| 5 臂消融（base/wm/noinject/notarget/nomem） | 同批同规则，预算天然一致 |

所以只有 Gemini base 的 AI2-THOR 这一行需要动。

---

## 2. 用户已拍板的口径（不要自行更改）

108 条上限写错的任务里：

| 子集 | 条数 | 处理 |
|---|---:|---|
| 自称 DONE 而提前结束（`Model claimed DONE but success conditions not met`） | 29 | 沿用原结果，按正确上限口径直接采用 |
| 其中 `05515`、`05528`（14/14 步贴边，最后一步也是模型自己输出 DONE） | 2 | 同上，沿用 |
| 撞上限而失败（`Reached maximum step limit (N steps)`） | **79** | 本次要重跑的就是这 79 条 |

依据：模型在提示词里看不到步数上限——提示词里没有任何 step limit / max steps 字样，
每步只给 `Step {step_count}`（`mllm_base_agent/agent/runner.py:232/254/431`），
上限只在 harness 侧 `runner.py:793` 判。所以自称 DONE 的 31 条与预算无关，
给多大预算结果都一样，不必重跑。

---

## 3. 要重跑的 79 条（照抄）

```
ai2thor03002 ai2thor03003 ai2thor03019 ai2thor03020 ai2thor03026 ai2thor03028
ai2thor03031 ai2thor03035 ai2thor03036 ai2thor03037 ai2thor03040 ai2thor03051
ai2thor03053 ai2thor03057 ai2thor03058 ai2thor03061 ai2thor03068 ai2thor03070
ai2thor03071 ai2thor03072 ai2thor03075 ai2thor04001 ai2thor04004 ai2thor04005
ai2thor04006 ai2thor04009 ai2thor04036 ai2thor04042 ai2thor04046 ai2thor04053
ai2thor04055 ai2thor04057 ai2thor04059 ai2thor04062 ai2thor04064 ai2thor04066
ai2thor04068 ai2thor04072 ai2thor04102 ai2thor04117 ai2thor04119 ai2thor04223
ai2thor04225 ai2thor04226 ai2thor04227 ai2thor04235 ai2thor04240 ai2thor04500
ai2thor04501 ai2thor04502 ai2thor05004 ai2thor05014 ai2thor05022 ai2thor05024
ai2thor05025 ai2thor05027 ai2thor05045 ai2thor05058 ai2thor05060 ai2thor05065
ai2thor05069 ai2thor05501 ai2thor05508 ai2thor05512 ai2thor05533 ai2thor05534
ai2thor05536 ai2thor05537 ai2thor05541 ai2thor05545 ai2thor05547 ai2thor05555
ai2thor05562 ai2thor05563 ai2thor05564 ai2thor05565 ai2thor05566 ai2thor05570
ai2thor05571
```

沿用不跑的 31 条：
`02436 03012 03017 03034 03042 03052 03064 03065 04044 04054 04065 04118 04228
04236 04238 04241 04242 05002 05026 05028 05029 05056 05077 05515 05521 05524
05528 05546 05577`

### 预算与花费（用户已知晓）

* 单价（用户给的代理价）：输入 1.8 美元 / 1M tokens，输出 10.8 美元 / 1M tokens
* 期望花费约 **53.8 美元（约 387 元）**；期望用量 27.5M 输入 + 0.39M 输出
* 参考：Gemini+WM 同样那 120 条当初花了 109.7 美元，base 重跑约为它的一半
* harness 单次实验红线是 500 元——不得扩范围，不要跑全量 311（那是 163 美元）

### 开跑前必须做的自检（别省）

1. 先只跑 1 条，例如 `ai2thor03020`（官方 94 步 / 旧值 18，差距最大），
   跑完立刻读该 episode 的 `max_steps`，**必须是 94**：

   ```
   python3 -c "import glob,json;f=glob.glob('/home/sudidaren/spatialworld_eval/runs/<RUN>/ai2thor/ai2thor03020/**/episode_*.json',recursive=True)[0];print(json.load(open(f))['max_steps'])"
   ```

2. 若仍是 18，说明解析路径又走了旧 fallback（task.json 没被合并进 task_config），
   先修解析再放量：`configs/ai2thor/load_config.py:190-210` 负责把 task.json
   合并进 `config["task"]` 再调用 `resolve_max_steps_from_task`；只读 YAML 不行
   （YAML 里没有 golden_actions，会退回 `max_steps: 30`）。
3. 自检通过再放量 79 条，跑完逐条断言 `episode.max_steps == 10 + 2*黄金动作数`。

### 怎么跑

* 环境：AI2-THOR 渲染机（卡或本机 + Xvfb）；模型走 Gemini API
  （`gemini-3.1-pro-preview`，代理 `apic1.ohmycdn.com`，key 见用户或 `eval_config.py`），
  不需要 GPU 起 vLLM。
* 复用冻结管线 `spatialworld_eval/` + `PROFILE=llm`，任务清单只写上面这 79 条。
* 建议 run 名：`gemini31pro_base_fix79_budget`（写表要用）。
* 并发 4–6；按每条约 23 步估，十几小时量级。
* 不要为了省钱把 31 条也塞进去重跑——那 31 条已定为沿用。

---

## 4. 跑完之后要做的事

1. 写回主表：`eval_tables/build_main_table.py` 的 `BATCHES` 里，Gemini base 的
   AI2-THOR 行追加新 run（放在 `gemini31pro_ai2thor_procthor_438_v1` 之后，
   让新结果覆盖旧 cell），然后依次跑
   `python3 eval_tables/build_main_table.py --mode sample`、
   `--mode full`、`build_main_table_tex.py`；确认 Coverage 仍 120/120、
   `Avg invalid actions` 不是 `-`。
2. 加脚注（必须写清，免得下一个 codex 重查）：
   * 2026-09-13 那批 base 的 `max_steps` 用了 `10+2×target_object_types` 旧 fallback，
     与官方 `10+2×golden_actions` 不符（120 条里 108 条偏小，平均少 13.7 步）；
   * 其中 31 条（29 条自称 DONE + `05515`/`05528` 两条贴边）经核实与预算无关，
     按正确上限口径直接采用；
   * 其余 79 条已于（重跑日期）用官方预算重跑，run 名
     `gemini31pro_base_fix79_budget`。
3. 重算翻转清单：`base 失败 → +WM 成功` 必须在同预算下重新数
   （WM 一侧本来就合规）。此前用「WM 步数 ≤ base 预算」这个保守上界算过一版：
   22 条翻转里剩 8 条（`05558 05573 05004 03019 04001 04500 05065 05555`）；
   重跑后用实测数据替换这个估计值，并同步更新 PPT / 视频里引用的案例。
4. 主表缺口清单（用户要求每次汇报都带）：当前主表与附表 0 个空格，
   消融表 15 行齐全；本次修完只有 Gemini base 这一行口径变化。

---

## 5. 不要做的事

* 不要重跑那 31 条（用户已明确）。
* 不要跑全量 311（163 美元，超红线），除非用户另外点头。
* 不要改 `FROZEN.sha256` 覆盖的文件；不要量化权重；不要跑 TVR。
* 推 GitHub、开卡、花钱开跑之前先报备并等用户同意（harness 第 0 节）。
* 不要再对 `/mnt/d` 做递归 grep（9p 盘很慢，会把机器拖住 20 分钟；
   当时那条 `grep -rln -- --max_steps /mnt/d/lightwm_out` 就被迫手工 kill）。

---

## 6. 顺带待办：PPT 修正（上一件事的尾巴）

用户给了 `演示文稿1(1).pptx`（微信目录，只有 1 页），让我核对。结论：

* 架构部分准确：约 55M 感知参数、AI2-THOR/ProcTHOR、提示在图像之前
  （`runner.py` 里 `mem_pending` 文本先于 image 追加）、决策权在冻结 MLLM、
  四条读出口、动作位姿估计（不读真值）。
* 示例部分不准确：那页右下的 "place the plate in the microwave" 对应
  `ai2thor03073`，而这条任务所有臂都没成功过（base 50/50 撞上限、
  Gemini+WM 50/50 撞上限、30B base 50/50 失败）；PPT 里 3 张截图
  （image4/5/6，像素 mse 约 3.8）命中
  `wm_gemini31pro_fix_rest156/ai2thor03073` 的 step_18/30/31/32/48，
  即来自那个失败回合；"Step 14 Plate held" 也对不上（base 第 14 步是
  `RotateRight`，WM 第 14 步是 `MoveBack`）。
* 用户已说「失败了那就换」，但还没做完。替换建议：
  * 同预算最干净：`ai2thor05573`（抹布扔垃圾桶；两边预算都是 16 步，
    base 16/16 撞上限失败 vs WM 5/16 成功）；
  * 或 `ai2thor05558`（信用卡；12/12 vs 2/12）；
  * 想保留长程叙事：`04223`（WM 49/50 成功 vs base 16/16 撞上限）或
    `04062`（45/50），但图上必须标明两边预算不同。
* 已产出的对比视频（可直接截图进 PPT）：
  `/mnt/d/lightwm_out/wingman_video_20260923/case1_05573_base_vs_wingmanwm.mp4`（28 秒）、
  `case2_05004_base_vs_wingmanwm.mp4`（24 秒）。
  左侧红框 base、右侧绿框 +WM，每步叠动作与报错，WM 侧还叠它的中文注入文本。
* 出片工具：本机没有 ffmpeg 和 PIL，用 `python3 -m venv` 加 tuna 源装
  `imageio-ffmpeg pillow` 即可（tuna 实测 11.7MB/s，aliyun 只有 0.84MB/s）；
  中文字体用 `/mnt/c/Windows/Fonts/msyh.ttc`。用完记得删临时帧目录。

---

## 7. 复现本次审计（需要时再用）

```bash
# 官方上限：n = golden_actions 里非终止动作数
python3 -c "import json;d=json.load(open('SpatialWorld/data/ai2thor/tasks/ai2thor03020/task.json'));a=[x for x in d['golden_actions']['actions'] if x.upper() not in ('DONE','FAIL')];print(10+2*len(a))"

# base 那批实际的上限
python3 -c "import glob,json;f=glob.glob('spatialworld_eval/runs/gemini31pro_ai2thor_procthor_438_v1/ai2thor/ai2thor03020/**/episode_*.json',recursive=True)[0];print(json.load(open(f))['max_steps'])"
```

判定规则一句话：base 的 `max_steps` 必须等于 `10 + 2 × (golden_actions 里非终止动作数)` 才算对。
