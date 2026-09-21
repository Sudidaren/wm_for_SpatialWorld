# TVRBench 加题集：判定改写 + WM 接入（2026-09-21）

这份目录存的是 2026-09-21 对 `~/tvrbench_addon`（255 条 TVR 目标视角复现任务）
做的三件事的快照与说明。addon 本身不是 git 仓库，所以关键文件抄一份在这里。

---

## 一、判定：从"位姿精确匹配"改成"目标锚点都在画面里"

**为什么改。** 官方 TVRBench 判据是 `agent_pose`，容差 **位置 0.01 m /
朝向 1° / 俯仰 1°**，而 agent 的动作粒度是 **0.25 m / 45° / 30°** ——
等于要求"落在同一格、同一角度"，**差一次 LookDown 就是 0 分**。实测 GPT-5
在 iTHOR 5 条上的最近一次是 `0.35 m / 0° / 30°`，全判 0，看不出任何区分度。

**改成什么。** 新条件类型 `objects_visible`：**目标位姿能看到的那组"可判别
锚点"，在最终画面里是否也都在**。锚点清单不是现编的，是生成任务时用
`audit_anchors.py` 在目标位姿量出来的（剔除硬结构面 + 软装饰，掩码 ≥0.5%，
中位 6 个），当年量它就是为了评估这个更松的判据会不会变成送分题 ——
结论是不会：随机可达位姿满足它的概率**中位仅 2%**、p90 8%、最大 28%。

**实现。**

* `pose_eval.py` 新增 `type_fracs()` / `check_objects_visible()` /
  `judge()`（按 `condition["type"]` 分派），旧的 `check_agent_pose` 保留。
* `apply_visibility_condition.py` 把 255 条的 `success_conditions` 换成
  `objects_visible`，**原位姿条件原样存进 `tvr.pose_success_condition`**，
  并把每个 `task.json` 备份成 `task.json.bak.json`（可 `--revert` 回滚）。
* 判定必须看**实例分割掩码**：实测同一帧下 `object["visible"]` 只报 1 个
  物体，而掩码能给出 21 个（Chair/DiningTable/Plate/Vase…），两者不等价。
  所以 `run_tvr_agent.build_env` 里开了 `env.render_instance_segmentation`。
  **它只影响渲染出的模态，不进 agent 的 prompt** —— first-person 文本里只有
  动作反馈，没有物体可见性。判定端读模拟器渲染结果属于裁判权限，
  **绝不进 agent 的输入**。

**回归。** `replay_golden.py` 也改走 `judge()`，所以 golden 必须在新判据下
仍然通过：实测混合抽样 **6/6 全过**（iTHOR + ProcTHOR easy/hard）。

**顺带排掉一个坑**：`area1pct` 那一档（每个锚点占画面 ≥1%）**不能当判据** ——
锚点是按 ≥0.5% 选的，area1pct 要求每个 ≥1%，自相矛盾，**golden 自己在它下面
只有 2/6 通过**。正式判据用 `presence`；`vis_area1pct` 只作为附加列记录。

---

## 二、WM 接入：`--wm`

官方 agent graph 的 `act_node` 是靠 `config.memory_probe.enabled` +
`config.memory_probe.world_model.enabled` 决定建不建 MemoryProbe / WorldModel
的（`mllm_base_agent/agent/runner.py:545` 起），所以 addon 只要把这一块塞进
cfg 就能接上 —— 与官方 `config_builder` + `wm_config_patch` 生成的那份
**逐字段对齐**（`obj_thr=0.40`、`resolution=224`、`variant=small`、
`depth_source=da2`、`target_hint`/`object_query` 两个通道）。

**注意这是个隐患**：这套配置是**手工对齐**，不是同一份代码生成。将来官方改
`wm_config_patch._apply_wingman_block`，这里不会自动跟着变。

---

## 三、"WM 该盯什么"：目标图认领（这批独有）

TVR 的指令里**没有任何物体名**（"Reproduce the target viewpoint…"），而
官方 `TargetHinter` 是靠 `relevant_types(指令文本, 检测到的类型)` 建立目标集的
—— 不喂它，**WM 的目标通道整条是哑的**（实测：WM 在跑、感知头也加载了，
但一条提示都不发，等于跑了个空 WM）。

做法（用户提的"让 WM 先认领一波相关图"）：开局渲染目标图后，
**用 WM 自己的检测头（RF-DETR）读那张图**，把检出的物体登记成它的目标集
（`seed_targets_from_goal_image` + 运行时并进 `TargetHinter.relevant`）。

* **不引入模拟器真值**：目标图本来就是任务输入（就摆在 prompt 里），
  用 agent 自己的感知读它；检测器不认识的物体它自然认不出来。
* 实测：一条 iTHOR 任务认领到 12 类物体，提示行数 **0 → 12**，内容形如
  "视野内：CoffeeTable（正前方约 1.2m）…" / "任务相关记忆（当前看不见的）：
  Drawer（任务目标）：记住的位置在左前方约 2.0m ±0.2m（step 1 看到过）"。

**这条目标来源在官方 311/127 和 gemini/gpt5 那两批里都不存在**，是目前
TVR 这批与另两批在 WM 上唯一实质性的行为差异。

---

## 四、与主表对齐的配置

addon 默认读的基底是 `configs/ai2thor/config.yaml`，与主表用的
`experiments/configs/ai2thor/config_close_gpt-5.yaml` **不是同一个文件**：

| 项 | 主表 | addon 原样 | 处理 |
|---|---|---|---|
| `short_term_history_window_size` | 29 | 50 | **已在 addon 里钉成 29**（`TVR_HISTORY_TURNS=29`），对齐 `WM_HISTORY_TURNS` |
| `visibility_distance` | 1.0 | 1.5 | **不动** —— TVR 的 golden 与锚点都是按 1.5 量出来的，改了等于换环境 |
| `move_ahead/back/left/right_magnitude` | 0.5 | 0.25 | **不动** —— TVR 的 golden 用 0.25 走格，0.5 会让位姿对不上 |
| `render_instance_segmentation` | false | true | 判定需要，保留 |
