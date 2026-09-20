# 参数策略：相机 / 深度参数不许用评测集来定

生效：2026-09-20 ｜ 适用范围：WingmanWM 的所有运行时旋钮（相机、深度、几何）

> 本文件是**新加**的策略文件，不在 `FROZEN.sha256` 清单里，没有改动任何冻结文件。
> 校验它有没有被遵守的是 `tests/test_param_provenance.py`（在交付验证里跑）。

---

## 1. 两条硬禁止

**禁止 A —— 不得用评测任务（311 条 AI2-THOR / 127 条 ProcTHOR / 任何评测房）
的反饋去拟合相机或深度参数。** 包括但不限于 `LIGHTWM_DA2_SCALE`、
`LIGHTWM_FOV_CONVENTION`、深度探针方式、三角化基线。

**禁止 B —— 不得用任务真值去挑参数。** 包括 `success_conditions`、
`golden_actions`、`target_object_types`，以及模拟器的物体表 / 位姿 / 分割。

**评估 ≠ 拟合。** 判据很简单：

* 规则**先定好**、评测集只用来**报告**这个既定配置的成绩 → 允许（评估）；
* 反过来看评测结果再回头调值 → 禁止（拟合）。

---

## 2. 参数只能来自这三种来源

| 来源 | 含义 | 例 |
|---|---|---|
| `definition` | 由模拟器 / 渲染器的**定义**直接导出，不涉及任何数据 | AI2-THOR 的 `fieldOfView` 是**垂直**视场，所以 `fx = fy = (height/2)/tan(fov/2)` → `LIGHTWM_FOV_CONVENTION=vertical` |
| `non_eval_rooms` | 在**非评测房**上标定：深度头没训过、也不是评测用的房子 | classic 家族 `FloorPlan1/2/4/5/7/10/19`（`tools/eval_wm_positions.py` 的默认房间） |
| `geometry` | 由 WM 自身的几何 / 里程计推出的**无标签**量 | `LIGHTWM_SCALE_ANCHOR`：只用"三角化距离 / 深度距离"的比值 |

**"哪个误差小"不是理由。** 误差小只能用来**印证**一个已经由上面三种来源
支持的选择；先看误差再定值，就是禁止 A。

---

## 3. 当前旋钮与来源（与代码默认值逐字对齐）

| 旋钮 | 代码默认 | 位置 | 来源 | 依据 |
|---|---|---|---|---|
| `LIGHTWM_FOV_CONVENTION` | `vertical` | `mllm_base_agent/agent/world_model.py` | `definition` | AI2-THOR 的 `fieldOfView` 定义为垂直视场；用**真值深度**在 3088 个记录视角上印证（3D 0.189m vs 旧约定 0.356m，与检测/深度头无关） |
| `LIGHTWM_DA2_SCALE` | `1.3816` | `lightwm_phases/phase_b/rfdetr_depth_runtime.py` | `non_eval_rooms` | 深度头在**非评测房**上的尺度常数（交接文档 §5：0.807 → 0.613 → 0.210m） |
| `LIGHTWM_TRI_WEIGHT` | `1.0` | `world_model.py` | `non_eval_rooms` | 707 个留出物体：纯几何 0.236m vs 纯深度 0.8m |
| `LIGHTWM_TRI_BASELINE` | `6.0` | `world_model.py` | `non_eval_rooms` | 12 个 classic 非评测 episode 的扫描（1.0m→1.005m … 6.0m→0.803m） |
| `LIGHTWM_DEPTH_PROBE` | `center` | `world_model.py` | （出厂值，未标定） | 改变它必须先拿 non_eval 数据说话 |
| `LIGHTWM_SCALE_ANCHOR` | `0` | `world_model.py` | `geometry` | 零标签，但任何批次都还没验证过 → 默认关 |

---

## 4. 怎么校验（机械可查）

`tests/test_param_provenance.py`：

1. 从源码里**正则抽出**每个旋钮的默认值，和上表逐字比对 —— 改了默认值却
   不更新声明，测试直接失败；
2. 每个旋钮必须声明来源，且来源必须在 `definition` / `non_eval_rooms` /
   `geometry` 三者之内（**不允许出现任何指向评测集的来源**）；
3. 断言 FOV 约定是垂直（`definition`），防止有人用"哪个误差小"把它换掉。

它和 `tests/test_wm_delivery.py` 的信息隔离审计一起，构成"不改评测集、
不越红线"的两道机械闸门。

---

## 5. 允许 / 不允许清单

| 允许 | 不允许 |
|---|---|
| 在非评测房上做 A/B，据此**决定**开关 | 在评测集上 A/B 再据此决定 |
| 在评测集上**报告**某个已定配置的成绩 | 看评测成绩回头改参数 |
| 用评测帧算**误差指标**（例如"WM 报够得着但真值 >1.5m"的 49%） | 用那些误差去反解常数 |
| 用模拟器定义推导相机内参 | 用拟合误差来"选"内参约定 |
