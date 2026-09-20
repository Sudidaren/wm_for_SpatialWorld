# 云上卡：最大的几栋 ProcTHOR 房子起不来（环境外部失败，不是模型失败）

2026-09-21 凌晨，30b 卡（`wm_q30b_438_v2`）上出现成批
`Failed to create environment: Reading from AI2-THOR backend timed out
(using 300.0s)`。**判定是 failed_external / env_error，不是模型没做对。**

## 现象

按 Scene 统计 30b 卡已判定的 199 条（同一批里其它场景都正常跑完了）：

| Scene | 结果 |
|---|---|
| FloorPlan301 / 310 / 320 | 有 success |
| FloorPlan303 / 311 / 313 / 319 / 321 / 322 / 323 / 324 / 329 / 330 | 全是 failed_model（**环境正常，模型没做对**） |
| FloorPlan306 / 307 / 312 / 314 / **315** / FloorPlan3 | 全是 failed_external（**环境没起来**） |

即：失败的只有最大的那几栋（FloorPlan306/307/312/314/315）。同一场景的
连续几个任务（如 05021-05024 全是 FloorPlan312、05031-05035 全是
FloorPlan315）**无一例外全部超时**，没有一条"慢但最终起来了"。

## 已经排除的原因

1. **不是构建缺失/损坏**：本地与四张卡的
   `~/.ai2thor/releases/thor-Linux64-f0825767.../` 文件数都是 162，
   `data.unity3d` 都是 977,372,770 字节，目录结构逐项一致。
2. **不是并发抢资源导致的偶发**：单独在 30b 卡上跑
   `tools/newcard_scene_probe.py FloorPlan315 900`（只 1 个 Controller），
   在 6 worker 并行的情况下 **>15 分钟没有返回**，远超 300s，也超过给它的
   900s。说明不是"再给点时间就行"。
3. **不是场景本身不可用**：本机跑的 `wingman_wm_qwen3vl30b_438_v1`
   里 `ai2thor05021`（FloorPlan312）是 **success**，`ai2thor05034`
   （FloorPlan315）是正常的 failed_model —— 同样的场景在本机起得来。

## 结论

卡上是 Xvfb + `thor-Linux64`（X11 构建）软件渲染；本机 WSLg 那条路有 GPU
加速。这几栋大房子在软件渲染下加载不出来。**修法是换成官方 headless 的
CloudRendering 构建**（`config_builder._apply_headless` 在
`HEADLESS=1` 时把 `env.platform` 设成 CloudRendering，本地
`~/.ai2thor/releases/thor-CloudRendering-f0825767.../` 就有这个构建），
但那会换掉渲染器、和已经跑出来的结果口径不一致，**没有在跑批中途动**。

所以这批结果里，上述场景的任务应作为「环境不可用」标注，而不是算进
模型成功率的分母。
