# 云卡"环境不可用"的真相：每个任务白烧一次全场景重载

2026-09-21 白天查清。结论先行：**不是场景跑不了，也不是并发抢资源，
是 `run_task.py` 对同一场景连续 reset 了两次，而云卡软件渲染下一次 reset
要 ~180 秒。**

## 一、以前是怎么跑通的（查 `docs/RESULTS_2026-09-19_local_and_cloud.md`）

云卡上**确实跑通过全链路**：

* 卡 C `base_q30_a311`：311/311 完成，18 成功（Qwen3-VL-30B 纯基线）
* 卡 D `wm_q30_da2_a311`：311/311 完成，13 成功（WM+30B）
* 口径注明是"**云上口径（10 worker、卡上 vLLM）**"

同一份文档也记录了**当年环境故障同样是常态**：一批批次直接被命名成
`_xvfb_failed` / `_errored_xvfb` 然后**作废重跑**，例如
`wm_q30_da2_a311_xvfb_failed`（216/311，成功 0）、
`wm_kimi_da2_a311_errored_1`（311/311，成功 0）。
provision 脚本 `tools/c2c_provision_remote.py` 里还加了 Xvfb 健康检查，
注释写着"这条检查能避免再烧 229 个任务"。

**所以当年的打法是：撞上环境故障就把整臂作废、重跑，直到拿到干净的 311/311。**

## 二、这次量到的真因

在空闲的 gpt5 卡上做隔离测量（无其它负载、Xvfb 刚重启过）：

| 步骤 | 本机（WSLg, llvmpipe+Vulkan） | 云卡（Xvfb, llvmpipe+OpenGL 4.5） |
|---|---|---|
| `Controller(scene=FloorPlan14)` | 3.4s | 9.4s |
| 第 1 次 `reset(scene)` | 1.0s | **183.2s** |
| 第 2 次 `reset(scene)` | 1.0s | 超时未返回 |

注意 `Controller(...)` 只差 3 倍，但 **`reset` 差 180 倍** —— 瓶颈在 reset
这条路径上（它才是真正重新加载场景的那一下）。重启 Xvfb、换 ICD、装
Vulkan 都没用：卡上 Unity 始终 `Renderer: llvmpipe` +
`OPENGL LOG: Creating OpenGL 4.5 graphics device`，而本机是
`Forcing GfxDevice: Vulkan`。Unity 2020 在无 `/dev/dri` 的容器里选不到
Vulkan，只能退回 OpenGL 软件路径。

## 三、真正的浪费：每个任务 reset 两次

`scripts/ai2thor/work/run_task.py`：

```python
observation = env.reset(task_description, scene=task_scene)   # (1)
...
init_data, init_scene = load_init_actions_from_folder(...)
if init_scene:                                                # (2)
    task_scene = init_scene
    observation = env.reset(task_description, scene=task_scene)
```

而 `AI2ThorEnvWrapper.reset()` 是**无条件** `controller.reset(scene=self.scene)`，
也就是一整次全场景重载。查了全部 311 条任务目录：**309 条 init.json 带 scene，
其中 308 条和 task.json 的 scene 完全相同**（唯一不同的是 ai2thor05571，
FloorPlan412 → FloorPlan430）。

也就是说 308/311 条任务的第二次 reset 是**纯冗余**：不改状态、不提供信息，
只是把场景从零再装一遍。云卡上 ~180s × 2 = 6 分钟/任务，直接撞穿
环境创建超时。

## 四、修法（对两边同时生效，语义是 no-op）

```python
if init_scene and init_scene != getattr(env, "scene", None):
    task_scene = init_scene
    observation = env.reset(task_description, scene=task_scene)
```

补丁：`patches/ai2thor_run_task_skip_redundant_reset.patch`
（文件本体在 `SpatialWorld/scripts/ai2thor/work/run_task.py`，已同步到四张卡）。

**实测效果**：30b 卡上原本必挂的 ai2thor03015/03016/03017 立刻跑完，
`failed_external` 从 36 一路往下掉。

## 五、还剩下的

即便砍掉一半，云卡一次 reset 仍要 ~180s（本机 1s）。所以补跑那批剩余
环境失败任务时把超时一并放长（`tools/newcard_refill_ai2thor.sh`）：
`AI2THOR_SERVER_TIMEOUT=900`、`AI2THOR_START_TIMEOUT=900`、
`WM_STALL_TIMEOUT=1800`（默认 600s 的 stall 看门狗会在场景还没 init 完时
就把 worker SIGKILL 掉）。

要让云上真正追上本机，还得让 Unity 用上 Vulkan —— 那需要容器里能看到
`/dev/dri`，属于宿主机层面的事，配置绕不过去。
