# -*- coding: utf-8 -*-
"""
LightWM / SpatialWorld unified evaluation - user-editable configuration.

只改这个文件即可完成 90% 的实验配置切换：
  1. SCENES           选择评测哪几类场景（默认三类；去掉一个就只测两类）
  2. PROFILE          选择模型族（llm / wingman_llm / rl）
  3. LLM_PRESET       选择底层 LLM（GPT / Gemini / Qwen ...）
  4. TASK_IDS         选择任务（[] = 测全部；填 id 列表 = 只测这几个）
  5. 其余（输出目录、重试、并行度、是否 golden 复核等）都在下方常量里。

命令行参数可以临时覆盖其中大部分（python supervisor.py --help）。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from phase_b.perception_defaults import perception_defaults

_perception = perception_defaults()

# ---------------------------------------------------------------------------
# 1) 场景选择：三类家庭任务场景。要只测两类就删掉其中一行。
#    可选值：ai2thor / procthor / virtualhome
#    注意：默认 wingman_llm 目前只在 AI2-THOR / ProcTHOR 上可用
#    （VirtualHome 没有米制深度）；测 VH 时请把 PROFILE 改成 "llm"。
# ---------------------------------------------------------------------------
SCENES: list[str] = ["ai2thor", "procthor"]

# ---------------------------------------------------------------------------
# 2) 模型族选择。
#    llm        : 纯 MLLM（GPT / Gemini / Qwen ...），官方 agent loop
#    wingman_llm : WingmanWM(LightWM 世界模型) + 上述 MLLM，官方 agent loop
#    rl         : DreamerV3 / DIAMOND 等 RL 策略（见 rl_agents.py 接入说明）
#    VirtualHome 没有米制深度，目前只能用 llm 评测。
# ---------------------------------------------------------------------------
PROFILE: str = "wingman_llm"

# ---------------------------------------------------------------------------
# 3) 底层 LLM 预设。
#    presets 里的字段会整体覆盖每个场景官方配置中的 model.vlm 块。
#    常用预设：gpt-5 / gpt-5.4 / gemini-3.1-pro / qwen3.5
#    自定义：直接改 LLM_OVERRIDES（base_url / api_key / model_name / provider ...）
# ---------------------------------------------------------------------------
LLM_PRESET: str = "gpt-5"

LLM_PRESETS: dict = {
    "gpt-5": {
        "provider": "openai",
        "model_name": "gpt-5",
        "temperature": 1.0,
        "max_tokens": 4096,
        "base_url": None,
        "api_key": None,
    },
    "gpt-5.4": {
        "provider": "openai",
        "model_name": "gpt-5.4",
        "temperature": 1.0,
        "max_tokens": 4096,
        "base_url": None,
        "api_key": None,
    },
    "gemini-3.1-pro": {
        "provider": "openai",
        "model_name": "Gemini-3.1-Pro-Preview",
        "temperature": 1.0,
        "top_p": 0.9,
        "max_tokens": 4096,
        "base_url": None,
        "api_key": None,
    },
    "qwen3.5": {
        "provider": "openai",
        "model_name": "Qwen3.5-397B-A17B",
        "temperature": 1.0,
        "top_p": 0.9,
        "max_tokens": 4096,
        "base_url": None,
        "api_key": None,
    },
}

# 叠加字段（优先级最高）。例如：
# LLM_OVERRIDES = {"base_url": "https://xxx/v1", "api_key": "sk-..."}
LLM_OVERRIDES: dict = {}

# ---------------------------------------------------------------------------
# WingmanWM / LightWM 感知权重（wingman_llm profile 用）。
# 运行时由 RGB 预测检测框与深度，并结合动作日志更新空间记忆。
# ---------------------------------------------------------------------------
PERCEPTION_CKPT: str = _perception['perception_ckpt']
WINGMAN_OPTIONS: dict = {
    # 注意：目标是 per-task 自动从 task.json 的 target_object_types 读入，
    # 不需要在这里手填。下面只是世界模型/门控的运行时选项。
    "variant": "small",      # small@224（本地权重）；云端 336 权重改 "base"
    "resolution": 224,
    "width": 256,
    "obj_thr": _perception['obj_thr'],
    "detector_backend": _perception['detector_backend'],
    "detector_path": _perception['detector_path'],
    "perception_runtime_root": str(REPO_ROOT),
    "navigation_directive": True,
    "interact_soft_dist": 1.2,
    "done_gate": True,
    "pose_from_action_log": True,
    "pose_initial": "origin",
}

# ---------------------------------------------------------------------------
# RL 策略（profile == "rl" 时生效；接入说明见 rl_agents.py）
#   RL_POLICY_IMPORT 空 = 使用内置 GoldenActionsPolicy 做自检/演示；
#   接真实 DreamerV3 / DIAMOND 时填 "module.path:ClassName"，
#   该类只需实现 act(observation, step) -> unified action dict。
# ---------------------------------------------------------------------------
RL_ENGINE: str = "dreamer_v3"      # dreamer_v3 / diamond（仅记录用途）
RL_POLICY_IMPORT: str = ""         # 空 = golden 自检；否则 "pkg.mod:Policy"
RL_CHECKPOINT: str = ""            # 传给策略 __init__ 的 checkpoint 路径

# ---------------------------------------------------------------------------
# 4) 任务选择：[] = 对应 SCENES 的全部任务；否则只跑列出的任务。
#    任务 id 示例：ai2thor03073 / procthor100 / virtualhome00020
# ---------------------------------------------------------------------------
TASK_IDS: list[str] = []

# 可选：按前缀过滤（例如只测官方考试子集时用正则 "ai2thor0(3|4)"）。
TASK_ID_FILTER: str = ""

# ---------------------------------------------------------------------------
# 路径与运行参数
# ---------------------------------------------------------------------------
SPATIALWORLD_ROOT: str = os.environ.get("SPATIALWORLD_ROOT", "/home/sudidaren/SpatialWorld")

# 每个场景使用其官方 uv venv 来执行（保证依赖/模拟器与官方一致）。
ENV_VENV_PYTHON: dict[str, str] = {
    "ai2thor": f"{SPATIALWORLD_ROOT}/envs/ai2thor/.venv/bin/python",
    "procthor": f"{SPATIALWORLD_ROOT}/envs/procthor/.venv/bin/python",
    "virtualhome": f"{SPATIALWORLD_ROOT}/envs/virtualhome/.venv/bin/python",
}
if os.environ.get('LIGHTWM_EVAL_PYTHON'):
    ENV_VENV_PYTHON = {env: os.environ['LIGHTWM_EVAL_PYTHON'] for env in ENV_VENV_PYTHON}

# 每个场景的官方基准配置（只作为模型块/日志模板的基底）。
BASE_CONFIG: dict[str, str] = {
    "ai2thor": (
        f"{SPATIALWORLD_ROOT}/experiments/configs/ai2thor/"
        "config_close_gpt-5.yaml"
    ),
    "procthor": (
        f"{SPATIALWORLD_ROOT}/experiments/configs/procthor/"
        "config_close_gpt-5.yaml"
    ),
    "virtualhome": (
        f"{SPATIALWORLD_ROOT}/experiments/configs/virtualhome/"
        "config_close_gpt-5.yaml"
    ),
}

OUTPUT_ROOT: str = "/home/sudidaren/spatialworld_eval/runs"
RUN_NAME: str = ""                    # 空 = 自动生成 20260909_<profile>_<model>

HEADLESS: bool = True                 # 服务器无显示器务必 True
WORKERS: int = 1                      # 并行任务数（默认 1，最稳）
MAX_EXTERNAL_RETRIES: int = 2         # api/env 故障自动重试次数
TASK_TIMEOUT_SEC: int = 3600          # 单任务硬超时（秒）
# 每次 LLM API 请求的硬超时（秒）。官方 provider 通过环境变量
# VLM_API_TIMEOUT 读取（默认 300），我们注入更短的值，避免 TCP 连接
# 卡死导致整个任务无限等待——这是"卡在第 N 步不动"的根因级修复。
LLM_API_TIMEOUT_SEC: int = 120

# 兜底看门狗：超过该秒数没有任何 stdout/stderr 输出才判定卡死并杀掉重试。
# 现在 API 请求本身已有 120s 超时 + 官方自动重试，这里的阈值放宽到
# 10 分钟，只用于捕获 Unity/环境层真正僵死的情况，不会误杀正常慢请求。
STALL_TIMEOUT_SEC: int = 600

# 官方 run_benchmark 对成功结果会再用 golden actions 复核一遍；
# 这里默认开启以保持一致（费时，可关）。
VERIFY_GOLDEN_REPLAY: bool = True

# 已产生"决定性结果"（true/false）的任务在重跑时默认跳过（断点续跑）。
RESUME_SKIP_DECIDED: bool = True

ENV_CLASSIFICATION = {
    "ai2thor": "ai2thor",
    "procthor": "procthor",
    "virtualhome": "virtualhome",
}


def resolve_llm_model() -> dict:
    preset = dict(LLM_PRESETS.get(LLM_PRESET) or LLM_PRESETS["gpt-5"])
    preset.update(LLM_OVERRIDES)
    return preset


def output_root() -> Path:
    return Path(OUTPUT_ROOT)
