#!/usr/bin/env python
"""End-to-end loop check for the state-check action (WM_STATE_CHECK).

Drives the real think_node / act_node pair with a scripted model so the runner
hooks are exercised for real:

    MoveAhead     -> normal environment step (nothing extra in the prompt)
    CheckState()  -> one-shot state summary on the next prompt
    DONE          -> accepted immediately, never intercepted

Run:  python tests/test_object_query_loop.py
"""

from __future__ import annotations

import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mllm_base_agent.agent import runner as R  # noqa: E402


class StubVLM:
    """Returns the scripted replies in order and records the prompts it saw."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def invoke(self, messages):
        content = messages[-1].content
        text = " ".join(p.get("text", "") for p in content
                        if isinstance(p, dict) and p.get("type") == "text")
        self.prompts.append(text)
        return type("R", (), {"content": self.replies.pop(0),
                              "response_metadata": {}})


class StubObservation:
    def __init__(self, image_path, metadata):
        self.image_path = image_path
        self.metadata = metadata
        self.reward = 0
        self.text_state = ""


class StubEnv:
    def __init__(self, image_path, metadata):
        self.image_path = image_path
        self.metadata = metadata
        self.calls = []

    def step_with_action_dict(self, action):
        self.calls.append(action)
        return StubObservation(self.image_path, self.metadata), None


def write_png(path):
    # 1x1 transparent PNG
    data = bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
        "890000000a49444154789c6360000002000154a24f5f0000000049454e44ae42"
        "6082")
    with open(path, "wb") as fh:
        fh.write(data)
    return path


def metadata(agent_x, laptop_x):
    return {
        "agent": {"position": {"x": agent_x, "y": 0.9, "z": 0.0},
                  "rotation": {"x": 0.0, "y": 0.0, "z": 0.0}},
        "objects": [{"objectType": "Laptop",
                     "position": {"x": laptop_x, "y": 0.9, "z": 2.0},
                     "visible": True, "distance": 2.0}],
        "inventoryObjects": [],
    }


def main():
    tmp = tempfile.mkdtemp(prefix="oq_loop_")
    image = write_png(os.path.join(tmp, "frame0.png"))
    current = {"meta": metadata(0.0, 0.0)}

    vlm = StubVLM([
        "<THINK>Walk towards the laptop.</THINK><ACTION>MoveAhead(1)</ACTION>",
        "<THINK>Let me check the state before finishing.</THINK><ACTION>CheckState()</ACTION>",
        "<THINK>WM confirms the laptop is visible.</THINK><ACTION>DONE</ACTION>",
    ])
    env = StubEnv(image, current["meta"])
    state = {
        "vlm": vlm,
        "env": env,
        "observation": StubObservation(image, current["meta"]),
        "task_prompt": "open the laptop",
        "step_count": 0,
        "max_steps": 20,
        "config": {
            "env": {"type": "ai2thor"},
            "task": {"instruction": "open the laptop", "max_steps": 20},
            "actions": {},
            "memory_probe": {
                "enabled": True,
                "target_source": "keyword",
                "object_query": {"enabled": True},
            },
        },
        "should_continue": True,
        "success": None,
    }

    checks = []

    # step 1: a normal move; the prompt must not carry any per-step memory block
    state = R.think_node(state)
    first_prompt = vlm.prompts[-1]
    checks.append(("no per-step memory block for a plain move",
                   "WM 物体状态汇总" not in first_prompt
                   and "WM 记忆" not in first_prompt))
    state = R.act_node(state)
    checks.append(("move reached the environment",
                   len(env.calls) == 1 and env.calls[0]["action_name"] == "MoveAhead"))
    checks.append(("step counter advanced", state["step_count"] == 1))

    # step 2: CheckState() is a no-op step, answered on the next prompt
    state = R.think_node(state)
    checks.append(("CheckState is a no-op action",
                   state["next_action"]["action_name"] == "CheckState"
                   and state["should_continue"] is True))
    state = R.act_node(state)
    checks.append(("no environment call for CheckState", len(env.calls) == 1))

    # step 3: the summary is shown once, and DONE passes straight through
    state = R.think_node(state)
    summary_prompt = vlm.prompts[-1]
    checks.append(("state summary shown after CheckState()",
                   "物体状态汇总" in summary_prompt
                   and "确认完成就输出 DONE" in summary_prompt))
    checks.append(("DONE accepted immediately (not intercepted)",
                   state["next_action"]["action_name"] == "DONE"
                   and state["should_continue"] is False))
    checks.append(("no extra API call beyond the script",
                   vlm.replies == [] and len(vlm.prompts) == 3))

    prompts_with_protocol = sum("CheckState()" in p for p in vlm.prompts)
    checks.append(("protocol injected exactly once", prompts_with_protocol == 1))

    failed = 0
    for name, ok in checks:
        print(("ok   " if ok else "FAIL ") + name)
        failed += 0 if ok else 1
    print(f"\n{len(checks) - failed}/{len(checks)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
