"""Online task decomposition for embodied agents.

The planner model (usually a cheap local model) decomposes the raw task
instruction into an ordered list of single-action subgoals.  The runner
injects this plan (and the current progress) into every step's prompt so the
executing VLM always knows what it should be doing next.

This module is pure incremental functionality: if decomposition fails or is
disabled, the agent simply runs without the plan (identical to the official
behaviour).
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from mllm_base_agent.llm.messages import HumanMessage, SystemMessage


DECOMPOSE_SYSTEM_PROMPT = """You are a task planner for an embodied agent acting in a 3D environment. Given a task instruction, decompose it into an ordered, executable step-by-step plan.

Requirements:
1. Produce 3 to 8 ordered subgoals.
2. Each subgoal must correspond to ONE executable action phase (find/approach, interact, or verify). Never merge multiple actions into a single subgoal. For example, "pick up the apple and put it in the fridge" must be split into separate subgoals.
3. Explicitly include "find/look for" subgoals whenever the agent may not know where an object is.
4. Each subgoal must carry a hint that reminds the agent which objects are relevant, where they may be located, and any state requirement (e.g. "remember to close the fridge door afterwards").
5. Use the same language as the task instruction for all subgoals and hints.
6. Output ONLY strict JSON, no markdown, no commentary:
{"subgoals": [{"goal": "subgoal description", "hint": "reminder for the agent"}, ...]}
"""


_ACTION_KEYWORDS: Dict[str, List[str]] = {
    "OpenObject": ["open", "打开", "拉开"],
    "CloseObject": ["close", "关上", "关闭", "合上"],
    "PickupObject": ["pick", "pickup", "grab", "拿起", "拾取", "捡起", "拿"],
    "PutObject": ["put", "place", "放", "放置", "摆"],
    "DropHandObject": ["drop", "放下", "丢下"],
    "ThrowObject": ["throw", "扔", "丢"],
    "ToggleObjectOn": ["turn on", "switch on", "开启", "打开", "开灯"],
    "ToggleObjectOff": ["turn off", "switch off", "关闭", "关掉", "关灯"],
    "SliceObject": ["slice", "切", "切片"],
    "BreakObject": ["break", "打破", "打碎", "砸"],
    "CookObject": ["cook", "煮", "烹饪", "加热"],
    "CleanObject": ["clean", "清洗", "清洁", "擦干净", "洗"],
    "DirtyObject": ["dirty", "弄脏", "弄污"],
    "FillObjectWithLiquid": ["fill", "倒", "装满", "注满"],
    "EmptyLiquidFromObject": ["empty", "倒空", "清空"],
    "UseUpObject": ["use up", "用完", "用光"],
}


def _extract_json(text: str) -> Optional[Any]:
    """Extract a JSON object/array from a model response, tolerating noise."""
    if not text:
        return None
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except (ValueError, TypeError):
        pass
    # Strip markdown fences if present.
    fenced = re.search(r"```(?:json)?\s*(.+?)```", stripped, re.S)
    candidate = fenced.group(1).strip() if fenced else stripped
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = candidate.find(opener), candidate.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(candidate[start : end + 1])
            except (ValueError, TypeError):
                continue
    return None


def _validate_plan(data: Any) -> Optional[List[Dict[str, str]]]:
    if isinstance(data, dict):
        subgoals = data.get("subgoals")
    elif isinstance(data, list):
        subgoals = data
    else:
        return None
    if not isinstance(subgoals, list):
        return None
    plan: List[Dict[str, str]] = []
    for item in subgoals:
        if not isinstance(item, dict):
            continue
        goal = str(item.get("goal") or "").strip()
        if not goal:
            continue
        hint = str(item.get("hint") or "").strip()
        plan.append({"goal": goal, "hint": hint})
    return plan or None


def decompose_task(
    vlm: Any,
    task_prompt: str,
    env_type: str = "ai2thor",
    max_retries: int = 2,
) -> Optional[Dict[str, Any]]:
    """Ask the planner model to decompose a task instruction.

    Returns ``{"subgoals": [...], "raw_response": "..."}`` on success, or
    ``None`` if every attempt fails (caller should degrade gracefully).
    """
    messages = [
        SystemMessage(content=DECOMPOSE_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Task instruction: {task_prompt}\n"
                f"Environment type: {env_type}\n"
                "Output the JSON plan now."
            )
        ),
    ]
    last_response = ""
    last_usage: Dict[str, Any] = {}
    for attempt in range(max_retries):
        try:
            response = vlm.invoke(messages)
            last_response = str(getattr(response, "content", response or ""))
            usage_metadata = getattr(response, "usage_metadata", None) or {}
            if isinstance(usage_metadata, dict):
                last_usage = usage_metadata
            data = _extract_json(last_response)
            plan = _validate_plan(data)
            if plan:
                return {
                    "subgoals": plan,
                    "raw_response": last_response[:4000],
                    "attempts": attempt + 1,
                    "token_usage": last_usage,
                }
        except Exception:
            # Fall through to the next attempt; the runner never crashes on
            # a failed decomposition.
            continue
    return None


def format_subgoal_plan(plan: List[Dict[str, str]], current_index: int) -> str:
    """Render the full plan with progress markers for prompt injection."""
    if not plan:
        return ""
    lines = [
        "**Decomposed Task Plan (reference only; if it conflicts with what you "
        "observe, trust your observations):**"
    ]
    for idx, item in enumerate(plan):
        goal = item.get("goal", "")
        hint = item.get("hint", "")
        if idx < current_index:
            marker = "✅"
        elif idx == current_index:
            marker = "▶️"
        else:
            marker = "⏳"
        line = f"{marker} {idx + 1}. {goal}"
        if hint and idx == current_index:
            line += f"  (提示/hint: {hint})"
        lines.append(line)
    lines.append(
        f"\nWork through the plan in order. After each subgoal is done, state in "
        f"your THINK that it is complete before moving to the next one."
    )
    return "\n".join(lines)


def format_current_subgoal(plan: List[Dict[str, str]], current_index: int) -> str:
    """Render a short line highlighting the current subgoal."""
    if not plan:
        return ""
    index = max(0, min(current_index, len(plan) - 1))
    goal = plan[index].get("goal", "")
    hint = plan[index].get("hint", "")
    line = f"**当前子目标/Current subgoal ({index + 1}/{len(plan)}): {goal}**"
    if hint:
        line += f"\n提示/hint: {hint}"
    return line


def subgoal_matches_action(goal_text: str, action_dict: Dict[str, Any]) -> bool:
    """Check whether an executed action plausibly completes a subgoal."""
    action_name = str(action_dict.get("action_name") or "")
    keywords = _ACTION_KEYWORDS.get(action_name)
    if not keywords:
        return False
    text = str(goal_text or "").lower()
    return any(kw.lower() in text for kw in keywords)


def advance_subgoal_index(
    plan: List[Dict[str, str]], current_index: int, action_dict: Dict[str, Any]
) -> int:
    """Return the next subgoal index after a successful matching action.

    Searches forward from the current subgoal so navigation/exploration steps
    (which have no matching action) do not block progress once the agent
    performs the next interaction action.
    """
    if not plan:
        return current_index
    start = max(0, int(current_index or 0))
    for idx in range(start, len(plan)):
        if subgoal_matches_action(plan[idx].get("goal", ""), action_dict):
            return idx + 1
    return current_index
