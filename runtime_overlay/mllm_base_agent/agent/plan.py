"""Task-level subgoal decomposition, wired into the per-step hint.

Two prompt variants are kept so the ablation can compare them:

  "old" -- the first, plain version (3-8 subgoals, find/interact/verify,
           no environment mechanics);
  "new" -- the refined version (mandatory action verbs, one subgoal per
           object, openable containers must be opened, close right after use,
           verification last).

The model receives ONLY the task instruction.  Progress is tracked from the
agent's own action history (keyword match against the subgoal text), never
from the model's claim.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from mllm_base_agent.llm.messages import HumanMessage, SystemMessage

PLAN_PROMPT_OLD = """You are a task planner for an embodied agent acting in a 3D house.
Decompose the task instruction into an ordered, executable plan.

Rules:
1. 3 to 8 subgoals.
2. Each subgoal is ONE action phase: find/approach, interact, or verify.
   "pick up X and put it in Y" must be split into separate subgoals.
3. Always include a "find/look for" subgoal for any object whose location is
   not stated in the instruction.
4. Every subgoal carries a short hint naming the relevant object/container.
5. Use the SAME LANGUAGE as the instruction.
6. Output ONLY strict JSON, no markdown:
{"subgoals": [{"goal": "...", "hint": "..."}, ...]}
"""

PLAN_PROMPT_NEW = """You are a task planner for an embodied agent acting in a 3D house.
Decompose the task instruction into an ordered, executable plan.

Rules:
1. 3 to 8 subgoals.
2. EVERY subgoal must have the form: <action verb> + <object> (+ target
   container/state). Allowed verbs: find, go to, pick up, put into, open,
   close, turn on, turn off, slice, fill, clean, cook, drop, verify.
   NEVER write vague steps such as "interact with X", "approach X",
   "handle X", "deal with X".
3. Split compound instructions: "pick up X and put it in Y" becomes
   "pick up X" and "put X into Y" as separate subgoals.
4. Add a "find" subgoal for every object whose location is not given.
5. If one action applies to several objects ("slice all the vegetables"),
   list them ONE PER SUBGOAL: "slice the tomato", "slice the lettuce".
6. End with a verification subgoal that names the exact state to confirm,
   e.g. "verify the fridge door is closed", "verify the egg is in the pot".
7. Use the SAME LANGUAGE as the instruction.
8. Output ONLY strict JSON, no markdown:
{"subgoals": [{"goal": "...", "hint": "..."}, ...]}
9. Environment mechanics of this simulator (generic, same for every task):
   (a) slicing does NOT require a knife -- "slice the bread" is one step;
   (b) an object inside a closed openable container (Fridge, Microwave,
       Cabinet, Drawer, Box, Safe) can neither be taken out nor put in:
       include "open the <container>" BEFORE interacting with its contents.
       If the instruction says the object is inside that container (or asks
       to close it afterwards), the open step is mandatory;
   (c) close the container IMMEDIATELY after you finish taking from or
       putting into it -- do not carry the object away first and come back;
   (d) when you refer to an object in a later step, use its CURRENT state
       ("the cleaned mug", not "the dirty mug");
   (e) do not invent other tools or actions.
"""

PROMPTS = {"old": PLAN_PROMPT_OLD, "new": PLAN_PROMPT_NEW}

#: action keyword -> canonical action family, for matching a subgoal text
_ACTION_WORDS = {
    "find": ["find", "look for", "locate", "search", "go to"],
    "pickup": ["pick up", "take", "grab", "hold"],
    "put": ["put", "place", "drop", "throw", "insert"],
    "open": ["open"],
    "close": ["close", "shut"],
    "toggle_on": ["turn on", "switch on", "turning on"],
    "toggle_off": ["turn off", "switch off", "turning off"],
    "slice": ["slice", "cut", "chop"],
    "fill": ["fill"],
    "empty": ["empty", "pour out"],
    "clean": ["clean", "wash", "rinse"],
    "dirty": ["dirty", "make dirty"],
    "cook": ["cook", "heat", "bake", "fry"],
    "break": ["break", "crack"],
    "useup": ["use up", "consume"],
    "verify": ["verify", "confirm", "check"],
}

#: AI2-THOR action name -> canonical family
ACTION_FAMILY = {
    "PickupObject": "pickup", "PutObject": "put", "DropHandObject": "put",
    "ThrowObject": "put", "OpenObject": "open", "CloseObject": "close",
    "ToggleObjectOn": "toggle_on", "ToggleObjectOff": "toggle_off",
    "SliceObject": "slice", "BreakObject": "break", "CookObject": "cook",
    "CleanObject": "clean", "DirtyObject": "dirty",
    "FillObjectWithLiquid": "fill", "EmptyLiquidFromObject": "empty",
    "UseUpObject": "useup",
}

def _obj_matches(obj: str, text: str) -> bool:
    """Does the plan text refer to this simulator object type?"""
    if not obj:
        return True
    base = re.sub(r"(sliced|cracked|cooked|filled|usedup|dirty|clean)$", "",
                  obj.lower())
    forms = {obj.lower(), base}
    return any(f and f in text for f in forms)


def _extract_json(text: str):
    try:
        return json.loads(str(text).strip())
    except Exception:
        m = re.search(r"\{.*\}", str(text), re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None


def decompose(vlm, instruction: str, variant: str = "new",
              max_tokens: int = 600) -> List[Dict[str, str]]:
    """One text-only call.  Returns [] on any failure (caller just runs without
    a plan, exactly like the baseline)."""
    prompt = PROMPTS.get(variant, PLAN_PROMPT_NEW)
    messages = [SystemMessage(content=prompt),
                HumanMessage(content=f"Instruction: {instruction}\n"
                                     "Output the JSON plan now.")]
    try:
        response = vlm.invoke(messages)
    except Exception:
        return []
    data = _extract_json(getattr(response, "content", response))
    if not isinstance(data, dict):
        return []
    out = []
    for item in data.get("subgoals") or []:
        if isinstance(item, dict) and item.get("goal"):
            out.append({"goal": str(item["goal"])[:160],
                        "hint": str(item.get("hint") or "")[:160]})
    return out[:8]


def _family_of_goal(goal: str) -> Optional[str]:
    low = goal.lower()
    for fam, words in _ACTION_WORDS.items():
        for w in sorted(words, key=len, reverse=True):
            if w in low:
                return fam
    return None


def advance(plan: List[Dict[str, str]], done: int, action_name: Optional[str],
            object_type: Optional[str]) -> int:
    """Mark subgoals done from the agent's own action history.

    A step counts as done when the action family matches AND (when the plan
    step names an object) the object type appears in the subgoal text.
    """
    if not plan or not action_name:
        return done
    fam = ACTION_FAMILY.get(action_name)
    if fam is None:
        return done
    obj = (object_type or "").lower()
    for idx in range(max(0, min(done, len(plan) - 1)), len(plan)):
        step = plan[idx]
        if _family_of_goal(step.get("goal", "")) != fam:
            continue
        text = (step.get("goal", "") + " " + step.get("hint", "")).lower()
        if obj and not _obj_matches(obj, text):
            continue            # a different object -> keep looking
        return idx + 1
    return done


def render(plan: List[Dict[str, str]], done: int, max_shown: int = 8) -> str:
    """One short line for the prompt: plan + which steps the WM says are done."""
    if not plan:
        return ""
    parts = []
    for i, step in enumerate(plan[:max_shown]):
        mark = "✓" if i < done else ("← 当前" if i == done else "")
        goal = step.get("goal", "").strip()
        parts.append(f"{i + 1}){goal}{(' ' + mark) if mark else ''}")
    tail = f"（已完成 {done}/{len(plan)}）"
    return "任务计划：" + "；".join(parts) + tail
