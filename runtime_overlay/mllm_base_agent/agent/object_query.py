"""On-demand world-model state summary (opt-in: WM_STATE_CHECK=1).

Scope (2026-09-16, after review): the *generic* object query was cut -- no
per-object position lookups, no watch list, nothing injected every step.  What
remains is one optional action the agent may use before ending the episode::

    CheckState()        # equivalent spelling: Query(all)

The world model then answers with a summary of the objects it remembers: where
it believes they are (bearing recomputed against the agent's *current* pose),
how stale that sighting is, how uncertain the triangulated position is, the
state it thinks they are in, and which ones it never saw.  ``DONE`` is never
intercepted: the agent may end the episode on any step, and whether to look at
the summary -- and whether to believe it -- is entirely the model's call.

Everything here is disabled unless ``WM_STATE_CHECK=1``; with the flag off the
runner behaves exactly as before.
"""

from __future__ import annotations

import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple

#: injected once, at step 0, in front of the first image
PROTOCOL_TEXT = """**收尾前可以查一次 WM 的状态（可选）**
如果你想在结束任务前确认各物体的状态，输出 CheckState()（等价写法 Query(all)），
WM 会把它记得的物体位置与状态汇总给你；这个汇总只给一次，不会每步重复。
WM 的记忆来自它自己的检测与画面变化判断，可能有错，"没见过"就是没见过。
注意：DONE 不受任何限制——你可以随时直接输出 DONE，也可以先 CheckState() 再决定。"""

_CHECK_ACTION_RE = re.compile(
    r"^\s*(?:checkstate|check_state|check\s+state|checkstatus|check\s+status|"
    r"verifystate|verify\s+state)\s*(?:\(\s*\))?\s*$",
    re.IGNORECASE,
)
_QUERY_RE = re.compile(
    r'^\s*(?:query|ask|where\s+is|find)\s*[\(\s:]\s*"?([^")]*?)"?\s*\)?\s*$',
    re.IGNORECASE,
)
_STOPWORDS = {"the", "a", "an", "my", "our", "your", "some", "this", "that"}
_ACTION_TAG_RE = re.compile(r"<ACTION>(.*?)</ACTION>", re.IGNORECASE | re.DOTALL)

_DIRECTIONS = ("正前方", "右前方", "正右方", "右后方",
               "正后方", "左后方", "正左方", "左前方")

#: interaction name -> (state when the frame changed, state when it did not)
_STATE_WORDS = {
    "ToggleObjectOn": ("WM 记为已开启", "WM 未看到画面变化，开启状态存疑"),
    "ToggleObjectOff": ("WM 记为已关闭", "WM 未看到画面变化，关闭状态存疑"),
    "OpenObject": ("WM 记为已打开", "WM 未看到画面变化，打开状态存疑"),
    "CloseObject": ("WM 记为已关闭", "WM 未看到画面变化，关闭状态存疑"),
    "PickupObject": ("WM 记为已被拿起", "WM 未看到画面变化，可能没拿起来"),
    "PutObject": ("WM 记为已放下（放下后的位置还没重新看到）", "WM 未看到画面变化"),
}


def extract_action_string(response_text: str) -> str:
    """The ``<ACTION>`` payload, or the raw text when the tag is missing."""
    match = _ACTION_TAG_RE.search(response_text or "")
    return (match.group(1) if match else (response_text or "")).strip()


def parse_query(action_string: str) -> Optional[str]:
    """``CheckState()`` (or any ``Query(...)``) -> the summary sentinel.

    Every accepted spelling means the same thing: "show me what you remember".
    ``Query(<name>)`` is accepted as an alias of ``CheckState()`` so that a
    model that spells the request that way still gets the summary instead of
    burning a parse-error retry.
    """
    plain = re.sub(r"\s+", " ", (action_string or "").strip().rstrip(".")).strip()
    if _CHECK_ACTION_RE.match(plain):
        return "__all__"
    match = _QUERY_RE.match(plain)
    if not match:
        return None
    name = match.group(1).strip()
    if not name or normalize(name) in {"all", "*", "everything", "summary",
                                       "summarize", "states"}:
        return "__all__"
    return name


def normalize(name: str) -> str:
    """Lowercase, drop articles, drop a trailing plural ``s``."""
    words = [w for w in re.split(r"[^a-z0-9]+", (name or "").lower()) if w]
    words = [w for w in words if w not in _STOPWORDS]
    if words and len(words[-1]) > 3 and words[-1].endswith("s"):
        words[-1] = words[-1][:-1]
    return " ".join(words)


def _direction_word(dx: float, dz: float, yaw: float) -> str:
    rad = math.radians(float(yaw))
    fwd = dx * math.sin(rad) + dz * math.cos(rad)
    right = dx * math.cos(rad) - dz * math.sin(rad)
    ang = math.degrees(math.atan2(right, fwd))
    return _DIRECTIONS[int(round(ang / 45.0)) % 8]


class ObjectMemory:
    """What the world model remembers, for the on-demand state summary."""

    def __init__(self) -> None:
        self.seen: Dict[str, Dict[str, Any]] = {}       # type -> last sighting
        self.acts: Dict[str, List[Tuple[int, str, Optional[bool]]]] = {}
        self.held: str = ""
        self.agent_xy: Optional[Tuple[float, float]] = None
        self.agent_yaw: float = 0.0

    # -- updates -----------------------------------------------------
    def observe(self, wm_metadata: Dict, step: int) -> None:
        """Fold one step of the world model's own metadata into the memory."""
        if not isinstance(wm_metadata, dict):
            return
        agent = wm_metadata.get("agent") or {}
        apos = agent.get("position") or {}
        ax, az = apos.get("x"), apos.get("z")
        yaw = float((agent.get("rotation") or {}).get("y") or 0.0)
        if ax is not None and az is not None:
            self.agent_xy = (float(ax), float(az))
            self.agent_yaw = yaw
        for obj in wm_metadata.get("objects") or []:
            if not isinstance(obj, dict):
                continue
            otype = str(obj.get("objectType") or "")
            if not otype:
                continue
            pos = obj.get("position") or {}
            entry = self.seen.setdefault(otype, {})
            entry["last_seen_step"] = int(step)
            entry["visible"] = bool(obj.get("visible"))
            entry["distance"] = float(obj.get("distance") or 0.0)
            if pos.get("x") is not None:
                entry["pos"] = (float(pos["x"]), float(pos.get("z") or 0.0))
            if obj.get("sigma") is not None:
                entry["sigma"] = float(obj["sigma"])
            if ax is not None and az is not None and pos.get("x") is not None:
                entry["direction"] = _direction_word(
                    float(pos["x"]) - float(ax),
                    float(pos.get("z") or 0.0) - float(az), yaw)
        inv = wm_metadata.get("inventoryObjects") or []
        self.held = str((inv[0] or {}).get("objectType") or "") if inv else ""

    def record_interaction(self, step: int, action_name: Optional[str],
                           object_type: Optional[str],
                           ok: Optional[bool]) -> None:
        """Remember the frame-difference verdict of an interaction."""
        if not action_name or not object_type:
            return
        self.acts.setdefault(str(object_type), []).append(
            (int(step), str(action_name), ok))

    # -- rendering ---------------------------------------------------
    def _entry_line(self, otype: str, entry: Dict[str, Any]) -> str:
        """One remembered object, bearing recomputed for the current pose."""
        chosen = False
        direction = entry.get("direction", "某处")
        dist = float(entry.get("distance") or 0.0)
        pos = entry.get("pos")
        if pos and self.agent_xy is not None:
            dx = pos[0] - self.agent_xy[0]
            dz = pos[1] - self.agent_xy[1]
            direction = _direction_word(dx, dz, self.agent_yaw)
            dist = math.hypot(dx, dz)
            chosen = True
        sigma = entry.get("sigma")
        band = f" ±{sigma:.1f}m" if sigma else ""
        if entry.get("visible"):
            head = f"{otype}：当前可见，{direction}约 {dist:.1f}m"
        else:
            head = (f"{otype}：记住的位置在{direction}约 {dist:.1f}m{band}"
                    f"（step {entry.get('last_seen_step')} 看到过，之后没再看到）")
            if not chosen:
                head = (f"{otype}：上次看到在{direction}约 {dist:.1f}m{band}"
                        f"（step {entry.get('last_seen_step')}，位置未经重算）")
        acts = self.acts.get(otype) or []
        if acts:
            step_i, name, ok = acts[-1]
            on, off = _STATE_WORDS.get(name, (f"WM 记为{name}完成", "WM 未看到画面变化"))
            head += f"；最近交互 step {step_i} {name} → {on if ok else off}"
        return head

    def summary(self, step: int, limit: int = 12,
                relevant: Optional[set] = None) -> str:
        """Per-object state summary, interacted-with objects first.

        ``relevant`` narrows the dump to the objects this task actually
        mentions (plus whatever was interacted with): a state dump that lists
        every mug and statue the WM ever saw is mostly noise, and the point of
        this block is to let the model check the task's own objects before it
        says DONE.  ``LIGHTWM_STATE_SCOPE=all`` restores the full dump.
        """
        lines = ["**WM 物体状态汇总（来自 WM 自己的检测与画面变化判断，可能有错）**"]
        seen = dict(self.seen)
        if relevant is not None:
            scope = os.environ.get("LIGHTWM_STATE_SCOPE", "relevant").lower()
            if scope != "all":
                keep = {t for t in seen if t in relevant or t in self.acts}
                if self.held:
                    keep.add(str(self.held))
                seen = {t: seen[t] for t in keep}
        if not seen:
            lines.append("- WM 还没有记住任何物体")
        else:
            acted = [t for t in self.acts if t in seen]
            rest = [t for t in seen if t not in acted]
            rest.sort(key=lambda t: -int(seen[t].get("last_seen_step") or 0))
            shown = (acted + rest)[:limit]
            for otype in shown:
                lines.append("- " + self._entry_line(otype, seen[otype]))
            hidden = len(seen) - len(shown)
            if hidden > 0:
                lines.append(f"- （另有 {hidden} 个 WM 记得的物体未列出）")
        lines.append(f"- 手持：{self.held or '空手'}")
        return "\n".join(lines)


def config_of(state: Dict) -> Dict[str, Any]:
    probe = ((state.get("config") or {}).get("memory_probe") or {})
    return probe.get("object_query") or {}


def memory_of(state: Dict) -> ObjectMemory:
    mem = state.get("_object_memory")
    if mem is None:
        mem = ObjectMemory()
        state["_object_memory"] = mem
    return mem


def render_block(state: Dict, step: int) -> str:
    """The one-shot state summary, shown only on the step after CheckState()."""
    mem = state.get("_object_memory")
    if mem is None:
        return ""
    if state.pop("_summary_pending", False):
        # 只汇总任务相关（+ 交互过 + 手持）的物体；没开目标提示时保持旧的全量。
        relevant = state.get("_wm_relevant_types")
        return (mem.summary(step, relevant=relevant)
                + "\n你可以据此判断任务是否已经完成：确认完成就输出 DONE，"
                  "否则继续下一步动作。")
    return ""


def apply_query(state: Dict, query: str, step: int) -> Dict[str, Any]:
    """The single remaining query: ask for the state summary (a no-op step)."""
    memory_of(state)                # make sure the memory exists
    state["_summary_pending"] = True
    state["should_continue"] = True
    state["_pending_query"] = query
    return {
        "action_type": "internal_noop",
        "action_name": "CheckState",
        "object_type": None,
    }
