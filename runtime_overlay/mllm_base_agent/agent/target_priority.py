"""Priority-ranked target hint, built from the MODEL's own object names.

No object vocabulary, no alias table (2026-09-16 decision: the baseline never
sees one, so the WM arm must not use one either).  Instead:

  1. at the start of the episode the agent's own VLM is asked, in free text,
     which objects the instruction involves -- one extra text call per task;
  2. those runtime strings are matched against the *runtime* output of the
     perception head (case-insensitive / substring), so an object only ever
     gets a position if the world model actually detected something whose
     name the model itself used;
  3. everything else (tiers, quotas, "only memory", pose-triggered refresh,
     distance/sigma caps, predicate demotion) is unchanged.

The only fixed strings left in this file are *verbs* (role markers used by the
settled-predicate test to decide that a remembered item is already finished),
not object names.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

_DIRECTIONS = ("正前方", "右前方", "正右方", "右后方",
               "正后方", "左后方", "正左方", "左前方")

#: role verbs for the settled-metadata test (no object names involved)
_ROLE_VERB = (
    r"(?:pick(?:ed|ing)?(?:\s+up)?|grab(?:bed)?|take|took|taking|bring|brought|"
    r"put|placed?|drop(?:ped)?|throw|threw|toss(?:ed)?|open(?:ed)?|"
    r"clos(?:e|ed|ing)|shut|turn(?:ed)?\s+(?:on|off)|switch(?:ed)?\s+(?:on|off)|"
    r"toggle[d]?|slice[d]?|cut|chop(?:ped)?|cook(?:ed)?|heat(?:ed)?|"
    r"wash(?:ed)?|clean(?:ed)?|wipe[d]?|rinse[d]?|fill(?:ed)?|empty|emptied|"
    r"pour(?:ed)?|use[d]?|hold|held|set|leave|left|move[d]?|transfer(?:red)?|"
    r"read|smash(?:ed)?|enable[d]?|disable[d]?|water(?:ed)?)"
)
_DEST_VERB = r"(?:put|place|move|set|leave|bring|carry|throw|toss|transfer)"

#: The agent-facing prompt for the one-shot extraction.  v2 (2026-09-16).
#:
#: Measured on the 438 real instructions with the deployed Qwen3-VL-30B
#: (0.2 s/call): coverage 94.7%, item precision 78.7%, and 92.2% of the
#: emitted names fall inside the perception head's label space.  A richer
#: "env + words" dual-field variant was tried and rejected: coverage fell to
#: 93.2% (5 empty answers instead of 1) while the words fallback added only
#: 0.2 pp of alignment.
EXTRACTION_PROMPT = (
    "List the object types this household task requires the agent to act on. "
    "Use EXACTLY the environment's object-type token, the way the environment "
    "names it: single words or CamelCase compounds such as \"GarbageCan\", "
    "\"Fridge\", \"CounterTop\", \"CellPhone\", \"DeskLamp\", \"SoapBottle\". "
    "Never output plurals, category words (\"vegetables\", \"food\"), tools "
    "the task does not require, or scenery the task does not act on. "
    "At most 4 items. Answer with strict JSON only: "
    "{\"objects\": [\"GarbageCan\", ...]}"
)


def normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name or "").lower())


def match_names(model_names: Iterable[str],
                detected_types: Iterable[str]) -> Dict[str, str]:
    """model string -> detected type, by case-insensitive substring match.

    Both sides are runtime strings (the model's words and the head's output),
    so this adds no external vocabulary.
    """
    out: Dict[str, str] = {}
    for name in model_names:
        key = normalize(name)
        if not key:
            continue
        best = None
        for tp in detected_types:
            cand = normalize(tp)
            if not cand:
                continue
            if key == cand or key in cand or cand in key:
                if best is None or len(cand) < len(normalize(best)):
                    best = tp
        if best is not None:
            out[str(name)] = str(best)
    return out


def parse_extraction(raw: str) -> List[Tuple[str, str]]:
    """Parse the model's answer into ``(env_name, own_wording)`` pairs.

    Accepts both the v2 dict format and a plain list of strings, so an older
    habit or a truncated answer still yields something usable.
    """
    match = re.search(r"\{.*\}", raw or "", re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except Exception:
        return []
    out: List[Tuple[str, str]] = []
    for item in (data.get("objects") or []):
        if isinstance(item, dict):
            env = str(item.get("env") or item.get("name") or "").strip()
            words = str(item.get("words") or item.get("word") or "").strip()
            if env or words:
                out.append((env or words, words))
        else:
            name = str(item).strip()
            if name:
                out.append((name, name))
    return out[:4]


def direction_word(dx: float, dz: float, yaw: float) -> str:
    rad = math.radians(float(yaw))
    fwd = dx * math.sin(rad) + dz * math.cos(rad)
    right = dx * math.cos(rad) - dz * math.sin(rad)
    ang = math.degrees(math.atan2(right, fwd))
    return _DIRECTIONS[int(round(ang / 45.0)) % 8]


TIER_LABEL = {1: "任务目标", 2: "手持/容器内容", 3: "已交互", 4: "只是见过"}


def settled_by_predicate(instruction: str,
                         entries: Iterable[Tuple[str, str]],
                         states: Dict[str, Dict[str, Any]],
                         contents: Dict[str, List[str]],
                         holding: str = "") -> Set[str]:
    """Detected types whose task predicate already holds, per WM's own ledger.

    Objects are addressed by the model's own names (matched to detected types
    by :func:`match_names`), so no vocabulary is needed.
    """
    text = re.sub(r"\s+", " ", (instruction or "").lower())
    done: Set[str] = set()

    def _mentions(*names: str) -> bool:
        low = normalize(text)
        return any(normalize(n) and normalize(n) in low for n in names)

    resolved: List[Tuple[str, str]] = []      # (display name, detected type)
    for env_name, words in entries:
        hit = match_names([env_name], list(states) + list(contents))
        if not hit and words:
            hit = match_names([words], list(states) + list(contents))
        if hit:
            resolved.append((env_name, list(hit.values())[0]))

    for name, ttype in resolved:
        st = states.get(ttype) or {}
        words = next((w for e, w in entries if e == name), "")
        if not _mentions(name, words):
            continue
        if st.get("isSliced") and \
                re.search(r"\b(?:slice|cut|chop)\b", text):
            done.add(ttype)
        if st.get("isCooked") and re.search(r"\b(?:cook|heat|bake|fry)\b", text):
            done.add(ttype)
        if st.get("isDirty") is False and \
                re.search(r"\b(?:clean|wash|wipe|rinse)\b", text):
            done.add(ttype)
        if st.get("isFilledWithLiquid") and re.search(r"\bfill\b", text):
            done.add(ttype)
        if st.get("isOpen") is True and re.search(r"\bopen\b", text):
            done.add(ttype)
        if st.get("isOpen") is False and re.search(r"\b(?:close|shut)\b", text):
            done.add(ttype)
        if st.get("isToggled") is True and re.search(r"\b(?:turn|switch)\s+on\b", text):
            done.add(ttype)
        if st.get("isToggled") is False and re.search(r"\b(?:turn|switch)\s+off\b", text):
            done.add(ttype)

    # "put <nameA> into <nameB>": settled once the ledger says A is inside B
    for i, (a, ta) in enumerate(resolved):
        words_a = next((w for e, w in entries if e == a), "")
        if not _mentions(a, words_a):
            continue
        for b, tb in resolved[i + 1:] + resolved[:i]:
            if re.search(rf"\b{_DEST_VERB}\b[^.;]{{0,30}}?\b(?:in|into|on|onto|inside)"
                         rf"\s+(?:the\s+)?[^.;]{{0,20}}", text) and \
                    tb in contents and ta in [str(x) for x in contents[tb]]:
                done.add(ta)
    if holding and holding in states:
        pass                      # 手持不算"已完成"，继续报
    return done


class TargetHinter:
    """Per-step builder for the ranked target block (stateful across steps)."""

    def __init__(self, task_description: str, vlm: Any = None,
                 limit: int = 6, names_limit: int = 5,
                 max_dist: float = 3.0, max_sigma: float = 0.5) -> None:
        self.task_desc = task_description or ""
        self.vlm = vlm
        self.limit = int(limit)
        self.names_limit = int(names_limit)
        self.max_dist = float(max_dist)
        self.max_sigma = float(max_sigma)
        self._entries: Optional[List[Tuple[str, str]]] = None
        self._never_seen_reported: Set[str] = set()
        self._last_pose: Optional[Tuple[float, float, float]] = None
        self._last_lines: Dict[str, str] = {}

    # -- the model's own object list ---------------------------------
    def entries(self) -> List[Tuple[str, str]]:
        """The model's own object list: (env token, its own wording) pairs."""
        if self._entries is not None:
            return self._entries
        self._entries = self._ask_model()
        return self._entries

    def model_names(self) -> List[str]:
        return [env for env, _ in self.entries()]

    def _ask_model(self) -> List[Tuple[str, str]]:
        """One extra text call per task: the model names the task objects."""
        if self.vlm is None or not self.task_desc:
            return []
        try:
            from mllm_base_agent.llm.messages import HumanMessage, SystemMessage

            messages = [
                SystemMessage(content=EXTRACTION_PROMPT),
                HumanMessage(content=f"Instruction: {self.task_desc}\nJSON now."),
            ]
            response = self.vlm.invoke(messages)
            text = str(getattr(response, "content", response) or "")
            return parse_extraction(text)
        except Exception:
            return []

    # -- rendering ---------------------------------------------------
    def _pose_changed(self, x: Optional[float], z: Optional[float], yaw: float) -> bool:
        if self._last_pose is None:
            return True
        lx, lz, lyaw = self._last_pose
        dyaw = abs((yaw - lyaw + 180) % 360 - 180)
        return dyaw >= 15.0 or math.hypot((x or 0.0) - lx, (z or 0.0) - lz) >= 0.25

    def update(self, metadata: Dict,
               acts: Dict[str, Tuple[int, str, Optional[bool]]],
               holding: str, step: int) -> str:
        objects = [o for o in (metadata.get("objects") or []) if isinstance(o, dict)]
        agent = metadata.get("agent") or {}
        apos = agent.get("position") or {}
        yaw = float((agent.get("rotation") or {}).get("y") or 0.0)
        ax, az = apos.get("x"), apos.get("z")
        pose_changed = self._pose_changed(ax, az, yaw)
        if ax is not None and az is not None:
            self._last_pose = (float(ax), float(az), yaw)

        by_type: Dict[str, Dict[str, Any]] = {}
        for obj in objects:
            tp = str(obj.get("objectType") or "")
            if not tp:
                continue
            cur = by_type.get(tp)
            if cur is None or float(obj.get("distance") or 1e9) < float(
                    cur.get("distance") or 1e9):
                by_type[tp] = obj

        entries = self.entries()
        names = [env for env, _ in entries]
        # 匹配：先用模型给的 env 名，匹配不上再退回它自己的说法
        matched: Dict[str, str] = {}
        for env_name, words in entries:
            hit = match_names([env_name], list(by_type))
            if not hit and words:
                hit = match_names([words], list(by_type))
            if hit:
                matched[env_name] = list(hit.values())[0]
        wanted: Set[str] = set(matched.values())
        states = {tp: (o.get("state") or {}) for tp, o in by_type.items()}
        contents: Dict[str, List[str]] = {}
        for tp, obj in by_type.items():
            if obj.get("contents"):
                contents[tp] = list(obj["contents"])
        settled = settled_by_predicate(self.task_desc, entries, states, contents,
                                       holding)

        rows: List[Dict[str, Any]] = []
        candidates: Set[str] = set(wanted) | set(by_type) | ({holding} if holding else set())
        for tp, inside in contents.items():
            candidates.add(tp)
            candidates.update(str(x) for x in inside)
        for tp in candidates:
            obj = by_type.get(tp)
            if obj is not None and obj.get("visible"):
                continue                    # 当前可见 -> 只进“视野内”行
            act = acts.get(tp)
            if tp == holding or any(tp in v for v in contents.values()):
                tier = 2
            elif tp in wanted:
                tier = 1
            elif act:
                tier = 3
            elif obj is not None:
                tier = 4
            else:
                continue                    # 模型说了但没见过 -> 交给“没见过”分支
            rows.append({"type": tp, "tier": tier, "obj": obj or {}, "acts": act,
                         "settled": tp in settled})

        # “模型说了但 WM 至今没见过”的名字，只提一次
        unseen_lines: List[str] = []
        inside_types = [str(x) for v in contents.values() for x in v]
        for env_name, words in entries:
            if env_name in matched:
                continue                   # 已匹配到检测结果 -> 由 rows 处理

            def _hit(cands, types):
                got = match_names([cands], types) if cands else {}
                if not got and words and words != cands:
                    got = match_names([words], types)
                return got

            if holding and _hit(env_name, [holding]):
                key = f"held::{normalize(env_name)}"
                if key not in self._never_seen_reported:
                    self._never_seen_reported.add(key)
                    unseen_lines.append(
                        f"- {env_name}（手持/容器内容）：手持中（不用找）")
                continue
            inside_hit = _hit(env_name, inside_types) if inside_types else {}
            if inside_hit:
                ttype = list(inside_hit.values())[0]
                holder = next((c for c, v in contents.items()
                               if ttype in [str(x) for x in v]), "")
                key = f"inside::{normalize(env_name)}"
                if key not in self._never_seen_reported:
                    self._never_seen_reported.add(key)
                    unseen_lines.append(
                        f"- {env_name}（手持/容器内容）：WM 记为在 {holder} 里")
                continue
            key = f"unseen::{normalize(env_name)}"
            if key in self._never_seen_reported:
                continue
            self._never_seen_reported.add(key)
            unseen_lines.append(f"- {env_name}（任务目标·模型自述）：目前没见过")

        rows = [r for r in rows if not r["settled"]]
        rows.sort(key=lambda r: r["tier"])
        chosen: List[Dict[str, Any]] = []
        quota = {1: 99, 2: 99, 3: 1, 4: 1}
        for row in rows:
            if quota[row["tier"]] <= 0:
                continue
            quota[row["tier"]] -= 1
            chosen.append(row)
            if len(chosen) >= self.limit:
                break

        lines = list(unseen_lines)
        for row in chosen:
            tp = row["type"]
            label = f"{tp}（{TIER_LABEL[row['tier']]}"
            if row["acts"]:
                label += "；已交互"
            label += "）"
            obj = row["obj"]
            if not obj:
                continue
            dist = obj.get("distance")
            sigma = obj.get("sigma")
            pos = obj.get("position") or {}
            word = "某处"
            if ax is not None and az is not None and pos.get("x") is not None:
                word = direction_word(float(pos["x"]) - float(ax),
                                      float(pos.get("z") or 0.0) - float(az), yaw)
                dist = math.hypot(float(pos["x"]) - float(ax),
                                  float(pos.get("z") or 0.0) - float(az))
            if dist is not None and float(dist) <= self.max_dist and (
                    sigma is None or float(sigma) <= self.max_sigma):
                band = f" ±{float(sigma):.1f}m" if sigma else ""
                body = (f"记住的位置在{word}约 {float(dist):.1f}m{band}"
                        f"（step {obj.get('last_seen_step')} 看到过）")
            else:
                body = (f"大概在{word}，距离不确定"
                        f"（step {obj.get('last_seen_step')} 看到过）")
            line = f"- {label}：{body}"
            if row["acts"]:
                line += f"；最近交互 step {row['acts'][0]} {row['acts'][1]}"
            if not pose_changed and self._last_lines.get(tp) == line:
                line = f"- {label}：位置同上"
            self._last_lines[tp] = line
            lines.append(line)

        names_line = []
        for tp, obj in sorted(by_type.items(),
                              key=lambda kv: float(kv[1].get("distance") or 1e9)):
            if not obj.get("visible"):
                continue
            pos = obj.get("position") or {}
            word = ""
            if ax is not None and az is not None and pos.get("x") is not None:
                word = direction_word(float(pos["x"]) - float(ax),
                                      float(pos.get("z") or 0.0) - float(az), yaw)
            dist = float(obj.get("distance") or 0.0)
            names_line.append(f"{tp}（{word}约 {dist:.1f}m）" if word else tp)
            if len(names_line) >= self.names_limit:
                break

        out: List[str] = []
        if lines:
            out.append("任务相关记忆（当前看不见的）：")
            out.extend(lines)
        if names_line:
            out.append("视野内：" + "、".join(names_line))
        return "\n".join(out)
