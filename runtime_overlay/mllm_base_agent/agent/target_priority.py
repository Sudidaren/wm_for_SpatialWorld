"""Priority-ranked target hint, built by matching the instruction to detections.

2026-09-19: the one-shot VLM extraction was replaced by a deterministic
*object-to-object* match, because it costs one API call per task and, measured
on our own runs, covered fewer targets than plain string matching did.

The rule now is:

  1. take the instruction's own words (1-, 2- and 3-grams, singular and
     plural);
  2. a detected object is task-relevant iff its own name -- the whole
     CamelCase name, spaced or joined, singular or plural -- appears in the
     instruction.  No object vocabulary, no alias table, no word similarity,
     no category matching: object to object only;
  3. once an object has been recognised it stays relevant for the rest of the
     episode, even after it leaves the view (that is what makes the memory
     useful);
  4. everything else (tiers, quotas, "only memory", pose-triggered refresh,
     distance/sigma caps, predicate demotion) is unchanged.

An object the agent has already acted on is *not* relevant by that fact alone:
the agent knows what it just did, and promoting a chance interaction to "task
target" would keep re-reporting whatever it happened to open and can even tell
it to walk back to it.  ``LIGHTWM_TARGET_FROM_ACTION=1`` turns that channel
back on as an ablation (measured: +0.22 correct and +0.07 wrong targets per
task, rescuing 20 of the 28 tasks whose instruction names no detected object);
it is off in the released configuration.

``LIGHTWM_TARGET_MODE=llm`` restores the previous behaviour (ask the VLM to
name the objects, then substring-match its answer); it is kept only so the old
arm can still be reproduced.  The default is ``exact``.

The only fixed string left in this file is one *verb* pattern
(``_DEST_VERB``, used by the settled-predicate test to read
"put A into B" out of the instruction).  It contains no object names.
"""

from __future__ import annotations

import json
import math
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

try:
    from . import state_variants
except ImportError:                      # script / flat import fallback
    import state_variants

_DIRECTIONS = ("正前方", "右前方", "正右方", "右后方",
               "正后方", "左后方", "正左方", "左前方")

#: destination verb, for the "put A into B" settled test (no object names)
_DEST_VERB = r"(?:put|place|move|set|leave|bring|carry|throw|toss|transfer)"

#: Every interaction action the environment accepts.  Used by "whatever the
#: agent acts on is relevant by construction"; navigation is not in here.
#: These are action *verbs* -- they appear with this exact spelling in the
#: system prompt every arm receives -- so they are API vocabulary, not a
#: table of object names.
_INTERACTION_ACTIONS = frozenset({
    "OpenObject", "CloseObject", "ToggleObjectOn", "ToggleObjectOff",
    "PickupObject", "SliceObject", "BreakObject", "CookObject",
    "DirtyObject", "CleanObject", "FillObjectWithLiquid",
    "EmptyLiquidFromObject", "UseUpObject",
})
#: A ``PutObject(X)`` names the *destination*, not the object the agent
#: decided to act on: the object itself was already counted at pickup, and the
#: destination is a place the agent is standing in front of.  Measured on the
#: 311 AI2-THOR traces, dropping it removes 0.02 wrong targets per task and
#: costs 0.01 right ones.  (``DropHandObject``/``ThrowObject`` are not in the
#: list above either, so they add nothing.)
_DESTINATION_ONLY_ACTIONS = frozenset({"PutObject"})

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


def _singulars(tok: str) -> Set[str]:
    """Cheap English singulariser (tomatoes -> tomato, knives -> knife...)."""
    out = {tok}
    for suf, rep in (("ies", "y"), ("ves", "f"), ("ves", "fe"), ("es", ""), ("s", "")):
        if tok.endswith(suf) and len(tok) > len(suf) + 1:
            out.add(tok[: -len(suf)] + rep)
    return out


def instruction_grams(text: str) -> Set[str]:
    """The instruction's own words and short phrases, singular and plural."""
    toks = re.sub(r"[^a-z0-9 ]", " ", (text or "").lower()).split()
    grams: Set[str] = set(toks)
    grams |= {" ".join(toks[i:i + 2]) for i in range(len(toks) - 1)}
    grams |= {" ".join(toks[i:i + 3]) for i in range(len(toks) - 2)}
    grams |= {s for g in list(grams) for s in _singulars(g)}
    return grams


def type_name_forms(cls: str) -> Set[str]:
    """The spellings of one object type: whole name only, singular and plural."""
    words = [w.lower() for w in re.split(r"(?<=[a-z])(?=[A-Z])|[\s_-]+", cls) if w]
    if not words:
        return set()
    out = {" ".join(words), "".join(words)}
    out |= {f + "s" for f in list(out)}
    return out


def relevant_types(text: str, detected: Iterable[str]) -> Set[str]:
    """Detected types whose own name appears in the instruction text."""
    grams = instruction_grams(text)
    return {str(t) for t in detected if type_name_forms(str(t)) & grams}


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
                         contents: Dict[str, List[str]]) -> Set[str]:
    """Detected types whose task predicate already holds, per WM's own ledger.

    Objects are addressed by the model's own names (matched to detected types
    by :func:`match_names`), so no vocabulary is needed.  A held object is
    deliberately never marked settled: the predicate tests below only look at
    the ledger's state fields and container contents.
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
    return done


class TargetHinter:
    """Per-step builder for the ranked target block (stateful across steps)."""

    def __init__(self, task_description: str, vlm: Any = None,
                 limit: int = 6, names_limit: int = 5,
                 max_dist: float = 3.0, max_sigma: float = 0.5,
                 memory_frames: Optional[int] = None,
                 mode: Optional[str] = None,
                 extra_slots: Optional[int] = None,
                 visible_all: Optional[bool] = None) -> None:
        self.task_desc = task_description or ""
        self.vlm = vlm
        #: "exact" (default): the instruction's own words are matched against
        #: the detected type names, object to object.  "llm": the previous
        #: behaviour, one extraction call per task.  Kept for reproducing the
        #: earlier arm; the default is the deterministic one.
        self.mode = (mode or os.environ.get("LIGHTWM_TARGET_MODE", "exact")).lower()
        #: 2026-09-19：块里只留任务相关物体。旧版每步还会各塞 1 个"已交互"和
        #: 1 个"只是见过"的物体（tier 3/4），那既占 token 又与"只提示相关物"
        #: 的规则冲突；要用回旧行为就设 LIGHTWM_HINT_EXTRA_SLOTS=1。
        self.extra_slots = int(extra_slots if extra_slots is not None
                               else os.environ.get("LIGHTWM_HINT_EXTRA_SLOTS", "0") or 0)
        #: "视野内"那一行默认也只列任务相关物体（其余的模型自己看得见）。
        #: LIGHTWM_HINT_VISIBLE_ALL=1 恢复旧行为。
        self.visible_all = bool(visible_all if visible_all is not None
                                else os.environ.get("LIGHTWM_HINT_VISIBLE_ALL", "0") == "1")
        #: 2026-09-19：模型已经对某个物体下过手（PickupObject / SliceObject /
        #: OpenObject ...），那它算不算任务相关物体——**默认不算**。
        #: 理由：那是模型自己刚做的动作，它自己知道；而把"我碰过的东西"升格成
        #: 任务目标，一旦它只是随手开了个柜子，我们就会一直报那个柜子、还可能
        #: 主动建议它走回去，等于把模型的误操作固化成目标。
        #: 代码留着用于消融（LIGHTWM_TARGET_FROM_ACTION=1 打开，离线实测：
        #: 每任务 +0.22 正确 / +0.07 错误目标，救回 28 个零注入任务里的 20 个），
        #: 但发布口径是关。
        self.from_action = os.environ.get("LIGHTWM_TARGET_FROM_ACTION", "0") == "1"
        #: Types recognised so far.  Once an object is task-relevant it stays
        #: relevant -- that persistence is the whole point of the memory.
        self._relevant: Set[str] = set()
        self.limit = int(limit)
        self.names_limit = int(names_limit)
        self.max_dist = float(max_dist)
        self.max_sigma = float(max_sigma)
        #: Ablation switch: how many *steps* back an object may have been last
        #: seen and still be reported.  ``None`` = the world model's memory as
        #: shipped (anything it ever anchored); ``0`` = only objects visible in
        #: the current frame, i.e. the same perception stack with the memory
        #: removed.  That arm is the one that decides whether the WM's gain
        #: comes from remembering or merely from seeing.
        self.memory_frames = memory_frames
        self._entries: Optional[List[Tuple[str, str]]] = None
        self._never_seen_reported: Set[str] = set()
        self._last_pose: Optional[Tuple[float, float, float]] = None
        #: last step's distance to the nearest not-visible relevant object, so
        #: the advice line can say "you are moving away" when that is true.
        #: 上一次给出"建议先转向它再靠近"时的 (物体类型, 距离)。
        #: 必须带上物体类型：只存距离的话，两次建议指向不同物体时就会拿
        #: A 的距离去和 B 的距离比，然后说"你在远离它"。实测 616 次
        #: "你在远离它" 里有 167 次（27%）是这种跨物体比较。
        self._last_gap: Optional[Tuple[str, float]] = None

    # -- the model's own object list ---------------------------------
    def entries(self) -> List[Tuple[str, str]]:
        """The model's own object list (only used by ``mode='llm'``)."""
        if self._entries is not None:
            return self._entries
        self._entries = self._ask_model()
        return self._entries

    def relevant(self, detected: Iterable[str]) -> Set[str]:
        """Which detected objects does this instruction name?  Accumulates."""
        if self.mode == "llm":
            by_type = {str(t): str(t) for t in detected}
            for env_name, words in self.entries():
                hit = match_names([env_name], list(by_type))
                if not hit and words:
                    hit = match_names([words], list(by_type))
                if hit:
                    self._relevant.add(list(hit.values())[0])
        else:
            self._relevant |= relevant_types(self.task_desc, detected)
        return set(self._relevant)

    def acted_on(self, detected: Iterable[str],
                 acts: Dict[str, Tuple[int, str, Optional[bool]]]) -> Set[str]:
        """Detected types the agent has already issued an interaction on.

        Off in the released configuration (see ``from_action``); this is the
        ablation channel.  When it is on, the agent's own action stream decides
        which object the task is about: "slice all the vegetables" names no
        object at all, and the agent still reaches for the Lettuce.  Names are
        grounded against the slots the world model actually perceived (through
        the prompt's own rename rule, so ``PickupObject(LettuceSliced)``
        attributes to the ``Lettuce`` slot), so an invented name cannot create
        a hint line out of nothing.
        """
        out: Set[str] = set()
        if not self.from_action:
            return out
        have = {str(t) for t in detected}
        for target, value in (acts or {}).items():
            action = str(value[1]) if len(value) > 1 else ""
            if action in _DESTINATION_ONLY_ACTIONS:
                continue
            if action not in _INTERACTION_ACTIONS:
                continue
            name = str(target)
            if name in have:
                out.add(name)
                continue
            base = state_variants.base_of(name)
            if base in have:
                out.add(base)
        return out

    def nearest(self, wm_metadata: Dict, visible: Optional[bool] = None
                ) -> Optional[Tuple[str, float, str]]:
        """(type, distance, direction word) of the nearest relevant object.

        ``visible`` filters to objects the current frame shows (True) or to
        objects the memory has but the frame does not (False).
        """
        agent = wm_metadata.get("agent") or {}
        apos = agent.get("position") or {}
        yaw = float((agent.get("rotation") or {}).get("y") or 0.0)
        ax, az = apos.get("x"), apos.get("z")
        best: Optional[Tuple[str, float, str]] = None
        for obj in (wm_metadata.get("objects") or []):
            tp = str(obj.get("objectType") or "")
            if tp not in self._relevant:
                continue
            if visible is not None and bool(obj.get("visible")) != visible:
                continue
            dist = obj.get("distance")
            word = "某处"
            pos = obj.get("position") or {}
            if ax is not None and az is not None and pos.get("x") is not None:
                dx = float(pos["x"]) - float(ax)
                dz = float(pos.get("z") or 0.0) - float(az)
                dist = math.hypot(dx, dz)
                word = direction_word(dx, dz, yaw)
            if dist is None:
                continue
            dist = float(dist)
            if best is None or dist < best[1]:
                best = (tp, dist, word)
        return best

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

    def update(self, wm_metadata: Dict,
               acts: Dict[str, Tuple[int, str, Optional[bool]]],
               holding: str, step: int,
               suppress_nav: bool = False) -> str:
        """``suppress_nav``：这一步不要给"往哪走"的建议。

        被挡住的那一步要用它：脱困指令和"建议先转向它再靠近"是两条互相打架的
        移动指令（实测 530 次被挡提示里有 193 次同时挂了这条，占 36%）。压掉
        的只是**这一步**的导航建议，记忆行里的位置和距离照常给。
        """
        objects = [o for o in (wm_metadata.get("objects") or []) if isinstance(o, dict)]
        if self.memory_frames is not None:
            # Ablation: keep only what the current view supports.  memory_frames
            # == 0 means "visible right now"; N > 0 also keeps anything seen in
            # the last N steps.  The perception, the anchors and the rendering
            # are untouched -- only the age of the evidence changes.
            horizon = int(self.memory_frames)
            objects = [o for o in objects
                       if o.get("visible")
                       or (horizon > 0
                           and step - int(o.get("last_seen_step") or 0) <= horizon)]
        agent = wm_metadata.get("agent") or {}
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

        # 任务相关物体：exact 模式下就是"指令里点了名、而且被检测到"的那些，
        # 识别过一次就永久保留（规则 3）。
        relevance = self.relevant(list(by_type))
        # 模型自己下过手的物体也算相关（见 acted_on 的说明）。永久保留。
        self._relevant |= self.acted_on(list(by_type), acts)
        relevance |= self._relevant
        wanted: Set[str] = {t for t in relevance if t in by_type}
        entries: List[Tuple[str, str]] = (self.entries() if self.mode == "llm"
                                          else [(t, t) for t in sorted(wanted)])
        # exact 模式下每个 entry 都已经落在检测结果里，下面的“说了但没见过”
        # 分支因此自然为空；llm 模式下保留原来的 matched 语义。
        matched: Dict[str, str] = ({t: t for t in wanted} if self.mode != "llm"
                                   else {})
        if self.mode == "llm":
            for env_name, words in entries:
                hit = match_names([env_name], list(by_type))
                if not hit and words:
                    hit = match_names([words], list(by_type))
                if hit:
                    matched[env_name] = list(hit.values())[0]
        states = {tp: (o.get("state") or {}) for tp, o in by_type.items()}
        contents: Dict[str, List[str]] = {}
        for tp, obj in by_type.items():
            if obj.get("contents"):
                contents[tp] = list(obj["contents"])
        settled = settled_by_predicate(self.task_desc, entries, states, contents)

        rows: List[Dict[str, Any]] = []
        rows_gap: List[Tuple[float, str, str]] = []
        candidates: Set[str] = set(wanted) | set(by_type) | ({holding} if holding else set())
        for tp, inside in contents.items():
            candidates.add(tp)
            candidates.update(str(x) for x in inside)
        for tp in candidates:
            obj = by_type.get(tp)
            if tp == holding:
                # 手里拿着的东西不需要"记住的位置"——「手持：X」那行已经说了，
                # 再报一次位置等于让模型去找自己手上的物体。
                continue
            if obj is not None and obj.get("visible"):
                continue                    # 当前可见 -> 只进“视野内”行
            act = acts.get(tp)
            if any(tp in v for v in contents.values()):
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
        quota = {1: 99, 2: 99, 3: self.extra_slots, 4: self.extra_slots}
        for row in rows:
            if quota[row["tier"]] <= 0:
                continue
            quota[row["tier"]] -= 1
            chosen.append(row)
            if len(chosen) >= self.limit:
                break

        lines = list(unseen_lines)
        nav_best: Optional[Tuple[float, int]] = None      # (距离, 行号)
        for row in chosen:
            tp = row["type"]
            obj = row["obj"]
            if not obj:
                continue
            # 报给模型的名字：环境改名后（切/打碎）必须用新名字，否则它下一
            # 步的输出会指向一个已经不存在的类型名。匹配仍用 objectType。
            shown = str(obj.get("objectTypeDisplay") or tp)
            label = f"{shown}（{TIER_LABEL[row['tier']]}"
            if row["acts"]:
                label += "；已交互"
            word = state_variants.state_word(obj.get("state"))
            if word:
                label += f"；{word}"
            label += "）"
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
            lines.append(line)
            if row["tier"] == 1 and dist is not None:
                rows_gap.append((float(dist), tp, word))
                if nav_best is None or float(dist) < nav_best[0]:
                    nav_best = (float(dist), len(lines) - 1, tp)

        # ② 导航建议：贴在"看不见的那个最近的任务目标"那一行的末尾。
        #    不另起一行——同一段文字里说两遍位置只会给模型添负担。
        #    候选来自上面的 rows：已经可见的、以及谓词已满足的（比如已经关掉
        #    的那盏灯）都不在内，否则会指使模型再跑一趟。
        if (os.environ.get("LIGHTWM_NAV_HINT", "1") != "0"
                and not suppress_nav):
            if nav_best is not None:
                dist, idx, tp_now = nav_best
                tail = ""
                # 只有**上一次建议的也是同一个物体**时，两次距离才可比。
                if (self._last_gap is not None
                        and self._last_gap[0] == tp_now
                        and dist > self._last_gap[1] + 0.05):
                    tail = (f"（注意：你在远离它，上一步 "
                            f"{self._last_gap[1]:.1f}m → 现在 {dist:.1f}m）")
                lines[idx] += f"——建议先转向它再靠近{tail}"
                self._last_gap = (tp_now, dist)
            else:
                self._last_gap = None

        names_line = []
        for tp, obj in sorted(by_type.items(),
                              key=lambda kv: float(kv[1].get("distance") or 1e9)):
            if holding and tp == holding:
                # 拿在手上的东西不报位置：「手持：X」那行已经说了，再报一次
                # "X（正前方约 0.3m）"等于让模型去找自己手里那个东西，而且
                # 手上物体的距离本来就测不准（贴着相机）。
                continue
            if not obj.get("visible"):
                continue
            if not self.visible_all and tp not in wanted:
                continue                    # 只报任务相关物体
            shown = str(obj.get("objectTypeDisplay") or tp)
            pos = obj.get("position") or {}
            word = ""
            if ax is not None and az is not None and pos.get("x") is not None:
                word = direction_word(float(pos["x"]) - float(ax),
                                      float(pos.get("z") or 0.0) - float(az), yaw)
            dist = float(obj.get("distance") or 0.0)
            names_line.append(f"{shown}（{word}约 {dist:.1f}m）" if word else shown)
            if len(names_line) >= self.names_limit:
                break

        out: List[str] = []
        if lines:
            out.append("任务相关记忆（当前看不见的）：")
            out.extend(lines)
        if names_line:
            out.append("视野内：" + "、".join(names_line))
        return "\n".join(out)
