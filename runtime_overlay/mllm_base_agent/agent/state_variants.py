"""State-variant names, read out of the prompt the agent is already given.

The perception head labels pixels with the environment's *base* types (its
117-class list has ``Lettuce`` but no ``LettuceSliced``).  The environment,
however, renames an object once it is sliced or cracked, and the agent then
has to act on the new name: the step log of ``ai2thor03001`` shows
``SliceObject(Lettuce)`` followed by ``PickupObject(LettuceSliced)``.

So an action's target string and the world model's own slot name disagree by
a suffix, and that disagreement silently breaks three things at once: the
hand state, the frame-diff verdict (the WM refuses to judge an action whose
target it cannot find) and the relevant-object match.

This module closes the gap **without adding vocabulary**: the production rule
is written in the system prompt that both arms receive --
``SliceObject(X) produces XSliced``, ``BreakObject(Egg) produces EggCracked``
and the explicit variant list -- so the mapping is parsed from that prompt,
not hard-coded and not looked up in an external dictionary.  See
``mllm_base_agent/prompts/ai2thor.py``, section "Object State Variants".

``LIGHTWM_STATE_VARIANTS="Base:Variant Base:Variant"`` overrides the parsed
rules (used by the unit tests, and the escape hatch if a prompt changes).
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Tuple

_PROMPT_FILES = (
    "ai2thor.py",
    "ai2thor_continuous.py",
    "procthor.py",
    "procthor_continuous.py",
)

#: ``**SliceObject(X)** produces **XSliced**`` -> ("X", "XSliced")
_PRODUCES_RE = re.compile(
    r"([A-Za-z]+)\(([A-Za-z]+)\)\*\*\s*produces\s*\*\*([A-Za-z]+)\*\*")
#: ``State variants (..): BreadSliced, TomatoSliced, ...``
_LIST_RE = re.compile(r"[Vv]ariants[^:\n]*:\s*([A-Za-z][^\n]*)")

_RULES: Optional[Tuple[Dict[str, str], Dict[str, str]]] = None


def _prompt_dir() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "prompts")


def _rule_text() -> str:
    """The text of every shipped agent prompt, concatenated."""
    base = _prompt_dir()
    chunks: List[str] = []
    for name in _PROMPT_FILES:
        path = os.path.join(base, name)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                chunks.append(fh.read())
        except OSError:
            continue
    return "\n".join(chunks)


def _parse(text: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    """-> (suffix rules, explicit pairs) as stated in the prompt."""
    suffixes: Dict[str, str] = {}
    explicit: Dict[str, str] = {}
    for verb, argument, produced in _PRODUCES_RE.findall(text):
        if (len(argument) == 1 and argument.isupper()
                and produced.startswith(argument) and len(produced) > 1):
            # The prompt writes the general rule with a placeholder -- "X"
            # produces "XSliced" -- so the suffix applies to every base type
            # the verb accepts.
            suffixes[produced[len(argument):]] = verb
        else:
            # A one-off rename, spelled with a real type name: the prompt's
            # "BreakObject(Egg) produces EggCracked (not EggSliced)".
            explicit[argument] = produced
    for listed in _LIST_RE.findall(text):
        for name in re.split(r"[,;\s]+", listed):
            name = name.strip().strip("()")
            if not name or not name[:1].isupper():
                continue
            for suffix in list(suffixes):
                if name.endswith(suffix) and len(name) > len(suffix):
                    explicit.setdefault(name[: -len(suffix)], name)
    return suffixes, explicit


def rules() -> Tuple[Dict[str, str], Dict[str, str]]:
    """(suffix -> verb, base -> variant), parsed once per process."""
    global _RULES
    if _RULES is None:
        _RULES = _parse(_rule_text())
        override = (os.environ.get("LIGHTWM_STATE_VARIANTS") or "").strip()
        if override:
            suffixes, explicit = _RULES
            for pair in override.split():
                if ":" not in pair:
                    continue
                base, variant = (x.strip() for x in pair.split(":", 1))
                if base and variant:
                    explicit[base] = variant
    return _RULES


def base_of(name: str) -> str:
    """``LettuceSliced`` -> ``Lettuce``; anything unknown is returned as is."""
    text = str(name or "")
    if not text:
        return text
    suffixes, explicit = rules()
    for base, variant in explicit.items():
        if text.lower() == variant.lower():
            return base
    for suffix in suffixes:
        if text.lower().endswith(suffix.lower()) and len(text) > len(suffix):
            return text[: -len(suffix)]
    return text


def is_variant(name: str) -> bool:
    return base_of(name) != str(name or "")


def variant_name(base: str, state: Dict) -> str:
    """The name the *agent* should use for a base type in this state.

    Only the transitions the prompt describes are applied: a sliced object is
    named ``<Base>Sliced``.  Nothing is invented for states the prompt does
    not rename (a cooked potato stays ``Potato`` as far as we can tell).
    """
    text = str(base or "")
    if not text:
        return text
    suffixes, explicit = rules()
    if state.get("isSliced") and "Sliced" in suffixes:
        return text + "Sliced"
    return text


def display_name(base: str, state: Optional[Dict] = None,
                 observed: Optional[str] = None) -> str:
    """What to print for a slot: the model's own word when we have it.

    ``observed`` is the variant the agent itself used in an action (so the
    hint repeats the agent's spelling back to it); otherwise the name is
    derived from the world model's own ledger.
    """
    if observed:
        return str(observed)
    if state:
        return variant_name(base, state)
    return str(base or "")


def state_word(state: Optional[Dict]) -> str:
    """A short Chinese word for the ledger state, for the per-step hint."""
    st = state or {}
    words = []
    if st.get("isSliced"):
        words.append("已切")
    if st.get("isCooked"):
        words.append("已煮/已加热")
    if st.get("isOpen") is True:
        words.append("已打开")
    elif st.get("isOpen") is False:
        words.append("已关上")
    if st.get("isToggled") is True:
        words.append("已开启")
    elif st.get("isToggled") is False:
        words.append("已关闭")
    if st.get("isFilledWithLiquid"):
        words.append("已装液体")
    if st.get("isDirty") is True:
        words.append("是脏的")
    elif st.get("isDirty") is False:
        words.append("已洗干净")
    if st.get("isUsedUp"):
        words.append("已用完")
    return "、".join(words)
