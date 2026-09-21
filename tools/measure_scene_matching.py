#!/usr/bin/env python3
"""Match the instruction against the objects this scene actually contains.

The alias problem came from forcing an instruction word into a global
125-class vocabulary.  This measures the alternative: parse the instruction
into content words, then match those words against the object types that
actually appear in *this* scene (taken from the perception index, i.e. the
detector's own observations of that scene), using WordNet for the
synonym/hypernym hops.

Run with the interpreter that has ``wn`` + Open English WordNet:

    /tmp/lexenv/bin/python tools/measure_scene_matching.py

Reports coverage against the task's own target list, plus the ambiguity cost
(how many candidates a task yields and what share of them are real targets),
because a matcher that lights up half the room is not a matcher.
"""

from __future__ import annotations

import ast
import json
import pickle
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import wn

TASK_ROOT = Path("/home/sudidaren/SpatialWorld/data/ai2thor/tasks")
INDEX = Path("/home/sudidaren/lightwm_phases/data/frame_index.pkl")
STOP = set("""a an the i you he she it we they me my your his her its our their
of to in on at for with and or but if then so please now want need would like
is are was were be been being do does did done have has had can could should
will just help me how when where what which that this these those there here
it's i'm i'd i'll please up out into from by as only all any some more most
other another well very really because while after before again also too
find take get put make turn bring give look go going gone see seen come want
there's don't doesn't didn't isn't aren't can't won't wouldn't shouldn't""".split())


def split_words(t: str) -> list[str]:
    return [w.lower() for w in re.split(r"(?<=[a-z])(?=[A-Z])|[\s_-]+", t) if w]


def scene_types() -> dict[str, set[str]]:
    with INDEX.open("rb") as fh:
        d = pickle.load(fh)
    out: dict[str, set[str]] = defaultdict(set)
    for f in d["frames"]:
        sc = f.get("scene")
        if not sc:
            continue
        for v in f.get("visible") or []:
            t = v.get("type")
            if t:
                out[sc].add(str(t))
    return out


def related(word: str, cls: str, w: wn.Wordnet, hops: int = 2) -> bool:
    """Is instruction word ``word`` plausibly naming object type ``cls``?"""
    cw = split_words(cls)
    if not cw:
        return False
    # obvious surface forms of the class name
    if word in set(cw) or word == "".join(cw) or word == " ".join(cw):
        return True
    # synonyms / hypernyms of the class, and of the word, up to `hops`
    frontier = [((" ".join(cw)), 0), (cw[-1], 0), (word, 0)]
    seen = set()
    while frontier:
        term, depth = frontier.pop(0)
        if not term or depth > hops or term in seen:
            continue
        seen.add(term)
        for s in w.synsets(term)[:2]:
            lemmas = {l.replace("_", " ") for l in s.lemmas()}
            if word in lemmas or " ".join(cw) in lemmas or cw[-1] in lemmas:
                return True
            for other in (s.hypernyms(), s.hyponyms()):
                for h in list(other)[:8]:
                    for l in h.lemmas():
                        if word == l.replace("_", " "):
                            return True
                    frontier.append((h.lemmas()[0].replace("_", " "), depth + 1))
    return False


def main() -> int:
    w = wn.Wordnet("oewn:2022")
    stypes = scene_types()
    print(f"scenes in index: {len(stypes)}")

    rows = []
    for p in sorted(TASK_ROOT.glob("*/task.json")):
        d = json.loads(p.read_text())
        raw = d.get("target_object_types")
        gt = {str(x).lower() for x in (ast.literal_eval(raw) if isinstance(raw, str) else (raw or []))}
        if not gt:
            continue
        scene = str(d.get("scene") or "")
        cands = stypes.get(scene) or set()
        instr = str(d.get("instruction") or "")
        words = {x for x in re.sub(r"[^a-z ]", " ", instr.lower()).split()
                 if x not in STOP and len(x) > 2}
        found = set()
        for cls in cands:
            if any(related(word, cls, w) for word in words):
                found.add(cls.lower())
        rows.append({"tid": p.parent.name, "scene": scene, "gt": gt,
                     "cand": {c.lower() for c in cands}, "found": found})

    n = len(rows)
    no_scene = sum(1 for r in rows if not r["cand"])
    full = sum(1 for r in rows if r["gt"] <= r["found"])
    part = sum(1 for r in rows if r["gt"] & r["found"])
    rec = statistics.mean(len(r["gt"] & r["found"]) / len(r["gt"]) for r in rows)
    amb = statistics.mean(len(r["found"]) for r in rows)
    prec = [len(r["found"] & r["gt"]) / len(r["found"]) for r in rows if r["found"]]
    print(f"tasks: {n}  (scene unknown for {no_scene})")
    print(f"  all targets hit {full}/{n} = {full/n:.1%} | at least one {part}/{n} = {part/n:.1%}"
          f" | target recall {rec:.1%}")
    print(f"  candidates per task {amb:.2f} | share of candidates that are real targets "
          f"{statistics.mean(prec):.0%}" if prec else "")
    missed = Counter(g for r in rows for g in r["gt"] - r["found"])
    print("  still missed (top 10):", ", ".join(f"{k}:{v}" for k, v in missed.most_common(10)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
