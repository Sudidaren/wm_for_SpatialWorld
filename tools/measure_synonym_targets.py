#!/usr/bin/env python3
"""Target coverage when the dictionary is expanded with WordNet, not aliases.

Run with the interpreter that has ``wn`` + the Open English WordNet added:

    /tmp/lexenv/bin/python tools/measure_synonym_targets.py

For every environment class name (e.g. ``Fridge``, ``GarbageCan``) this asks
WordNet for the class's own lemma set ("fridge" -> refrigerator, icebox) and
matches the instruction text against that expanded set.  Nothing here knows
the evaluation task's targets: the expansion comes from a public lexicon, and
the resulting class -> surface-form map is printed so it can be audited.
"""

from __future__ import annotations

import ast
import json
import pickle
import re
import statistics
from collections import Counter
from pathlib import Path

import wn

TASK_ROOT = Path("/home/sudidaren/SpatialWorld/data/ai2thor/tasks")
INDEX = Path("/home/sudidaren/lightwm_phases/data/frame_index.pkl")


def load_vocab() -> list[str]:
    with INDEX.open("rb") as fh:
        d = pickle.load(fh)
    types = d.get("object_types") or []
    if isinstance(types, dict):
        types = list(types.keys())
    return sorted({str(t) for t in types if t})


def split_words(t: str) -> list[str]:
    return [w.lower() for w in re.split(r"(?<=[a-z])(?=[A-Z])|[\s_-]+", t) if w]


def base_forms(t: str) -> set[str]:
    ws = split_words(t)
    out = {"".join(ws), " ".join(ws), ws[-1]} if ws else set()
    for f in list(out):
        out.add(f[:-1] if f.endswith("s") else f + "s")
    return {f for f in out if f}


def expand(w: wn.Wordnet, t: str, hops: int) -> set[str]:
    """Class name -> surface forms, via WordNet synonyms (and optional hyponyms)."""
    forms = base_forms(t)
    for surface in (" ".join(split_words(t)), split_words(t)[-1] if split_words(t) else t):
        for s in w.synsets(surface)[:2]:
            forms |= {l.replace("_", " ") for l in s.lemmas()}
            if hops:
                for h in list(s.hypernyms())[:2]:
                    forms |= {l.replace("_", " ") for l in h.lemmas()}
                if surface in ("vegetable", "veg"):
                    for h in list(s.hyponyms())[:60]:
                        forms |= {l.replace("_", " ") for l in h.lemmas()}
    for f in list(forms):
        forms.add(f[:-1] if f.endswith("s") else f + "s")
    return {f for f in forms if f}


def expand_components(w: wn.Wordnet, t: str, hops: int = 2) -> set[str]:
    """Same as expand, plus every component word of every form found.

    The compounds AI2-THOR uses (LightSwitch, CounterTop, DeskLamp) have no
    WordNet entry at all, so their only bridge to instruction wording is the
    head/modifier words -- which is exactly what makes them ambiguous, and why
    the caller has to look at precision as well as coverage.
    """
    forms = expand(w, t, hops)
    words = set(forms)
    for f in forms:
        words |= set(split_words(f))
    return {f for f in words if f}


def main() -> int:
    w = wn.Wordnet("oewn:2022")
    vocab = load_vocab()
    print(f"classes: {len(vocab)}  (perception collection vocabulary)")

    expanded = {t: {"strict": base_forms(t), "syn": expand(w, t, 0),
                    "syn+hypo": expand(w, t, 1),
                    "comp": expand_components(w, t)} for t in vocab}

    rows = []
    for p in sorted(TASK_ROOT.glob("*/task.json")):
        d = json.loads(p.read_text())
        raw = d.get("target_object_types")
        gt = {str(x).lower() for x in (ast.literal_eval(raw) if isinstance(raw, str) else (raw or []))}
        if not gt:
            continue
        instr = str(d.get("instruction") or "")
        text = " " + re.sub(r"[^a-zA-Z ]", " ", instr).lower() + " "
        words = set(text.split())
        hits = {}
        for mode in ("strict", "syn", "syn+hypo", "comp"):
            found = {t.lower() for t in vocab
                     if any((" " + f + " ") in text if " " in f else f in words
                            for f in expanded[t][mode])}
            hits[mode] = gt & found
        rows.append({"tid": p.parent.name, "gt": gt, "instr": instr, **hits})

    n = len(rows)
    print(f"tasks: {n}")
    for mode in ("strict", "syn", "syn+hypo", "comp"):
        full = sum(1 for r in rows if len(r[mode]) == len(r["gt"]))
        part = sum(1 for r in rows if r[mode])
        rec = statistics.mean(len(r[mode]) / len(r["gt"]) for r in rows)
        print(f"  [{mode:9s}] all targets hit {full}/{n} = {full/n:.1%} | "
              f"at least one {part}/{n} = {part/n:.1%} | target recall {rec:.1%}")

    # Ambiguity cost: with component-word expansion a single word like "table"
    # lights up several classes, so a hit is only useful if the WM can then
    # disambiguate with what it actually sees.
    print("\n  ambiguity (matched classes per task, and share of matches that are ground truth):")
    for mode in ("strict", "comp"):
        matched, gt_share = [], []
        for r in rows:
            text = " " + re.sub(r"[^a-zA-Z ]", " ", r["instr"]).lower() + " "
            words = set(text.split())
            found = {t.lower() for t in vocab
                     if any((" " + f + " ") in text if " " in f else f in words
                            for f in expanded[t][mode])}
            matched.append(len(found))
            if found:
                gt_share.append(len(found & r["gt"]) / len(found))
        print(f"    [{mode:9s}] 平均命中类数 {statistics.mean(matched):.2f} | "
              f"命中里属于真值的比例 {statistics.mean(gt_share):.0%}" if gt_share else "")

    print("\n  auto-derived surface forms worth eyeballing:")
    for t in ("Fridge", "GarbageCan", "Television", "CellPhone", "SideTable",
              "LightSwitch", "CounterTop", "Mug", "Bowl"):
        extra = sorted(expanded[t]["syn"] - expanded[t]["strict"])
        print(f"    {t:12s} <- {extra[:10]}")

    missed = Counter()
    for r in rows:
        for g in r["gt"] - r["comp"]:
            missed[g] += 1
    print("\n  still missed (top 10):")
    for name, k in missed.most_common(10):
        print(f"    {k:4d}  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
