#!/usr/bin/env python3
"""Threshold sweep for semantic matching against what is actually in view.

The user's proposal: parse the instruction, then match it against the objects
the perception head reports *at that moment*, instead of forcing every word
into a global class vocabulary.  Matching everything related maximises recall
but floods the hint; this measures the precision/recall trade-off as the
similarity threshold moves.

Similarity is Wu-Palmer over WordNet hypernym chains (this ``wn`` build has no
shortest_path_distance), computed between each instruction content word and
each object type visible in a sampled frame of the task's scene.

    /tmp/lexenv/bin/python tools/measure_wup_matching.py
"""

from __future__ import annotations

import ast
import json
import pickle
import random
import re
import statistics
from pathlib import Path

import wn

TASK_ROOT = Path("/home/sudidaren/SpatialWorld/data/ai2thor/tasks")
INDEX = Path("/home/sudidaren/lightwm_phases/data/frame_index.pkl")
STOP = set("""a an the i you he she it we they me my your his her its our their
of to in on at for with and or but if then so please now want need would like
is are was were be been being do does did done have has had can could should
will just help how when where what which that this these those there here
up out into from by as only all any some more most other another well very
really because while after before again also too find take get put make turn
bring give look go going gone see seen come""".split())


def split_words(t: str) -> list[str]:
    return [w.lower() for w in re.split(r"(?<=[a-z])(?=[A-Z])|[\s_-]+", t) if w]


class Sim:
    def __init__(self, w: wn.Wordnet):
        self.w = w
        self._path: dict[str, list] = {}
        self._syn: dict[str, list] = {}
        self._pair: dict[tuple[str, str], float] = {}

    def synsets(self, word: str) -> list:
        # WordNet lookups hit SQLite every time; the sweep makes hundreds of
        # thousands of them, so both the synset lists and the pair scores are
        # cached (uncached this ran 15+ minutes without finishing).
        if word not in self._syn:
            self._syn[word] = self.w.synsets(word)[:3]
        return self._syn[word]

    def path(self, s) -> list:
        key = s.id
        if key not in self._path:
            chain, cur, seen = [], s, set()
            while cur is not None and cur.id not in seen:
                chain.append(cur)
                seen.add(cur.id)
                parents = list(cur.hypernyms())
                cur = parents[0] if parents else None
            self._path[key] = chain
        return self._path[key]

    def wup(self, a: str, b: str) -> float:
        key = (a, b)
        if key in self._pair:
            return self._pair[key]
        best = 0.0
        for sa in self.synsets(a):
            pa = self.path(sa)
            for sb in self.synsets(b):
                pb = self.path(sb)
                ids = {s.id for s in pb}
                common = [s for s in pa if s.id in ids]
                if not common:
                    continue
                lch = max(common, key=lambda s: len(self.path(s)))
                depth = len(self.path(lch))
                d = len(pa) + len(pb)
                if d:
                    best = max(best, 2.0 * depth / d)
        self._pair[key] = best
        return best


def main() -> int:
    w = wn.Wordnet("oewn:2022")
    sim = Sim(w)
    print("Wu-Palmer 抽查:")
    for a, b in (("tomato", "vegetable"), ("lettuce", "vegetable"), ("mug", "cup"),
                 ("fridge", "refrigerator"), ("lamp", "desk lamp"),
                 ("counter", "countertop"), ("egg", "vegetable"), ("apple", "fridge")):
        print(f"   {a:10s} ~ {b:14s} {sim.wup(a, b):.2f}")

    with INDEX.open("rb") as fh:
        idx = pickle.load(fh)
    frames: dict[str, list[set[str]]] = {}
    for f in idx["frames"]:
        sc = f.get("scene")
        if sc:
            frames.setdefault(sc, []).append(
                {str(v["type"]).lower() for v in (f.get("visible") or []) if v.get("type")})

    random.seed(7)
    data = []
    for p in sorted(TASK_ROOT.glob("*/task.json")):
        t = json.loads(p.read_text())
        raw = t.get("target_object_types")
        gt = {str(x).lower() for x in (ast.literal_eval(raw) if isinstance(raw, str) else (raw or []))}
        if not gt:
            continue
        fr = frames.get(str(t.get("scene") or "")) or []
        if not fr:
            continue
        words = {x for x in re.sub(r"[^a-z ]", " ", str(t.get("instruction") or "").lower()).split()
                 if x not in STOP and len(x) > 2}
        sample = [random.choice(fr) for _ in range(25)]
        data.append((gt, words, sample))

    print(f"\n任务数 {len(data)}（每任务抽 25 帧）")
    print(f"{'阈值':>6} {'目标可见时召回':>14} {'每帧候选':>9} {'候选真值占比':>12}")
    for thr in (0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0):
        rec, cand, prec = [], [], []
        for gt, words, sample in data:
            r, c, pp = [], [], []
            for vis in sample:
                found = {cl for cl in vis if any(sim.wup(x, cl) >= thr for x in words)}
                if gt & vis:
                    r.append(1.0 if (gt & found) else 0.0)
                c.append(len(found))
                if found:
                    pp.append(len(found & gt) / len(found))
            if r:
                rec.append(statistics.mean(r))
            cand.append(statistics.mean(c))
            if pp:
                prec.append(statistics.mean(pp))
        print(f"{thr:6.2f} {statistics.mean(rec):13.1%} {statistics.mean(cand):9.2f} "
              f"{statistics.mean(prec):11.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
