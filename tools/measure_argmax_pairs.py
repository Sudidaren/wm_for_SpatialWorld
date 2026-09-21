#!/usr/bin/env python3
"""The user's matching rule, measured: direct first, then argmax + threshold.

Rule:
  1. a detected object whose name appears directly in the instruction is
     task-relevant;
  2. for every instruction noun that has no direct counterpart, score it
     against the detected objects that are not yet relevant, take the single
     best match, and accept it only if that best score clears a threshold.

This sweeps the threshold and reports, per frame: how much of the task's real
target set gets paired (recall) and how much of what we pair is noise
(precision).  Run with the ``wn`` interpreter:

    /tmp/lexenv/bin/python tools/measure_argmax_pairs.py
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

from measure_wup_matching import STOP, Sim, split_words

TASK_ROOT = Path("/home/sudidaren/SpatialWorld/data/ai2thor/tasks")
INDEX = Path("/home/sudidaren/lightwm_phases/data/frame_index.pkl")
COMPONENT_SCORE = 0.80   # score given to an exact component-word hit (lamp~DeskLamp)


def direct_tokens(cls: str) -> set[str]:
    ws = split_words(cls)
    return {"".join(ws), " ".join(ws)} | {w + "s" for w in ws} | {w for w in ws}


def component_words(cls: str) -> set[str]:
    ws = split_words(cls)
    return set(ws) | {w + "s" for w in ws}


def pair_frame(sim: Sim, instruction_words: set[str], detected: set[str],
               threshold: float, use_components: bool = True):
    detected = {d.lower() for d in detected}
    relevant: dict[str, str] = {}          # class -> how it was matched
    unmatched_words = set()
    for word in instruction_words:
        hit = None
        for cls in detected:
            if word in direct_tokens(cls):
                hit = cls
                break
        if hit:
            relevant[hit] = "direct"
        else:
            unmatched_words.add(word)

    pool = detected - set(relevant)
    paired = []
    for word in sorted(unmatched_words):
        best, best_score = None, 0.0
        for cls in pool:
            score = sim.wup(word, " ".join(split_words(cls)))
            if use_components and word in component_words(cls):
                score = max(score, COMPONENT_SCORE)
            if score > best_score:
                best, best_score = cls, score
        if best is not None and best_score >= threshold:
            relevant[best] = f"sim:{best_score:.2f}"
            paired.append((word, best, best_score))
            pool.discard(best)
    return relevant, paired


def main() -> int:
    w = wn.Wordnet("oewn:2022")
    sim = Sim(w)
    with INDEX.open("rb") as fh:
        idx = pickle.load(fh)
    frames: dict[str, list[set[str]]] = {}
    for f in idx["frames"]:
        sc = f.get("scene")
        if sc:
            frames.setdefault(sc, []).append(
                {str(v["type"]).lower() for v in (f.get("visible") or []) if v.get("type")})

    random.seed(11)
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
        data.append((gt, words, [random.choice(fr) for _ in range(25)]))

    print(f"tasks {len(data)}, 25 frames each")
    print(f"{'thr':>5} {'recall(target seen)':>19} {'pairs/frame':>11} "
          f"{'noise/frame':>11} {'precision':>9}")
    for thr in (0.0, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95):
        rec, pairs, noise, prec = [], [], [], []
        for gt, words, sample in data:
            r, npair, nnoise, pp = [], [], [], []
            for vis in sample:
                rel, _ = pair_frame(sim, words, vis, thr)
                got = {c for c in rel}
                if gt & vis:
                    r.append(1.0 if (gt & got) else 0.0)
                npair.append(len(got - gt))
                if got:
                    pp.append(len(got & gt) / len(got))
            if r:
                rec.append(statistics.mean(r))
            pairs.append(statistics.mean(npair) + statistics.mean(
                [len({c for c in pair_frame(sim, words, v, thr)[0]} & gt) for v in sample]))
            noise.append(statistics.mean(npair))
            if pp:
                prec.append(statistics.mean(pp))
        print(f"{thr:5.2f} {statistics.mean(rec):18.1%} {statistics.mean(pairs):11.2f} "
              f"{statistics.mean(noise):11.2f} {statistics.mean(prec):9.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
