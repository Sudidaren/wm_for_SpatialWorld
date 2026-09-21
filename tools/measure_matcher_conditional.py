#!/usr/bin/env python3
"""Compare matchers on the frames where the target is actually in view.

The earlier version sampled random frames of the task's scene and diluted the
denominator: the target was in view in only 48% of them, and collection sweeps
are not the evaluation trajectory.  The question a matcher has to answer is
"when the target is in front of me, do I recognise it as task-relevant", so
this samples *until* the target is visible, and then measures:

  pairing recall  -- target in view -> target marked relevant
  noise           -- non-target objects marked relevant in the same frame
  precision       -- share of marked objects that are real targets

and repeats it with persistence over a 10-frame trajectory, because rule 3
says once something is recognised it stays.

    /tmp/lexenv/bin/python tools/measure_matcher_conditional.py
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

from measure_argmax_pairs import pair_frame
from measure_wup_matching import STOP, Sim, split_words

TASK_ROOT = Path("/home/sudidaren/SpatialWorld/data/ai2thor/tasks")
INDEX = Path("/home/sudidaren/lightwm_phases/data/frame_index.pkl")


def exact_forms(cls: str, components: bool) -> set[str]:
    ws = split_words(cls)
    out = {" ".join(ws), "".join(ws), " ".join(ws) + "s", "".join(ws) + "s"}
    out |= {w + "s" for w in ws}
    if components:
        out |= set(ws)
    return out


def singulars(tok: str) -> set[str]:
    """Cheap English singulariser for instruction tokens.

    Without this, "slice the tomatoes" / "wash the dishes" / "put the knives"
    never reach Tomato / Dish / Knife: the plural rules only ever *added* an
    's', they never took one off.
    """
    out = {tok}
    for suf, rep in (("ies", "y"), ("ves", "f"), ("ves", "fe"), ("es", ""), ("s", "")):
        if tok.endswith(suf) and len(tok) > len(suf) + 1:
            out.add(tok[: -len(suf)] + rep)
    return out


def main() -> int:
    sim = Sim(wn.Wordnet("oewn:2022"))
    with INDEX.open("rb") as fh:
        idx = pickle.load(fh)
    frames: dict[str, list[set[str]]] = {}
    for f in idx["frames"]:
        sc = f.get("scene")
        if sc:
            # keep the original CamelCase ("SideTable"): lowercasing here
            # destroys the word boundary and makes "side table" unmatchable --
            # that bug was inflating the "missed targets" list by ~40 cases.
            vis = {str(v["type"]) for v in (f.get("visible") or []) if v.get("type")}
            if vis:
                frames.setdefault(sc, []).append(vis)

    random.seed(5)
    data = []
    for p in sorted(TASK_ROOT.glob("*/task.json")):
        t = json.loads(p.read_text())
        raw = t.get("target_object_types")
        gt = {str(x).lower() for x in (ast.literal_eval(raw) if isinstance(raw, str) else (raw or []))}
        if not gt:
            continue
        pool = frames.get(str(t.get("scene") or "")) or []
        if not pool:
            continue
        toks = re.sub(r"[^a-z ]", " ", str(t.get("instruction") or "").lower()).split()
        grams = set(toks)
        grams |= {" ".join(toks[i:i + 2]) for i in range(len(toks) - 1)}
        grams |= {" ".join(toks[i:i + 3]) for i in range(len(toks) - 2)}   # PaperTowelRoll
        grams |= {s for t in list(grams) for s in singulars(t)}
        words = {x for x in toks if x not in STOP and len(x) > 2}
        with_target = [v for v in pool if gt & {x.lower() for x in v}]
        data.append({"tid": t["task_id"], "gt": gt, "grams": grams, "words": words,
                     "vis": with_target or pool, "has_target_frames": bool(with_target)})

    n_with = sum(1 for d in data if d["has_target_frames"])
    print(f"tasks {len(data)}; 场景里有'目标可见帧'的任务 {n_with} "
          f"({n_with / len(data):.0%})，其余用全部帧")
    print(f"{'matcher':>22} {'pair recall':>12} {'noise/frame':>12} {'precision':>10} "
          f"{'traj recall':>12} {'traj noise':>11}")

    def run(label, fn):
        rec, noise, prec, trec, tnoise = [], [], [], [], []
        for d in data:
            per, pn = [], []
            for _ in range(25):
                vis = random.choice(d["vis"])
                vis_l = {c.lower() for c in vis}
                got = {c.lower() for c in fn(d, vis)}
                if d["gt"] & vis_l:
                    per.append(1.0 if (d["gt"] & got) else 0.0)
                pn.append(len(got - d["gt"]))
                if got:
                    prec.append(len(got & d["gt"]) / len(got))
            rec.append(statistics.mean(per))
            noise.append(statistics.mean(pn))
            rel = set()
            for _ in range(10):
                rel |= {c.lower() for c in fn(d, random.choice(d["vis"]))}
            trec.append(1.0 if (d["gt"] & rel) else 0.0)
            tnoise.append(len(rel - d["gt"]))
        print(f"{label:>22} {statistics.mean(rec):11.1%} {statistics.mean(noise):12.2f} "
              f"{statistics.mean(prec):10.0%} {statistics.mean(trec):11.1%} "
              f"{statistics.mean(tnoise):11.2f}")

    run("exact object-only", lambda d, vis:
        {c for c in vis if any(g in exact_forms(c, False) for g in d["grams"])})
    run("exact + components", lambda d, vis:
        {c for c in vis if any(g in exact_forms(c, True) for g in d["grams"])})
    run("semantic 0.85", lambda d, vis: set(pair_frame(sim, d["words"], vis, 0.85)[0]))
    run("semantic 0.90", lambda d, vis: set(pair_frame(sim, d["words"], vis, 0.90)[0]))
    run("exact|semantic 0.85", lambda d, vis:
        {c for c in vis if any(g in exact_forms(c, True) for g in d["grams"])}
        | set(pair_frame(sim, d["words"], vis, 0.85)[0]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
