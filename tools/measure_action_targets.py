#!/usr/bin/env python3
"""How many task targets does "whatever the model acts on" recover?

Proposal B: an object the agent has already issued an interaction action on
(``PickupObject(X)``, ``SliceObject(X)``, ``OpenObject(X)`` ...) is by
definition task-relevant, so it joins the relevant set and stays there.

The signal is the model's own output, so it needs no oracle and no vocabulary.
The only open question is its noise: a model that grabs a Knife it does not
need would drag the Knife into the hint for the rest of the episode.  This
measures candidate gate policies against each task's ``target_object_types``
on real runs, using the detector's own class list as the "already matched by
the instruction" baseline.

    python3 tools/measure_action_targets.py --run local:/home/.../runs/<name>
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import statistics
from collections import Counter
from pathlib import Path

TASK_ROOT = Path("/home/sudidaren/SpatialWorld/data/ai2thor/tasks")

#: every non-navigation action the environment accepts
INTERACTIONS = frozenset({
    "OpenObject", "CloseObject", "ToggleObjectOn", "ToggleObjectOff",
    "PickupObject", "PutObject", "ThrowObject", "DropHandObject",
    "SliceObject", "BreakObject", "CookObject", "DirtyObject", "CleanObject",
    "FillObjectWithLiquid", "EmptyLiquidFromObject", "UseUpObject",
})
#: of those, the ones that change an object's state rather than a container's
STATE_CHANGES = INTERACTIONS - {"OpenObject", "CloseObject", "PickupObject",
                                "PutObject", "ThrowObject", "DropHandObject",
                                "ToggleObjectOn", "ToggleObjectOff"}


def normalise(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name or "").lower())


def load_detector_classes() -> set[str]:
    """The perception head's label space (its checkpoint's own class list)."""
    import torch
    for cand in sorted(Path("/home/sudidaren/lightwm_phases/checkpoints").rglob(
            "checkpoint_best_total.pth")):
        ck = torch.load(str(cand), map_location="cpu", weights_only=False)
        args = ck.get("args") or {}
        names = (args.get("class_names") if isinstance(args, dict)
                 else getattr(args, "class_names", None))
        if names:
            return {str(x) for x in names}
    return set()


def outcome_of(hint: str, action: str):
    """``False`` when the step's hint published a failure, else ``None``.

    The step log records the hint the model actually received; a failed step
    is the only one that says so ("上一个动作：X（失败：画面未变化）"), so a
    missing line means "not known to have failed", not "succeeded".
    """
    if not hint or "（失败" not in hint:
        return None
    return False


def ground_truth(tid: str) -> set[str]:
    f = TASK_ROOT / tid / "task.json"
    if not f.exists():
        return set()
    d = json.loads(f.read_text())
    raw = d.get("target_object_types")
    if isinstance(raw, str):
        try:
            raw = ast.literal_eval(raw)
        except Exception:
            raw = [raw]
    return {normalise(x) for x in (raw or [])}


def instruction_hits(instr: str, vocab: set[str]) -> set[str]:
    """What the shipped exact matcher already keeps: instruction vs labels."""
    toks = re.sub(r"[^a-z0-9 ]", " ", (instr or "").lower()).split()
    grams = set(toks) | {" ".join(toks[i:i + 2]) for i in range(len(toks) - 1)}

    def forms(t: str) -> set[str]:
        ws = [w.lower() for w in re.split(r"(?<=[a-z])(?=[A-Z])|[\s_-]+", t) if w]
        if not ws:
            return set()
        out = {"".join(ws), " ".join(ws)}
        out |= {f + "s" for f in out}
        out |= {f[:-1] if f.endswith("s") else f for f in out}
        return out

    return {normalise(t) for t in vocab if forms(t) & grams}


def parse_action(action: str):
    m = re.match(r"^\s*([A-Za-z]+)\s*(?:\(\s*([A-Za-z0-9_]*)\s*\))?\s*$",
                 action or "")
    if not m:
        return None, None
    return m.group(1), m.group(2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="local:PATH")
    ap.add_argument("--show-noise", type=int, default=0,
                    help="print this many wrongly-added targets with context")
    args = ap.parse_args()
    source, _, root = args.run.partition(":")
    if source != "local":
        raise SystemExit("only local: is supported")

    vocab = load_detector_classes()
    print(f"detector classes: {len(vocab)}")

    # tid -> list of (action, object_type, frame_changed)
    traces: dict[str, list[tuple[str, str, object]]] = {}
    for f in sorted(Path(root).rglob("steps.jsonl")):
        m = re.search(r"ai2thor\d+", str(f))
        if not m:
            continue
        tid = m.group(0)
        rows = []
        for line in f.read_text(errors="ignore").splitlines():
            try:
                d = json.loads(line)
            except Exception:
                continue
            name, obj = parse_action(d.get("action_string") or "")
            if name:
                rows.append((name, obj or "",
                             outcome_of(d.get("mem_hint") or "", name)))
        if len(rows) > len(traces.get(tid, [])):
            traces[tid] = rows

    policies = {
        "P1 所有交互动作": lambda a, o, ok: a in INTERACTIONS,
        "P2 排除明确失败": lambda a, o, ok: a in INTERACTIONS and ok is not False,
        "P3 只要判定成功": lambda a, o, ok: a in INTERACTIONS and ok is True,
        "P4 只算状态改变类": lambda a, o, ok: a in STATE_CHANGES,
        "P5 不要放置目的地": lambda a, o, ok: a in INTERACTIONS - {
            "PutObject", "ThrowObject", "DropHandObject"},
        "P6 只算拿起+状态改变": lambda a, o, ok: a in STATE_CHANGES | {
            "PickupObject"},
    }

    results = {}
    for pname, keep in policies.items():
        rows = []
        for tid, trace in sorted(traces.items()):
            gt = ground_truth(tid)
            if not gt:
                continue
            task_file = TASK_ROOT / tid / "task.json"
            instr = json.loads(task_file.read_text()).get("instruction", "")
            base = instruction_hits(instr, vocab)
            added = {normalise(o) for a, o, ok in trace if o and keep(a, o, ok)}
            new = added - base
            rows.append({
                "tid": tid, "base": base, "new": new, "gt": gt,
                "correct": len(new & gt), "wrong": len(new - gt),
                "base_correct": len(base & gt),
            })
        if not rows:
            continue
        results[pname] = rows

    for pname, rows in results.items():
        n = len(rows)
        corr = statistics.mean(r["correct"] for r in rows)
        wrong = statistics.mean(r["wrong"] for r in rows)
        zero_noise = sum(1 for r in rows if r["wrong"] == 0) / n
        rescued = [r for r in rows
                   if not r["base"] and r["base_correct"] == 0 and r["correct"]]
        still_empty = [r for r in rows
                       if r["base_correct"] == 0 and r["correct"] == 0]
        print(f"\n== {pname}  (n={n})")
        print(f"   每任务新增 正确 {corr:.2f} / 噪声 {wrong:.2f}；"
              f"零噪声任务 {zero_noise*100:.0f}%")
        print(f"   指令零命中的 {sum(1 for r in rows if not r['base'])} 个任务里，"
              f"救回 {len(rescued)} 个；仍无正确目标 {len(still_empty)} 个")
        top = Counter()
        for r in rows:
            for x in r["new"] - r["gt"]:
                top[x] += 1
        print("   噪声 top: " + ", ".join(f"{k}×{v}" for k, v in top.most_common(8)))
        if args.show_noise:
            shown = 0
            for r in rows:
                for x in sorted(r["new"] - r["gt"]):
                    instr = json.loads((TASK_ROOT / r["tid"] / "task.json")
                                       .read_text()).get("instruction", "")
                    acts = [(a, o) for a, o, _ in traces[r["tid"]]
                            if normalise(o) == x]
                    print(f"      {r['tid']} 「{instr[:70]}」 "
                          f"gt={sorted(r['gt'])} 加了 {x} via {acts[:2]}")
                    shown += 1
                    if shown >= args.show_noise:
                        break
                if shown >= args.show_noise:
                    break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
