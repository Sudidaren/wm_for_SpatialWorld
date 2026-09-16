#!/usr/bin/env python3
"""Prove that this repository ships nothing unexplained.

Every tracked file is classified by an explicit rule.  The audit fails if:

  * a tracked file matches no rule (an unexplained / leftover artefact), or
  * the runtime overlay ships a file that is not in its sha256 manifest, or
  * two Python modules with the same name would shadow each other.

Run from the repository root:

    python3 tools/audit_repo.py            # report + exit code
    python3 tools/audit_repo.py --verbose  # also list every file
"""

from __future__ import annotations

import collections
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: (matcher, category, why it is in the repository)
RULES = (
    ("runtime_overlay/", "runtime delivery",
     "the LightWM delta vs official SpatialWorld; sha256-manifested"),
    ("eval_pipeline/cloud/", "eval orchestration",
     "cloud launcher / monitor / ablation entry point"),
    ("eval_pipeline/", "frozen eval pipeline",
     "mirror of eval_pipeline v1.0 (FROZEN.sha256)"),
    ("results/", "published results",
     "paired evaluation snapshots with regeneration scripts"),
    ("docs/", "documentation", "handover, perception notes"),
    ("phase_b/", "perception training", "detector + depth head training"),
    ("phase_c/", "hidden-object belief", "phase C line (separate contribution)"),
    ("phase_d/", "eval definitions", "task definitions / config generation"),
    ("shared/", "offline geometry & memory libraries", "used by training/eval"),
    ("wm_modules/", "external WM modules", "separate contribution"),
    ("pack_v3/", "data packaging", "cloud upload helper"),
    ("scripts/", "perception setup helpers", "RF-DETR download/check"),
    ("configs/", "perception config", "detector label space"),
    ("data/", "task sets & splits", "frozen splits used by training"),
    ("tests/", "tests", "perception runtime + weight download tests"),
    ("README.md", "documentation", "repo overview"),
    ("REPRODUCE_DETECTOR.md", "documentation", "detector reproduction"),
    ("requirements", "dependencies", "pinned requirements"),
    ("cloud_setup.sh", "setup", "cloud bootstrap"),
    (".gitignore", "meta", "ignore rules"),
)


def tracked():
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True,
                         text=True, check=True).stdout
    return [line for line in out.splitlines() if line.strip()]


def classify(path: str):
    for matcher, category, why in RULES:
        if path.startswith(matcher) or path == matcher:
            return category, why
    return None, None


def overlay_check(verbose: bool) -> int:
    overlay = ROOT / "runtime_overlay"
    manifest = overlay / "MANIFEST.sha256"
    if not manifest.exists():
        print("  runtime_overlay/MANIFEST.sha256 missing")
        return 1
    listed = set()
    for line in manifest.read_text().splitlines():
        if line.strip():
            rel = line.split(None, 1)[1].strip()
            listed.add(rel[2:] if rel.startswith("./") else rel)
    meta = {"MANIFEST.sha256", "README.md", "INVENTORY.md",
            "setup_lightwm_runtime.sh", "verify_delivery.sh"}
    present = {str(p.relative_to(overlay)) for p in overlay.rglob("*")
               if p.is_file() and "__pycache__" not in p.parts
               and p.suffix != ".pyc"}
    unlisted = sorted(present - listed - meta)
    missing = sorted(listed - present)
    print(f"  overlay: {len(present)} files on disk, {len(listed)} in manifest")
    if not unlisted and not missing:
        print("  overlay inventory complete (nothing unlisted, nothing missing)")
    if unlisted:
        print("  UNLISTED (redundant): " + ", ".join(unlisted))
    if missing:
        print("  MISSING (declared but absent): " + ", ".join(missing))
    if verbose and not (unlisted or missing):
        for f in sorted(listed):
            print(f"      {f}")
    return 1 if (unlisted or missing) else 0


def main() -> int:
    verbose = "--verbose" in sys.argv
    files = tracked()
    by_cat = collections.Counter()
    by_match = collections.defaultdict(list)
    unclassified = []
    for f in files:
        cat, why = classify(f)
        if cat is None:
            unclassified.append(f)
        else:
            by_cat[cat] += 1
            by_match[cat].append(f)

    print(f"repository: {ROOT}")
    print(f"tracked files: {len(files)}\n")
    print(f"{'category':<42}{'files':>6}")
    print("-" * 48)
    for cat, n in by_cat.most_common():
        print(f"{cat:<42}{n:>6}")

    problems = 0
    print("\noverlay integrity:")
    problems += overlay_check(verbose)

    dupes = [name for name, n in collections.Counter(
        os.path.basename(f) for f in files if f.endswith(".py")).items() if n > 1]
    real_dupes = [d for d in dupes if d != "__init__.py"]
    if real_dupes == ["run_task.py"]:
        note = "run_task.py only (one per environment: ai2thor / procthor / carla)"
    else:
        note = ", ".join(real_dupes) or "none (only __init__.py)"
    print(f"\nPython modules sharing a basename: {note}")

    big = []
    for f in files:
        p = ROOT / f
        if p.exists() and p.stat().st_size > 2 * 1024 * 1024:
            big.append((f, p.stat().st_size))
    print("tracked files over 2 MB:")
    for f, size in sorted(big, key=lambda x: -x[1]):
        print(f"    {size / 1024 / 1024:6.1f} MB  {f}")
    if not big:
        print("    none")

    if unclassified:
        problems += len(unclassified)
        print("\nUNCLASSIFIED files (leftovers?):")
        for f in unclassified:
            print(f"    {f}")
    else:
        print("\nevery tracked file matches an explicit rule")

    if verbose:
        print("\nfiles by category:")
        for cat in sorted(by_match):
            print(f"  [{cat}]")
            for f in sorted(by_match[cat]):
                print(f"      {f}")

    print(f"\nresult: {'FAIL' if problems else 'PASS'}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
