# -*- coding: utf-8 -*-
"""Freeze the household single-agent evaluation set (476 tasks).

Outputs (under eval_pipeline/eval_sets/):
  - eval_set_v1.json    : full manifest + counts + source hashes
  - eval_set_v1.csv     : one row per task
  - eval_set_v1.sha256  : sha256 of the json (freeze id)

Scope fixed by team decision (2026-09-12):
  household, single-agent only = AI2-THOR 311 + ProcTHOR 127 + VirtualHome 38
  Excluded: multi-agent 46, CARLA 80, EmbodiedCity 53, Game 105.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval_config as cfg
from env_spec import select_tasks

SCENES = ["ai2thor", "procthor", "virtualhome"]
EXCLUDED = {"mutil-ai2thor": 29, "mutil-procthor": 17,
            "carla": 80, "embodiedcity": 53, "game": 105}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    root = Path(cfg.SPATIALWORLD_ROOT)
    tasks = select_tasks(SCENES, None, "")
    try:
        upstream = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        upstream = ""
    cls_csv = root / "task_classification_detail.csv"
    rows = []
    for t in sorted(tasks, key=lambda x: (x.env, x.task_id)):
        rows.append({
            "task_id": t.task_id,
            "environment": t.env,
            "category": t.category,
            "task_type": t.task_type,
            "scene": t.scene,
            "golden_steps": t.golden_steps,
            "target_object_types": t.target_types,
            "task_json": str(t.task_json_path.relative_to(root)),
            "task_json_sha256": sha256_file(t.task_json_path),
        })
    by_env = Counter(r["environment"] for r in rows)
    by_cat = Counter(r["category"] for r in rows)
    by_type = Counter(r["task_type"] for r in rows)
    manifest = {
        "version": "eval_set_v1",
        "created_at": datetime.now().isoformat(),
        "scope": "household single-agent tasks only",
        "scenes": SCENES,
        "counts": {
            "total": len(rows),
            "by_environment": dict(by_env),
            "by_category": dict(by_cat),
            "by_task_type": dict(by_type),
        },
        "excluded": EXCLUDED,
        "source": {
            "spatialworld_root": str(root),
            "spatialworld_commit": upstream,
            "classification_csv": str(cls_csv.relative_to(root)),
            "classification_csv_sha256": sha256_file(cls_csv),
        },
        "tasks": rows,
    }
    out = Path(__file__).resolve().parent / "eval_sets"
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "eval_set_v1.json"
    json_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    with (out / "eval_set_v1.csv").open("w", encoding="utf-8-sig",
                                        newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()),
                                extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            r2 = dict(r)
            r2["target_object_types"] = "|".join(r["target_object_types"])
            writer.writerow(r2)
    digest = hashlib.sha256(json_path.read_bytes()).hexdigest()
    (out / "eval_set_v1.sha256").write_text(
        f"{digest}  eval_set_v1.json\n", encoding="utf-8")
    print(f"total={len(rows)} by_env={dict(by_env)}")
    print(f"sha256={digest}")
    print(f"written: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
