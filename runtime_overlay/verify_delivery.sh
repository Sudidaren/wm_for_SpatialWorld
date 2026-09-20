#!/usr/bin/env bash
# One-command verification of a LightWM delivery.
#
#   bash verify_delivery.sh --repo <spatialworld_dir> --overlay <overlay_dir>
#
# Proves, on a checkout that has the overlay applied:
#   1. every runtime file is byte-identical to the published overlay (sha256)
#   2. the overlay ships no unlisted / redundant file
#   3. the runtime compiles
#   4. the information-isolation audit passes (no simulator channel reaches WM)
#   5. the functional smoke passes (memory readout, target hint, CheckState,
#      world-model anchors + dead-reckoned pose) with stubbed perception/VLM
#   6. the unit suites that back those claims pass
#
# No GPU, no simulator, no network, no model weights are needed.
set -uo pipefail

REPO=""; OVERLAY=""
while [ $# -gt 0 ]; do
  case "$1" in
    --repo)    REPO="$2"; shift 2 ;;
    --overlay) OVERLAY="$2"; shift 2 ;;
    *) echo "usage: $0 --repo <spatialworld_dir> --overlay <overlay_dir>"; exit 2 ;;
  esac
done
if [ -z "$REPO" ] || [ -z "$OVERLAY" ]; then
  echo "usage: $0 --repo <spatialworld_dir> --overlay <overlay_dir>"; exit 2
fi
REPO="$(cd "$REPO" && pwd)"
OVERLAY="$(cd "$OVERLAY" && pwd)"

PY="${PY:-}"
if [ -z "$PY" ]; then
  for cand in "$REPO/envs/ai2thor/.venv/bin/python" python3 python; do
    if command -v "$cand" >/dev/null 2>&1; then PY="$cand"; break; fi
  done
fi
if [ -z "$PY" ]; then echo "no python interpreter found"; exit 2; fi

PASS=0; FAIL=0
step() { printf '\n=== %s ===\n' "$1"; }
ok()   { echo "  PASS  $1"; PASS=$((PASS + 1)); }
no()   { echo "  FAIL  $1"; FAIL=$((FAIL + 1)); }

echo "repo    : $REPO"
echo "overlay : $OVERLAY"
echo "python  : $PY ($($PY -V 2>&1))"

# ---------------------------------------------------------------- 1. sha256
step "1/6 overlay integrity (sha256)"
if (cd "$REPO" && sha256sum -c --quiet "$OVERLAY/MANIFEST.sha256"); then
  ok "$(grep -c . "$OVERLAY/MANIFEST.sha256") files match the manifest"
else
  no "manifest mismatch"
fi

# ------------------------------------------------------------- 2. inventory
step "2/6 no unlisted / redundant file in the overlay"
if "$PY" - "$OVERLAY" <<'PY'
import hashlib, sys
from pathlib import Path

overlay = Path(sys.argv[1])
listed = set()
for line in (overlay / "MANIFEST.sha256").read_text().splitlines():
    if not line.strip():
        continue
    rel = line.split(None, 1)[1].strip()
    listed.add(rel[2:] if rel.startswith("./") else rel)
meta = {"MANIFEST.sha256", "README.md", "setup_lightwm_runtime.sh",
        "verify_delivery.sh", "INVENTORY.md"}
present = set()
for p in overlay.rglob("*"):
    if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc":
        present.add(str(p.relative_to(overlay)))
unlisted = sorted(present - listed - meta)
missing = sorted(listed - present)
print(f"  overlay files: {len(present)}  listed in manifest: {len(listed)}")
if unlisted:
    print("  unlisted files (redundant?): " + ", ".join(unlisted)); sys.exit(1)
if missing:
    print("  listed but absent: " + ", ".join(missing)); sys.exit(1)
print("  every shipped file is accounted for")
PY
then ok "overlay inventory complete"; else no "overlay inventory"; fi

# ---------------------------------------------------------- 3. compiles
step "3/6 runtime compiles"
if "$PY" -m py_compile \
      "$REPO"/mllm_base_agent/agent/*.py \
      "$REPO"/scripts/ai2thor/work/run_task.py ; then
  ok "py_compile"
else
  no "py_compile"
fi

# ------------------------------------------------------------- 4/5. checks
run_suite() {                       # $1 = label, $2 = path
  step "$1"
  if (cd "$REPO" && "$PY" "$2"); then ok "$2"; else no "$2"; fi
}

run_suite "4/6 information isolation + functional smoke" "tests/test_wm_delivery.py"
run_suite "5/6 object state check (CheckState)"          "tests/test_object_query.py"
run_suite "5/6 object state check, agent loop"           "tests/test_object_query_loop.py"
run_suite "5/6 target hint (no vocabulary)"              "tests/test_target_priority.py"
run_suite "5/6 state variants (sliced/cracked rename)"   "tests/test_state_variants.py"
run_suite "5/6 hand state (unseen but held)"             "tests/test_held_state.py"
run_suite "5/6 blocked move (one instruction only)"      "tests/test_blocked_hint.py"
run_suite "5/6 parameter provenance (no eval-set fitting)" "tests/test_param_provenance.py"

# ------------------------------------------------------------- 6. summary
step "6/6 summary"
echo "  passed: $PASS   failed: $FAIL"
if [ "$FAIL" -eq 0 ]; then
  echo
  echo "OK: this checkout reproduces the delivered WingmanWM behaviour"
  echo "    and no simulator-only channel reaches the world model."
  exit 0
fi
echo
echo "FAILED: see the steps above."
exit 1
