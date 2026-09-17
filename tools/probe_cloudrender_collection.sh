#!/bin/bash
# One-house smoke test for GPU (CloudRendering) data collection.
# Run it ON the machine whose GPU you want to test; it prints where it hangs.
WM_ROOT=${WM_ROOT:-/home/sudidaren/lightwm_phases}
cd "$WM_ROOT" || exit 1
PY=${PY:-/home/sudidaren/SpatialWorld/envs/ai2thor/.venv/bin/python}
OUT=${OUT:-/tmp/_crtest}
START=${START:-109}
echo "=== CloudRendering probe: house val#$START, sparse policy ==="
timeout "${TIMEOUT:-240}" env PYTHONUNBUFFERED=1 "$PY" phase_b/collect_procthor.py \
  --split val --start "$START" --houses 1 --yaws 2 --horizons 1 --position-step 15 \
  --platform CloudRendering --out "$OUT" 2>&1 | tail -12
echo "frames: $(find "$OUT" -name 'step_*_rgb.png' 2>/dev/null | wc -l)"
