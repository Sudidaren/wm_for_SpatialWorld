#!/bin/bash
# Start the local subgoal-decomposition planner (Ollama + Qwen2.5-7B).
# Models and binaries live on D: (/mnt/d/ollama), port 11435.
#
# Usage:
#   bash scripts/start_local_planner.sh          # foreground
#   nohup bash scripts/start_local_planner.sh > /mnt/d/ollama/ollama_11435.log 2>&1 &
set -e
export OLLAMA_MODELS=/mnt/d/ollama/models
export OLLAMA_HOST=127.0.0.1:11435
if ! command -v /mnt/d/ollama/root/bin/ollama >/dev/null 2>&1; then
    echo "Ollama binary not found at /mnt/d/ollama/root/bin/ollama" >&2
    exit 1
fi
echo "Starting Ollama planner on 127.0.0.1:11435 (models: $OLLAMA_MODELS)"
exec /mnt/d/ollama/root/bin/ollama serve
