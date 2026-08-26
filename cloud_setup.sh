#!/usr/bin/env bash
# One-command cloud bootstrap: clone repo, install env, download data.
# Run on the cloud GPU machine (Ubuntu, NVIDIA driver, 50GB+ disk).
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
REPO_URL="${REPO_URL:-git@github.com:Sudidaren/wm_for_SpatialWorld.git}"
HF_DATASET="${HF_DATASET:-Sudidaren/lightwm-data}"
DATA_DIR="${DATA_DIR:-/data/lightwm}"

echo "== 1/4 clone code =="
if [ ! -d lightwm_phases ]; then
  git clone "$REPO_URL" lightwm_phases
fi
cd lightwm_phases

echo "== 2/4 python env =="
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install transformers numpy pillow tqdm pyyaml scipy huggingface_hub

echo "== 3/4 download data (23GB) =="
mkdir -p "$DATA_DIR"
export HF_HUB_ENABLE_HF_TRANSFER=0
huggingface-cli download "$HF_DATASET" --repo-type=dataset \
  --local-dir "$DATA_DIR"

echo "== 4/4 set paths (add to ~/.bashrc) =="
echo "export LIGHTWM_DATA_ROOT=$DATA_DIR/lightwm_data" >> ~/.bashrc
echo "export LIGHTWM_COV_ROOT=$DATA_DIR/lightwm_data_cov" >> ~/.bashrc
echo "export LIGHTWM_FD_ROOT=$DATA_DIR/fd_benchmark_full_20260811_224644" >> ~/.bashrc
echo "export HF_ENDPOINT=$HF_ENDPOINT" >> ~/.bashrc

echo ""
echo "SETUP DONE. Next:"
echo "  cd ~/lightwm_phases && bash phase_b/cloud_train.sh"
