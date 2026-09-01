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

echo "== 3/5 download data (~37GB) =="
python3 - "$DATA_DIR" "$HF_DATASET" <<'PYEOF'
import os, sys, tarfile, shutil
from huggingface_hub import hf_hub_download, list_repo_tree, HfApi

data_dir, repo = sys.argv[1], sys.argv[2]
os.makedirs(data_dir, exist_ok=True)
tars = [
    "tarballs/lightwm_data_episodes.tar",
    "tarballs/lightwm_data_cov_episodes.tar",
    "tarballs/lightwm_data_objviews_episodes.tar",
    "tarballs/fd_ai2thor.tar",
]
for t in tars:
    p = hf_hub_download(repo, t, repo_type="dataset",
                        local_dir=os.path.join(data_dir, "dl"))
    print("downloaded", t, flush=True)

def extract(tar_rel, dest):
    p = os.path.join(data_dir, "dl", tar_rel)
    os.makedirs(dest, exist_ok=True)
    with tarfile.open(p) as tf:
        tf.extractall(dest)

extract(tars[0], data_dir)                                  # lightwm_data/episodes
extract(tars[1], data_dir)                                  # lightwm_data_cov/episodes
extract(tars[2], data_dir)                                  # lightwm_data_objviews/episodes
fd_dest = os.path.join(data_dir, "fd_benchmark_full_20260811_224644")
extract(tars[3], fd_dest)                                   # fd/ai2thor

# scene_gt + manifests (small files, download directly)
api = HfApi()
for f in list_repo_tree(repo, repo_type="dataset", recursive=True):
    if f.path.startswith(("lightwm_data/scene_gt/", "lightwm_data_cov/scene_gt/",
                          "lightwm_data/manifest.jsonl",
                          "lightwm_data_cov/manifest.jsonl")):
        dest = os.path.join(data_dir, f.path)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        hf_hub_download(repo, f.path, repo_type="dataset",
                        local_dir=data_dir)
shutil.rmtree(os.path.join(data_dir, "dl"), ignore_errors=True)
print("DATA EXTRACTED", flush=True)
PYEOF

echo "== 5/5 set paths (add to ~/.bashrc) =="
echo "export LIGHTWM_DATA_ROOT=$DATA_DIR/lightwm_data" >> ~/.bashrc
echo "export LIGHTWM_COV_ROOT=$DATA_DIR/lightwm_data_cov" >> ~/.bashrc
echo "export LIGHTWM_OBJVIEW_ROOT=$DATA_DIR/lightwm_data_objviews" >> ~/.bashrc
echo "export LIGHTWM_FD_ROOT=$DATA_DIR/fd_benchmark_full_20260811_224644" >> ~/.bashrc
echo "export HF_ENDPOINT=$HF_ENDPOINT" >> ~/.bashrc

echo ""
echo "SETUP DONE. Next:"
echo "  cd ~/lightwm_phases && bash phase_b/cloud_train.sh"
