#!/usr/bin/env bash
# One-command cloud bootstrap: clone repo, install env, download data.
# Run on the cloud GPU machine (Ubuntu, NVIDIA driver, 50GB+ disk).
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
# Force mirror HTTP downloads (xet bypasses HF_ENDPOINT and hits slow US CDN)
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
REPO_URL="${REPO_URL:-git@github.com:Sudidaren/wm_for_SpatialWorld.git}"
HF_DATASET="${HF_DATASET:-Sudidaren/lightwm-data}"
DATA_DIR="${DATA_DIR:-/data/lightwm}"
SYS_VENV="${SYS_VENV:-0}"   # 1 = reuse the machine's preinstalled torch

echo "== 1/4 clone code =="
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/README.md" ] && [ -f "$SCRIPT_DIR/cloud_setup.sh" ]; then
  # running from inside the extracted repo (e.g. tar archive, no .git)
  echo "repo already present: $SCRIPT_DIR"
  cd "$SCRIPT_DIR"
elif [ -f "$PWD/lightwm_phases/README.md" ]; then
  echo "repo already present: $PWD/lightwm_phases"
  cd lightwm_phases
else
  git clone "$REPO_URL" lightwm_phases
  cd lightwm_phases
fi

echo "== 2/4 python env =="
if [ "$SYS_VENV" = "1" ]; then
  [ -d .venv ] || python3 -m venv --system-site-packages .venv
  source .venv/bin/activate
  if python -c "import torch, torchvision" 2>/dev/null; then
    echo "reusing system torch/torchvision (SYS_VENV=1)"
  else
    echo "no system torch found; installing cu128 wheels"
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
  fi
else
  [ -d .venv ] || python3 -m venv .venv
  source .venv/bin/activate
  pip install -U pip
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
fi
pip install transformers numpy pillow tqdm pyyaml scipy huggingface_hub

echo "== 3/5 download data (~40GB, streamed tar-by-tar) =="
python3 - "$DATA_DIR" "$HF_DATASET" <<'PYEOF'
import os, sys, shutil, tarfile
from huggingface_hub import hf_hub_download, list_repo_tree, HfApi

data_dir, repo = sys.argv[1], sys.argv[2]
os.makedirs(data_dir, exist_ok=True)
targets = [
    ("tarballs/lightwm_data_episodes.tar", data_dir),
    ("tarballs/lightwm_data_cov_episodes.tar", data_dir),
    ("tarballs/lightwm_data_objviews_episodes.tar", data_dir),
    ("tarballs/fd_ai2thor.tar",
     os.path.join(data_dir, "fd_benchmark_full_20260811_224644")),
    ("tarballs/lightwm_data_procthor.tar", data_dir),
    ("tarballs/lightwm_data_virtualhome.tar", data_dir),
]
for tar_rel, dest in targets:
    p = hf_hub_download(repo, tar_rel, repo_type="dataset",
                        local_dir=os.path.join(data_dir, "dl"))
    print("downloaded", tar_rel, flush=True)
    os.makedirs(dest, exist_ok=True)
    with tarfile.open(p) as tf:
        tf.extractall(dest)
    os.remove(p)
    print("extracted", tar_rel, flush=True)

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
echo "export LIGHTWM_PROCTHOR_ROOT=$DATA_DIR/lightwm_data_procthor" >> ~/.bashrc
echo "export LIGHTWM_VIRTUALHOME_ROOT=$DATA_DIR/lightwm_data_virtualhome" >> ~/.bashrc
echo "export HF_ENDPOINT=$HF_ENDPOINT" >> ~/.bashrc

echo ""
echo "SETUP DONE. Next:"
echo "  cd ~/lightwm_phases && bash phase_b/cloud_train.sh"
