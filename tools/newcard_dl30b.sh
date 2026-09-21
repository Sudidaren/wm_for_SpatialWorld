#!/bin/bash
# 在卡上用 ModelScope 下 30B（实测 12-14 MB/s，比 hf-mirror 快 20 倍以上）。
# ModelScope 在 AutoDL 的 no_proxy 里，直连即可，不要 source network_turbo。
set -u
LOG=/root/autodl-tmp/dl30b.log
exec >>"$LOG" 2>&1
echo "================ [$(date '+%F %T')] 开始下载 Qwen3-VL-30B-A3B-Instruct"
/root/miniconda3/bin/pip install -q -i https://pypi.tuna.tsinghua.edu.cn/simple \
    modelscope 2>&1 | tail -2
/root/miniconda3/bin/python - <<'EOF'
import os, time
from modelscope import snapshot_download
t = time.time()
kwargs = dict(local_dir='/root/autodl-tmp/models/qwen3vl-30b-bf16')
try:
    path = snapshot_download('Qwen/Qwen3-VL-30B-A3B-Instruct', max_workers=8, **kwargs)
except TypeError:
    path = snapshot_download('Qwen/Qwen3-VL-30B-A3B-Instruct', **kwargs)
print('done', path, f'{time.time()-t:.0f}s')
EOF
echo "[$(date '+%F %T')] 下载结束，核对："
du -sh /root/autodl-tmp/models/qwen3vl-30b-bf16
ls /root/autodl-tmp/models/qwen3vl-30b-bf16 | wc -l
ls /root/autodl-tmp/models/qwen3vl-30b-bf16/*.safetensors | wc -l
