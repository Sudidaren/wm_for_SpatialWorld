#!/usr/bin/env bash
# 依次并行上传所有 tar（8 路分片）。独立脚本，避免 pgrep 自匹配。
set -uo pipefail
S=/home/sudidaren/_newcard_stage
U=/home/sudidaren/lightwm_phases/tools/upload_parallel.py
for f in spatialworld_code spatialworld_eval lightwm_phases procthor10k spatialworld_envs; do
    [ -f "$S/$f.tar" ] || { echo "跳过 $f（没有包）"; continue; }
    python3 "$U" "$S/$f.tar" 8
done
echo "全部上传完成"
