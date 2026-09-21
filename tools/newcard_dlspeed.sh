#!/bin/bash
# 在评测卡上测三种下载方式的真实吞吐（各跑 ~45 秒），决定 62GB 从哪儿下。
set -u
shard1="https://hf-mirror.com/Qwen/Qwen3-VL-30B-A3B-Instruct/resolve/main/model-00001-of-00013.safetensors"
rm -rf /tmp/dltest && mkdir -p /tmp/dltest && cd /tmp/dltest

echo "=== 0) 有没有 aria2 / network_turbo ==="
command -v aria2c || echo "  no aria2c"
ls /etc/network_turbo 2>/dev/null || echo "  no network_turbo"

echo "=== 1) curl 单连接（45s）==="
timeout 45 curl -s -o s1.bin -w '  curl×1: %{speed_download} B/s\n' --max-time 44 "$shard1"

echo "=== 2) curl 8 连接（45s，各下一个 range）==="
rm -f p*.bin
for i in $(seq 0 7); do
  start=$((i * 20000000)); end=$((start + 19999999))
  curl -s -o "p$i.bin" -r "${start}-${end}" --max-time 44 "$shard1" &
done
t0=$(date +%s); wait; t1=$(date +%s)
sz=$(du -cb p*.bin 2>/dev/null | tail -1 | cut -f1)
echo "  8×range: $((sz / 1000000)) MB in $((t1 - t0))s -> $((sz / (t1 - t0 + 1) / 1000000)) MB/s"

echo "=== 3) aria2c 16 连接（45s，装了才测）==="
if command -v aria2c >/dev/null; then
  rm -f a.bin a.bin.aria2
  timeout 45 aria2c -x16 -s16 -k1M --summary-interval=0 -d . -o a.bin "$shard1" >/dev/null 2>&1
  sz=$(stat -c %s a.bin 2>/dev/null || echo 0)
  echo "  aria2c×16: $((sz / 1000000)) MB in 45s -> $((sz / 45 / 1000000)) MB/s"
else
  echo "  （跳过，先 apt 装）"
fi
echo "=== 4) 版本 / 磁盘 ==="
/root/miniconda3/bin/python -V; df -h /root/autodl-tmp | tail -1
