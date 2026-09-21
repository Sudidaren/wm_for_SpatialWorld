#!/bin/bash
# 开了 AutoDL 学术加速（/etc/network_turbo）之后再测一次吞吐。
set -u
echo "=== network_turbo 里是什么 ==="
cat /etc/network_turbo
echo "=== source 它 ==="
source /etc/network_turbo
env | grep -iE "proxy" | head
echo "=== 直连 huggingface.co（20s）==="
timeout 25 curl -s -o /tmp/hf.bin -w '  hf 直连: %{speed_download} B/s  http=%{http_code}\n' \
  --max-time 20 -r 0-52428800 \
  "https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct/resolve/main/model-00001-of-00013.safetensors"
echo "=== hf-mirror（20s，开加速后）==="
timeout 25 curl -s -o /tmp/mir.bin -w '  hf-mirror: %{speed_download} B/s  http=%{http_code}\n' \
  --max-time 20 -r 0-52428800 \
  "https://hf-mirror.com/Qwen/Qwen3-VL-30B-A3B-Instruct/resolve/main/model-00001-of-00013.safetensors"
echo "=== hf 直连 · 8 进程并行（20s）==="
source /etc/network_turbo
rm -f /tmp/pp*.bin
for i in $(seq 0 7); do
  s=$((i * 30000000)); e=$((s + 29999999))
  curl -s -o "/tmp/pp$i.bin" -r "${s}-${e}" --max-time 19 \
    "https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct/resolve/main/model-00001-of-00013.safetensors" &
done
t0=$(date +%s); wait; t1=$(date +%s)
sz=$(du -cb /tmp/pp*.bin 2>/dev/null | tail -1 | cut -f1)
echo "  8 并行: $((sz / 1000000)) MB in $((t1 - t0))s -> $((sz / (t1 - t0 + 1) / 1000000)) MB/s"
