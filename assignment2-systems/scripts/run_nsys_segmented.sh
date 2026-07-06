#!/usr/bin/env bash
# Targeted nsys profiles WITH forward/backward/optimizer NVTX ranges, so we can
# attribute total forward vs backward GPU time and answer nsys_profile (a)/(d).
# forward mode uses --inference (true inference, no saved activations).
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
GPU="${1:-2}"
mkdir -p out/nsys

for spec in "medium 1024" "xl 1024"; do
  set -- $spec; size=$1; ctx=$2
  for mode in forward full; do
    inf=""; [ "$mode" = forward ] && inf="--inference"
    echo ">>> nsys-seg $size ctx=$ctx $mode"
    CUDA_VISIBLE_DEVICES="$GPU" nsys profile --trace=cuda,nvtx --force-overwrite=true \
      --output="out/nsys/${size}_ctx${ctx}_${mode}seg" \
      "$PY" -m cs336_systems.benchmark --size "$size" --context-length "$ctx" --mode "$mode" $inf \
        --warmup 2 --steps 3 --nvtx > "out/nsys/${size}_${ctx}_${mode}seg.log" 2>&1 \
      && echo "  ok" || echo "  FAIL"
  done
done
echo "=== nsys-seg done ==="
