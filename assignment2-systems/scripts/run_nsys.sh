#!/usr/bin/env bash
# Problem (nsys_profile): profile forward/backward/optimizer with nsys for two
# model sizes × three power-of-two context lengths (>128). NVTX ranges (warmup /
# measure / step_i / attention sub-steps) let us attribute kernels to each phase.
# NOTE: this nsys (2024.2.1) does NOT use `--` as the app separator — the app
# command follows the flags directly, and long options take `=value`.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
GPU="${1:-2}"
mkdir -p out/nsys

for size in medium xl; do
  for ctx in 256 1024 4096; do
    rep="out/nsys/${size}_ctx${ctx}_full"
    echo ">>> nsys $size ctx=$ctx full"
    CUDA_VISIBLE_DEVICES="$GPU" nsys profile \
      --trace=cuda,nvtx --force-overwrite=true --output="$rep" \
      "$PY" -m cs336_systems.benchmark --size "$size" --context-length "$ctx" \
        --mode full --dtype fp32 --warmup 2 --steps 3 --nvtx \
      > "out/nsys/${size}_ctx${ctx}.log" 2>&1 \
      && echo "  ok -> ${rep}.nsys-rep" || echo "  (failed/OOM: $size ctx=$ctx)"
  done
done
echo "=== nsys profiling done -> out/nsys ==="
