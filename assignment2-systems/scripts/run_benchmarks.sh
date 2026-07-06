#!/usr/bin/env bash
# Section 2 timing sweep: every model size × {forward, forward_backward, full}
# × {fp32, bf16}, 5 warmup + 10 measured steps, ctx 512 batch 4.
# Each config runs in its own process so an OOM (e.g. 10B full) is isolated.
# Feeds benchmarking_script(b) [fp32] and benchmarking_mixed_precision(c) [fp32 vs bf16].
set -u
cd "$(dirname "$0")/.."
UV=~/.local/bin/uv
OUT=out/bench_section2.jsonl
mkdir -p out
: > "$OUT"

for size in small medium large xl 10B; do
  for mode in forward forward_backward full; do
    for dtype in fp32 bf16; do
      echo ">>> $size $mode $dtype"
      CUDA_VISIBLE_DEVICES=0 $UV run python -m cs336_systems.benchmark \
        --size "$size" --mode "$mode" --dtype "$dtype" \
        --warmup 5 --steps 10 --json >> "$OUT" 2>/dev/null \
        || echo "{\"size\":\"$size\",\"mode\":\"$mode\",\"dtype\":\"$dtype\",\"oom\":true}" >> "$OUT"
    done
  done
done
echo "=== done -> $OUT ==="
