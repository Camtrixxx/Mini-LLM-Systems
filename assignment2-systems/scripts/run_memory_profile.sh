#!/usr/bin/env bash
# Problem (memory_profiling): xl model, context lengths 128 and 2048, forward-only
# and full training step, fp32 and bf16. Reports peak memory and dumps memory_viz
# snapshot pickles for the forward and full-step timelines.
set -u
cd "$(dirname "$0")/.."
UV=~/.local/bin/uv
GPU="${1:-1}"
OUT=out/mem_xl.jsonl
mkdir -p out out/mem_snapshots
: > "$OUT"

for ctx in 128 2048; do
  for mode in forward full; do
    # "forward" here means inference (no_grad, no saved activations); "full" is a
    # training step that must retain activations for backward.
    inf=""; [ "$mode" = "forward" ] && inf="--inference"
    for dtype in fp32 bf16; do
      snap="out/mem_snapshots/xl_ctx${ctx}_${mode}_${dtype}.pickle"
      echo ">>> xl ctx=$ctx $mode $dtype"
      CUDA_VISIBLE_DEVICES="$GPU" $UV run python -m cs336_systems.benchmark \
        --size xl --context-length "$ctx" --mode "$mode" --dtype "$dtype" $inf \
        --warmup 3 --steps 3 --memory-profile "$snap" --json >> "$OUT" 2>/dev/null \
        || echo "{\"size\":\"xl\",\"context_length\":$ctx,\"mode\":\"$mode\",\"dtype\":\"$dtype\",\"oom\":true}" >> "$OUT"
    done
  done
done
echo "=== mem done -> $OUT ==="
