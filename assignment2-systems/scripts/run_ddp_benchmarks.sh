#!/usr/bin/env bash
# distributed_communication (all-reduce 1MB..1GB × 2/4/6 GPUs) + DDP variant
# comparison (naive/flat/overlap on xl, 2 GPUs).
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python

: > out/allreduce.jsonl
for np in 2 4 6; do
  gpus=$(seq -s, 0 $((np-1)))
  echo ">>> all-reduce world_size=$np"
  CUDA_VISIBLE_DEVICES="$gpus" "$PY" -m torch.distributed.run --nproc_per_node="$np" \
    --master_port=29561 scripts/bench_allreduce.py >> out/allreduce.jsonl 2>/dev/null \
    || echo "  (failed ws=$np)"
done

: > out/ddp_variants.jsonl
for v in naive flat overlap; do
  echo ">>> ddp xl $v"
  CUDA_VISIBLE_DEVICES=0,1 "$PY" -m torch.distributed.run --nproc_per_node=2 \
    --master_port=29562 scripts/bench_ddp.py --ddp "$v" --size xl --batch-size 4 \
    --warmup 3 --iters 10 >> out/ddp_variants.jsonl 2>/dev/null \
    || echo "  (failed $v)"
done
echo "=== ddp benchmarks done ==="
