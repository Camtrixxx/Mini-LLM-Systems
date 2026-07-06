#!/usr/bin/env bash
# Leaderboard LR sweep, round 2: 6e-3 (round-1 winner) sat at the low edge of
# the grid, so probe below it. Same 109M / eff-batch-2048 / 1200-step config.
set -euo pipefail
cd "$(dirname "$0")/.."
UV=~/.local/bin/uv
ARCH="--vocab-size 32000 --context-length 256 --d-model 768 --num-layers 12 --num-heads 12 --d-ff 2048 --tie-embeddings"
COMMON="--train-data out/data/owt_train.npy --val-data out/data/owt_valid.npy $ARCH \
  --batch-size 256 --max-iters 1200 --warmup-iters 150 --eval-interval 200 \
  --checkpoint-interval 1200 --log-interval 20 --compile"

for LR in 3e-3 4.5e-3; do
  echo "=== sweep2 lr=$LR $(date +%T) ==="
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 $UV run torchrun --nproc_per_node=8 --master_port=29531 \
    scripts/train.py --run-name owt-lb-lr-$LR $COMMON --lr $LR \
    > out/runs/log-owt-lb-lr-$LR.txt 2>&1 || echo "lr=$LR FAILED"
done
echo "=== sweep2 done $(date +%T) ==="
