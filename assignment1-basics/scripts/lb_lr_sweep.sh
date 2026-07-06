#!/usr/bin/env bash
# Leaderboard LR sweep: 109M model on OWT, 8-GPU DDP at the *final* effective
# batch (8×256=2048) so the winning LR transfers directly to the 45-min run.
# Short (1200 steps) and sequential — each run uses all 8 GPUs.
set -euo pipefail
cd "$(dirname "$0")/.."
UV=~/.local/bin/uv
ARCH="--vocab-size 32000 --context-length 256 --d-model 768 --num-layers 12 --num-heads 12 --d-ff 2048 --tie-embeddings"
COMMON="--train-data out/data/owt_train.npy --val-data out/data/owt_valid.npy $ARCH \
  --batch-size 256 --max-iters 1200 --warmup-iters 150 --eval-interval 200 \
  --checkpoint-interval 1200 --log-interval 20 --compile"

for LR in 6e-3 1.2e-2 2e-2; do
  echo "=== sweep lr=$LR $(date +%T) ==="
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 $UV run torchrun --nproc_per_node=8 --master_port=29530 \
    scripts/train.py --run-name owt-lb-lr-$LR $COMMON --lr $LR \
    > out/runs/log-owt-lb-lr-$LR.txt 2>&1 || echo "lr=$LR FAILED"
done
echo "=== sweep done $(date +%T) ==="
