#!/bin/bash
# TinyStories ablations + batch-size experiments, one run per GPU.
# Usage: bash scripts/run_ablations.sh <BEST_LR>
set -e
BEST_LR=${1:?usage: run_ablations.sh <best_lr>}
TRAIN=out/data/tinystories_train.npy
VAL=out/data/tinystories_valid.npy
COMMON="--train-data $TRAIN --val-data $VAL --compile"

# --- Round 1: architecture ablations (7 runs) ---
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --run-name ts-abl-noln       $COMMON --lr $BEST_LR --norm-position none  > out/runs/log-abl-noln.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 python scripts/train.py --run-name ts-abl-noln-lowlr $COMMON --lr 1e-4     --norm-position none  > out/runs/log-abl-noln-lowlr.txt 2>&1 &
CUDA_VISIBLE_DEVICES=2 python scripts/train.py --run-name ts-abl-postnorm  $COMMON --lr $BEST_LR --norm-position post  > out/runs/log-abl-postnorm.txt 2>&1 &
CUDA_VISIBLE_DEVICES=3 python scripts/train.py --run-name ts-abl-nope      $COMMON --lr $BEST_LR --no-rope              > out/runs/log-abl-nope.txt 2>&1 &
CUDA_VISIBLE_DEVICES=4 python scripts/train.py --run-name ts-abl-silu      $COMMON --lr $BEST_LR --ffn-type silu --d-ff 2048 > out/runs/log-abl-silu.txt 2>&1 &
# --- Batch size experiment, part 1 (batch sizes sharing round 1 GPUs) ---
CUDA_VISIBLE_DEVICES=5 python scripts/train.py --run-name ts-bs-1   $COMMON --lr 1e-4    --batch-size 1   --max-iters 10000 > out/runs/log-bs-1.txt 2>&1 &
CUDA_VISIBLE_DEVICES=6 python scripts/train.py --run-name ts-bs-32  $COMMON --lr 5e-4    --batch-size 32  > out/runs/log-bs-32.txt 2>&1 &
CUDA_VISIBLE_DEVICES=7 python scripts/train.py --run-name ts-bs-64  $COMMON --lr 7e-4    --batch-size 64  > out/runs/log-bs-64.txt 2>&1 &
wait

# --- Round 2: remaining batch sizes ---
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --run-name ts-bs-256  $COMMON --lr 2e-3 --batch-size 256  > out/runs/log-bs-256.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 python scripts/train.py --run-name ts-bs-512  $COMMON --lr 3e-3 --batch-size 512  > out/runs/log-bs-512.txt 2>&1 &
CUDA_VISIBLE_DEVICES=2 python scripts/train.py --run-name ts-bs-1024 $COMMON --lr 4e-3 --batch-size 1024 > out/runs/log-bs-1024.txt 2>&1 &
wait
echo ABLATIONS_DONE
