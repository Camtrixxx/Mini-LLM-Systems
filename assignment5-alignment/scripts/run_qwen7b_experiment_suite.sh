#!/usr/bin/env bash
set -euo pipefail

# Full-ish Qwen 7B experiment suite for the local 8xA800 machine.
#
# GPU0 is intentionally left for the vLLM baseline server.  This script uses
# GPU1-GPU7 for independent GRPO experiment runs in parallel.  Each run trains
# the last transformer layer only; this is the version that fits robustly in the
# current single-process training script without adding FSDP/ZeRO.

MODEL=${MODEL:-models/Qwen2.5-Math-7B-Instruct}
OUT_DIR=${OUT_DIR:-out/a5_full_qwen7b}
TRAIN_LIMIT=${TRAIN_LIMIT:-2048}
EVAL_LIMIT=${EVAL_LIMIT:-128}
COMMON_STEPS=${COMMON_STEPS:-50}
STANDARD_STEPS=${STANDARD_STEPS:-100}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-256}
PROMPTS_PER_STEP=${PROMPTS_PER_STEP:-2}
GROUP_SIZE=${GROUP_SIZE:-4}
TRAINABLE_LAST_LAYERS=${TRAINABLE_LAST_LAYERS:-1}

mkdir -p "${OUT_DIR}/logs"

run_on_gpu() {
  local gpu=$1
  local run_name=$2
  shift 2
  echo "[launch] gpu=${gpu} run=${run_name}" | tee -a "${OUT_DIR}/logs/launcher.log"
  (
    CUDA_VISIBLE_DEVICES="${gpu}" uv run python scripts/train_grpo_gsm8k.py \
      --model "${MODEL}" \
      --reward-mode numeric \
      --rollout-backend transformers \
      --device cuda:0 \
      --trainable-last-layers "${TRAINABLE_LAST_LAYERS}" \
      --train-limit "${TRAIN_LIMIT}" \
      --eval-limit "${EVAL_LIMIT}" \
      --prompts-per-step "${PROMPTS_PER_STEP}" \
      --group-size "${GROUP_SIZE}" \
      --max-new-tokens "${MAX_NEW_TOKENS}" \
      --rollout-batch-size 1 \
      --gradient-accumulation-steps 1 \
      --gradient-checkpointing \
      --eval-every 25 \
      --out-dir "${OUT_DIR}" \
      --run-name "${run_name}" \
      "$@"
  ) > "${OUT_DIR}/logs/${run_name}.stdout.log" 2>&1 &
}

# Wave 1: standard, LR sweep, prompt ablation, one variant.
run_on_gpu 1 standard_grpo_100step \
  --prompt question_only --variant grpo --lr 1e-6 --steps "${STANDARD_STEPS}"
run_on_gpu 2 lr_3e-7_50step \
  --prompt question_only --variant grpo --lr 3e-7 --steps "${COMMON_STEPS}"
run_on_gpu 3 lr_3e-6_50step \
  --prompt question_only --variant grpo --lr 3e-6 --steps "${COMMON_STEPS}"
run_on_gpu 4 lr_1e-5_50step \
  --prompt question_only --variant grpo --lr 1e-5 --steps "${COMMON_STEPS}"
run_on_gpu 5 prompt_three_shot_50step \
  --prompt r1_zero_three_shot_gsm8k --variant grpo --lr 1e-6 --steps "${COMMON_STEPS}"
run_on_gpu 6 variant_dr_grpo_50step \
  --prompt question_only --variant dr_grpo --lr 1e-6 --steps "${COMMON_STEPS}"
run_on_gpu 7 variant_rft_50step \
  --prompt question_only --variant rft --lr 1e-6 --steps "${COMMON_STEPS}"
wait

# Wave 2: remaining variants, off-policy, try-your-own.
run_on_gpu 1 variant_maxrl_50step \
  --prompt question_only --variant maxrl --lr 1e-6 --steps "${COMMON_STEPS}"
run_on_gpu 2 offpolicy_noclip_50step \
  --prompt question_only --variant grpo --lr 1e-6 --steps "${COMMON_STEPS}" \
  --importance-reweighting-method noclip
run_on_gpu 3 offpolicy_grpo_clip_50step \
  --prompt question_only --variant grpo --lr 1e-6 --steps "${COMMON_STEPS}" \
  --importance-reweighting-method grpo --cliprange 0.2
run_on_gpu 4 offpolicy_gspo_clip_50step \
  --prompt question_only --variant grpo --lr 1e-6 --steps "${COMMON_STEPS}" \
  --importance-reweighting-method gspo --cliprange 0.2
run_on_gpu 5 try_group8_temp09_50step \
  --prompt question_only --variant grpo --lr 1e-6 --steps "${COMMON_STEPS}" \
  --group-size 8 --temperature 0.9
wait

echo "[done] experiment suite complete: ${OUT_DIR}"
