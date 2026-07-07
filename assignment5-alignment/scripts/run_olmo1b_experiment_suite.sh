#!/usr/bin/env bash
# OLMo-2-0425-1B full-parameter GRPO suite — the "with learning signal" companion
# to the Qwen7B last-layer suite. Weak base model + full-param updates + bigger
# rollout batches so reward/accuracy movement is actually measurable.
#
# Stage 0 (this script assumes the model is downloaded to models/OLMo-2-0425-1B):
#   baselines on the full 1319-question test set (vllm, fast)
# Stage 1: standard on-policy GRPO, full-param, r1_zero strict reward,
#   vllm rollouts (GPU pair 0/1), 200 steps, eval 256 every 25.
# Stage 2 (parallel, single GPU each, transformers rollouts):
#   LR sweep x3, Dr.GRPO, RFT — 100 steps each on GPUs 2..6.
set -u
cd "$(dirname "$0")/.."
UV=~/.local/bin/uv
MODEL=models/OLMo-2-0425-1B
OUT=out/a5_full_olmo1b
mkdir -p "$OUT/logs"

COMMON_TRAIN="--model $MODEL --train-limit 4096 --eval-limit 256 --eval-every 25 \
  --prompts-per-step 8 --group-size 8 --rollout-batch-size 16 \
  --max-new-tokens 512 --temperature 0.7 --gradient-accumulation-steps 4 \
  --out-dir $OUT"

stage_baselines() {
  for prompt in r1_zero question_only; do
    for mode in strict numeric; do
      echo ">>> baseline $prompt $mode"
      CUDA_VISIBLE_DEVICES=0 $UV run --no-sync python scripts/eval_gsm8k.py \
        --model "$MODEL" --prompt "$prompt" --reward-mode "$mode" \
        --backend vllm --vllm-gpu 0 --limit 1319 \
        --out-dir "$OUT/baselines" > "$OUT/logs/baseline_${prompt}_${mode}.log" 2>&1 \
        && echo "  ok" || echo "  FAIL (see log)"
    done
  done
}

stage_main() {
  echo ">>> standard GRPO 200 step (train GPU0, vllm rollout GPU1)"
  CUDA_VISIBLE_DEVICES=0,1 $UV run --no-sync python scripts/train_grpo_gsm8k.py \
    $COMMON_TRAIN --run-name standard_grpo_200step \
    --prompt r1_zero --reward-mode strict --variant grpo --lr 1e-6 --steps 200 \
    --rollout-backend vllm --vllm-gpu 1 --vllm-port 8100 --device cuda:0 \
    > "$OUT/logs/standard_grpo_200step.log" 2>&1 && echo "  ok" || echo "  FAIL"
}

stage_ablations() {
  declare -A runs=(
    [lr_3e-7]="--lr 3e-7 --variant grpo"
    [lr_3e-6]="--lr 3e-6 --variant grpo"
    [lr_1e-5]="--lr 1e-5 --variant grpo"
    [variant_dr_grpo]="--lr 1e-6 --variant dr_grpo"
    [variant_rft]="--lr 1e-6 --variant rft"
  )
  gpu=2
  for name in lr_3e-7 lr_3e-6 lr_1e-5 variant_dr_grpo variant_rft; do
    echo ">>> ablation $name (GPU $gpu)"
    CUDA_VISIBLE_DEVICES=$gpu $UV run --no-sync python scripts/train_grpo_gsm8k.py \
      $COMMON_TRAIN --run-name "${name}_100step" \
      --prompt r1_zero --reward-mode strict --steps 100 \
      --rollout-backend transformers --device cuda:0 ${runs[$name]} \
      > "$OUT/logs/${name}_100step.log" 2>&1 && echo "  [done] $name" || echo "  [FAIL] $name" &
    gpu=$((gpu+1))
  done
  wait
}

case "${1:-all}" in
  baselines) stage_baselines ;;
  main) stage_main ;;
  ablations) stage_ablations ;;
  all) stage_baselines; stage_main; stage_ablations ;;
esac
echo "=== suite stage(s) done ==="
