"""Run a compact GRPO experiment on GSM8K.

This script intentionally uses the tested GRPO primitives from
``tests/adapters.py``.  It supports two rollout paths:

* ``--rollout-backend transformers``: simplest smoke test, no vLLM required.
* ``--rollout-backend vllm``: faster rollouts through ``VLLMServer`` and NCCL
  weight sync after each policy update.

For a 7B model, full-parameter AdamW may exceed a single A800 once activations
and optimizer state are included.  Use ``--trainable-last-layers`` for a first
end-to-end run, then move to FSDP/ZeRO for full-parameter runs.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Literal

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cs336_alignment.gsm8k import (
    GSM8K_DIR,
    evaluate_responses,
    get_reward_fn,
    load_gsm8k_jsonl,
    load_prompt_template,
    render_prompts,
)
from tests.adapters import (
    run_get_response_log_probs,
    run_grpo_train_step,
    run_tokenize_prompt_and_output,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen2.5-Math-7B")
    p.add_argument("--prompt", default="r1_zero")
    p.add_argument("--reward-mode", choices=["strict", "numeric"], default="strict")
    p.add_argument("--train-path", type=Path, default=GSM8K_DIR / "train.jsonl")
    p.add_argument("--eval-path", type=Path, default=GSM8K_DIR / "test.jsonl")
    p.add_argument("--out-dir", type=Path, default=Path("out/a5/grpo"))
    p.add_argument("--run-name", default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--train-limit", type=int, default=None)
    p.add_argument("--eval-limit", type=int, default=128)
    p.add_argument("--prompts-per-step", type=int, default=4)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--trainable-last-layers", type=int, default=None)
    p.add_argument(
        "--variant",
        choices=["grpo", "grpo_constant", "dr_grpo", "rft", "maxrl"],
        default="grpo",
    )
    p.add_argument(
        "--importance-reweighting-method",
        choices=["none", "noclip", "grpo", "gspo"],
        default="none",
    )
    p.add_argument("--cliprange", type=float, default=0.2)
    p.add_argument("--loss-normalization-constant", type=int, default=None)
    p.add_argument("--rollout-backend", choices=["transformers", "vllm"], default="transformers")
    p.add_argument("--rollout-batch-size", type=int, default=4)
    p.add_argument("--vllm-port", type=int, default=8000)
    p.add_argument("--vllm-gpu", default="1")
    p.add_argument("--vllm-tensor-parallel-size", type=int, default=1)
    p.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--sync-vllm-weights", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--save-every", type=int, default=0)
    return p.parse_args()


def dtype_from_arg(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def set_trainable_last_layers(model: torch.nn.Module, n_layers: int | None) -> None:
    if n_layers is None:
        return
    for param in model.parameters():
        param.requires_grad_(False)

    layers = None
    for path in ("model.layers", "transformer.h", "gpt_neox.layers"):
        obj = model
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None:
            layers = obj
            break
    if layers is None:
        raise ValueError("Could not find transformer layers for --trainable-last-layers.")

    for layer in list(layers)[-n_layers:]:
        for param in layer.parameters():
            param.requires_grad_(True)
    for name, module in model.named_modules():
        if name.endswith(("lm_head", "norm", "ln_f")):
            for param in module.parameters(recurse=False):
                param.requires_grad_(True)


def variant_kwargs(args: argparse.Namespace, rollout_batch_size: int) -> dict:
    constant = args.loss_normalization_constant or rollout_batch_size
    table = {
        "grpo": {
            "baseline": "mean",
            "advantage_normalizer": "std",
            "loss_normalization": "sequence",
        },
        "grpo_constant": {
            "baseline": "mean",
            "advantage_normalizer": "std",
            "loss_normalization": "constant",
            "normalization_constant": constant,
        },
        "dr_grpo": {
            "baseline": "mean",
            "advantage_normalizer": "none",
            "loss_normalization": "constant",
            "normalization_constant": constant,
        },
        "rft": {
            "baseline": "none",
            "advantage_normalizer": "none",
            "loss_normalization": "constant",
            "normalization_constant": constant,
        },
        "maxrl": {
            "baseline": "mean",
            "advantage_normalizer": "mean",
            "loss_normalization": "constant",
            "normalization_constant": constant,
        },
    }
    return table[args.variant]


@torch.no_grad()
def generate_with_transformers(
    model: torch.nn.Module,
    tokenizer,
    prompts: list[str],
    *,
    group_size: int,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> list[str]:
    model.eval()
    old_use_cache = getattr(model.config, "use_cache", None)
    if old_use_cache is not None:
        model.config.use_cache = True
    outputs: list[str] = []
    do_sample = temperature > 0
    try:
        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start : start + batch_size]
            encoded = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(model.device)
            if encoded["input_ids"].shape[1] == 0:
                fallback_id = tokenizer.eos_token_id or tokenizer.pad_token_id or 0
                encoded["input_ids"] = torch.full(
                    (len(batch_prompts), 1), fallback_id, dtype=torch.long, device=model.device
                )
                encoded["attention_mask"] = torch.ones_like(encoded["input_ids"])
            generated = model.generate(
                **encoded,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                max_new_tokens=max_new_tokens,
                num_return_sequences=group_size,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                remove_invalid_values=True,
                renormalize_logits=True,
            )
            input_width = encoded["input_ids"].shape[1]
            for i in range(len(batch_prompts)):
                for sample_idx in range(group_size):
                    row = i * group_size + sample_idx
                    outputs.append(tokenizer.decode(generated[row, input_width:], skip_special_tokens=True))
    finally:
        if old_use_cache is not None:
            model.config.use_cache = old_use_cache
        model.train()
    return outputs


@torch.no_grad()
def compute_old_log_probs(model, tokenizer, prompts: list[str], responses: list[str]) -> torch.Tensor:
    device = next(model.parameters()).device
    tok = run_tokenize_prompt_and_output(prompts, responses, tokenizer)
    input_ids = tok["input_ids"].to(device)
    labels = tok["labels"].to(device)
    return run_get_response_log_probs(model, input_ids, labels, False)["log_probs"].detach().cpu()


def evaluate_policy(args, model, tokenizer, eval_prompts, eval_ground_truths, reward_fn) -> dict[str, float]:
    if not eval_prompts:
        return {}
    responses = generate_with_transformers(
        model,
        tokenizer,
        eval_prompts,
        group_size=1,
        batch_size=args.rollout_batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
        top_p=1.0,
    )
    _, summary = evaluate_responses(responses, eval_ground_truths, reward_fn)
    return {f"eval_{k}": v for k, v in summary.items()}


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    prompt_name = Path(args.prompt).stem
    run_name = args.run_name or f"{Path(args.model).name}_{prompt_name}_{args.variant}"
    run_dir = args.out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype_from_arg(args.dtype),
        trust_remote_code=True,
    ).to(args.device)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    set_trainable_last_layers(model, args.trainable_last_layers)
    model.train()

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    template = load_prompt_template(args.prompt)
    train_examples = load_gsm8k_jsonl(args.train_path, limit=args.train_limit, seed=args.seed, shuffle=True)
    eval_examples = load_gsm8k_jsonl(args.eval_path, limit=args.eval_limit)
    eval_prompts = render_prompts(eval_examples, template)
    eval_ground_truths = [example.final_answer for example in eval_examples]
    reward_fn = get_reward_fn(prompt_name, args.reward_mode)

    vllm_server = None
    if args.rollout_backend == "vllm":
        from cs336_alignment.vllm_utils import VLLMServer

        vllm_server = VLLMServer(
            model_id=args.model,
            port=args.vllm_port,
            gpu=args.vllm_gpu,
            tensor_parallel_size=args.vllm_tensor_parallel_size,
            seed=args.seed,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        )
        vllm_server.start()
        if args.sync_vllm_weights:
            vllm_server.init_weight_sync(args.device)
            vllm_server.sync_policy_weights(model)

    logs_path = run_dir / "log.jsonl"
    t0 = time.perf_counter()
    try:
        for step in tqdm(range(1, args.steps + 1), desc="grpo"):
            batch = random.sample(train_examples, args.prompts_per_step)
            prompts = render_prompts(batch, template)
            if args.rollout_backend == "vllm":
                completions = vllm_server.generate_completions(
                    prompts,
                    {
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "max_tokens": args.max_new_tokens,
                        "n": args.group_size,
                        "seed": args.seed + step,
                    },
                    batch_size=args.rollout_batch_size,
                )
                responses = [completion.text for completion in completions]
            else:
                responses = generate_with_transformers(
                    model,
                    tokenizer,
                    prompts,
                    group_size=args.group_size,
                    batch_size=args.rollout_batch_size,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )

            repeated_prompts = [prompt for prompt in prompts for _ in range(args.group_size)]
            repeated_ground_truths = [example.final_answer for example in batch for _ in range(args.group_size)]
            old_log_probs = None
            if args.importance_reweighting_method != "none":
                old_log_probs = compute_old_log_probs(model, tokenizer, repeated_prompts, responses)

            rollout_batch_size = len(responses)
            loss, metadata = run_grpo_train_step(
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                max_grad_norm=args.max_grad_norm,
                reward_fn=reward_fn,
                repeated_prompts=repeated_prompts,
                rollout_responses=responses,
                repeated_ground_truths=repeated_ground_truths,
                group_size=args.group_size,
                importance_reweighting_method=args.importance_reweighting_method,
                old_log_probs=old_log_probs,
                cliprange=None if args.importance_reweighting_method in ("none", "noclip") else args.cliprange,
                **variant_kwargs(args, rollout_batch_size),
            )
            if vllm_server is not None and args.sync_vllm_weights:
                vllm_server.sync_policy_weights(model)

            _, rollout_summary = evaluate_responses(responses, repeated_ground_truths, reward_fn)
            record = {
                "step": step,
                "wall": round(time.perf_counter() - t0, 2),
                "loss": float(loss.detach().cpu()),
                **rollout_summary,
                **{
                    key: (float(value.detach().cpu()) if isinstance(value, torch.Tensor) else value)
                    for key, value in metadata.items()
                },
            }
            if args.eval_every and step % args.eval_every == 0:
                record.update(evaluate_policy(args, model, tokenizer, eval_prompts, eval_ground_truths, reward_fn))
            with open(logs_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(json.dumps(record, ensure_ascii=False))

            if args.save_every and step % args.save_every == 0:
                ckpt_dir = run_dir / f"checkpoint-{step:06d}"
                model.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)
    finally:
        if vllm_server is not None:
            vllm_server.stop()


if __name__ == "__main__":
    main()
