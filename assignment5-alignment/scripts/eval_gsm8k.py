"""Evaluate a causal LM on the bundled GSM8K split.

Examples:
    uv run python scripts/eval_gsm8k.py \
        --model Qwen/Qwen2.5-Math-7B \
        --prompt r1_zero --backend vllm --limit 1319

    CUDA_VISIBLE_DEVICES=0 uv run --no-sync python scripts/eval_gsm8k.py \
        --model Qwen/Qwen2.5-Math-7B \
        --prompt question_only --backend transformers --limit 32
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from cs336_alignment.gsm8k import (
    GSM8K_DIR,
    evaluate_responses,
    get_reward_fn,
    load_gsm8k_jsonl,
    load_prompt_template,
    render_prompts,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen2.5-Math-7B")
    p.add_argument("--split", choices=["train", "test"], default="test")
    p.add_argument("--data-path", type=Path, default=None)
    p.add_argument("--prompt", default="r1_zero", help="prompt name or path")
    p.add_argument("--reward-mode", choices=["strict", "numeric"], default="strict")
    p.add_argument("--backend", choices=["vllm", "transformers"], default="vllm")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--n", type=int, default=1, help="samples per prompt")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="bfloat16")
    p.add_argument("--out-dir", type=Path, default=Path("out/a5"))
    p.add_argument("--vllm-port", type=int, default=8000)
    p.add_argument("--vllm-gpu", default="0")
    p.add_argument("--vllm-tensor-parallel-size", type=int, default=1)
    p.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--launch-server", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def _dtype(name: str):
    return {
        "auto": "auto",
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def generate_transformers(args: argparse.Namespace, prompts: list[str]) -> list[str]:
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    if args.device == "auto":
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=_dtype(args.dtype),
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=_dtype(args.dtype),
            trust_remote_code=True,
        ).to(args.device)
    model.eval()

    outputs: list[str] = []
    do_sample = args.temperature > 0
    for start in tqdm(range(0, len(prompts), args.batch_size), desc="generate"):
        batch_prompts = prompts[start : start + args.batch_size]
        encoded = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(model.device)
        if encoded["input_ids"].shape[1] == 0:
            fallback_id = tokenizer.eos_token_id or tokenizer.pad_token_id or 0
            encoded["input_ids"] = torch.full(
                (len(batch_prompts), 1), fallback_id, dtype=torch.long, device=model.device
            )
            encoded["attention_mask"] = torch.ones_like(encoded["input_ids"])
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                do_sample=do_sample,
                temperature=args.temperature if do_sample else None,
                top_p=args.top_p if do_sample else None,
                max_new_tokens=args.max_new_tokens,
                num_return_sequences=args.n,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                remove_invalid_values=True,
                renormalize_logits=True,
            )
        input_width = encoded["input_ids"].shape[1]
        for i in range(len(batch_prompts)):
            for sample_idx in range(args.n):
                row = i * args.n + sample_idx
                outputs.append(tokenizer.decode(generated[row, input_width:], skip_special_tokens=True))
    return outputs


def generate_vllm(args: argparse.Namespace, prompts: list[str]) -> list[str]:
    from cs336_alignment.vllm_utils import VLLMServer

    server = VLLMServer(
        model_id=args.model,
        port=args.vllm_port,
        gpu=args.vllm_gpu,
        tensor_parallel_size=args.vllm_tensor_parallel_size,
        seed=args.seed,
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        launch_server=args.launch_server,
    )
    server.start()
    try:
        completions = server.generate_completions(
            prompts,
            {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_tokens": args.max_new_tokens,
                "n": args.n,
                "seed": args.seed,
            },
            batch_size=args.batch_size,
        )
        return [completion.text for completion in completions]
    finally:
        if args.launch_server:
            server.stop()


def main() -> None:
    args = parse_args()
    data_path = args.data_path or GSM8K_DIR / f"{args.split}.jsonl"
    examples = load_gsm8k_jsonl(data_path, limit=args.limit)
    template = load_prompt_template(args.prompt)
    prompts = render_prompts(examples, template)
    prompt_name = Path(args.prompt).stem
    reward_fn = get_reward_fn(prompt_name, args.reward_mode)

    if args.backend == "vllm":
        responses = generate_vllm(args, prompts)
    else:
        responses = generate_transformers(args, prompts)

    repeated_examples = [example for example in examples for _ in range(args.n)]
    repeated_prompts = [prompt for prompt in prompts for _ in range(args.n)]
    ground_truths = [example.final_answer for example in repeated_examples]
    records, summary = evaluate_responses(responses, ground_truths, reward_fn)

    rows = []
    for prompt, example, record in zip(repeated_prompts, repeated_examples, records):
        rows.append(
            {
                "question": example.question,
                "prompt": prompt,
                "response": record.pop("response"),
                **record,
            }
        )

    run_name = f"eval_{Path(args.model).name}_{prompt_name}_{args.reward_mode}_{args.backend}_{args.split}"
    if args.limit is not None:
        run_name += f"_n{args.limit}"
    result_dir = args.out_dir / run_name
    result_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(result_dir / "samples.jsonl", rows)
    summary = {
        **summary,
        "model": args.model,
        "prompt": prompt_name,
        "reward_mode": args.reward_mode,
        "backend": args.backend,
        "split": args.split,
        "num_examples": len(examples),
        "num_completions": len(responses),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "n": args.n,
    }
    (result_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
