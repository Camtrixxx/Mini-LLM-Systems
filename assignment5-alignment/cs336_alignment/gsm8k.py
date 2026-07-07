"""GSM8K helpers for Assignment 5 experiments.

The assignment tests keep most GRPO primitives in ``tests/adapters.py``.  This
module holds experiment-facing utilities: loading the bundled GSM8K jsonl files,
rendering prompt templates, extracting final answers, and grading completions.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import re

from cs336_alignment.drgrpo_grader import grade, question_only_reward_fn, r1_zero_reward_fn


REPO_ROOT = Path(__file__).resolve().parents[1]
PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
GSM8K_DIR = REPO_ROOT / "data" / "gsm8k"


@dataclass(frozen=True)
class GSM8KExample:
    question: str
    answer: str
    final_answer: str


def extract_gsm8k_final_answer(answer: str) -> str:
    """Return the final answer after GSM8K's ``####`` marker."""
    marker = "####"
    if marker in answer:
        return answer.split(marker)[-1].strip().replace(",", "")
    return answer.strip().replace(",", "")


def load_gsm8k_jsonl(
    path: str | Path,
    *,
    limit: int | None = None,
    seed: int = 0,
    shuffle: bool = False,
) -> list[GSM8KExample]:
    examples: list[GSM8KExample] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            answer = record["answer"]
            examples.append(
                GSM8KExample(
                    question=record["question"],
                    answer=answer,
                    final_answer=extract_gsm8k_final_answer(answer),
                )
            )
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(examples)
    if limit is not None:
        examples = examples[:limit]
    return examples


def load_prompt_template(prompt: str | Path) -> str:
    """Load either a named prompt from ``cs336_alignment/prompts`` or a path."""
    prompt_path = Path(prompt)
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8")
    if not prompt_path.suffix:
        prompt_path = prompt_path.with_suffix(".prompt")
    prompt_path = PROMPT_DIR / prompt_path.name
    return prompt_path.read_text(encoding="utf-8")


def render_prompts(examples: Iterable[GSM8KExample], template: str) -> list[str]:
    return [template.format(question=example.question) for example in examples]


def reward_fn_for_prompt(prompt_name: str) -> Callable[[str, str], dict[str, float]]:
    """Choose the strict reward used by the corresponding assignment prompt."""
    if prompt_name.startswith("question_only"):
        return question_only_reward_fn
    return r1_zero_reward_fn


def numeric_fallback_reward_fn(response: str, ground_truth: str) -> dict[str, float]:
    """Lenient GSM8K reward: grade the last numeric answer in the response.

    This is useful for instruction-tuned models that solve the problem but do
    not obey the assignment's strict XML/boxed answer format.
    """
    strict = question_only_reward_fn(response, ground_truth)
    if strict["reward"] > 0:
        return strict
    numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", response)
    if not numbers:
        return {"format_reward": 0.0, "answer_reward": 0.0, "reward": 0.0}
    answer = numbers[-1].replace(",", "")
    correct = grade(answer, ground_truth, fast=True)
    return {
        "format_reward": 1.0,
        "answer_reward": float(correct),
        "reward": float(correct),
    }


def get_reward_fn(prompt_name: str, mode: str = "strict") -> Callable[[str, str], dict[str, float]]:
    if mode == "strict":
        return reward_fn_for_prompt(prompt_name)
    if mode == "numeric":
        return numeric_fallback_reward_fn
    raise ValueError(f"unknown reward mode: {mode}")


def evaluate_responses(
    responses: list[str],
    ground_truths: list[str],
    reward_fn: Callable[[str, str], dict[str, float]],
) -> tuple[list[dict], dict[str, float]]:
    records: list[dict] = []
    for response, gt in zip(responses, ground_truths):
        reward = reward_fn(response, gt)
        records.append({"response": response, "ground_truth": gt, **reward})

    n = max(len(records), 1)
    summary = {
        "accuracy": sum(r["answer_reward"] for r in records) / n,
        "format_accuracy": sum(r["format_reward"] for r in records) / n,
        "mean_reward": sum(r["reward"] for r in records) / n,
    }
    return records, summary


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
