from __future__ import annotations

import json
import os
import random
import re
from typing import Any, Callable, Literal

import torch
from torch import Tensor
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase



def run_tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    """Tokenize the prompt and output strings, and construct a mask aligned with
    labels that is 1 for response tokens and 0 for other tokens (prompt or padding).

    Args:
        prompt_strs: list[str]
            List of prompt strings.
        output_strs: list[str]
            List of output strings.
        tokenizer: PreTrainedTokenizer
            Tokenizer to use for tokenization.

    Returns:
        dict[str, torch.Tensor].
            Let prompt_and_output_lens be a list containing the lengths of the
            concatenated tokenized prompt and output strings. Then the returned
            dictionary should have the following keys:

            input_ids
                torch.Tensor of shape
                (batch_size, max(prompt_and_output_lens) - 1): the tokenized
                prompt and output strings, with the final token sliced off.
            labels
                torch.Tensor of shape
                (batch_size, max(prompt_and_output_lens) - 1): shifted input
                ids, i.e., the input ids without the first token.
            response_mask
                torch.Tensor of shape
                (batch_size, max(prompt_and_output_lens) - 1): a mask aligned
                with labels, with value 1 where the corresponding label token
                is part of the response and 0 otherwise.
    """
    prompt_ids = [tokenizer.encode(p) for p in prompt_strs]
    output_ids = [tokenizer.encode(o) for o in output_strs]
    full = [p + o for p, o in zip(prompt_ids, output_ids)]
    max_len = max(len(f) for f in full)
    pad_id = tokenizer.pad_token_id
    batch = len(full)
    # Pad every sequence to max_len, then slice the final/first column off the
    # whole batch (input_ids = padded[:, :-1], labels = padded[:, 1:]).
    padded = torch.full((batch, max_len), pad_id, dtype=torch.long)
    response_mask = torch.zeros((batch, max_len - 1), dtype=torch.long)
    for i, f in enumerate(full):
        n, p = len(f), len(prompt_ids[i])
        padded[i, :n] = torch.tensor(f, dtype=torch.long)
        # labels[j] = padded[j+1]; a response token when j+1 in [p, n-1] -> j in [p-1, n-2].
        response_mask[i, p - 1 : n - 1] = 1
    return {"input_ids": padded[:, :-1], "labels": padded[:, 1:], "response_mask": response_mask}


def run_get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool,
) -> dict[str, torch.Tensor]:
    """Get per-token conditional log-probabilities (given the previous tokens)
    from a causal language model, and optionally the entropy of the model's
    next-token distribution.

    Args:
        model: PreTrainedModel
            HuggingFace model used for scoring (placed on the correct device
            and in inference mode if gradients should not be computed).
        input_ids: torch.Tensor
            shape (batch_size, sequence_length), concatenated prompt + response
            tokens as produced by your tokenization method.
        labels: torch.Tensor
            shape (batch_size, sequence_length), labels as produced by your
            tokenization method.
        return_token_entropy: bool
            If True, also return per-token entropy.

    Returns:
        dict[str, torch.Tensor].
            "log_probs"
                shape (batch_size, sequence_length), conditional
                log-probabilities log p_(theta)(x_t | x_(<t)).
            "token_entropy"
                optional, shape (batch_size, sequence_length), per-token
                entropy for each position (present only if
                return_token_entropy=True).
    """
    logits = model(input_ids).logits  # (batch, seq, vocab)
    log_probs_all = torch.log_softmax(logits, dim=-1)
    log_probs = torch.gather(log_probs_all, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    out = {"log_probs": log_probs}
    if return_token_entropy:
        # H = -sum_v p_v log p_v, computed from log-probs for stability.
        out["token_entropy"] = -(log_probs_all.exp() * log_probs_all).sum(dim=-1)
    return out


def run_compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute rewards for a list of rollout responses, along with metadata for
    the reward components.

    Args:
        reward_fn: Callable[[str, str], dict[str, float]]
            Scores the rollout responses against the ground truths, producing
            a dict with keys "reward", "format_reward", and "answer_reward".
        rollout_responses: list[str]
            Rollouts from the policy. The length of this list is
            rollout_batch_size = n_prompts_per_rollout_batch * group_size.
        repeated_ground_truths: list[str]
            The ground truths for the examples. The length of this list is
            rollout_batch_size, because the ground truth for each example is
            repeated group_size times.

    Returns:
        tuple[torch.Tensor, dict[str, float]].
            raw_rewards
                shape (rollout_batch_size,). Unnormalized rewards for each
                rollout response.
            metadata
                Reward statistics to log. At minimum, include the mean total
                and format rewards over the rollout batch.
    """
    rewards, formats, answers = [], [], []
    for response, gt in zip(rollout_responses, repeated_ground_truths):
        r = reward_fn(response, gt)
        rewards.append(r["reward"])
        formats.append(r.get("format_reward", 0.0))
        answers.append(r.get("answer_reward", 0.0))
    raw_rewards = torch.tensor(rewards, dtype=torch.float32)
    metadata = {
        "mean_reward": float(raw_rewards.mean()),
        "mean_format_reward": float(sum(formats) / len(formats)),
        "mean_answer_reward": float(sum(answers) / len(answers)),
    }
    return raw_rewards, metadata


def run_compute_group_normalized_rewards(
    raw_rewards: torch.Tensor,
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute advantages by applying the requested baseline and normalization
    within each group.

    Args:
        raw_rewards: torch.Tensor
            shape (rollout_batch_size,). Unnormalized rewards for each rollout
            response, where rollout_batch_size = n_prompts_per_rollout_batch *
            group_size.
        group_size: int
            Number of responses per question (group).
        baseline: Literal["mean", "none"]
            For this problem, support mean, which subtracts the per-group mean
            reward. Later, none will mean no baseline subtraction.
        advantage_eps: float
            Small constant to avoid division by zero in normalization.
        advantage_normalizer: Literal["std", "none", "mean"]
            For this problem, support std, which divides by the per-group
            standard deviation. Later, none will mean no normalization and
            mean will mean divide by the per-group mean reward.

    Returns:
        tuple[torch.Tensor, dict[str, float]].
            advantages
                shape (rollout_batch_size,). Group-normalized rewards for each
                rollout response.
            metadata
                your choice of other statistics to log (e.g. mean, std, max/min
                of rewards).
    """
    groups = raw_rewards.reshape(-1, group_size).float()
    group_mean = groups.mean(dim=1, keepdim=True)
    centered = groups - group_mean if baseline == "mean" else groups
    if advantage_normalizer == "std":
        group_std = groups.std(dim=1, keepdim=True)  # unbiased (Bessel), matches reference
        advantages = centered / (group_std + advantage_eps)
    elif advantage_normalizer == "mean":
        advantages = centered / (group_mean + advantage_eps)
    else:  # "none"
        advantages = centered
    advantages = advantages.reshape(-1)
    metadata = {
        "mean_reward": float(raw_rewards.mean()),
        "std_reward": float(raw_rewards.std()),
        "max_reward": float(raw_rewards.max()),
        "min_reward": float(raw_rewards.min()),
    }
    return advantages, metadata


def run_compute_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy-gradient loss at every token, where
    raw_rewards_or_advantages is either the raw reward or an
    already-normalized advantage.

    Args:
        raw_rewards_or_advantages: torch.Tensor
            Shape (batch_size,) or (batch_size, 1), scalar reward/advantage for
            each rollout response.
        policy_log_probs: torch.Tensor
            Shape (batch_size, sequence_length), logprobs for each token.
        importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"]
            "none": no importance reweighting; "noclip": apply importance
            reweighting without clipping; "grpo": do PPO/GRPO-style
            token-level reweighting and clipping; "gspo": do GSPO-style
            sequence-level reweighting and clipping.
        old_log_probs: torch.Tensor | None
            Required unless importance_reweighting_method = "none"; shape
            (batch_size, sequence_length).
        cliprange: float | None = None
            Clip parameter epsilon, required when importance_reweighting_method
            is "grpo" or "gspo".
        response_mask: torch.Tensor | None = None
            Optional shape (batch_size, sequence_length) mask over response
            tokens. Required for GSPO implementations that average the
            sequence-level log-ratio over response tokens only.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
            per_token_policy_gradient_loss
                Shape (batch_size, sequence_length), the per-token
                policy-gradient loss (to be aggregated across the batch and
                sequence dimensions in the training loop).
            metadata
                Statistics from the underlying loss call, such as
                clip-fraction components.
    """
    adv = raw_rewards_or_advantages
    if adv.dim() == 1:
        adv = adv.unsqueeze(-1)  # (batch, 1) -> broadcast over tokens
    metadata: dict[str, torch.Tensor] = {}

    if importance_reweighting_method == "none":
        per_token = -adv * policy_log_probs
        return per_token, metadata

    if importance_reweighting_method == "noclip":
        ratio = torch.exp(policy_log_probs - old_log_probs)
        return -adv * ratio, metadata

    if importance_reweighting_method == "grpo":
        # PPO-style token-level clipped surrogate.
        ratio = torch.exp(policy_log_probs - old_log_probs)
        unclipped = ratio * adv
        clipped = torch.clamp(ratio, 1 - cliprange, 1 + cliprange) * adv
        per_token = -torch.minimum(unclipped, clipped)
        metadata["clip_fraction"] = (unclipped > clipped).float().mean().detach()
        return per_token, metadata

    if importance_reweighting_method == "gspo":
        # Sequence-level importance ratio: mean log-ratio over response tokens.
        log_ratio = policy_log_probs - old_log_probs
        m = response_mask.float()
        seq_log_ratio = (log_ratio * m).sum(dim=-1, keepdim=True) / m.sum(dim=-1, keepdim=True).clamp_min(1.0)
        s = torch.exp(seq_log_ratio)  # (batch, 1)
        unclipped = s * adv
        clipped = torch.clamp(s, 1 - cliprange, 1 + cliprange) * adv
        per_token = -torch.minimum(unclipped, clipped) * torch.ones_like(policy_log_probs)
        metadata["clip_fraction"] = (unclipped > clipped).float().mean().detach()
        return per_token, metadata

    raise ValueError(f"unknown importance_reweighting_method: {importance_reweighting_method}")


def run_aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss: torch.Tensor,
    mask: torch.Tensor,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> torch.Tensor:
    """Aggregate the per-token policy-gradient loss according to the response
    mask and loss-normalization strategy.

    Args:
        per_token_policy_gradient_loss: torch.Tensor
            Shape (batch_size, sequence_length), the per-token policy-gradient
            loss (to be aggregated across the batch and sequence dimensions in
            the training loop).
        mask
            torch.Tensor of shape (batch_size, sequence_length) denoting which
            positions should be included in the loss.
        loss_normalization: Literal["sequence", "constant"] = "sequence"
            "sequence": average loss over each sequence, then average over
            sequences; "constant": normalize total loss by a constant.
        normalization_constant: int | None = None
            The constant to divide total loss by; required if
            loss_normalization = "constant".

    Returns:
        loss: torch.Tensor
            A scalar containing the average loss. Make sure you can later call
            backward on this loss.
    """
    m = mask.float()
    masked = per_token_policy_gradient_loss * m
    if loss_normalization == "sequence":
        # Average over each sequence's response tokens, then over sequences.
        per_seq = masked.sum(dim=-1) / m.sum(dim=-1).clamp_min(1.0)
        return per_seq.mean()
    # "constant": total masked loss divided by a fixed constant.
    return masked.sum() / normalization_constant


def run_grpo_train_step(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    optimizer: torch.optim.Optimizer,
    gradient_accumulation_steps: int,
    max_grad_norm: float | None,
    reward_fn: Callable[[str, str], dict[str, float]],
    repeated_prompts: list[str],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    """Execute forward-and-backward passes, with gradient_accumulation_steps
    microbatches.

    Args:
        model: PreTrainedModel
            HuggingFace model to train.
        tokenizer: PreTrainedTokenizer
            Tokenizer to use for tokenization.
        optimizer: Optimizer
            Optimizer for the model.
        gradient_accumulation_steps: int
            Number of microbatches per optimizer step.
        max_grad_norm: float | None
            If not None, clip the gradient norm to this value before calling
            optimizer.step().
        reward_fn: Callable[[str, str], dict[str, float]]
            Scores the rollout responses against the ground truths, producing
            a dict with keys "reward", "format_reward", and "answer_reward".
        repeated_prompts: list[str]
            The prompts for the examples. The length of this list is
            rollout_batch_size, because the prompt for each example is repeated
            group_size times.
        rollout_responses: list[str]
            Rollouts from the policy. The length of this list is
            rollout_batch_size = n_prompts_per_rollout_batch * group_size.
        repeated_ground_truths: list[str]
            The ground truths for the examples. The length of this list is
            rollout_batch_size, because the ground truth for each example is
            repeated group_size times.
        group_size: int
            Number of responses per question (group).
        baseline: Literal["mean", "none"]
            If mean, subtract the per-group mean reward; if none, do nothing.
        advantage_eps: float
            Small constant to avoid division by zero in normalization.
        advantage_normalizer: Literal["std", "none", "mean"]
            If std, divide by the per-group standard deviation; if none, do
            nothing; if mean, divide by the per-group mean reward.
        importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"]
            "none": no importance reweighting; "noclip": apply importance
            reweighting without clipping; "grpo": do PPO/GRPO-style token-level
            reweighting and clipping; "gspo": do GSPO-style sequence-level
            reweighting and clipping.
        old_log_probs: torch.Tensor | None
            Required unless importance_reweighting_method = "none"; shape
            (batch_size, sequence_length).
        cliprange: float | None = None
            Clip parameter epsilon, required when importance_reweighting_method
            is "grpo" or "gspo".
        loss_normalization: Literal["sequence", "constant"] = "sequence"
            "sequence": average loss over each sequence, then average over
            sequences; "constant": normalize total loss by a constant (fixed
            for all of training).
        normalization_constant: int | None = None
            The constant to divide total loss by; required if
            loss_normalization = "constant".

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
            loss
                scalar tensor. The batch loss, adjusted for gradient
                accumulation. We return this so we can log it.
            metadata
                Dict with metadata from the underlying loss call, gradient norm
                before clipping, and any other statistics you might want to log.
    """
    device = next(model.parameters()).device
    # Tokenize the full rollout batch and compute group-normalized advantages.
    tok = run_tokenize_prompt_and_output(repeated_prompts, rollout_responses, tokenizer)
    input_ids = tok["input_ids"].to(device)
    labels = tok["labels"].to(device)
    response_mask = tok["response_mask"].to(device)
    raw_rewards, _ = run_compute_rollout_rewards(reward_fn, rollout_responses, repeated_ground_truths)
    advantages, adv_meta = run_compute_group_normalized_rewards(
        raw_rewards, group_size, baseline, advantage_eps, advantage_normalizer
    )
    advantages = advantages.to(device)

    rollout_batch_size = input_ids.shape[0]
    micro = rollout_batch_size // gradient_accumulation_steps

    # "sequence" normalization makes each microbatch a mean, so combining across
    # microbatches averages (÷ accum). "constant" makes each microbatch a partial
    # sum with a shared denominator, so combining just sums (no ÷ accum).
    scale = gradient_accumulation_steps if loss_normalization == "sequence" else 1

    optimizer.zero_grad(set_to_none=True)
    total_loss = torch.tensor(0.0, device=device)
    metadata: dict[str, torch.Tensor | float] = dict(adv_meta)
    for step in range(gradient_accumulation_steps):
        sl = slice(step * micro, (step + 1) * micro)
        policy_log_probs = run_get_response_log_probs(
            model, input_ids[sl], labels[sl], return_token_entropy=False
        )["log_probs"]
        per_token_loss, loss_meta = run_compute_policy_gradient_loss(
            raw_rewards_or_advantages=advantages[sl],
            policy_log_probs=policy_log_probs,
            importance_reweighting_method=importance_reweighting_method,
            old_log_probs=None if old_log_probs is None else old_log_probs[sl].to(device),
            cliprange=cliprange,
            response_mask=response_mask[sl],
        )
        loss = run_aggregate_loss_across_microbatch(
            per_token_loss, response_mask[sl], loss_normalization, normalization_constant
        )
        (loss / scale).backward()
        total_loss = total_loss + loss.detach() / scale
        metadata.update({k: v for k, v in loss_meta.items()})

    if max_grad_norm is not None:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        metadata["grad_norm"] = grad_norm.detach()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return total_loss, metadata


"""
The below adapters are used in the optional 
RLHF / safety part of the Alignment assignment.
"""


def get_packed_sft_dataset(
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: str | os.PathLike,
    seq_length: int,
    shuffle: bool,
) -> Dataset:
    """
    Given a tokenizer and a path to a dataset with instruction-tuning examples,
    construct a PyTorch Dataset for language modeling. The examples should be
    packed, i.e., all sequences in the dataset are of a constant length (`seq_length`).

    Args:
        tokenizer: transformers.PreTrainedTokenizerBase
            Transformers tokenizer to use in tokenizing and encoding text.
        dataset_path: str
            Path to file with instruction-tuning examples.
        seq_length: int
            Number of tokens to include in each example.
        shuffle: bool
            If true, shuffle the documents before packing them into examples.

    Returns:
        PyTorch Dataset for language modeling. Each example in this dataset is a dictionary of
        with keys "input_ids" and "labels" (both tensors of shape (seq_length, )).
        "input_ids" contains the token IDs for the language modeling inputs, and "labels" contains
        the token IDs for the language modeling labels.
    """
    template = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:\n{response}"
    )
    records = [json.loads(line) for line in open(dataset_path)]
    if shuffle:
        random.shuffle(records)
    # Each document = BOS + Alpaca-formatted (prompt, response) + EOS, concatenated
    # into one token stream, then packed into non-overlapping seq_length windows.
    stream: list[int] = []
    for r in records:
        text = template.format(instruction=r["prompt"], response=r["response"])
        stream += [tokenizer.bos_token_id] + tokenizer.encode(text, add_special_tokens=False) + [tokenizer.eos_token_id]

    n = (len(stream) - 1) // seq_length
    examples = [
        {
            "input_ids": torch.tensor(stream[j * seq_length : (j + 1) * seq_length], dtype=torch.long),
            "labels": torch.tensor(stream[j * seq_length + 1 : (j + 1) * seq_length + 1], dtype=torch.long),
        }
        for j in range(n)
    ]

    class _PackedSFTDataset(Dataset):
        def __len__(self):
            return len(examples)

        def __getitem__(self, idx):
            return examples[idx]

    return _PackedSFTDataset()


def run_iterate_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
):
    """
    Given a PyTorch Dataset, return an iterable over batches of size `batch_size`.
    Iterating through the returned iterable should constitute one epoch over the Dataset.

    Args:
        dataset: Dataset
            Dataset to emit batches from.
        batch_size: int
            Number of examples to include per batch.
        shuffle: bool
            If true, shuffle examples before batching them.

    Returns:
        Iterable over batches, where each batch has size `batch_size`.
    """
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def run_parse_mmlu_response(
    mmlu_example: dict[str, Any],
    model_output: str,
) -> str | None:
    """
    Given an MMLU example and a model output, parse the model output into a
    predicted option letter (i.e., 'A', 'B', 'C', or 'D'). If the model output
    cannot be parsed into a prediction option letter, return None.

    mmlu_example: dict[str, Any]
        Dictionary with an MMLU example. Contains the following keys:
        - "subject": str with the subject of the question.
        - "question": str with the text of the question.
        - "options": list[str] with the four answer options (in order).
                     The first option refers to letter "A", the second to "B", etc.
        - "answer": str with the option of the correct answer (e.g., "A")
    model_output: str
        str with the model's output to the MMLU example.

    Returns:
        str (one of "A", "B", "C", or "D") if the model output can be parsed into a prediction,
        else None.
    """
    # Prefer an explicit "answer is X" style statement; fall back to a standalone letter.
    m = re.search(r"answer\s*(?:is|:)?\s*\(?\s*([A-D])\b", model_output, flags=re.IGNORECASE)
    if m is None:
        m = re.search(r"\b([A-D])\b", model_output)
    return m.group(1).upper() if m else None


def run_parse_gsm8k_response(
    model_output: str,
) -> str | None:
    """
    Given a GSM8K model output, parse the model output into a predicted numeric answer by
    taking the last number that occurs in the output.

    model_output: str
        str with the model's output to a GSM8K example.

    Returns:
        str with the predicted numeric answer if the model output can be parsed into a prediction,
        else None.
    """
    # Take the last number in the output (integers/decimals, optional thousands commas, sign).
    numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", model_output)
    if not numbers:
        return None
    return numbers[-1].replace(",", "")


def run_compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:
    """
    Given two language models (`lm`, and the "reference model" `lm_ref`),
    their tokenizer, the DPO beta hyperparameter, a prompt and a pair
    of responses to the prompt, computes the value of the DPO loss for this example.

    lm: torch.nn.Module
        Language model being trained.
    lm_ref: torch.nn.Module
        Reference language model.
    tokenizer: PreTrainedTokenizerBase
        Tokenizer for both language models.
    beta: float
        DPO beta hyperparameter.
    prompt: str
        Prompt for this instance of preference pair.
    response_chosen: str
        Preferred response to the prompt.
    response_rejected: str
        Rejected response to the prompt.

    Returns:
        torch.Tensor with the DPO loss for this example.
    """
    from pathlib import Path

    import cs336_alignment

    template = (Path(cs336_alignment.__file__).parent / "prompts_safety" / "alpaca_sft.prompt").read_text()

    def sequence_logprob(m: torch.nn.Module, text: str) -> torch.Tensor:
        # Unconditional log-prob of the full concat(prompt-template, response)+EOS;
        # the prompt part cancels in the chosen-minus-rejected difference.
        ids = tokenizer.encode(text) + [tokenizer.eos_token_id]
        ids = torch.tensor([ids], device=m.device)
        logits = m(ids).logits[0]  # (T, vocab)
        log_probs = torch.log_softmax(logits[:-1], dim=-1)
        targets = ids[0, 1:]
        return log_probs[torch.arange(targets.shape[0]), targets].sum()

    text_w = template.format(instruction=prompt, response=response_chosen)
    text_l = template.format(instruction=prompt, response=response_rejected)

    with torch.no_grad():
        ref_w = sequence_logprob(lm_ref, text_w).to(lm.device)
        ref_l = sequence_logprob(lm_ref, text_l).to(lm.device)
    pol_w = sequence_logprob(lm, text_w)
    pol_l = sequence_logprob(lm, text_l)

    logits = beta * ((pol_w - ref_w) - (pol_l - ref_l))
    return -torch.nn.functional.logsigmoid(logits)
