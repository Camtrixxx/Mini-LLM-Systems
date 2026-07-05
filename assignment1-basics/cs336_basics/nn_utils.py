"""Training utilities: loss, gradient clipping, batch sampling, checkpointing
(CS336 Assignment 1, sections 4-5)."""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, BinaryIO

import numpy.typing as npt
import torch
from torch import Tensor


def cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    """Average cross-entropy loss.

    Args:
        logits: (..., vocab_size) unnormalized logits.
        targets: (...) integer class indices.

    Computed as -logit[target] + logsumexp(logits), with the max subtracted
    first for numerical stability (cancels in both terms).
    """
    logits = logits - logits.amax(dim=-1, keepdim=True)
    log_sum_exp = torch.log(torch.exp(logits).sum(dim=-1))
    target_logits = torch.gather(logits, -1, targets.unsqueeze(-1)).squeeze(-1)
    return (log_sum_exp - target_logits).mean()


def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
    eps: float = 1e-6,
) -> None:
    """Clip the combined L2 norm of all gradients to at most `max_l2_norm`,
    modifying `parameter.grad` in place."""
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return
    total_norm = torch.sqrt(sum(g.pow(2).sum() for g in grads))
    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + eps)
        for g in grads:
            g.mul_(scale)


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[Tensor, Tensor]:
    """Sample `batch_size` random (input, target) sequence pairs for language
    modeling. Targets are the inputs shifted right by one position."""
    starts = torch.randint(0, len(dataset) - context_length, (batch_size,))
    x = torch.stack(
        [torch.from_numpy(dataset[i : i + context_length].astype("int64")) for i in starts]
    )
    y = torch.stack(
        [
            torch.from_numpy(dataset[i + 1 : i + 1 + context_length].astype("int64"))
            for i in starts
        ]
    )
    return x.to(device), y.to(device)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    """Serialize model/optimizer state and the current iteration number."""
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": iteration,
        },
        out,
    )


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """Restore model/optimizer state from a checkpoint; return the iteration."""
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]
