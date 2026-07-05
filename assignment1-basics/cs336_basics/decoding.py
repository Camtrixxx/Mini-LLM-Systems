"""Text generation from a trained TransformerLM (CS336 Assignment 1, section 6)."""

from __future__ import annotations

import torch
from torch import Tensor

from cs336_basics.model import TransformerLM, softmax


@torch.no_grad()
def generate(
    model: TransformerLM,
    prompt_ids: list[int],
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_id: int | None = None,
    device: str | torch.device = "cpu",
) -> list[int]:
    """Autoregressively sample a completion for `prompt_ids`.

    - temperature: softmax temperature (0 = greedy decoding).
    - top_p: nucleus sampling threshold (1.0 = no truncation).
    - Stops at `eos_id` (not included in the output) or `max_new_tokens`.

    Returns the generated token IDs (without the prompt).
    """
    model.eval()
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated: list[int] = []

    for _ in range(max_new_tokens):
        # Truncate the context to the model's maximum length (keep the tail).
        context = ids[:, -model.context_length :]
        logits = model(context)[0, -1]  # (vocab_size,)

        if temperature == 0.0:
            next_id = int(logits.argmax())
        else:
            probs = softmax(logits / temperature, dim=-1)
            if top_p < 1.0:
                probs = _truncate_top_p(probs, top_p)
            next_id = int(torch.multinomial(probs, num_samples=1))

        if eos_id is not None and next_id == eos_id:
            break
        generated.append(next_id)
        ids = torch.cat(
            [ids, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=-1
        )

    return generated


def _truncate_top_p(probs: Tensor, top_p: float) -> Tensor:
    """Zero out everything outside the smallest set of tokens whose cumulative
    probability reaches `top_p`, then renormalize."""
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    # Keep tokens up to and including the one that crosses the threshold.
    keep = cumulative - sorted_probs < top_p
    filtered = torch.zeros_like(probs)
    filtered[sorted_idx[keep]] = sorted_probs[keep]
    return filtered / filtered.sum()
