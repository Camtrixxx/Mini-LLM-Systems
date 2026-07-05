"""AdamW optimizer and LR schedule (CS336 Assignment 1, section 5)."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable

import torch


class AdamW(torch.optim.Optimizer):
    """AdamW (Loshchilov & Hutter, 2019): Adam with decoupled weight decay."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if not state:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                state["t"] += 1
                t = state["t"]
                m, v = state["m"], state["v"]

                # Update biased first/second moment estimates.
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias-corrected step size.
                step_size = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)
                p.addcdiv_(m, v.sqrt() + eps, value=-step_size)

                # Decoupled weight decay.
                p.mul_(1 - lr * weight_decay)
        return loss


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """Cosine annealing with linear warmup.

    - it < warmup_iters: linear ramp from 0 to max_learning_rate.
    - warmup_iters <= it <= cosine_cycle_iters: cosine decay to min_learning_rate.
    - it > cosine_cycle_iters: min_learning_rate.
    """
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    if it > cosine_cycle_iters:
        return min_learning_rate
    progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    return min_learning_rate + 0.5 * (1 + math.cos(math.pi * progress)) * (
        max_learning_rate - min_learning_rate
    )
