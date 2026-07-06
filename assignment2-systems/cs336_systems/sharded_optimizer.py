"""Optimizer state sharding (ZeRO stage 1 / P_os).

Each parameter is assigned to one rank (round-robin). Every rank still holds the
full parameters and computes full gradients, but only builds optimizer state
(e.g. AdamW's m, v) for the parameters it owns — cutting optimizer-state memory
by ~1/world_size. In step(), each rank updates only its shard, then the updated
parameters are broadcast from their owner so every rank holds the full, in-sync
model afterwards.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist


class ShardedOptimizer(torch.optim.Optimizer):
    def __init__(self, params, optimizer_cls: type[torch.optim.Optimizer], **kwargs: Any):
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = kwargs
        # Populated by add_param_group (called from super().__init__).
        self._all_params: list[torch.Tensor] = []
        self._param_owner: dict[int, int] = {}   # id(param) -> owner rank
        self.inner: torch.optim.Optimizer | None = None
        super().__init__(params, {})

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        # Register with the outer optimizer (so param_groups / zero_grad see all params).
        super().add_param_group(param_group)
        # Round-robin assign each new param to a rank; collect the ones we own.
        local: list[torch.Tensor] = []
        for p in param_group["params"]:
            owner = len(self._all_params) % self.world_size
            self._param_owner[id(p)] = owner
            self._all_params.append(p)
            if owner == self.rank:
                local.append(p)
        if not local:
            return
        if self.inner is None:
            self.inner = self.optimizer_cls(local, **self.optimizer_kwargs)
        else:
            self.inner.add_param_group({"params": local})

    def step(self, closure=None, **kwargs):
        loss = self.inner.step(closure, **kwargs) if self.inner is not None else None
        # Each parameter was updated only on its owner rank; sync to all ranks.
        for p in self._all_params:
            dist.broadcast(p.data, src=self._param_owner[id(p)])
        return loss
