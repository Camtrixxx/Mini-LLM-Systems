"""Distributed Data Parallel containers.

Three variants used across Section 5:
  - DDPIndividual: overlaps gradient all-reduce with the backward pass by firing
    an async all-reduce per parameter as its grad becomes ready (this is what the
    adapter get_ddp returns; also the ddp_overlap_individual_parameters answer).
  - DDPNaive: all-reduces each parameter's grad after the whole backward pass.
  - DDPFlat: a single all-reduce over all flattened gradients after backward.

All average gradients across ranks (all-reduce SUM / world_size, since gloo has
no AVG op) so the averaged gradient matches a single-process step over the union
of the per-rank shards. Parameters are broadcast from rank 0 at construction so
every rank starts from identical weights.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn


def _broadcast_params(module: nn.Module) -> None:
    for p in module.parameters():
        dist.broadcast(p.data, src=0)
    for b in module.buffers():
        dist.broadcast(b.data, src=0)


class DDPIndividual(nn.Module):
    """Overlapping DDP: async all-reduce per parameter during backward."""

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()
        self._handles: list[tuple] = []
        _broadcast_params(module)
        for p in module.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(self._grad_hook)

    def _grad_hook(self, param: torch.Tensor) -> None:
        handle = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
        self._handles.append((handle, param))

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        for handle, param in self._handles:
            handle.wait()
            param.grad /= self.world_size
        self._handles.clear()


class DDPNaive(nn.Module):
    """Naive DDP: all-reduce every parameter grad after the backward pass."""

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()
        _broadcast_params(module)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        for p in self.module.parameters():
            if p.requires_grad and p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                p.grad /= self.world_size


class DDPFlat(nn.Module):
    """Minimal DDP: one all-reduce over all flattened grads after backward."""

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()
        _broadcast_params(module)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        grads = [p.grad for p in self.module.parameters() if p.requires_grad and p.grad is not None]
        if not grads:
            return
        flat = torch._utils._flatten_dense_tensors(grads)
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat /= self.world_size
        for g, synced in zip(grads, torch._utils._unflatten_dense_tensors(flat, grads)):
            g.copy_(synced)
