"""Fully-Sharded Data Parallel (FSDP), a.k.a. ZeRO stage 3.

Weights of Linear/Embedding submodules are sharded along dim 0 across ranks; all
other parameters (e.g. RMSNorm) are replicated. Each rank stores only its shard
as the master (fp32) parameter. A forward_pre_hook all-gathers the shards into
the full weight (optionally cast to compute_dtype to save bandwidth/compute) just
in time for the layer; the full weight is kept through backward so autograd can
accumulate a full-shaped gradient, which finish_gradient_synchronization() then
reduce-scatters back to a per-rank shard gradient (in fp32). Replicated parameter
gradients are all-reduced. All collectives average by dividing by world_size.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from cs336_basics.model import Embedding, Linear


def _pad_rows(t: torch.Tensor, padded_rows: int) -> torch.Tensor:
    if t.shape[0] == padded_rows:
        return t
    return F.pad(t, (0, 0, 0, padded_rows - t.shape[0]))


class FSDP(nn.Module):
    def __init__(self, module: nn.Module, compute_dtype: torch.dtype | None = None):
        super().__init__()
        self.module = module
        self.compute_dtype = compute_dtype
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self._sharded = []      # list of (submodule, full_rows, padded_rows, shard_rows)
        self._replicated = []   # list of Parameters

        sharded_ids = set()
        for m in module.modules():
            if isinstance(m, (Linear, Embedding)) and m.weight.requires_grad:
                self._shard_module(m)
                sharded_ids.add(id(m.weight))
        for p in module.parameters():
            if p.requires_grad and id(p) not in sharded_ids:
                self._replicated.append(p)

    def _shard_module(self, m: nn.Module) -> None:
        w = m.weight.data                      # (full_rows, cols) fp32 master
        full_rows, cols = w.shape
        shard_rows = (full_rows + self.world_size - 1) // self.world_size
        padded_rows = shard_rows * self.world_size
        w_pad = _pad_rows(w, padded_rows)
        shard = w_pad[self.rank * shard_rows:(self.rank + 1) * shard_rows].contiguous()
        m.weight.data = shard                  # master = this rank's shard (fp32)
        self._sharded.append((m, full_rows, padded_rows, shard_rows))
        m.register_forward_pre_hook(self._make_gather_hook())

    def _make_gather_hook(self):
        def hook(m, _inp):
            master = m.weight.data             # current fp32 shard (post optimizer step)
            shard_c = master.to(self.compute_dtype) if self.compute_dtype else master
            gathered = torch.empty(
                (shard_c.shape[0] * self.world_size, shard_c.shape[1]),
                dtype=shard_c.dtype, device=shard_c.device,
            )
            dist.all_gather_into_tensor(gathered, shard_c.contiguous())
            info = self._info(m)
            full = gathered[:info[1]]          # unpad to full_rows
            m._fsdp_master = master            # keep fp32 shard for restore
            m.weight.data = full               # compute uses full (compute_dtype) weight
        return hook

    def _info(self, m):
        for rec in self._sharded:
            if rec[0] is m:
                return rec
        raise KeyError

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        # Sharded weights: reduce-scatter the full grad into this rank's shard grad.
        for m, full_rows, padded_rows, shard_rows in self._sharded:
            grad = m.weight.grad                       # (full_rows, cols), compute_dtype
            cols = m._fsdp_master.shape[1]
            grad_pad = _pad_rows(grad.float(), padded_rows).contiguous()
            shard_grad = torch.empty((shard_rows, cols), dtype=torch.float32, device=grad.device)
            dist.reduce_scatter_tensor(shard_grad, grad_pad, op=dist.ReduceOp.SUM)
            shard_grad /= self.world_size
            m.weight.data = m._fsdp_master             # restore fp32 shard
            m.weight.grad = shard_grad                 # shard-shaped fp32 grad
            del m._fsdp_master
        # Replicated params: all-reduce (average) the grad.
        for p in self._replicated:
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                p.grad /= self.world_size


def fsdp_gather_full_params(fsdp_model: FSDP) -> dict[str, torch.Tensor]:
    """Reconstruct full (unsharded) parameters, keyed by module.named_parameters() names."""
    sharded = {id(m.weight): (fr, pr, sr) for m, fr, pr, sr in fsdp_model._sharded}
    out: dict[str, torch.Tensor] = {}
    for name, p in fsdp_model.module.named_parameters():
        if id(p) in sharded:
            full_rows, padded_rows, shard_rows = sharded[id(p)]
            gathered = torch.empty((padded_rows, p.shape[1]), dtype=p.dtype, device=p.device)
            dist.all_gather_into_tensor(gathered, p.data.contiguous())
            out[name] = gathered[:full_rows].clone()
        else:
            out[name] = p.data.clone()
    return out
