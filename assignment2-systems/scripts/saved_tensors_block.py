"""memory_profiling(f) support: measure bytes saved-for-backward by one xl TransformerBlock,
and the size of its parameter-gradient tensors, for comparison."""
import torch
from cs336_basics.model import RotaryEmbedding, TransformerBlock
from cs336_systems.model_configs import MODEL_SIZES

cfg = MODEL_SIZES["xl"]; d_model, d_ff, nh = cfg.d_model, cfg.d_ff, cfg.num_heads
ctx, batch = 2048, 4
dev = "cuda"
block = TransformerBlock(d_model=d_model, d_ff=d_ff, num_heads=nh,
                         positional_encoder=RotaryEmbedding(context_length=ctx, dim=d_model // nh)).to(dev)
x = torch.randn((batch, ctx, d_model), device=dev, requires_grad=True)

total = 0
def pack(t):
    global total
    if not isinstance(t, torch.nn.Parameter):
        total += t.numel() * t.element_size()
    return t
def unpack(t): return t

with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
    y = block(x)
print(f"saved-for-backward per xl TransformerBlock: {total/1024**2:.1f} MiB")

n_params = sum(p.numel() for p in block.parameters())
print(f"block params: {n_params/1e6:.1f}M  -> grad tensors (fp32): {n_params*4/1024**2:.1f} MiB")
print(f"all {cfg.num_layers} layers saved: {total*cfg.num_layers/1024**3:.1f} GiB")
