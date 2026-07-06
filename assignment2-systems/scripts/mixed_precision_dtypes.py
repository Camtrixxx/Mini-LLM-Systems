"""Problem (benchmarking_mixed_precision)(a): observe component dtypes under FP16 autocast."""
import torch
from torch import nn

class ToyModel(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()
    def forward(self, x):
        h1 = self.fc1(x); print("fc1 output:", h1.dtype)
        h1 = self.relu(h1)
        h2 = self.ln(h1); print("ln  output:", h2.dtype)
        logits = self.fc2(h2); print("logits    :", logits.dtype)
        return logits

dev = "cuda"
m = ToyModel(20, 5).to(dev)
x = torch.randn(8, 20, device=dev)
tgt = torch.randint(0, 5, (8,), device=dev)
print("param dtype (outside autocast):", m.fc1.weight.dtype)
with torch.autocast(device_type="cuda", dtype=torch.float16):
    print("param dtype (inside autocast):", m.fc1.weight.dtype)
    logits = m(x)
    loss = torch.nn.functional.cross_entropy(logits, tgt)
    print("loss      :", loss.dtype)
loss.backward()
print("fc1.grad  :", m.fc1.weight.grad.dtype)
