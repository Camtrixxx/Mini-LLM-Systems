"""Problem (mixed_precision_accumulation): run the given accumulation snippet."""
import torch

print("fp32 acc, fp32 addend:")
s = torch.tensor(0, dtype=torch.float32)
for _ in range(1000):
    s += torch.tensor(0.01, dtype=torch.float32)
print(" ", s.item())

print("fp16 acc, fp16 addend:")
s = torch.tensor(0, dtype=torch.float16)
for _ in range(1000):
    s += torch.tensor(0.01, dtype=torch.float16)
print(" ", s.item())

print("fp32 acc, fp16 addend (implicit upcast on +=):")
s = torch.tensor(0, dtype=torch.float32)
for _ in range(1000):
    s += torch.tensor(0.01, dtype=torch.float16)
print(" ", s.item())

print("fp32 acc, fp16 addend explicitly cast to fp32:")
s = torch.tensor(0, dtype=torch.float32)
for _ in range(1000):
    x = torch.tensor(0.01, dtype=torch.float16)
    s += x.type(torch.float32)
print(" ", s.item())
