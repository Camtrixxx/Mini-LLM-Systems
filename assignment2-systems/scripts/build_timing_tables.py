"""Build markdown timing tables from out/bench_section2.jsonl."""
import json
from collections import defaultdict

rows = defaultdict(dict)  # (size) -> {(mode,dtype): (mean_ms, std_ms) or "OOM"}
order = ["small", "medium", "large", "xl", "10B"]
with open("out/bench_section2.jsonl") as f:
    for line in f:
        d = json.loads(line)
        key = (d["mode"], d["dtype"])
        if d.get("oom"):
            rows[d["size"]][key] = "OOM"
        else:
            rows[d["size"]][key] = (d["mean_s"] * 1000, d["std_s"] * 1000)


def cell(size, mode, dtype):
    v = rows.get(size, {}).get((mode, dtype))
    if v is None:
        return "—"
    if v == "OOM":
        return "OOM"
    return f"{v[0]:.1f}±{v[1]:.1f}"


print("### (b) fp32 timing (ms/step, 5 warmup + 10 measured, ctx=512 batch=4)\n")
print("| size | forward | forward+backward | full (w/ AdamW) |")
print("|---|---|---|---|")
for s in order:
    print(f"| {s} | {cell(s,'forward','fp32')} | {cell(s,'forward_backward','fp32')} | {cell(s,'full','fp32')} |")

print("\n### (c) fp32 vs BF16 mixed precision (ms/step)\n")
print("| size | fwd fp32 | fwd bf16 | fwd+bwd fp32 | fwd+bwd bf16 |")
print("|---|---|---|---|---|")
for s in order:
    print(f"| {s} | {cell(s,'forward','fp32')} | {cell(s,'forward','bf16')} | "
          f"{cell(s,'forward_backward','fp32')} | {cell(s,'forward_backward','bf16')} |")
