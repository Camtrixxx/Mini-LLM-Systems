"""memory_profiling(e): find the largest allocations in a memory snapshot pickle."""
import pickle
import sys
snap = pickle.load(open(sys.argv[1], "rb"))
# Collect allocated block sizes with their top frames from segments.
allocs = []
for seg in snap.get("segments", []):
    addr = seg.get("address")
    for b in seg.get("blocks", []):
        if b.get("state") == "active_allocated":
            frames = b.get("frames", [])
            top = "; ".join(f"{fr.get('name','?')}" for fr in frames[:3]) if frames else "?"
            allocs.append((b["size"], top))
allocs.sort(reverse=True)
print(f"{len(allocs)} active allocations; top 8 by size:")
for size, top in allocs[:8]:
    print(f"  {size/1024**2:8.1f} MiB | {top[:90]}")
