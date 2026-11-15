import pickle
from pathlib import Path

with Path("outputs/tinystories_5M_10k.pkl").open("rb") as stream:
    original = pickle.load(stream)
with Path("outputs/bpe-experiments/optimized-5m/bpe.pkl").open("rb") as stream:
    optimized = pickle.load(stream)
assert original[0] == optimized[0] and original[1] == optimized[1]
print("Exact vocab dict and ordered merge equality")

root = Path("outputs/bpe-experiments")
with (root / "tiny-prefix.txt").open("rb") as stream:
    for size in (50, 200):
        stream.seek(0)
        sample = stream.read(size * 1024**2)
        boundary = sample.rindex(b"<|endoftext|>") + len(b"<|endoftext|>")
        with (root / f"tiny-{size}m.txt").open("xb") as output:
            output.write(sample[:boundary])
