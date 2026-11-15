import gzip
from pathlib import Path

root = Path("outputs/bpe-experiments")
with gzip.open(root / "owt-prefix.gz") as stream:
    sample = stream.read(16 * 1024**2)
marker = b"<|endoftext|>"
boundary = sample.rindex(marker) + len(marker)
with (root / "owt-16m.txt").open("xb") as stream:
    stream.write(sample[:boundary])
