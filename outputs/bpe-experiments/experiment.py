import argparse
import pickle
import resource
import time
from pathlib import Path

from cs336_basics.bpe import train_bpe


def validate(vocab, merges):
    special = b"<|endoftext|>"
    assert list(vocab) == list(range(len(vocab)))
    assert len(vocab) == 257 + len(merges)
    assert all(vocab[i] == bytes([i]) for i in range(256))
    assert vocab[256] == special and list(vocab.values()).count(special) == 1
    available = set(vocab[i] for i in range(256))
    for index, (left, right) in enumerate(merges, 257):
        assert left in available and right in available
        assert vocab[index] == left + right
        available.add(vocab[index])
    longest = max(len(token) for token in available)
    print(f"vocab={len(vocab)} merges={len(merges)} max_bytes={longest}")
    for token in sorted(token for token in available if len(token) == longest):
        print(repr(token.decode("utf-8", errors="backslashreplace")))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--vocab-size", type=int, choices=(10000, 32000), default=10000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is None:
        with args.input.open("rb") as stream:
            validate(*pickle.load(stream))
        return
    args.output.mkdir(parents=True)
    start = time.perf_counter()
    vocab, merges = train_bpe(args.input, args.vocab_size, ["<|endoftext|>"])
    print(f"input_bytes={args.input.stat().st_size} train_seconds={time.perf_counter()-start:.3f} peak_rss_kib={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}")
    with (args.output / "bpe.pkl").open("xb") as stream:
        pickle.dump((vocab, merges), stream)
    with (args.output / "bpe.pkl").open("rb") as stream:
        validate(*pickle.load(stream))
    if len(vocab) < args.vocab_size:
        print("Corpus exhausted before reaching requested vocabulary size.")


if __name__ == "__main__":
    main()
