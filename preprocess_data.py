import argparse
import pickle
import shutil
import tempfile
from itertools import islice
from pathlib import Path

import numpy as np

from cs336_basics.tokenizer import Tokenizer

END_OF_TEXT = "<|endoftext|>"
DEFAULT_TOKENIZER = "outputs/tinystories_5M_10k.pkl"


def load_tokenizer(path):
    # Pickle artifacts must come from a trusted local BPE training run.
    with open(path, "rb") as source:
        vocab, merges = pickle.load(source)
    if END_OF_TEXT.encode() not in vocab.values():
        raise ValueError("Tokenizer vocabulary is missing <|endoftext|>")
    return Tokenizer(vocab, merges, special_tokens=[END_OF_TEXT])


def iter_documents(source):
    pending = []
    limit = 8 * 1024 * 1024
    pending_size = 0
    while line := source.readline(limit + 1):
        if len(line) > limit:
            raise ValueError("Input line exceeds the 8 Mi-character limit")
        parts = line.split(END_OF_TEXT)
        for part in parts[:-1]:
            if pending_size + len(part) > limit:
                raise ValueError("Document exceeds the 8 Mi-character limit")
            pending.append(part)
            yield "".join(pending)
            yield END_OF_TEXT
            pending.clear()
            pending_size = 0
        pending_size += len(parts[-1])
        if pending_size > limit:
            raise ValueError("Document exceeds the 8 Mi-character limit")
        pending.append(parts[-1])
    if pending:
        yield "".join(pending)


def encode_file(source, destination, tokenizer):
    if source.suffix == ".partial":
        raise ValueError(f"Incomplete input: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    if min(tokenizer.vocab) < 0 or max(tokenizer.vocab) > np.iinfo(np.uint16).max:
        raise ValueError("Token IDs must fit in uint16")

    dtype = np.dtype("<u2")
    count = 0
    with tempfile.TemporaryDirectory(dir=destination.parent) as directory:
        staged = Path(directory) / "tokens.npy"
        with tempfile.TemporaryFile(dir=directory) as raw:
            with source.open(encoding="utf-8", newline="") as text:
                ids = tokenizer.encode_iterable(iter_documents(text))
                while (batch := np.fromiter(islice(ids, 65536), dtype=dtype)).size:
                    count += batch.size
                    raw.write(batch.tobytes())
            if not count:
                raise ValueError(f"No tokens in {source}")
            raw.seek(0)
            with staged.open("wb") as output:
                np.lib.format.write_array_header_2_0(
                    output, {"descr": dtype.str, "fortran_order": False, "shape": (count,)}
                )
                shutil.copyfileobj(raw, output, length=1024 * 1024)
        staged.rename(destination)
    return count


def main():
    parser = argparse.ArgumentParser(description="Encode separate train and validation text files.")
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.train.resolve() == args.val.resolve():
        parser.error("Train and validation must be separate files")
    for source, name in ((args.train, "train"), (args.val, "val")):
        if not source.is_file() or source.suffix == ".partial":
            parser.error(f"Expected a complete input file: {source}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("train", "val"):
        if (args.output_dir / f"{name}.npy").exists():
            parser.error(f"Output already exists: {args.output_dir / (name + '.npy')}")
    tokenizer = load_tokenizer(args.tokenizer)
    for source, name in ((args.train, "train"), (args.val, "val")):
        output = args.output_dir / f"{name}.npy"
        print(f"{output}: {encode_file(source, output, tokenizer):,} uint16 tokens", flush=True)


if __name__ == "__main__":
    main()

