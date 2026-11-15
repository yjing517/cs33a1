import gzip
import pickle
from pathlib import Path

from cs336_basics.bpe import merge_word, pre_tokenize

root = Path(__file__).parent
marker = b"<|endoftext|>"


def held_out(stream, offset):
    stream.seek(offset)
    docs = stream.read(1024**2).split(marker)[1:-1]
    assert docs
    return docs


def count_tokens(doc, ranks):
    count = 0
    for text in pre_tokenize(doc.decode("utf-8")):
        word = tuple(bytes([byte]) for byte in text.encode("utf-8"))
        while len(word) > 1:
            pair = min(zip(word, word[1:]), key=lambda pair: ranks.get(pair, len(ranks)))
            if pair not in ranks:
                break
            word = merge_word(word, pair)
        count += len(word)
    return count


with (root / "tiny-prefix.txt").open("rb") as stream:
    tiny = held_out(stream, 200 * 1024**2)
with gzip.open(root / "owt-prefix.gz") as stream:
    owt = held_out(stream, 16 * 1024**2)

for name in ("optimized-200m-recovered", "owt-32k"):
    with (root / name / "bpe.pkl").open("rb") as stream:
        _, merges = pickle.load(stream)
    ranks = {pair: rank for rank, pair in enumerate(merges)}
    assert len(ranks) == len(merges)
    for corpus, documents in (("TinyStories", tiny), ("OpenWebText", owt)):
        size = sum(map(len, documents))
        count = sum(count_tokens(doc, ranks) for doc in documents)
        print(f"{name} on {corpus}: docs={len(documents)} bytes={size} tokens={count} bytes_per_token={size/count:.4f}")

