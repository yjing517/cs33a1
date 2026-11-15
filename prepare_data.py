
from pathlib import Path
import pickle

import numpy as np

from cs336_basics.tokenizer import Tokenizer


VOCAB_PATH = Path("outputs/tinystories_5M_10k.pkl")
TEXT_PATH = Path("tests/fixtures/tinystories_sample_5M.txt")
OUTPUT_DIR = Path("outputs/lm_data")

SPECIAL_TOKEN = "<|endoftext|>"


def main():
    # 1. 加载 BPE 词表
    with open(VOCAB_PATH, "rb") as f:
        vocab, merges = pickle.load(f)

    tokenizer = Tokenizer(
        vocab=vocab,
        merges=merges,
        special_tokens=[SPECIAL_TOKEN],
    )

    print("Vocab size:", len(tokenizer.vocab))

    # 2. 读取文本
    text = TEXT_PATH.read_text(encoding="utf-8")

    # 3. 按故事结束标记划分
    stories = text.split(SPECIAL_TOKEN)
    stories = [s for s in stories if s.strip()]

    assert len(stories) >= 2, "Not enough stories"

    # 固定随机种子，保证划分可复现
    rng = np.random.default_rng(42)
    rng.shuffle(stories)

    split = int(len(stories) * 0.9)
    train_stories = stories[:split]
    val_stories = stories[split:]

    # 4. 分别编码，避免训练集和验证集重叠
    def encode_stories(items):
        ids = []
        for story in items:
            ids.extend(tokenizer.encode(story))
            ids.extend(tokenizer.encode(SPECIAL_TOKEN))
        return np.asarray(ids, dtype=np.uint16)

    train_ids = encode_stories(train_stories)
    val_ids = encode_stories(val_stories)

    assert len(train_ids) > 17
    assert len(val_ids) > 17

    assert train_ids.max() < len(tokenizer.vocab)
    assert val_ids.max() < len(tokenizer.vocab)

    # 5. 保存
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.save(OUTPUT_DIR / "train.npy", train_ids)
    np.save(OUTPUT_DIR / "val.npy", val_ids)

    print("Train stories:", len(train_stories))
    print("Val stories:", len(val_stories))
    print("Train tokens:", len(train_ids))
    print("Val tokens:", len(val_ids))
    print("Saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
