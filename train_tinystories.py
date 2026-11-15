
import time
import pickle
from pathlib import Path

from cs336_basics.bpe import train_bpe

input_path = "tests/fixtures/tinystories_sample_5M.txt"
vocab_size = 10000

output_path = Path("outputs/tinystories_5M_10k.pkl")
output_path.parent.mkdir(parents=True, exist_ok=True)

start = time.perf_counter()

vocab, merges = train_bpe(
    input_path=input_path,
    vocab_size=vocab_size,
    special_tokens=["<|endoftext|>"],
)

elapsed = time.perf_counter() - start

with open(output_path, "wb") as f:
    pickle.dump((vocab, merges), f)

longest = max(vocab.values(), key=len)

print(f"词表大小：{len(vocab)}")
print(f"合并次数：{len(merges)}")
print(f"训练耗时：{elapsed:.2f} 秒")
print(f"最长 token：{longest!r}")
print(f"最长 token 字节数：{len(longest)}")
print(f"保存成功：{output_path}")
