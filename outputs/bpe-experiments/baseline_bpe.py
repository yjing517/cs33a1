from pathlib import Path
import regex as re
from collections import Counter, defaultdict

#        uv run python -m cs336_basics.bpe

def init_vocab() -> dict[int, bytes]:
    vocab = {}

    for i in range(256):
        vocab[i] = bytes([i])

    return vocab


def pre_tokenize(text: str) -> list[str]:
    pattern = (
        r"""'(?:[sdmt]|ll|ve|re)"""
        r"""| ?\p{L}+"""
        r"""| ?\p{N}+"""
        r"""| ?[^\s\p{L}\p{N}]+"""
        r"""|\s+(?!\S)"""
        r"""|\s+"""
    )

    return re.findall(pattern, text)

def to_byte_tokens(tokens:list[str])->list[tuple[bytes,...]]:
    result=[]

    for token in tokens:
        data=token.encode("utf-8")
        byte_tuple=tuple(bytes([b])for b in data)
        result.append(byte_tuple
        
        )
    return result
def read_corpus(input_path:str | Path) -> str:
    with open(input_path,"r",encoding="utf-8") as f:
        text=f.read()
    return text

def count_pairs(words: Counter) -> Counter:
    counts = Counter()

    for word, freq in words.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            counts[pair] += freq

    return counts

def find_best_pair(counts: Counter):
    return max(counts, key=lambda pair:(counts[pair],pair))

def merge_word(word,pair):
    result=[]
    i=0
    while i < len(word):
        if (
            i + 1 < len(word)
            and (word[i], word[i + 1]) == pair
        ):
            result.append(word[i] + word[i + 1])
            i += 2
        else:
            result.append(word[i])
            i += 1

    return tuple(result)

def word_pairs(word):
    return Counter(zip(word, word[1:]))


def train_bpe(input_path, vocab_size, special_tokens):
    # ① 初始化 256 个单字节 token
    vocab = init_vocab()

    # ② 添加特殊 token
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")

    # ③ 读取训练语料
    text = read_corpus(input_path)

    # ④ 预分词并转换成字节 token
    segments = split_special_tokens(text, special_tokens)

    tokens = []
    for segment in segments:
        tokens.extend(pre_tokenize(segment))

    words = Counter(to_byte_tokens(tokens))

    word_list = list(words.items())
    pair_counts = Counter()
    pair_to_words = defaultdict(set)
    for wid, (word, freq) in enumerate(word_list):
        for pair, count in word_pairs(word).items():
            pair_counts[pair] += count * freq
            pair_to_words[pair].add(wid)

    merges = []
    while len(vocab) < vocab_size and pair_counts:
        best_pair = find_best_pair(pair_counts)
        merges.append(best_pair)
        vocab[len(vocab)] = best_pair[0] + best_pair[1]

        for wid in list(pair_to_words[best_pair]):
            word, freq = word_list[wid]
            old_counts = word_pairs(word)
            new_word = merge_word(word, best_pair)
            new_counts = word_pairs(new_word)
            for pair in old_counts.keys() | new_counts.keys():
                old = old_counts[pair]
                new = new_counts[pair]
                pair_counts[pair] += (new - old) * freq
                if old and not new:
                    pair_to_words[pair].remove(wid)
                elif new and not old:
                    pair_to_words[pair].add(wid)
                if not pair_counts[pair]:
                    del pair_counts[pair]
                    del pair_to_words[pair]
            word_list[wid] = (new_word, freq)

    return vocab, merges

def split_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    if not special_tokens:
        return [text]

    pattern = "|".join(
        re.escape(token)
        for token in sorted(special_tokens, key=len, reverse=True)
    )

    return re.split(pattern, text)

if __name__ == "__main__":
    test_path = Path("/tmp/bpe_test.txt")
    test_path.write_text(
        "abab<|endoftext|>abab",
        encoding="utf-8"
    )

    vocab, merges = train_bpe(
        input_path=test_path,
        vocab_size=259,
        special_tokens=["<|endoftext|>"]
    )

    print("词表大小:", len(vocab))
    print("特殊 token:", vocab[256])
    print("合并规则:", merges)
    print("新增 token:", {
        i: vocab[i]
        for i in range(257, len(vocab))
    })