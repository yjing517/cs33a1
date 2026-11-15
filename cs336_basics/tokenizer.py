
import regex as re

import json
from pathlib import Path

#   uv run python -m cs336_basics.tokenizer


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

vocab = {
    97: b"a",
    98: b"b",
    99: b"c",
    256: b"ab",
    257: b"bc",
    258: b"abc",
}

merges = [
    (b"a", b"b"),   # rank 0
    (b"b", b"c"),   # rank 1
    (b"ab", b"c"),  # rank 2
]

class Tokenizer:
    def __init__(
        self, 
        vocab: dict[int, bytes],
        merges:list[tuple[bytes, bytes]],
        special_tokens: list[str] |None = None,
    ):

        self.vocab = vocab.copy()
        self.merges = merges    
        self.special_tokens = special_tokens or []

        #bytes->tokenId
        self.byte_to_token_id = {
            token:idx
            for idx, token in self.vocab.items()
        }

        #左字节串，右字节串->合并顺序
        self.merge_ranks = {
            merge:rank
            for rank, merge in enumerate(merges)
        }

        # 添加特殊标记    

        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes not in self.byte_to_token_id:
                new_id=max(self.vocab,default=-1)+1
                self.vocab[new_id] = token_bytes
                self.byte_to_token_id[token_bytes] = new_id 
    


    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ):
        # GPT-2 byte -> Unicode 映射
        def bytes_to_unicode():
            bs = (
                list(range(ord("!"), ord("~") + 1))
                + list(range(ord("¡"), ord("¬") + 1))
                + list(range(ord("®"), ord("ÿ") + 1))
            )
            cs = bs.copy()
            n = 0

            for b in range(256):
                if b not in bs:
                    bs.append(b)
                    cs.append(256 + n)
                    n += 1

            return dict(zip(bs, map(chr, cs)))

        unicode_to_byte = {
            char: byte
            for byte, char in bytes_to_unicode().items()
        }

        def to_bytes(token: str) -> bytes:
            return bytes(
                unicode_to_byte[char]
                for char in token
            )

        # 1. 加载 vocab.json
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            raw_vocab = json.load(f)

        vocab = {
            token_id: to_bytes(token)
            for token, token_id in raw_vocab.items()
        }

        # 2. 加载 merges.txt
        merges = []

        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line or line.startswith("#"):
                    continue

                left, right = line.split()
                merges.append(
                    (to_bytes(left), to_bytes(right))
                )

        # 3. 创建 Tokenizer
        return cls(vocab, merges, special_tokens)







    
    def encode_iterable(self, iterable):
        for text in iterable:
            yield from self.encode(text)
    def _encode_piece(self, text: str) -> list[int]:
        tokens=[
            bytes([b])
            for b in text.encode("utf-8")
        ]

        while len(tokens)>1:
            candidates = []

            for i in range(len(tokens)-1):
                pair=(tokens[i], tokens[i+1])

                if pair in self.merge_ranks:
                    candidates.append((self.merge_ranks[pair], i))

            if not candidates:
                break
            
            _,index = min(candidates)

            tokens[index:index+2]=[
                tokens[index]+tokens[index+1]
            ]

        return [
            self.byte_to_token_id[token]
            for token in tokens
        ]

    def decode(self, ids: list[int]) -> str:
        data = b"".join(
            self.vocab[idx] for idx in ids
        )
        return data.decode("utf-8", errors="replace")

    def encode(self, text: str) -> list[int]:
        result = []

        if not self.special_tokens:
            for piece in pre_tokenize(text):
                result.extend(self._encode_piece(piece))
            return result

        specials=sorted(
            self.special_tokens,
            key=len,
            reverse=True,
        )

        pattern = "(" + "|".join(
            re.escape(token) for token in specials
        ) + ")"

        # 分割普通文本和特殊 token
        parts = re.split(pattern, text)

        for part in parts:
            if not part:
                continue

            # 特殊 token 直接查 ID
            if part in self.special_tokens:
                result.append(
                    self.byte_to_token_id[part.encode("utf-8")]
                )
            else:
                # 普通文本走预分词 + BPE
                for piece in pre_tokenize(part):
                    result.extend(self._encode_piece(piece))

        return result

if __name__ == "__main__":
    tokenizer = Tokenizer(vocab, merges)

    print(tokenizer._encode_piece("abc"))  # [258]
    print(tokenizer._encode_piece("ab"))   # [256]
    print(tokenizer._encode_piece("bc"))   # [257]

    print(tokenizer.encode("abc"))        # [258]
    print(tokenizer.decode([258]))        # abc
    tokenizer = Tokenizer(
        vocab,
        merges,
        special_tokens=["<|endoftext|>"]
    )

    print(tokenizer.encode("ab<|endoftext|>c"))
