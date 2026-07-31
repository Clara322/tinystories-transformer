from ast import literal_eval
import regex as re
from typing import Iterable, Iterator

from cs336_basics.pretokenization_example import find_chunk_boundaries


class Tokenizer:
    vocab: dict[int, bytes]
    byte_to_id: dict[bytes, int]
    merges: list[tuple[bytes, bytes]]
    special_tokens: list[str] | None = None

    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        next_id = max(self.vocab.keys()) + 1
        for tok in (special_tokens or []):
            b = tok.encode("utf-8")
            if b not in self.vocab.values():
                self.vocab[next_id] = b
                next_id += 1
        self.merges = merges
        self.special_tokens = special_tokens
        self.byte_to_id = {v: k for k, v in self.vocab.items()}

    def encode(self, text: str) -> list[int]:
        encoded = []
        if self.special_tokens:
            sorted_specials = sorted(self.special_tokens, key=len, reverse=True)
            pattern = "|".join(re.escape(token) for token in sorted_specials)
            sub_chunks = re.split("(" + pattern + ")", text)
        else:
            sub_chunks = [text]
        
        special_set = set(self.special_tokens or [])
        for chunk in sub_chunks:
            if chunk in special_set:
                encoded.append(self.byte_to_id[chunk.encode("utf-8")])
                continue
            # print(chunk)
            PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
            matches = (re.finditer(PAT, chunk))
            for match in matches:
                elem = match.group()
                utf_encoded = elem.encode("utf-8")

                i = 0
                new_token = []
                found = False
                token_bytes = [bytes([b]) for b in utf_encoded]
                k = 0
                while (k < len(self.merges)):
                    merge = self.merges[k]
                    i = 0
                    new_token = []
                    found = False
                    while i < len(token_bytes):
                        new_token.append(token_bytes[i])
                        if (i + 1 < len(token_bytes) and not found and (token_bytes[i] == merge[0] and token_bytes[i + 1] == merge[1])):
                            new_token[i] = token_bytes[i] + token_bytes[i+1]
                            j = i + 2
                            while j < len(token_bytes):
                                new_token.append(token_bytes[j])
                                j += 1
                            found = True
                            # print(token_bytes[i], token_bytes[i + 1], new_token)
                            break
                        i += 1
                    if found:
                        found = False
                        token_bytes = new_token
                        new_token = []
                        k = -1
                    k += 1
                i = 0
                # print(encoded)
                while i < len(new_token):
                    encoded.append(self.byte_to_id[bytes(new_token[i])])
                    i += 1
        # print(encoded)
        return encoded


        
    # def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
    #     raise NotImplementedError()
    def decode(self, ids: list[int]) -> str:
        if ids == []:
            return ""
        encoded = self.vocab[ids[0]]
        for i in range(1, len(ids)):
            val = self.vocab[ids[i]]
            encoded = encoded + val
        return encoded.decode("utf-8", errors='replace')


def from_files(vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None) -> Tokenizer:
    vocab: dict[int, bytes] = {}
    merges: list[tuple[bytes, bytes]] = []
    with open(vocab_filepath, "rb") as f:
        lines = [line.rstrip() for line in f]
    for line in lines:
        s = line.split(None, 1)
        vocab[int((s[0]).decode("utf-8"))] = literal_eval((s[1]).decode("utf-8"))
        # print(int((s[0]).decode("utf-8")), literal_eval((s[1]).decode("utf-8")))
    
    with open(merges_filepath, "rb") as f:
        lines = [line.rstrip() for line in f]
    
    for l in lines:
        line = (l.decode("utf-8"))[1:-1]
        s = line.split(", ", 1)
        print(s[0], s[1])
        merges.append((literal_eval(s[0]), literal_eval(s[1])))

    print(vocab)


    return Tokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)


tokenizer = from_files('data/vocab-test.txt', 'data/merges-test.txt')
tokenizer.encode('the cat ate')