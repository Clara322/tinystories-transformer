import itertools
import os
from typing import Optional

import regex as re

from cs336_basics.pretokenization_example import find_chunk_boundaries

def create_vocab():
    vocab_set: set[tuple[bytes, ...]] = set()
    vocab: dict[int, bytes] = {}
    vocab[0] = b"<|endoftext|>"
    count = 1
    vocab_set.add((b"<|endoftext|>",))
    for i in range(0, 256):
        vocab_set.add(bytes((i,)))
        vocab[count] = bytes((i,))
        count += 1
    # print(vocab_set)
    # print(vocab)
    return (vocab_set, vocab)

def token_frequencies(frequencies, utf_encoded) -> dict[tuple[bytes, ...], int]:
    key = tuple([bytes([b]) for b in utf_encoded])
    if key in frequencies.keys():
        count = frequencies[key]
        frequencies[key] = count + 1
    else:
        frequencies[key] = 1
    #print(frequencies)
    return frequencies

def merges(bp_freq, frequencies):
    
    max_count = 0
    max_token = (b"0", b"0")
    for token in frequencies.keys():
        # if (new_token and new_token not in token):
        #     continue
        #print(token)
        no_appearances = frequencies[token]
        token_bytes = [b for b in token]
        i = 0
        while i < len(token_bytes) - 1:
            f = token_bytes[i]
            s = token_bytes[i + 1]
            ##print(f, s)

            count = 0
            if (f, s) in bp_freq.keys():
                count = bp_freq[(f, s)]
            bp_freq[(f, s)] = count + no_appearances
            if ((count + no_appearances) > max_count):
                max_count = count + no_appearances
                max_token = (f, s)
            elif ((count + no_appearances) == max_count):
                max_token = max(max_token, (f, s))

            i += 1
    #print(max_count, max_token)
    #print(bp_freq)

    (max_f, max_s) = max_token
    (f, s) = max_f, max_s
    new_token = (max_f + max_s)
    #print("NEW TOKEN ADDED " + ("" + max_f + max_s))
    return (f, s, new_token, max_count)

def new_merges(bp_freq, frequencies, new_token, old_f, old_s):
    max_count = 0
    max_token = (b"0", b"0")
    del bp_freq[(old_f, old_s)]
    for token in frequencies.keys():
        no_appearances = frequencies[token]
        token_bytes = [b for b in token]
        i = 0
        while i < len(token_bytes):
            if (token_bytes[i] == new_token):
                if (i - 1 >= 0 and token_bytes[i - 1] != new_token):
                    f = token_bytes[i - 1]
                    if ((f, old_f) in bp_freq):
                        bp_freq[(f, old_f)] -= no_appearances
                    s = token_bytes[i]
                    count = bp_freq.get((f, s), 0)
                    bp_freq[(f, s)] = count + no_appearances
                if (i + 1 < len(token_bytes)):
                    s = token_bytes[i + 1]
                    if (s == new_token):
                        if ((old_s, old_f) in bp_freq):
                            bp_freq[(old_s, old_f)] -= no_appearances
                    else:
                        if ((old_s, s) in bp_freq):
                            bp_freq[(old_s, s)] -= no_appearances
                    f = token_bytes[i]
                    count = bp_freq.get((f, s), 0)
                    bp_freq[(f, s)] = count + no_appearances
            i += 1
    for key in bp_freq.keys():
        val = bp_freq[key]
        if (val > max_count):
            max_count = val
            max_token = key
        elif(val == max_count):
            max_token = max(max_token, key)
    (max_f, max_s) = max_token
    (f, s) = max_f, max_s
    new_token = (max_f + max_s)
    return (f, s, new_token, max_count)

def new_vocab(frequencies, f, s, new_token):
    for token in list(frequencies.keys()):
        i = 0
        count = 0
        bytes_f = []
        found = False
        while i < len(token):
            if ((i + 1) < len(token) and token[i] == f and token[i + 1] == s):
                count = frequencies[token]
                bytes_f.append(new_token)
                i += 2
                found = True
            else:
                bytes_f.append(token[i])
                i += 1
        if found:
            del frequencies[token]
            frequencies[tuple(bytes_f)] = count
    return frequencies


from multiprocessing import Pool

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def process_chunk(args):
    file_path, start, end, special_tokens = args

    with open(file_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    if special_tokens:
        pattern = "|".join(re.escape(tok) for tok in special_tokens)
        sub_chunks = re.split(pattern, chunk)
    else:
        sub_chunks = [chunk]

    frequencies: dict[tuple[bytes, ...], int] = {}
    for sub_chunk in sub_chunks:
        for match in re.finditer(PAT, sub_chunk):
            utf_encoded = match.group().encode("utf-8")
            frequencies = token_frequencies(frequencies, utf_encoded)
    return frequencies

def run_tokenizer(input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str]):
    text_input = "low low low low low lower lower widest widest widest newest newest newest newest newest newest"

    (vocab_set, vocab) = create_vocab()
    merges_list: list[tuple[bytes, bytes]] = []
    size_vocab = len(vocab_set)

    num_processes = 4

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        tasks = [
            (input_path, start, end, special_tokens)
            for start, end in zip(boundaries[:-1], boundaries[1:])
        ]

        frequencies: dict[tuple[bytes, ...], int] = {}
        with Pool(num_processes) as pool:
            for partial in pool.map(process_chunk, tasks):
                for key, count in partial.items():
                    frequencies[key] = frequencies.get(key, 0) + count

        bp_freq: dict[tuple[bytes, ...], int] = {}
        t = True
        while True:
            if (t):
                bp_freq: dict[tuple[bytes, ...], int] = {}
                (first, second, new_token, max_count) = merges(bp_freq, frequencies)
                t = False
            else:
                (first, second, new_token, max_count) = new_merges(bp_freq, frequencies, n, f, s)
            if (max_count <= 1 or size_vocab >= vocab_size):
                break
            frequencies = new_vocab(frequencies, first, second, new_token)
            merges_list.append((first, second))
    
            if ((new_token,) not in vocab_set):
                vocab_set.add((new_token,))
                vocab[size_vocab] = new_token
                size_vocab += 1
                n = new_token
                f = first
                s = second

    return (vocab, merges_list)


    



run_tokenizer('data/TinyStoriesV2-GPT4-valid.txt', 500, ["<|endoftext|>"])
