import itertools
import os

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

def pre_tokenize(text_input):
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pre_tokenized_words = (re.findall(PAT, text_input))
    # pre_tokenized_words = (text_input.split(" "))
    #print(pre_tokenized_words)
    return pre_tokenized_words

def token_encode(pre_tokenized_words):
    utf_encoded = [x.encode("utf-8") for x in pre_tokenized_words] 
    #print(utf_encoded)
    return utf_encoded

def token_frequencies(frequencies, utf_encoded) -> dict[tuple[bytes, ...], int]:
    for i in utf_encoded:
        key = tuple([bytes([b]) for b in i])
        if key in frequencies.keys():
            count = frequencies[key]
            frequencies[key] = count + 1
        else:
            frequencies[key] = 1
    #print(frequencies)
    return frequencies

def merges(frequencies):
    bp_freq: dict[tuple[bytes, ...], int] = {}
    max_count = 0
    max_token = (b"0", b"0")
    for token in frequencies.keys():
        #print(token)
        no_appearances = frequencies[token]
        token_bytes = [b for b in token]
        i = 0
        while i < len(token_bytes) - 1:
            f = token_bytes[i]
            s = token_bytes[i + 1]
            ##print(f, s)

            if (f, s) in bp_freq.keys():
                count = bp_freq[(f, s)]
                bp_freq[(f, s)] = count + no_appearances
                if ((count + no_appearances) > max_count):
                    max_count = count + no_appearances
                    max_token = (f, s)
                elif ((count + no_appearances) == max_count):
                    max_token = max(max_token, (f, s))
            else: 
                bp_freq[(f, s)] = no_appearances
                if (no_appearances > max_count):
                    max_count = no_appearances
                    max_token = (f, s)
                elif ((no_appearances) == max_count):
                    max_token = max(max_token, (f, s))
            i += 1
    #print(max_count, max_token)
    #print(bp_freq)

    (max_f, max_s) = max_token
    (f, s) = max_f, max_s
    new_token = (max_f + max_s)
    #print("NEW TOKEN ADDED " + ("" + max_f + max_s))
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


def run_tokenizer(input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str]):
    text_input = "low low low low low lower lower widest widest widest newest newest newest newest newest newest"

    (vocab_set, vocab) = create_vocab()
    merges_list: list[tuple[bytes, bytes]] = []
    size_vocab = len(vocab_set)

    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        
        frequencies: dict[tuple[bytes, ...], int] = {}

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            if (size_vocab >= vocab_size):
                break
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            pattern = "|".join(re.escape(token) for token in special_tokens)
            sub_chunks = re.split(pattern, chunk)

            for chunk in sub_chunks:
                # print(chunk)

                pre_tokenized_words = pre_tokenize(chunk)
                utf_encoded = token_encode(pre_tokenized_words)
                frequencies = token_frequencies(frequencies, utf_encoded)

        while True:
            (first, second, new_token, max_count) = merges(frequencies)
            if (max_count <= 1 or size_vocab >= vocab_size):
                break
            frequencies = new_vocab(frequencies, first, second, new_token)
            merges_list.append((first, second))
    
            if ((new_token,) not in vocab_set):
                vocab_set.add((new_token,))
                vocab[size_vocab] = new_token
                size_vocab += 1

                    # print(frequencies)
    # print(vocab)
    # print(merges_list)
    return (vocab, merges_list)


    



run_tokenizer('data/TinyStoriesV2-GPT4-valid.txt', 500, ["<|endoftext|>"])
