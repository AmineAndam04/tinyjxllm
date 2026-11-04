import argparse
import json
import os
from collections import Counter
from multiprocessing import Pool
from time import perf_counter

import regex

GPT4_PATTERN = regex.compile(
    r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|"""
    r"""[^\r\n\p{L}\p{N}]?+\p{L}++|"""
    r"""\p{N}{1,3}+|"""
    r"""[^\s\p{L}\p{N}]++[\r\n]*+|"""
    r"""\s*+\r?\n|"""
    r"""\s+(?!\S)|"""
    r"""\s+"""
)


def merge(tokens: list[bytes], pair: tuple, new_id: int):
    """
    A function to merge old pairs
    """
    i = 0
    while i < (len(tokens) - 1):
        if tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
            tokens = tokens[:i] + [new_id] + tokens[i + 2 :]
        i += 1
    return tokens


def process_job(file_path: str, start: int, end: int, special_tokens: list[str]):
    """
    A Function that takes a file in "file_path", reads the chunk from "start" to "end"
    then it returns the word counts(e.g {"word1":n1,"word2":n2})
    """
    with open(file_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8")
    if special_tokens:
        special_pattern = "|".join(
            [regex.escape(special_token) for special_token in special_tokens]
        )
        chunks = regex.split(special_pattern, chunk)
    else:
        chunks = [chunk]
    words_stats = Counter()
    for chunk in chunks:
        words_stats.update(GPT4_PATTERN.findall(chunk))
    return words_stats


def chunk_file(file_path, num_chunks, search_size, pre_tok_special_token):
    if not isinstance(pre_tok_special_token, bytes):
        pre_tok_special_token = pre_tok_special_token.encode("utf-8")
    file_size = os.path.getsize(file_path)
    print("file_size = ", file_size)
    prelim_chunk_size = file_size // num_chunks
    search_step = search_size - (len(pre_tok_special_token) - 1)
    chunk_args = []
    with open(file_path, "rb") as f:
        chunk_start = 0
        while chunk_start < file_size:
            chunk_end = min(file_size, chunk_start + prelim_chunk_size)
            while True:
                f.seek(chunk_end)
                lookahead = f.read(search_size)
                if lookahead == b"":
                    chunk_end = file_size
                    break
                found_at = lookahead.find(pre_tok_special_token)
                if found_at != -1:
                    chunk_end = chunk_end + found_at
                    break
                chunk_end += search_step
            chunk_args.append((file_path, chunk_start, chunk_end))
            chunk_start = chunk_end
    return chunk_args


def pre_tokenization(
    file_path: str,
    special_tokens: list[str],
    pre_tok_special_token: str,
    num_chunks: int,
    search_size: int,
    num_processes: int,
):
    """
    A pre-tokenization function
    It:  1- Prepares text boundaries
         2- Runs process_job in parallel
         3- Computes pair count and pair indexes
    """
    # -------- Get boundaries
    chunk_args = chunk_file(file_path, num_chunks, search_size, pre_tok_special_token)
    chunk_args = [(*chunk_arg, special_tokens) for chunk_arg in chunk_args]

    word_stats = {}  # {"word":count}
    with Pool(processes=num_processes) as pool:
        pool_results = pool.starmap(process_job, chunk_args)
        # Aggregate stats
        for word_stat in pool_results:
            for word, stat in word_stat.items():
                word_stats[word] = word_stats.get(word, 0) + stat
    words = {}  # {"ID",word}
    word_counts = {}  # {ID:count}
    index = {}  # {pair: (id_word1,id_word2)}
    pair_stats = {}  # {pair: stat}
    for word_id, (word, count) in enumerate(word_stats.items()):
        word_bytes = list(word.encode("utf-8"))
        words[word_id] = word_bytes
        word_counts[word_id] = count
        for pair in zip(word_bytes, word_bytes[1:]):
            index.setdefault(pair, set()).add(word_id)
            pair_stats[pair] = pair_stats.get(pair, 0) + count
    return pair_stats, words, word_counts, index


def train(
    file_path: str,
    vocab_size: int,
    special_tokens: list[str],
    pre_tok_special_token: str,
    save_path: str,
    num_chunks: int,
    search_size: int,
    num_processes: int,
    verbose: bool = False,
    save_readable: bool = False,
):
    special_tokens = special_tokens or []
    min_vocab_size = 256 + len(special_tokens)

    assert vocab_size > min_vocab_size
    num_merges = vocab_size - min_vocab_size
    vocab = {idx: bytes([idx]) for idx in range(256)}
    merges = {}
    # Add special_tokens to the vocabulary
    special_tokens_ = {}
    if special_tokens:
        for idx, special_token in enumerate(special_tokens):
            vocab[256 + idx] = special_token.encode("utf-8")
            special_tokens_[special_token] = 256 + idx
    # pre-tokenization
    pair_counts, words, word_counts, index = pre_tokenization(
        file_path=file_path,
        special_tokens=special_tokens,
        pre_tok_special_token=pre_tok_special_token,
        num_chunks=num_chunks,
        search_size=search_size,
        num_processes=num_processes,
    )
    for i in range(num_merges):
        if not pair_counts:
            break
        pair = max(pair_counts, key=pair_counts.get)
        # We added this one bellow to have the same merges as the version that uses the heapq
        # pair = max(pair_counts, key=lambda p: (pair_counts[p], (vocab[p[0]], vocab[p[1]])))
        new_vocab_idx = len(vocab)
        vocab[new_vocab_idx] = vocab[pair[0]] + vocab[pair[1]]
        merges[pair] = new_vocab_idx
        if verbose:
            print(
                f"merge {i + 1}/{num_merges}: {pair} to {new_vocab_idx}, count  = {pair_counts[pair]} occurrences"
            )
        # merge
        ids = index[pair].copy()
        for id in ids:
            word = words[id]
            new_word = merge(tokens=word, pair=pair, new_id=new_vocab_idx)
            # Update words
            words[id] = new_word
            # Add the new pairs and decrease the count of adjacent pairs
            adjacent_pairs = []
            for new_pair in zip(new_word, new_word[1:]):
                if new_pair[0] == new_vocab_idx or new_pair[1] == new_vocab_idx:
                    # Add new pairs
                    pair_counts[new_pair] = pair_counts.get(new_pair, 0) + word_counts[id]
                    index.setdefault(new_pair, set()).add(id)
                    # Update the counts
                    if new_pair[1] == new_vocab_idx:
                        first = new_pair[0] if new_pair[0] != new_vocab_idx else pair[1]
                        old_pair = (first, pair[0])
                    else:
                        old_pair = (pair[1], new_pair[1])
                    pair_counts[old_pair] = pair_counts.get(old_pair) - word_counts[id]
                    adjacent_pairs.append(old_pair)
            # Update the index
            adjacent_pairs = set(adjacent_pairs)
            all_new_pairs = set(zip(new_word, new_word[1:]))
            for adj_pair in adjacent_pairs:
                if adj_pair not in all_new_pairs:
                    index[adj_pair].discard(id)
                    if adj_pair != pair and pair_counts[adj_pair] == 0:
                        del pair_counts[adj_pair]
                        del index[adj_pair]
        del pair_counts[pair]
        del index[pair]

    save(
        save_path=save_path,
        vocab=vocab,
        merges=merges,
        special_tokens=special_tokens_,
        save_readable=save_readable,
    )


def save(save_path, vocab, merges, special_tokens, save_readable):
    os.makedirs(save_path, exist_ok=True)
    with open(f"{save_path}/merges.json", "w") as file:
        json.dump([[a, b, new_id] for (a, b), new_id in merges.items()], file, indent=4)
    with open(f"{save_path}/vocab.json", "w") as file:
        json.dump({idx: token.hex() for idx, token in vocab.items()}, file, indent=4)
    with open(f"{save_path}/special_tokens.json", "w") as file:
        json.dump(special_tokens, file, indent=4)
    if save_readable:

        def readable(token: bytes) -> str:
            try:
                s = token.decode("utf-8")
            except UnicodeDecodeError:
                return "".join(f"\\x{b:02x}" for b in token)
            return (
                s.replace(" ", "␣").replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
            )

        vocab_readable = {idx: readable(token) for idx, token in vocab.items()}
        merges = [
            {
                "left": readable(vocab[a]),
                "right": readable(vocab[b]),
                "result": readable(vocab[new_id]),
                "new_id": new_id,
            }
            for (a, b), new_id in merges.items()
        ]

        with open(f"{save_path}/vocab_readable.json", "w", encoding="utf-8") as f:
            json.dump(vocab_readable, f, indent=2, ensure_ascii=False)
        with open(f"{save_path}/merges_readable.json", "w", encoding="utf-8") as f:
            json.dump(merges, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    print(os.getcwd())
    # Arguments
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer")
    parser.add_argument("--file-path", type=str, default="data/TinyStoriesV2-GPT4-train.txt")
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument(
        "--special-tokens",
        nargs="*",
        default=["<|endoftext|>"],
        help="Special tokens, separated by spaces",
    )
    parser.add_argument(
        "--pre-tok-special-token",
        type=str,
        default="<|endoftext|>",
        help="Token used to find chunk boundaries",
    )
    parser.add_argument("--save-path", type=str, default="tokenizer/old")
    parser.add_argument("--num-chunks", type=int, default=64)
    parser.add_argument("--search-size", type=int, default=4096)
    parser.add_argument("--num-processes", type=int, default=4)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--save-readable", action="store_true")
    args = parser.parse_args()

    # Start training
    start = perf_counter()
    train(
        file_path=args.file_path,
        vocab_size=args.vocab_size,
        special_tokens=args.special_tokens,
        pre_tok_special_token=args.pre_tok_special_token,
        save_path=args.save_path,
        num_chunks=args.num_chunks,
        search_size=args.search_size,
        num_processes=args.num_processes,
        verbose=args.verbose,
        save_readable=args.save_readable,
    )
    elapsed = perf_counter() - start
    print(f"Processing took {elapsed:.3f} seconds")
