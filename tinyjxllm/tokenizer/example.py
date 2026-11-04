# This code shows how to encode a full (large) dataset
import argparse
from multiprocessing import Pool

import numpy as np
from tokenizer import Tokenizer
from trainer import chunk_file


# We want to have one instance of tokenizer in each process (pros: shared cache )
def init_worker(tokenizer_path: str):
    global _tokenizer  # noqa: PLW0603
    _tokenizer = Tokenizer(files_path=tokenizer_path)


def process_job(file_path: str, start: int, end: int, idx):
    with open(file_path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8")
        tokenized = _tokenizer.encode(text)
    return {"idx": idx, "tokenized": tokenized}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode a dataset using PBE")
    parser.add_argument("--file-path", type=str, default="data/TinyStoriesV2-GPT4-train.txt")
    parser.add_argument("--tokenizer-path", type=str, default="tinyjxllm/tokenizer/tinystories")
    parser.add_argument(
        "--pre-tok-special-token",
        type=str,
        default="<|endoftext|>",
        help="Token used to find chunk boundaries",
    )
    parser.add_argument("--save-path", type=str, default="tokenizer")
    parser.add_argument("--num-chunks", type=int, default=64)
    parser.add_argument("--search-size", type=int, default=4096)
    parser.add_argument("--num-processes", type=int, default=4)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--save-readable", action="store_true")
    args = parser.parse_args()
    # Chunk data
    chunk_args = chunk_file(
        args.file_path, args.num_chunks, args.search_size, args.pre_tok_special_token
    )
    chunk_args = [(*chunk_arg, idx) for idx, chunk_arg in enumerate(chunk_args)]
    # Encode
    with Pool(
        processes=args.num_processes, initializer=init_worker, initargs=(args.tokenizer_path,)
    ) as pool:
        tokenized = pool.starmap(process_job, chunk_args)
    # Save
    tokenized.sort(key=lambda x: x["idx"])
    tokenized = np.concatenate([result["tokenized"] for result in tokenized])
    np.save(args.file_path.replace(".txt", ".npy"), tokenized)
