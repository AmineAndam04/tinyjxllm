import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import json
from types import SimpleNamespace

import jax
import jax.numpy as jnp
from checkpoint import load_model_only
from training import ACT_FUN

from tokenizer.tokenizer import Tokenizer
from transformer import TransformerLM


def greedy_sampling(logits, **kwargs):
    return jnp.argmax(logits)


def random_sampling(logits, key, temp, **kwargs):
    return jax.random.categorical(key, logits / temp, axis=-1)


def top_k_sampling(logits, key, top_k, temp, **kwargs):
    k_logits, k_indices = jax.lax.top_k(logits, top_k)
    choice = random_sampling(k_logits, key, temp)
    return k_indices[choice]


def top_p_sampling(logits, key, top_p, temp, **kwargs):
    logits = logits / temp
    sorted_idx = jnp.argsort(-logits)
    sorted_logits = logits[sorted_idx]
    probs = jax.nn.softmax(sorted_logits)
    cum_probs = jnp.cumsum(probs)
    keep = (cum_probs - probs) < top_p
    masked = jnp.where(keep, sorted_logits, -jnp.inf)
    choice = jax.random.categorical(key, masked)
    return sorted_idx[choice]


SAMPLING = {
    "greedy": greedy_sampling,
    "random": random_sampling,
    "top_k": top_k_sampling,
    "top_p": top_p_sampling,
}


def load_model_and_config(model_path):
    with open(f"{model_path}/config.json", "r") as file:
        config = json.load(file)
    config = SimpleNamespace(**config)
    model = TransformerLM(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        num_layers=config.num_layers,
        d_model=config.d_model,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        rope_theta=config.rope_theta,
        use_bias=config.use_bias,
        act_fun=ACT_FUN[config.act_fun],
        eps=config.eps,
        key=jax.random.key(0),
    )
    model = load_model_only(path=model_path, model=model)
    return model, config


def generate(
    model,
    tokenizer,
    prompt: str,
    max_seq_len: int,
    max_gen_toks: int,
    sampling: str,
    temp: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.8,
    seed: int = 42,
):
    key = jax.random.key(seed)
    sampling_fun = SAMPLING[sampling]
    # Tokenize the prompt
    tokens = list(tokenizer.encode(prompt))
    next_toks = []

    while len(next_toks) < max_gen_toks:
        logits = model(jnp.asarray(tokens[-max_seq_len:])[None, :])[0, -1]
        key_gn, key = jax.random.split(key)
        next_id = sampling_fun(
            logits=logits, key=key_gn, top_k=top_k, top_p=top_p, temp=temp
        ).item()
        tokens.append(next_id)
        next_toks.append(next_id)
        if (
            hasattr(tokenizer, "special_tokens")
            and next_id == tokenizer.special_tokens["<|endoftext|>"]
        ):
            break

    return tokenizer.decode(next_toks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Transformer LM")
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--tokenizer_path", type=str, default="llms-from-scratch/tokenizer")
    parser.add_argument("--max_gen_toks", type=int, default=128)
    parser.add_argument(
        "--sampling", choices=["greedy", "random", "top_k", "top_p"], default="greedy"
    )
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    # Load model:
    model, train_config = load_model_and_config(model_path=args.model_path)
    # Load tokenizer
    tokenizer = Tokenizer(files_path=args.tokenizer_path)
    prompts = [
        "Tom and Jane are friends. One day, Jane goes to Tom’s house. Tom has a big pot of soup.",
        "Alice was so tired when she got back home so she went",
        "Lily likes cats and dogs. She asked her mom for a dog and her mom said no, so instead she asked",
        "Alice and Jack walked up the street and met a girl in a red dress. The girl said to them,",
        "Once upon a time, in an ancient house, there lived a girl named Lily.",
        "Once upon a time, there was a beautiful cat.",
    ]
    outputs = {}
    for prompt in prompts:
        next_tokens = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_seq_len=train_config.context_length,
            max_gen_toks=args.max_gen_toks,
            sampling=args.sampling,
            temp=args.temp,
            top_k=args.top_k,
            top_p=args.top_p,
            seed=args.seed,
        )
        outputs[prompt] = next_tokens
        print("The prompt is :", prompt)
        print("Completion is :", next_tokens)
        name = f"{args.sampling}-temp_{args.temp}-top_k_{args.top_k}-top_p_{args.top_p}-seed_{args.seed}"
        with open(os.path.join(args.model_path, f"{name}.json"), "w", encoding="utf-8") as f:
            json.dump(outputs, f, indent=4, ensure_ascii=False)
