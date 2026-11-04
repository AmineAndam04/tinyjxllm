import argparse
import json
import os
from types import SimpleNamespace

import jax
import jax.numpy as jnp
from checkpoint import load_model_only
from tinyjxllm.tokenizer.tokenizer import Tokenizer
from training import ACT_FUN
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
        d_ff=None if config.d_ff < 0 else config.d_ff,
        rope_theta=config.rope_theta,
        use_bias=config.use_bias,
        act_fun=ACT_FUN[config.act_fun],
        eps=config.eps,
        key=jax.random.key(42),
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
    parser.add_argument("--tokenizer_path", type=str, default="tinyjxllm/tokenizer/tinystories")
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
    # prompts = [
    #     """
    #     Outfit 1: Typical
    #     This pairing was the first outfit I thought of when I bought the shoes. It’s like a summer version of this
    #     Jake Grantham outfit; in fact, my shoes are close to the colors of his Nike Racers! Instead of a heavy Harris Tweed
    #     jacket and denim shirt, I’m wearing a cotton DB jacket and and a linen shirt. Both fabrics (in these colors) are an
    #     absolute must for summer, as they go with both dark and and light pants! As you can see, they pair wonderfully
    #     with the dark jeans and shoes. It’s a pseudo menswear/prep outfit.
    #     Overall, this is a very casual outfit which is why I paired my sneakers with it. I’m not about wearing a
    #     full wool suit with sneakers (as GQ shows a lot) but I’m definitely open to keeping things casual, like this cotton
    #     DB. Casual fabrics are key to pulling off your sneakers in a dressed down menswear outfit. I’d even suggest to
    #     wear these sneakers with a khaki chino suit or a white linen suit. Just be sure to ditch the tie or wear a tee or polo;
    #     wearing a tie with sneakers is a bit too much
    #     """,
    #     """
    #     Some of the most glorious historical attractions in Spain date from the period of Muslim rule, including The
    #     Mezquita, built as the Great Mosque of C  ́ordoba and the Medina Azahara, also in C  ́ordoba and now in ruins but
    #     still visitable as such and built as the Madinat al-Zahra, the Palace of al-Andalus; and the Alhambra in Granada, a
    #     splendid, intact palace. There are also two synagogues still standing that were built during the era of Muslim Spain:
    #     Santa Mar ́ıa la Blanca in Toledo and the Synagogue of C ́ordoba, in the Old City.
    #     Reconquista and Imperial era
    #     """,
    #     """
    #     Pinky
    #     The pink ghost’s AI is designed to ”feel” opposite of the red ghost’s behavior. Pinky actually attempts to
    #     get out in front of Pac-Man. This is accomplished by setting the target 4 tiles ahead of Pac-Man’s current location in
    #     the direction that Pac-Man is travelling. One exception to this is when Pac-Man is traveling up. Due to an overflow
    #     bug in the code, the calculation includes a left offset equal to the expected up offset.
    #     The use of Pac-Man’s expected position allows the player to exploit the navigation code for Pinky. This was most
    #     often seen in situations where Pac-Man and Pinky are in close proximity to each other. For example, if they were
    #     headed toward one another, once they are less than 4 tiles apart, the target tile is now behind Pinky. If he encounters
    #     an intersection while in this configuration, he will attempt to double-back (by turning) in order to reach the target tile.
    #     Player’s would often perform what was referred to as a ”head fake” by changing directions rapidly just
    #     as Pinky was approaching an intersection. By displacing the target tile just as Pinky
    #     """,
    #     """
    #     Xmonad, Xresources, and Fonts
    #     Xresources can be an absolute pain (they were for me). So, this a bit of help:
    #     First of all, the way that fonts are generally handled in Xmonad is through XFT . Xft is the X FreeType
    #     Interface library and is a library that handles the actual rendering of fonts in Xmonad/urxvt (Xresources specifies
    #     a bunch of resources to things that launch under the X window manager. Settings for individual programs are
    #     specified in Xresources like so: Urxvt.background:background settins here ).
    #     Anyway, fonts are set using this syntax in Xresources :
    #     urxvt*font:xft:your font name goes here:size=the size of the font goes here,xft:fall-back font 1 goes here
    #     So, thats the most basic part of fonts in urxvt specified via Xresources.
    #     Fixing /.ssh/ Permissions in Cygwin
    #     In Cygwin on Windows I found that I could not set the permissions of my /.ssh/ folder to be 0600, as is
    #     required for ssh to allow you to use keys. The symptom I had was that I no matter what I did, it always modfied the
    #     owner and the group
    #     """,
    #     """
    #     Aaron loves mint chocolate cake, but he requires that it be paired with mini chocolate chips, so I threw some of
    #     those in between the layers. I also had a few Peppermint Jo Jos on hand so I crushed them up and threw some of
    #     those in along with some crushed meringue cookies because, why not? It’s a total smorgasbord of minty chocolate
    #     chippy cookie crunchy goodness. I didn’t measure how much of each topping I used, but after I tasted the finished
    #     product, I wish I had added more. You can add anything you want- crushed candy canes, peppermint bark, etc. And
    #     don’t be afraid to use a heavy hand. Texture = good.
    #     If you don’t have 7-inch cake pans, you can get 3 shorter 8-inch layers out of this
    #     """,
    #     """
    #     In a shocking finding, scientist discovered a herd of unicorns living
    #     in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the
    #     researchers was the fact that the unicorns spoke perfect English.
    #     """,
    #     """
    #     Prehistoric man sketched an incredible array of prehistoric beasts on the rough limestone walls of a cave in
    #     modern day France 36,000 years ago.
    #     Now, with the help of cutting-edge technology, those works of art in the Chauvet-Pont-d’Arc Cave have been
    #     reproduced to create the biggest replica cave in the world.
    #     The manmade cavern named the Caverne du Pont-d’Arc has been built a few miles from the original site in
    #     Vallon-Pont-D’arc in Southern France and contains 1,000 painstakingly-reproduced drawings as well as around 450
    #     bones and other features...
    #     Cavemen and women sketched an incredible array of prehistoric beasts on the rough limestone walls of a cave
    #     36,000 years ago and now a replica has been created
    #     """,
    # ]
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
