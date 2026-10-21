import argparse
import time
from datetime import datetime
from functools import partial

import jax
import jax.profiler
import numpy as np
from checkpoint import save_checkpoint
from data import data_loading
from logger import Logger
from optim import clip_grads, get_optim_by_name

from transformer import TransformerLM, cross_entropy_loss

ACT_FUN = {
    "silu": jax.nn.silu,
    "relu": jax.nn.relu,
}
if __name__ == "__main__":
    print(jax.devices(), jax.default_backend())
    parser = argparse.ArgumentParser(description="Train a Transformer LM")
    parser.add_argument(
        "--train_path",
        type=str,
        default="data/TinyStoriesV2-GPT4-train.npy",
    )
    parser.add_argument(
        "--valid_path",
        type=str,
        default="data/TinyStoriesV2-GPT4-valid.npy",
    )

    parser.add_argument("--seed", type=int, default=42)
    # Model
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=-1)
    parser.add_argument("--rope_theta", type=float, default=10000)
    parser.add_argument("--use_bias", action="store_true")
    parser.add_argument("--act_fun", type=str, default="silu")
    parser.add_argument("--eps", type=float, default=1e-7)
    # Optim
    parser.add_argument("--optim", type=str, default="adamw")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta_1", type=float, default=0.9)
    parser.add_argument("--beta_2", type=float, default=0.95)
    parser.add_argument("--decay", type=float, default=0.01)
    parser.add_argument("--optim_eps", type=float, default=1e-7)
    parser.add_argument("--use_lr_schedular", action="store_true")
    parser.add_argument("--clip_grads", type=float, default=-1.0)
    parser.add_argument("--lr_max", type=float, default=1e-3)
    parser.add_argument("--lr_min", type=float, default=1e-4)
    parser.add_argument("--t_warmup", type=float, default=10)
    parser.add_argument("--t_final", type=float, default=400)
    # Training
    parser.add_argument("--num_iterations", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=64)
    # Logger
    parser.add_argument("--logger", choices=["tensorboard", "wandb"], default="tensorboard")
    parser.add_argument("--wandb_project", type=str, default="JAX-LLMs")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--run_name", type=str, default="LM-Transformer")
    parser.add_argument("--log_every", type=int, default=20)
    # Eval
    parser.add_argument("--eval_every", type=int, default=50)
    parser.add_argument("--n_eval_batches", type=int, default=20)
    # SAVE
    parser.add_argument("--save_path", type=str, default="runs")
    parser.add_argument("--save_every", type=int, default=100)
    args = parser.parse_args()
    if args.d_ff < 0:
        args.d_ff = None
    # Random seeds
    rng = np.random.default_rng(args.seed)
    key = jax.random.key(args.seed)
    # Logger
    run_name = f"{args.run_name}_{datetime.now():%Y-%m-%d_%H-%M-%S}"
    logger = Logger(args, run_name)
    # Import data
    train_data = np.load(args.train_path, mmap_mode="r")
    valid_data = np.load(args.valid_path, mmap_mode="r")
    # Model
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        use_bias=args.use_bias,
        act_fun=ACT_FUN[args.act_fun],
        eps=args.eps,
        key=key,
    )
    # Optimizer + schedular if any
    optim_init, optim_update = get_optim_by_name(name=args.optim)
    optim_state = optim_init(
        model=model,
        lr=args.lr,
        beta_1=args.beta_1,
        beta_2=args.beta_2,
        decay=args.decay,
        eps=args.optim_eps,
    )
    if args.use_lr_schedular:
        from optim import cosine_annealing

        lr_schedular = partial(
            cosine_annealing,
            lr_max=args.lr_max,
            lr_min=args.lr_min,
            t_warmup=args.t_warmup,
            t_final=args.t_final,
        )

    @jax.jit
    def loss_fn(model, x, y):
        y_hat = model(token_ids=x)
        return cross_entropy_loss(logits=y_hat, targets=y)

    def evaluate_step(model, val_data, n_eval_batches):
        val_rng = np.random.default_rng(0)
        losses = []
        for _ in range(n_eval_batches):
            x, y = data_loading(val_data, args.batch_size, args.context_length, val_rng)
            losses.append(loss_fn(model, x, y))
        return np.mean(losses)

    @jax.jit
    def train_step(model, optim_state, x, y):
        loss, grads = jax.value_and_grad(loss_fn)(model, x, y)
        if args.clip_grads > 0:
            grads = clip_grads(grads, args.clip_grads, eps=args.optim_eps)
        model, optim_state = optim_update(model=model, grad=grads, state=optim_state)
        return model, optim_state, loss

    train_loss = 0
    window_start = time.perf_counter()
    elapsed_steps = 0
    for t in range(1, args.num_iterations + 1):
        if args.use_lr_schedular:
            optim_state = optim_state._replace(lr=lr_schedular(t))
            logger.log("train/lr", lr_schedular(t), t)
        # Train
        x, y = data_loading(
            data=train_data, batch_size=args.batch_size, context_length=args.context_length, rng=rng
        )
        model, optim_state, loss = train_step(model, optim_state, x, y)
        train_loss = train_loss + loss
        elapsed_steps += 1
        # Logging
        if t % args.log_every == 0:
            logger.log("train/loss", float(train_loss) / args.log_every, t)
            # logg tokens per second
            period = time.perf_counter() - window_start
            tok_per_s = (elapsed_steps * args.batch_size * args.context_length) / period
            logger.log("perf/tokens_per_sec", tok_per_s, t)
            train_loss, elapsed_steps = 0, 0
            window_start = time.perf_counter()
        # Evaluate
        if t % args.eval_every == 0:
            loss.block_until_ready()
            eval_start = time.perf_counter()
            val_loss = evaluate_step(model, valid_data, args.n_eval_batches)
            logger.log("val/loss", val_loss, t)
            eval_end = time.perf_counter()
            window_start += eval_end - eval_start
        # Save checkpoint
        if t % args.save_every == 0:
            save_checkpoint(
                path=logger.log_dir, model=model, optim_state=optim_state, extra=vars(args)
            )
    # Final eval and checkpointing
    val_loss = evaluate_step(model, valid_data, args.n_eval_batches)
    logger.log("val/loss", val_loss, args.num_iterations + 1)
    save_checkpoint(path=logger.log_dir, model=model, optim_state=optim_state, extra=vars(args))
    logger.close()
