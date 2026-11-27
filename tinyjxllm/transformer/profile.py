from tensorboardX import SummaryWriter
import time
import os
from datetime import datetime
from functools import partial

import jax
import numpy as np
from checkpoint import save_checkpoint
from logger import Logger
from optim import clip_grads, cosine_annealing, get_optim_by_name, global_norm
from transformer import TransformerLM, count_params, cross_entropy_loss

from data import data_loading, iterate_eval_data

ACT_FUN = {
    "silu": jax.nn.silu,
    "relu": jax.nn.relu,
}


SEED = 42
RUN_NAME = "profile"
VOCAB_SIZE = 10000
BATCH_SIZE = 32
CONTEXT_LENGTH = 256
NUM_LAYERS = 1
D_MODEL = 512
NUM_HEADS = 16
D_FF = None
EPS = 1e-5
NUM_ITERATIONS = 3

CLIP_GRADS = -1
if __name__ == "__main__":
    print(jax.devices(), jax.default_backend())
    # Random seeds
    rng = np.random.default_rng(SEED)
    key = jax.random.key(SEED)
    key_x, key_y = jax.random.split(key)
    # Logger
    run_name = f"{RUN_NAME}_{datetime.now():%Y-%m-%d_%H-%M-%S}"
    log_dir = os.path.join("runs", run_name)
    logger = SummaryWriter(log_dir)
    # Data
    x = jax.random.choice(key=key_x, a=VOCAB_SIZE, shape=(BATCH_SIZE, CONTEXT_LENGTH))
    y = jax.random.choice(key=key_y, a=VOCAB_SIZE, shape=(BATCH_SIZE, CONTEXT_LENGTH))
    # Model
    model = TransformerLM(
        vocab_size=VOCAB_SIZE,
        context_length=CONTEXT_LENGTH,
        num_layers=NUM_LAYERS,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        rope_theta=10000,
        use_bias=False,
        act_fun=ACT_FUN["silu"],
        eps=EPS,
        key=key,
    )
    # Optimizer
    optim_init, optim_update = get_optim_by_name(name="adamw")
    optim_state = optim_init(
        model=model,
        lr=1e-4,
        beta_1=0.99,
        beta_2=0.99,
        decay=0.01,
        eps=EPS,
    )

    @jax.jit
    def loss_fn(model, x, y):
        with jax.named_scope("loss_fn"):
            y_hat = model(token_ids=x)
            loss = cross_entropy_loss(logits=y_hat, targets=y)
        return loss

    @jax.jit(donate_argnums=(0, 1))
    def train_step(model, optim_state, x, y):
        with jax.named_scope("train_step"):
            with jax.named_scope("forward_backward"):
                loss, grads = jax.value_and_grad(loss_fn)(model, x, y)
            with jax.named_scope("grad_norm"):
                grad_norm = global_norm(grads)
            if CLIP_GRADS > 0:
                with jax.named_scope("grad_clip"):
                    grads = clip_grads(grads, CLIP_GRADS, eps=EPS, norm=grad_norm)
            with jax.named_scope("update"):
                model, optim_state = optim_update(model=model, grad=grads, state=optim_state)
        return model, optim_state, (loss, grad_norm)

    # compile
    for _ in range(3):
        model, optim_state, out = train_step(model, optim_state, x, y)
    jax.block_until_ready((model, optim_state, out))
    # Train
    with jax.profiler.trace(log_dir=log_dir, create_perfetto_trace=True):
        for t in range(NUM_ITERATIONS):
            with jax.profiler.StepTraceAnnotation("train", step_num=t):
                model, optim_state, out = train_step(model, optim_state, x, y)
                jax.block_until_ready((model, optim_state, out))
    # jax.profiler.save_device_memory_profile(os.path.join(log_dir, "memory.prof"))
