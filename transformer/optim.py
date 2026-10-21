from typing import Any, NamedTuple

import jax
import jax.numpy as jnp


def get_optim_by_name(name: str):
    """
    It returns the initializer and the update functions of the selected optimizer
    """
    if name == "sgd":
        return init_sgd, update_sgd
    elif name == "adamw":
        return init_adamw, update_adamw
    else:
        raise NotImplementedError


# Stochastic Gradient Descent
class SGDState(NamedTuple):
    lr: float


def init_sgd(model, lr, **kwargs):
    return SGDState(lr=lr)


def update_sgd(model, grad, state):
    return jax.tree.map(lambda p, g: p - state.lr * g, model, grad), state


# AdamW
class AdamWState(NamedTuple):
    lr: float
    beta_1: float
    beta_2: float
    decay: float
    eps: float
    step: int
    m: Any
    v: Any


def init_adamw(model, lr, beta_1, beta_2, decay, eps):
    return AdamWState(
        lr=lr,
        beta_1=beta_1,
        beta_2=beta_2,
        decay=decay,
        eps=eps,
        step=1,
        m=jax.tree.map(jnp.zeros_like, model),
        v=jax.tree.map(jnp.zeros_like, model),
    )


def update_adamw(model, grad, state):
    t, lr, decay = state.step, state.lr, state.decay
    beta_1, beta_2 = state.beta_1, state.beta_2

    alpha = lr * (1 - beta_2**t) ** 0.5 / (1 - beta_1**t)

    m = jax.tree.map(lambda m, g: beta_1 * m + (1 - beta_1) * g, state.m, grad)
    v = jax.tree.map(lambda v, g: beta_2 * v + (1 - beta_2) * g * g, state.v, grad)

    model = jax.tree.map(
        lambda p, m_, v_: p - lr * decay * p - alpha * m_ / (jnp.sqrt(v_) + state.eps), model, m, v
    )
    state = state._replace(m=m, v=v, step=t + 1)
    return model, state


# Gradient clipping
def clip_grads(grad, max_grad, eps=1e-6):  # to optimize : what types, and see what optax did
    """
    Clip gradient using global norm
    """
    global_norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in jax.tree.leaves(grad)))
    scale = jnp.where(global_norm <= max_grad, 1.0, max_grad / (global_norm + eps))
    return jax.tree.map(lambda g: g * scale, grad)


# Cosine annealing
def cosine_annealing(t: int, lr_max: float, lr_min: float, t_warmup: int, t_final: int):
    return jnp.where(
        t < t_warmup,
        (t / t_warmup) * lr_max,
        jnp.where(
            t < t_final,
            lr_min
            + 0.5
            * (1 + jnp.cos((t - t_warmup) * jnp.pi / (t_final - t_warmup)))
            * (lr_max - lr_min),
            lr_min,
        ),
    )
