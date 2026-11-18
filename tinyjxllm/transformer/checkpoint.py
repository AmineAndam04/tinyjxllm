"""To save and load our models"""

import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np


# TODO in the last iteration save just the params
def save_checkpoint(path, model, optim_state, extra: dict):
    os.makedirs(path, exist_ok=True)
    checkpoint = {
        "params": [np.asarray(x) for x in jax.tree.leaves(model)],
        "optim": [np.asarray(x) for x in jax.tree.leaves(optim_state)],
        "extra": extra,
    }
    with open(f"{path}/checkpoint.pkl", "wb") as f:
        pickle.dump(checkpoint, f)


def load_checkpoint(path, model, optim_state):
    with open(f"{path}/checkpoint.pkl", "rb") as f:
        ckpt = pickle.load(f)
    _, model_def = jax.tree.flatten(model)
    _, optim_def = jax.tree.flatten(optim_state)
    model = jax.tree.unflatten(model_def, [jnp.asarray(x) for x in ckpt["params"]])
    optim_state = jax.tree.unflatten(optim_def, [jnp.asarray(x) for x in ckpt["optim"]])
    return model, optim_state, ckpt["extra"]


def load_model_only(path, model):
    with open(f"{path}/checkpoint.pkl", "rb") as f:
        ckpt = pickle.load(f)
    _, model_def = jax.tree.flatten(model)
    model = jax.tree.unflatten(model_def, [jnp.asarray(x) for x in ckpt["params"]])
    return model
