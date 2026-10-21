from typing import Any

import jax.numpy as jnp
import numpy as np


def data_loading(data: Any, batch_size: int, context_length: int, rng):
    """
    Lazy loading of data
    """
    starts = rng.integers(0, len(data) - context_length, size=batch_size)
    idx = starts[:, None] + np.arange(context_length + 1)
    window = np.asarray(data[idx], dtype=np.int32)
    return jnp.asarray(window[:, :-1]), jnp.asarray(window[:, 1:])
