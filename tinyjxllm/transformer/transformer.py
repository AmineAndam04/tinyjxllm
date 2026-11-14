import math
from collections.abc import Callable
from functools import cache

import jax
import jax.numpy as jnp
from einops import rearrange
from jax.tree_util import GetAttrKey, register_pytree_with_keys_class
from jaxtyping import Array, ArrayLike, Int, PRNGKeyArray


def auto_pytree_with_keys(children: tuple[str, ...], aux_data: tuple[str, ...]):
    def decorator(cls):
        def tree_flatten_with_keys(obj):
            children_ = [(GetAttrKey(chld), getattr(obj, chld)) for chld in children]
            aux_data_ = tuple(getattr(obj, aux) for aux in aux_data)
            return children_, aux_data_

        def tree_unflatten(aux_data_, children_):
            obj = object.__new__(cls)
            for f, value in zip(children, children_, strict=True):
                setattr(obj, f, value)
            for f, value in zip(aux_data, aux_data_, strict=True):
                setattr(obj, f, value)
            return obj

        cls.tree_flatten_with_keys = tree_flatten_with_keys
        cls.tree_unflatten = tree_unflatten

        return register_pytree_with_keys_class(cls)

    return decorator


def uniform_init(key: PRNGKeyArray, shape: tuple, lim: float):
    return jax.random.uniform(key=key, shape=shape, minval=-lim, maxval=lim)


def normal_init(key: PRNGKeyArray, shape: tuple, std: float):
    return std * jax.random.normal(key=key, shape=shape)


def constant_init(shape: tuple, constant: float):
    return constant * jnp.ones(shape)


@auto_pytree_with_keys(
    children=("weight", "bias"),
    aux_data=("in_features", "out_features", "use_bias"),
)
class Linear:
    """
    Linear layer
    """

    def __init__(
        self, in_features: int, out_features: int, use_bias: bool = True, *, key: PRNGKeyArray
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        wkey, bkey = jax.random.split(key)
        lim = 1 / math.sqrt(in_features)
        self.weight = uniform_init(wkey, (out_features, in_features), lim)
        self.bias = constant_init(shape=(out_features,), constant=0) if use_bias else None

    def __call__(self, x: Array) -> Array:
        # jnp.einsum("...k,jk->...j",x,self.weight)
        x = x @ self.weight.T
        if self.use_bias:
            x = x + self.bias
        return x


@auto_pytree_with_keys(
    children=("embeddings",),
    aux_data=("num_embeddings", "dim_embedding"),
)
class Embedding:
    """
    Embedding layer
    """

    def __init__(self, num_embeddings: int, dim_embedding: int, *, key: PRNGKeyArray) -> None:
        self.num_embeddings = num_embeddings
        self.dim_embedding = dim_embedding
        self.embeddings = normal_init(key, (num_embeddings, dim_embedding), std=0.02)

    def __call__(self, x: Int[ArrayLike, "..."]) -> Array:
        if not jnp.issubdtype(x.dtype, jnp.integer):
            raise TypeError(f"Inputs should be integers, got {x.dtype}")
        return self.embeddings.at[x].get(mode="fill", fill_value=jnp.nan)

    def decode(self, x: Array):
        return (x @ self.embeddings.T) * (1 / math.sqrt(self.dim_embedding))


@auto_pytree_with_keys(
    children=("gain",),
    aux_data=("d_model", "eps"),
)
class RMSNorm:
    """
    Root Mean Square normalization
    """

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        self.d_model = d_model
        self.eps = eps
        self.gain = jnp.ones((d_model,))

    def __call__(self, x: Array) -> Array:
        ms = jnp.mean(x**2, axis=-1, keepdims=True)
        x = x * jax.lax.rsqrt(ms + self.eps) * self.gain
        return x


def find_d_ff_(d_model: int, multiple_of: int = 64) -> int:
    """
    A helper function to find a nearby multiple of 64
    """
    d_ff = 8 * d_model / 3
    return multiple_of * round(d_ff / multiple_of)


@auto_pytree_with_keys(
    children=("w1", "w2", "w3"),
    aux_data=("d_model", "d_ff", "act_fun"),
)
class FeedForward:
    """
    The position-wise feedforward layer of a Transformer
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        act_fun: Callable = jax.nn.silu,
        *,
        key: PRNGKeyArray,
    ) -> None:
        self.d_model = d_model
        self.act_fun = act_fun
        if d_ff is None:
            d_ff = find_d_ff_(d_model=d_model)
        self.d_ff = d_ff
        key1, key2, key3 = jax.random.split(key, 3)
        self.w1 = Linear(in_features=d_model, out_features=d_ff, use_bias=False, key=key1)
        self.w2 = Linear(in_features=d_ff, out_features=d_model, use_bias=False, key=key2)
        self.w3 = Linear(in_features=d_model, out_features=d_ff, use_bias=False, key=key3)

    def __call__(self, x: Array) -> Array:
        return self.w2(self.act_fun(self.w1(x)) * self.w3(x))


@cache
def rope_rotations(max_seq_len: int, d_h: int, theta: float):
    """
    A function to cache the "cos" and "sin" values of RoPE embeddings
    """
    with jax.ensure_compile_time_eval():
        exponents = -jnp.arange(0, d_h, step=2) / d_h
        thetas = theta**exponents
        position = jnp.arange(max_seq_len)
        rotates = jnp.outer(position, thetas)
        rotates = jnp.repeat(rotates, 2, axis=-1)
        return jnp.cos(rotates), jnp.sin(rotates)  # of shape (max_seq_len,d_h)


@auto_pytree_with_keys(
    children=(),
    aux_data=("theta", "d_h", "max_seq_len"),
)
class RoPE:
    """
    Rotary Position Embeddings
    """

    def __init__(self, theta: float, d_h: int, max_seq_len: int) -> None:
        self.theta = theta
        self.d_h = d_h
        self.max_seq_len = max_seq_len

    def __call__(self, x: Array, token_positions: Array) -> Array:
        """
        token_position is assumed of (seq,) shape
        """
        coses, sines = rope_rotations(max_seq_len=self.max_seq_len, d_h=self.d_h, theta=self.theta)
        coses = coses[token_positions]
        sines = sines[token_positions]
        frst, scnd = jnp.unstack(rearrange(x, " ... (i j) -> ... i j", j=2), axis=-1)
        x_sins = rearrange(jnp.stack([-scnd, frst], axis=-1), "... i j -> ... (i j)")
        return x * coses + x_sins * sines


def scaled_dot_product_attention(Q: Array, K: Array, V: Array, mask: None | Array = None):
    """
    Scaled attention
    """
    scores = (1 / math.sqrt(K.shape[-1])) * jnp.einsum("... a b, ... c b -> ... a c", Q, K)
    if mask is not None:
        scores = jnp.where(mask, scores, -jnp.inf)
    out = jnp.einsum("... a b, ... b c -> ... a c", jax.nn.softmax(scores, axis=-1), V)
    return out


@auto_pytree_with_keys(
    children=("w_query", "w_key", "w_value", "w_output", "rope"),
    aux_data=("d_model", "num_heads", "use_bias"),
)
class MultiHeadSelfAttention:
    """
    Multi-head attention layer
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        use_bias: bool = False,
        rope: RoPE | None = None,
        *,
        key: PRNGKeyArray,
    ) -> None:
        assert d_model % num_heads == 0
        if rope is not None:
            assert rope.d_h == d_model // num_heads
        self.d_model = d_model
        self.num_heads = num_heads
        self.use_bias = use_bias
        self.rope = rope
        key1, key2, key3, key4 = jax.random.split(key, 4)
        self.w_query = Linear(
            in_features=d_model, out_features=d_model, use_bias=use_bias, key=key1
        )
        self.w_key = Linear(in_features=d_model, out_features=d_model, use_bias=use_bias, key=key2)
        self.w_value = Linear(
            in_features=d_model, out_features=d_model, use_bias=use_bias, key=key3
        )
        self.w_output = Linear(
            in_features=d_model, out_features=d_model, use_bias=use_bias, key=key4
        )

    def __call__(self, x: Array, token_positions: Array | None = None) -> Array:
        query = rearrange(self.w_query(x), "... seq (h d) -> ... h seq d ", h=self.num_heads)
        key = rearrange(self.w_key(x), "... seq (h d) -> ... h seq d ", h=self.num_heads)
        value = rearrange(self.w_value(x), "... seq (h d) -> ... h seq d ", h=self.num_heads)
        if self.rope:
            if token_positions is None:
                token_positions = jnp.arange(x.shape[-2])
            token_positions = token_positions[..., None, :]
            query = self.rope(query, token_positions)
            key = self.rope(key, token_positions)
        mask = jnp.tril(jnp.ones((x.shape[-2], x.shape[-2]), dtype=bool))
        out = scaled_dot_product_attention(Q=query, K=key, V=value, mask=mask)
        out = rearrange(out, "... h seq d -> ... seq (h d)")
        return self.w_output(out)


@auto_pytree_with_keys(
    children=("norm1", "attn", "norm2", "ff"),
    aux_data=(),
)
class TransformerBlock:
    """
    Transformer block = pre-norm + multi-head attention + pre-norm + feedforward
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int | None = None,
        *,
        use_bias: bool = False,
        act_fun: Callable = jax.nn.silu,
        eps: float = 1e-5,
        rope: RoPE | None = None,
        key: PRNGKeyArray,
    ) -> None:
        key1, key2 = jax.random.split(key)
        self.norm1 = RMSNorm(d_model=d_model, eps=eps)
        self.attn = MultiHeadSelfAttention(
            d_model=d_model, num_heads=num_heads, use_bias=use_bias, rope=rope, key=key1
        )

        self.norm2 = RMSNorm(d_model=d_model, eps=eps)
        self.ff = FeedForward(d_model=d_model, d_ff=d_ff, act_fun=act_fun, key=key2)

    def __call__(self, x: Array, token_positions: None | Array = None) -> Array:
        x = x + self.attn(x=self.norm1(x), token_positions=token_positions)
        x = x + self.ff(self.norm2(x))
        return x


@auto_pytree_with_keys(
    children=("token_embeddings", "blocks", "ln_final", "lm_head"),
    aux_data=("context_length",),
)
class TransformerLM:
    """
    Transformer Language model with RoPE embeddings
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int | None = None,
        *,
        rope_theta: float = 10000.0,
        use_bias: bool = False,
        act_fun: Callable = jax.nn.silu,
        eps: float = 1e-5,
        key: PRNGKeyArray,
    ) -> None:
        assert d_model % num_heads == 0
        keys = jax.random.split(key, num_layers + 2)
        self.context_length = context_length

        rope = RoPE(theta=rope_theta, d_h=d_model // num_heads, max_seq_len=context_length)
        self.token_embeddings = Embedding(vocab_size, d_model, key=keys[0])
        self.blocks = tuple(
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                use_bias=use_bias,
                act_fun=act_fun,
                eps=eps,
                rope=rope,
                key=keys[i + 1],
            )
            for i in range(num_layers)
        )
        self.ln_final = RMSNorm(d_model=d_model, eps=eps)
        self.lm_head = Linear(d_model, vocab_size, use_bias=False, key=keys[-1])

    def __call__(self, token_ids: Array, token_positions: Array | None = None) -> Array:
        x = self.token_embeddings(token_ids)
        for block in self.blocks:
            x = block(x, token_positions)
        return self.lm_head(self.ln_final(x))


# Losses
def cross_entropy_loss(logits: Array, targets: Array) -> Array:
    """
    Cross entropy loss
    """
    loss = -jnp.take_along_axis(
        jax.nn.log_softmax(x=logits, axis=-1), indices=targets[..., None], axis=-1
    ).mean()
    return loss


def perplexity(logits: Array, targets: Array) -> Array:
    """
    Perplexity
    """
    return jnp.exp(cross_entropy_loss(logits, targets))


def jax_pprint(model):
    def visit(name, value, depth=0):
        indent = "  " * depth

        if hasattr(value, "shape"):
            print(f"{indent}{name}: shape={value.shape}, dtype={value.dtype}")

        elif hasattr(value, "tree_flatten_with_keys"):
            print(f"{indent}{name}: {type(value).__name__}")
            children, _ = value.tree_flatten_with_keys()

            for key, child in children:
                visit(key.name, child, depth + 1)

        elif isinstance(value, (tuple, list)):
            print(f"{indent}{name}:")

            for index, child in enumerate(value):
                visit(f"[{index}]", child, depth + 1)

        else:
            print(f"{indent}{name}: {value}")

    visit("Model", model)


def count_params(model):
    return sum(leaf.size for leaf in jax.tree.leaves(model))
