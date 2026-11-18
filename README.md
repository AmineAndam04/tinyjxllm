# tinyjxllm

An LLM built from scratch in JAX.

## Install

Tested with Python 3.14 and [uv](https://docs.astral.sh/uv/getting-started/installation/).

After cloning this repo, run these commands:

```bash
cd tinyjxllm
uv venv --python 3.14
source .venv/bin/activate
uv pip install -e .
```

To train on NVIDIA GPUs, you will need to install a compatible version of JAX. For example:

```bash
uv pip install -U "jax[cuda13]"
```
## Tokenization

The `tokenizer` module implements Byte Pair Encoding (BPE):

- `trainer.py`: trains a tokenizer.
- `trainer_v2.py`: an efficient version of `trainer.py`; it uses a max-heap instead of `max`.
- `tokenizer.py`: encodes text into token IDs and decodes token IDs back into text.
- `example.py`: encodes large datasets in parallel and saves the token IDs as `.npy` files.
- `tinystories/` and `owt/`: contain tokenizers trained on TinyStories and OpenWebText (OWT), respectively.

Tokenizer training on TinyStories took roughly 80 seconds in my setup (with max).

## Train

The `transformer` module contains a small training framework:

- `transformer.py`: neural network layers in pure JAX.
- `optim.py`: optimizers.
- `data.py`: data loaders.
- `logger.py`: logging with Weights & Biases or TensorBoard.
- `checkpoint.py`: saves and loads model parameters and optimizer state.
- `training.py`: the training loop.
- `eval.py`: generates text from a trained model.

An example:

```bash
python tinyjxllm/transformer/training.py \
    --log_perstep \
    --run_name=blog-lrmax_004-batch_32 \
    --use_lr_schedular \
    --lr_max=0.004 \
    --lr_min=0.0001 \
    --t_warmup=6400 \
    --num_iterations=64000 \
    --t_final=64000 \
    --batch_size=32 \
    --beta_2=0.99 \
    --d_ff=2048
```
