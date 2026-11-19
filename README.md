# Autoregressive Image Generation in Spectral Space

This research studies image generation as autoregressive modeling of discrete
spectral coefficients.

## Method

```text
image
  -> channel transform
  -> spectral decomposition
  -> coefficient ordering
  -> normalization
  -> quantization
  -> discrete token sequence
  -> causal Transformer
  -> generated token sequence
  -> inverse quantization and transforms
  -> generated image
```

### 1. Channel representation

Supported channel transforms:

- `grayscale`;
- `rgb`;
- `ycbcr`.

### 2. Spectral decomposition

The default representation uses an orthonormal two-dimensional DCT-II. The
research also supports block DCT, a real-valued DFT representation based on the
Hartley transform, and wavelet decomposition.

| Decomposition | Interpretation |
|---|---|
| `dct` | global image frequencies |
| `block_dct` | local spectra computed independently inside image blocks |
| `dft` | interleaved real-valued Fourier/Hartley coefficients |
| `wavelet` | multiscale spatial-frequency coefficients |


### 3. Coefficient ordering

A two-dimensional spectrum must be linearized before it can be modeled by a
causal sequence model. The default zigzag traversal places low-frequency DCT
coefficients first and gradually moves toward higher frequencies.

Available orderings are:

- `zigzag` — diagonal low-to-high frequency traversal;
- `radial` — increasing normalized distance from the spectrum origin;
- `dyadic` — frequency regions expanding by powers of two;
- `block` — image blocks first, then zigzag order inside each block.

### 4. Normalization and quantization

Spectral coefficients have highly non-uniform scales. The tokenizer can apply
a signed logarithm followed by min-max or standard scaling before fitting a
quantizer on the training split.

Implemented quantizers:

- **uniform** — scalar bins shared globally or fitted per sequence position;
- **scalar-channel K-means** — one scalar codebook per channel;
- **joint-channel K-means** — one vector codebook fitted to aligned
  cross-channel coefficients.

For an image with `C` channels and `H x W` spectral positions:

- uniform and scalar-channel K-means produce `C * H * W` tokens;
- joint-channel K-means produces `H * W` tokens because one token represents
  the complete channel vector at a frequency position.

The codebook size is the model vocabulary size. Current experiments compare
K=256 and K=768 for joint-channel K-means and K=256 for scalar-channel
K-means.

### 5. Autoregressive model

For a quantized sequence `z = (z_1, ..., z_L)`, the model learns

```text
p(z_1, ..., z_L) = product_t p(z_t | z_1, ..., z_{t-1})
```

using next-token cross-entropy. The implementation is a decoder-only Transformer
with causal self-attention, rotary positional embeddings, GELU feed-forward
layers, and a dedicated beginning-of-sequence token. A larger spectral iGPT
variant with gradient checkpointing and chunked attention is available for the
CelebA comparison configuration.

## Repository layout

```text
datasets/       dataset loaders and the spectral tokenizer
model/          autoregressive models, training, checkpoints, and generation
metrics/        image generation, encoders, metrics, and visualizations
experiments/    configuration loading and the shared experiment runner
configs/        runnable YAML experiments and reusable presets
compare/        CelebA preparation and reference-model comparison utilities
tests/          unit and integration tests
run_experiment.py
```

## Installation

The project targets Python 3.12 and runs on CPU or CUDA.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

Validate and inspect an experiment without loading data, creating a model, or
writing artifacts:

```bash
python run_experiment.py \
  --config configs/cifar10_ycbcr_dct_kmeans_joint_768.yaml \
  --dry-run
```

## Training and generation

Train an experiment:

```bash
python run_experiment.py --config configs/mnist.yaml
```

Evaluate an existing checkpoint without training:

```bash
python run_experiment.py \
  --config configs/mnist.yaml \
  --mode evaluate \
  --checkpoint runs/<experiment>/<run>/latest.pt
```

Resume training into a new run directory:

```bash
python run_experiment.py \
  --config configs/mnist.yaml \
  --checkpoint runs/<experiment>/<run>/latest.pt
```

Evaluation and resumed training restore `tokenizer.npz` from the checkpoint directory so the same
normalization statistics and codebooks are used.

## Data and tokenization

Dataset sources, split rules, local ImageNet layout, tensor shapes, spectral
transforms, and the tokenizer API are documented in
[datasets/README.md](datasets/README.md).

## Evaluation

Generated images can be evaluated with FID, Inception Score, precision, and
recall using Inception or CLIP features. The configured sample counts are
appropriate for pipeline checks; reliable model comparisons require larger
evaluation sets and controlled training budgets.

## Outputs

Each run is stored under:

```text
runs/<experiment.name>/<UTC timestamp>-<run id>/
```

## Tests

Run the complete suite:

```bash
python -m pytest
```

Run only the executable notebook contract:

```bash
python -m pytest tests/test_tokenizer.py
python -m pytest tests/test_model.py
python -m pytest tests/test_prepare_celeba.py
```

## Documentation

- [Dataset loading and spectral tokenization](datasets/README.md)
- [Experiment configurations and inheritance](configs/README.md)
- [Complete shared configuration reference](configs/base.yaml)
