# Image Datasets and Tokenization

This module loads images and converts them into sequences of discrete tokens for
the autoregressive model. The inverse transformation reconstructs images from
tokens; only the quantization stage is lossy.

The pipeline is:

```text
NCHW image
    -> spectral decomposition
    -> coefficient ordering
    -> sequence of length C * H * W
    -> normalization
    -> uniform quantization
    -> integer tokens
```

## Quick Start

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

Minimal example:

```python
from datasets.src.datasets_loader import load_images
from datasets.src.tokenizer import Tokenizer

train_images = load_images("mnist", "train", limit=10_000)
val_images = load_images("mnist", "val", limit=1_000)

tokenizer = Tokenizer().fit(train_images)
tokens = tokenizer.encode(val_images)
restored_images = tokenizer.decode(tokens)

print(train_images.shape)     # (10000, 1, 28, 28)
print(tokens.shape)           # (1000, 784)
print(restored_images.shape)  # (1000, 1, 28, 28)
```

## Loading Images

The main entry point is `datasets.src.datasets_loader.load_images`:

```python
images = load_images(
    ds_name="cifar10",
    split="train",
    gray=False,
    size=64,
    limit=None,
    val_fraction=0.1,
    seed=0,
    root="data",
)
```

It returns a `float32` NumPy array in the `[0, 1]` range and
`(N, C, H, W)` layout.

| Dataset | Data source | Size | Channels with `gray=False` |
|---|---|---:|---:|
| MNIST | downloaded automatically through `torchvision` | 28 × 28 | 1 |
| CIFAR-10 | downloaded automatically as Parquet from Hugging Face | 32 × 32 | 3 |
| CelebA | downloaded automatically as Parquet from Hugging Face | `size` × `size` | 3 |
| ImageNet | downloaded manually and stored locally in `ImageFolder` format | `size` × `size` | 3 |

`gray=True` is the default. An RGB image is converted to luminance using
`0.299 R + 0.587 G + 0.114 B`, producing an array of shape
`(N, 1, H, W)`.

For MNIST, CIFAR-10, and ImageNet, the training portion is split
deterministically into `train` and `val` according to `val_fraction` and
`seed`. CelebA uses its predefined splits. `limit` is applied after
shuffling and is useful for quick experiments.

ImageNet is not downloaded automatically. The following directory layout is
expected:

```text
data/
└── imagenet/
    ├── train/
    │   ├── class_a/
    │   └── class_b/
    └── val/
        ├── class_a/
        └── class_b/
```

ImageNet provides labeled `train` and `val` directories.
The loader splits the original `train` directory into the project’s training and validation subsets.
Because the official test labels are unavailable, the original `val` directory is used as the project’s test subset.

> The current CelebA loader uses one Parquet shard for each split rather than
> every shard in the full dataset.

## Tokenizer

`Tokenizer` combines four independent stages: decomposition, coefficient
ordering, normalization, and quantization.

### 1. Decomposition

| Value | Transformation | Additional parameters |
|---|---|---|
| `dct` | orthonormal two-dimensional DCT-II | — |
| `block_dct` | independent DCT inside each block | `block_size`, default: `8` |
| `dft` | real-valued DFT representation through the Hartley transform | — |
| `wavelet` | two-dimensional wavelet decomposition | `wavelet`, `level` |

All transformations operate only on the last two spatial axes. Batches and
channels are therefore processed independently, and the array shape is
preserved.

For `block_dct`, both the image height and width must be divisible by
`block_size`.

### 2. Coefficient Ordering

The following `ordering` values are available:

- `zigzag` — diagonal traversal from low to high frequencies, as in JPEG;
- `radial` — ordering by normalized distance from the origin;
- `dyadic` — gradual expansion through squares whose sides are powers of two;
- `block` — blocks first, followed by zigzag ordering within each block.

### 3. Normalization

The available transforms are `none` and `signed_log`. The available scaling
methods are `minmax` and `standard`.

### 4. Quantization

The current implementation provides uniform scalar quantization through the
`uniform` method. `num_bins` defines both the number of bins and the
vocabulary size. Token values range from `0` to `num_bins - 1`.

With `per_position=false`, all sequence values share the same bin boundaries.
With `per_position=true`, each sequence position gets its own boundaries and
centers, but the token at each position is still a single scalar.

## Multi-Channel Images

The tokenizer accepts grayscale images without a channel axis, in
`(N, H, W)` layout, or images in NCHW `(N, C, H, W)` layout. NHWC is not
supported.

For RGB input, each channel is decomposed independently. After ordering, the
channel sequences are concatenated into one sequence of length `C * H * W`:

```text
[R coefficients] + [G coefficients] + [B coefficients]
```

## Configuration

`Tokenizer` accepts either a `TokenizerConfig` instance or a plain
dictionary. Omitted fields retain their default values.

```yaml
tokenizer:
  decomposition: dct       # dct | block_dct | dft | wavelet
  decomposition_params: {}
  ordering: zigzag         # zigzag | radial | dyadic | block
  ordering_params: {}
  normalization:
    transform: signed_log  # none | signed_log
    log_scale: 255.0
    scaling: minmax        # minmax | standard
    per_position: false
  quantization:
    method: uniform
    num_bins: 256
    per_position: false
```

Python example:

```python
from datasets.src.tokenizer import Tokenizer

tokenizer = Tokenizer(
    {
        "decomposition": "block_dct",
        "decomposition_params": {"block_size": 8},
        "ordering": "block",
        "ordering_params": {"block_size": 8},
        "normalization": {
            "transform": "signed_log",
            "log_scale": 255.0,
            "scaling": "minmax",
            "per_position": True,
        },
        "quantization": {
            "method": "uniform",
            "num_bins": 256,
            "per_position": True,
        },
    }
)
```

The main properties available after `fit()` are:

- `tokenizer.image_shape` — the `(C, H, W)` shape of the training images;
- `tokenizer.sequence_length` — `C * H * W`;
- `tokenizer.vocab_size` — `num_bins`.

`encode()` rejects images whose shape differs from the shape used to fit the
tokenizer.

## Module Layout

```text
datasets/
├── configs/                 # standalone configuration examples
├── src/
│   ├── datasets_loader.py   # dataset loading and splitting
│   ├── decomposition.py     # DCT, block DCT, DFT/Hartley, and wavelets
│   ├── ordering.py          # coefficient ordering
│   ├── normalization.py     # signed log and scaling
│   ├── quantization.py      # uniform quantization
│   └── tokenizer.py         # complete forward and inverse pipeline
└── tests/                   # unit tests
```

## Tests

Run the module tests from the repository root:

```bash
python -m pytest datasets/tests
```

The tests cover transform invertibility, shape and energy preservation,
coefficient orderings, both `per_position` modes, quantization error, and the
full `encode/decode` cycle for grayscale and RGB input.

Tests that require real datasets are skipped automatically when the
corresponding files have not been downloaded. Running the test suite itself
does not download any data.

To test the module together with the model, shared configuration, and tokenizer
state persistence:

```bash
python -m pytest
```
