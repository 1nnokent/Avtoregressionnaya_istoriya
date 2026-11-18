# Experiment configurations

`run_experiment.py` accepts complete experiment configurations stored directly
in this directory:

```bash
python run_experiment.py --config configs/mnist.yaml --dry-run
```

## Directory roles

```text
configs/
  base.yaml       complete defaults and the MNIST DCT baseline
  *.yaml          runnable experiment entry points
  presets/        reusable fragments for independent experiment axes
  datasets/       original component settings; not runner configurations
  model/          original component settings; not runner configurations
```

The location of a file defines its role:

- `base.yaml` contains every section required by the runner. It is both the
  common parent and a complete MNIST experiment using DCT and uniform
  quantization with 256 bins.
- YAML files directly under `configs/` are runnable experiment entry points.
  Some also act as parents: `cifar10.yaml` and `celeba.yaml` define the dataset,
  the default RGB transform, and baseline settings for their experiment
  families.
- `presets/` contains partial dictionaries that represent one independent
  choice. `channel_*.yaml` selects a channel transform; `kmeans_*.yaml` selects
  K-means grouping and vocabulary size. Presets are not runnable on their own.
- `datasets/` and `model/` preserve settings from the original standalone
  components. They use different schemas and are not accepted by the shared
  experiment runner.

## Inheritance and merge order

An `extends` value may be one relative path or a list of relative paths. Each
path is resolved from the YAML file that declares it.

```yaml
extends:
  - cifar10.yaml
  - presets/channel_ycbcr.yaml
  - presets/kmeans_joint_768.yaml
experiment:
  name: cifar10_ycbcr_dct_kmeans_joint_768
```

Parents are loaded and merged from left to right. The child is merged last.
Dictionaries are merged recursively; any other value is replaced by the later
value.

The example resolves in four steps:

1. `cifar10.yaml` selects CIFAR-10, RGB, DCT, and the shared baseline settings.
2. `channel_ycbcr.yaml` replaces RGB with YCbCr.
3. `kmeans_joint_768.yaml` replaces uniform quantization with joint-channel
   K-means using 768 centroids.
4. The child assigns the final experiment name.

Later parents should override earlier parents only intentionally. Channel and
quantization presets currently modify different leaves of `tokenizer.params`,
so they compose without discarding one another.

## Available presets

| Preset | Effect |
|---|---|
| `channel_ycbcr.yaml` | sets `channel_transform: ycbcr` |
| `kmeans_scalar_256.yaml` | scalar-channel K-means, 256 centroids |
| `kmeans_joint_256.yaml` | joint-channel K-means, 256 centroids |
| `kmeans_joint_768.yaml` | joint-channel K-means, 768 centroids |

RGB is set by the CIFAR-10 and CelebA family configurations. MNIST omits
`channel_transform`; after fitting, the tokenizer infers the grayscale
transform from its single input channel.

## Naming

Extended experiment names follow this axis order:

```text
<dataset>[_<non-default-channel>]_<decomposition>_<quantizer>_<grouping>_<K>.yaml
```

RGB is the default transform for the color-dataset families and is omitted
from their file names. Non-default transforms such as `ycbcr` are explicit.
The short `mnist.yaml`, `cifar10.yaml`, and `celeba.yaml` files are baseline
entry points.

For a new extended experiment, `experiment.name` should match the file name
without the `.yaml` suffix.
