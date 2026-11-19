import io

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from compare import prepare_celeba


def _encoded_image(color):
    buffer = io.BytesIO()
    Image.new("RGB", (4, 6), color).save(buffer, format="PNG")
    return buffer.getvalue()


def _write_shard(path, colors):
    image_type = pa.struct([pa.field("bytes", pa.binary())])
    images = pa.array(
        [{"bytes": _encoded_image(color)} for color in colors],
        type=image_type,
    )
    pq.write_table(pa.table({"image": images}), path)


def test_build_cache_streams_all_project_parquet_shards(monkeypatch, tmp_path):
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    _write_shard(first, [(255, 0, 0), (0, 255, 0)])
    _write_shard(second, [(0, 0, 255)])
    monkeypatch.setattr(
        prepare_celeba,
        "celeba_parquet_paths",
        lambda split, root: [first, second],
    )

    path = prepare_celeba.build_cache("train", tmp_path / "cache")
    images = np.load(path)

    assert images.shape == (3, 64, 64, 3)
    assert images.dtype == np.uint8
    np.testing.assert_array_equal(images[:, 0, 0], [[255, 0, 0], [0, 255, 0], [0, 0, 255]])
    assert not path.with_suffix(".npy.part").exists()
