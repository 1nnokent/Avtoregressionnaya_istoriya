import ast
from pathlib import Path

import nbformat
from nbclient import NotebookClient


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "mnist_pipeline_demo.ipynb"
REQUIRED_SECTIONS = (
    "## 1. Подготовка",
    "## 2. Разложения",
    "## 3. Порядки",
    "## 4. Нормализация",
    "## 5. Квантизация",
    "## 6. Итоговый токенизатор",
)


def test_notebook_has_required_sections_and_compilable_code():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    markdown = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "markdown"
    )

    for section in REQUIRED_SECTIONS:
        assert section in markdown

    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            compile(cell.source, f"{NOTEBOOK.name}:cell-{index}", "exec")


def test_notebook_covers_all_decompositions_and_orderings(monkeypatch):
    monkeypatch.setenv("MNIST_TRAIN_LIMIT", "200")
    monkeypatch.setenv("MNIST_VAL_LIMIT", "80")
    monkeypatch.setenv("MNIST_TEST_LIMIT", "80")
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    notebook.cells.append(
        nbformat.v4.new_code_cell(
            """
assert tuple(FRACTIONS) == (0.05, 0.10, 0.25, 0.50)
assert set(decomposition_results) == set(DECOMPOSITIONS)
assert SELECTED_DECOMPOSITION == min(
    DECOMPOSITIONS,
    key=lambda name: decomposition_results[name]["mse_by_fraction"][0.25],
)
assert set(ordering_results) == set(ORDERINGS)
assert selected_ordering == min(
    ORDERINGS,
    key=lambda name: ordering_results[name]["mse_by_fraction"][0.25],
)
for result in (*decomposition_results.values(), *ordering_results.values()):
    assert set(result["mse_by_fraction"]) == set(FRACTIONS)
    assert all(np.isfinite(value) for value in result["mse_by_fraction"].values())
"""
        )
    )

    NotebookClient(
        notebook,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK.parent.parent)}},
    ).execute()


def test_notebook_covers_normalization_and_quantization_matrix(monkeypatch):
    monkeypatch.setenv("MNIST_TRAIN_LIMIT", "200")
    monkeypatch.setenv("MNIST_VAL_LIMIT", "80")
    monkeypatch.setenv("MNIST_TEST_LIMIT", "80")
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    notebook.cells.append(
        nbformat.v4.new_code_cell(
            """
expected_normalization = {
    (transform, scaling, per_position)
    for transform in TRANSFORMS
    for scaling in SCALINGS
    for per_position in (False, True)
}
assert set(normalization_results) == expected_normalization
assert selected_normalization["key"] == min(
    normalization_results,
    key=lambda key: normalization_results[key]["mse"],
)
assert selected_normalization["config"]["log_scale"] == 255.0

expected_quantization = {
    (num_bins, per_position)
    for num_bins in (16, 32, 64, 128, 256)
    for per_position in (False, True)
}
assert set(quantization_results) == expected_quantization
assert selected_quantization["key"] == min(
    quantization_results,
    key=lambda key: quantization_results[key]["mse"],
)
for result in (*normalization_results.values(), *quantization_results.values()):
    assert np.isfinite(result["mse"])
    assert np.isfinite(result["psnr"])
"""
        )
    )

    NotebookClient(
        notebook,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK.parent.parent)}},
    ).execute()


def test_notebook_final_tokenizer_contract(monkeypatch):
    monkeypatch.setenv("MNIST_TRAIN_LIMIT", "200")
    monkeypatch.setenv("MNIST_VAL_LIMIT", "80")
    monkeypatch.setenv("MNIST_TEST_LIMIT", "80")
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    notebook.cells.append(
        nbformat.v4.new_code_cell(
            """
assert psnr(test_images, test_images) == float("inf")
assert np.issubdtype(test_tokens.dtype, np.integer)
assert test_tokens.shape == (len(test_images), 28 * 28)
assert 0 <= test_tokens.min() <= test_tokens.max() < final_tokenizer.vocab_size
assert test_reconstruction.shape == (len(test_images), 1, 28, 28)
assert np.isfinite(test_reconstruction).all()
assert np.isfinite(final_metrics["mse"])
assert np.isfinite(final_metrics["psnr"])
assert final_config.decomposition == SELECTED_DECOMPOSITION
assert final_config.decomposition_params == selected_decomposition_params
assert final_config.ordering == selected_ordering
assert final_config.ordering_params == final_ordering_params
assert final_config.normalization == selected_normalization["config"]
assert final_config.quantization == selected_quantization["config"]

train_fitted_reference = Tokenizer(final_config).fit(train_images)
np.testing.assert_allclose(final_tokenizer.normalizer.offset, train_fitted_reference.normalizer.offset)
np.testing.assert_allclose(final_tokenizer.normalizer.spread, train_fitted_reference.normalizer.spread)
np.testing.assert_allclose(final_tokenizer.quantizer.thresholds, train_fitted_reference.quantizer.thresholds)
"""
        )
    )

    executed = NotebookClient(
        notebook,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK.parent.parent)}},
    ).execute()

    final_cell = next(cell for cell in executed.cells if cell.id == "fit-final-tokenizer")
    output = "".join(
        part
        for item in final_cell.outputs
        if item.output_type == "stream"
        for part in item.text
    )
    token_line = next(
        (line for line in output.splitlines() if line.startswith("Первые 32 токена первой test-цифры:")),
        None,
    )
    assert token_line is not None
    shown_tokens = ast.literal_eval(token_line.split(":", maxsplit=1)[1].strip())
    assert len(shown_tokens) == 32
    assert all(isinstance(token, int) and token >= 0 for token in shown_tokens)

