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
assert SELECTED_DECOMPOSITION == "dct"
assert set(ordering_results) == set(ORDERINGS)
assert selected_ordering in ORDERINGS
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

