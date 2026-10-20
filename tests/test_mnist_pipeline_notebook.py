from pathlib import Path

import nbformat


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

