# MNIST Pipeline Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Создать исполняемый русскоязычный Jupyter-ноутбук, который последовательно сравнивает и выбирает этапы полного MNIST-пайплайна проекта.

**Architecture:** Один ноутбук использует публичные интерфейсы `src` и держит экспериментальные вспомогательные функции локально, чтобы повествование оставалось читаемым. Выборы выполняются только по validation-метрикам; итоговая конфигурация собирается через `Tokenizer` и один раз оценивается на test.

**Tech Stack:** Python 3, Jupyter/nbformat/nbconvert, NumPy, Matplotlib, SciPy, PyWavelets, torchvision, pytest.

**Spec:** `docs/superpowers/specs/2026-10-20-mnist-pipeline-notebook-design.md`

## Global Constraints

- Использовать только MNIST и публичные функции/классы существующих модулей `src`.
- Все пояснения и подписи графиков должны быть на русском языке.
- Параметры выбираются на validation; test используется только в финальном разделе.
- Ноутбук должен выполняться сверху вниз из корня репозитория на CPU.
- Для `block_dct` и порядка `block` на изображениях 28×28 использовать `block_size=7`.
- Для `signed_log` во всех экспериментах и итоговом `Tokenizer` использовать `log_scale=255.0`.
- При сравнении нормализаций использовать общий `Quantizer(num_bins=64, per_position=False)`.
- Не добавлять новые алгоритмы обработки в библиотеку.

## Review Focus

- Чистая среда без `data/MNIST`: загрузка должна скачать данные, а не завершиться из-за отсутствующего каталога.
- Запуск из каталога `notebooks/`: ячейка импорта должна корректно определить корень репозитория.
- Конфигурация `block_dct` на 28×28: размер блока 7 должен исключить ошибку неделимости.
- Метрика PSNR при нулевом MSE: должна возвращать `inf`, а не деление на ноль с исключением.
- Значения test вне диапазона train: квантизатор должен выдавать токены внутри `[0, num_bins)` и конечную реконструкцию.

---

### Task 1: Каркас и контракт ноутбука

**Files:**
- Create: `notebooks/mnist_pipeline_demo.ipynb`
- Create: `tests/test_mnist_pipeline_notebook.py`

**Interfaces:**
- Consumes: публичные константы и классы из `src.decomposition`, `src.ordering`, `src.normalization`, `src.quantization`, `src.tokenizer`, `src.datasets_loader`.
- Produces: валидный ноутбук с разделами `Подготовка`, `Разложения`, `Порядки`, `Нормализация`, `Квантизация`, `Итоговый токенизатор`.

- [ ] **Step 1: Написать падающий структурный тест**

В `tests/test_mnist_pipeline_notebook.py` добавить `test_notebook_has_required_sections_and_compilable_code()`: загрузить JSON через `nbformat`, проверить наличие шести заголовков и скомпилировать исходный текст каждой кодовой ячейки через `compile(...)`.

- [ ] **Step 2: Запустить тест и подтвердить ожидаемое падение**

Run: `pytest tests/test_mnist_pipeline_notebook.py::test_notebook_has_required_sections_and_compilable_code -v`
Expected: FAIL, потому что `notebooks/mnist_pipeline_demo.ipynb` ещё не существует.

- [ ] **Step 3: Создать минимальный ноутбук с конфигурацией и разделами**

Добавить Markdown-заголовки, конфигурационную ячейку с `SEED=42`, `TRAIN_LIMIT=2000`, `VAL_LIMIT=500`, `TEST_LIMIT=500`, импорт публичных API и определение корня репозитория для запуска как из корня, так и из `notebooks/`.

- [ ] **Step 4: Запустить структурный тест**

Run: `pytest tests/test_mnist_pipeline_notebook.py::test_notebook_has_required_sections_and_compilable_code -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add notebooks/mnist_pipeline_demo.ipynb tests/test_mnist_pipeline_notebook.py
git commit -m "test: define MNIST notebook contract"
```

### Task 2: Разложения и порядки

**Files:**
- Modify: `notebooks/mnist_pipeline_demo.ipynb`
- Modify: `tests/test_mnist_pipeline_notebook.py`

**Interfaces:**
- Consumes: `load_images(...)`, `get_decomposition(...)`, `DECOMPOSITIONS`, `ordering_indices(...)`, `ORDERINGS`, `forward_ordering(...)`, `invert_ordering(...)`.
- Produces: `decomposition_results`, фиксированный `SELECTED_DECOMPOSITION = "dct"`, `ordering_results`, `selected_ordering`.

- [ ] **Step 1: Добавить падающий тест покрытия вариантов**

Добавить `test_notebook_covers_all_decompositions_and_orderings()`, который проверяет ссылки на `DECOMPOSITIONS`/`ORDERINGS`, `block_size=7`, доли `(0.05, 0.10, 0.25, 0.50)` и явную фиксацию DCT.

- [ ] **Step 2: Запустить тест и увидеть FAIL**

Run: `pytest tests/test_mnist_pipeline_notebook.py::test_notebook_covers_all_decompositions_and_orderings -v`
Expected: FAIL из-за отсутствующих экспериментов.

- [ ] **Step 3: Реализовать секции загрузки, разложений и порядков**

Добавить функции `mse(original, restored) -> float`, `psnr(original, restored, data_range=1.0) -> float`, `keep_largest(coeffs, fraction) -> np.ndarray`, `reconstruct_prefix(coeffs, order, fraction) -> np.ndarray`; визуализации коэффициентов, карт обхода, реконструкций и validation-графиков. Для `block_dct` передавать `block_size=7`; DCT фиксировать вручную, порядок выбирать по validation MSE при доле 0.25.

- [ ] **Step 4: Запустить тесты ноутбука**

Run: `pytest tests/test_mnist_pipeline_notebook.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add notebooks/mnist_pipeline_demo.ipynb tests/test_mnist_pipeline_notebook.py
git commit -m "feat: compare MNIST decompositions and orderings"
```

### Task 3: Нормализация и квантизация

**Files:**
- Modify: `notebooks/mnist_pipeline_demo.ipynb`
- Modify: `tests/test_mnist_pipeline_notebook.py`

**Interfaces:**
- Consumes: DCT-последовательности в выбранном порядке, `Normalizer`, `Quantizer`.
- Produces: `normalization_results`, `selected_normalization`, `quantization_results`, `selected_quantization`.

- [ ] **Step 1: Добавить падающий тест матрицы конфигураций**

Добавить `test_notebook_covers_normalization_and_quantization_matrix()`: проверить декартово произведение `TRANSFORMS × SCALINGS × (False, True)`, значения бинов `(16, 32, 64, 128, 256)`, оба режима `per_position` и выбор по validation MSE.

- [ ] **Step 2: Запустить тест и увидеть FAIL**

Run: `pytest tests/test_mnist_pipeline_notebook.py::test_notebook_covers_normalization_and_quantization_matrix -v`
Expected: FAIL из-за отсутствующих сравнений.

- [ ] **Step 3: Реализовать сравнение нормализации**

Для каждого сочетания обучить `Normalizer` только на train (`log_scale=255.0` для `signed_log`), применить общий uniform `Quantizer(num_bins=64, per_position=False)` и измерить validation MSE после полного обратного пути; показать распределения и таблицу, выбрать минимум.

- [ ] **Step 4: Реализовать сравнение квантизации**

Для каждой пары `num_bins × per_position` обучить квантизатор на нормализованном train, оценить validation MSE/PSNR после обратного пути, построить графики и выбрать минимум.

- [ ] **Step 5: Запустить тесты ноутбука**

Run: `pytest tests/test_mnist_pipeline_notebook.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add notebooks/mnist_pipeline_demo.ipynb tests/test_mnist_pipeline_notebook.py
git commit -m "feat: compare normalization and quantization"
```

### Task 4: Итоговый токенизатор и исполняемость

**Files:**
- Modify: `notebooks/mnist_pipeline_demo.ipynb`
- Modify: `tests/test_mnist_pipeline_notebook.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: `selected_ordering`, `selected_normalization`, `selected_quantization`, `Tokenizer`.
- Produces: `final_tokenizer`, `test_tokens`, `test_reconstruction`, итоговые test MSE/PSNR и галерею ошибок.

- [ ] **Step 1: Добавить падающий smoke-тест итогового контракта**

Добавить `test_notebook_final_tokenizer_contract()`, проверяющий наличие обучения только на `train_images`, кодирования `test_images`, проверок формы `(N, 1, 28, 28)`, целого типа токенов, диапазона словаря и конечности реконструкции. В тест метрики добавить случай `mse == 0`, для которого PSNR равен `inf`.

- [ ] **Step 2: Запустить тест и увидеть FAIL**

Run: `pytest tests/test_mnist_pipeline_notebook.py::test_notebook_final_tokenizer_contract -v`
Expected: FAIL из-за отсутствующего финального раздела.

- [ ] **Step 3: Собрать и визуализировать итоговый Tokenizer**

Сформировать `TokenizerConfig` с DCT и выбранными validation-конфигурациями, вызвать `fit(train_images)`, затем `encode/decode(test_images)`, проверить контракт токенов и показать итоговую таблицу и галерею `оригинал / реконструкция / |ошибка|`.

- [ ] **Step 4: Исключить временные макеты**

Добавить `.superpowers/` в `.gitignore`, не затрагивая существующие правила.

- [ ] **Step 5: Выполнить весь ноутбук**

Run: `jupyter nbconvert --to notebook --execute notebooks/mnist_pipeline_demo.ipynb --output mnist_pipeline_demo.executed.ipynb --ExecutePreprocessor.timeout=600`
Expected: завершение без traceback; все графики и таблицы сформированы.

- [ ] **Step 6: Запустить полный набор тестов**

Run: `pytest -q`
Expected: все тесты PASS (сетевые тесты могут быть SKIP только по существующим маркерам проекта).

- [ ] **Step 7: Commit**

```bash
git add .gitignore notebooks/mnist_pipeline_demo.ipynb notebooks/mnist_pipeline_demo.executed.ipynb tests/test_mnist_pipeline_notebook.py
git commit -m "feat: add executable MNIST pipeline notebook"
```

