# Авторегрессионная генерация изображений

Объединены ветки `model`, `data/baseline` и `metrics`: изображения → спектральные коэффициенты → токены → causal Transformer → изображения → метрики. Данные, архитектура, преобразования и метрики выбираются YAML-конфигом; запуск выполняет один скрипт.

Короткая [инструкция по запуску и изменению конфигов](QUICKSTART.md). Подробная [техническая спецификация](docs/TECHNICAL_SPEC.md). Для проверки без обучения используйте `validate`; подкоманда `train`, включая smoke-конфиг, действительно выполняет обучение.

## Окружение и запуск

Основная среда — **локально, Python 3.12**, CPU или NVIDIA CUDA. Первый бейзлайн рассчитан на CPU: доступен Intel i5-13500H без NVIDIA GPU. Colab/Kaggle для воспроизведения не требуются. Большие RGB-эксперименты следует запускать с CUDA: attention и генерация без KV-кэша дороги на длинных последовательностях.

Windows PowerShell из корня проекта, с установленным `uv`:

```powershell
./scripts/setup.ps1
.venv/Scripts/python.exe run_experiment.py train --config configs/mnist_baseline.yaml
```

Для NVIDIA с драйвером, поддерживающим CUDA 12.8: `./scripts/setup.ps1 -Device cu128`. Совместимые версии PyTorch 2.9.1 / torchvision 0.24.1 выбраны по [официальной таблице PyTorch](https://pytorch.org/get-started/previous-versions/).

Linux/macOS или установка без `uv`:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
# CPU Linux/Windows; для macOS эти версии устанавливаются из обычного PyPI
python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-dev.txt
python run_experiment.py train --config configs/mnist_baseline.yaml
```

`requirements.txt` фиксирует основные зависимости; `requirements-dev.txt` добавляет pytest/ruff; `requirements-metrics.txt` — необязательный CLIP. `requirements-lock-cpu.txt` содержит **все точные версии** проверенного CPU-окружения, включая транзитивные зависимости. Воспроизведение: `uv pip install --python .venv/Scripts/python.exe -r requirements-lock-cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu`. Setup-скрипт хранит кэш в `data/.cache/uv`, рядом с проектом.

```powershell
# Проверить конфиг без скачивания данных
.venv/Scripts/python.exe run_experiment.py validate --config configs/mnist_baseline.yaml
# Короткая проверка всего конвейера на настоящем MNIST
.venv/Scripts/python.exe run_experiment.py train --config configs/smoke.yaml
# Новый эксперимент через конфиг/переопределения
.venv/Scripts/python.exe run_experiment.py train --config configs/mnist_baseline.yaml --set experiment.name=mnist_wavelet --set tokenizer.decomposition=wavelet --set training.max_steps=600
# Повторная оценка сохранённого запуска без обучения
.venv/Scripts/python.exe run_experiment.py evaluate --run-dir runs/mnist_dct64_baseline/<run-id> --device cpu
# Просмотр логов
.venv/Scripts/tensorboard.exe --logdir runs
```

Совместимые точки входа: `train.py --config ...` и `python -m src.evalute --run-dir ...`. Каталоги запусков не перезаписываются. При ошибке в `metadata.json` записывается `status: failed`.

## Структура проекта и конфигов

```text
configs/
  base.yaml               общие настройки
  mnist_baseline.yaml     локальный DCT64-бейзлайн
  mnist_dct_zigzag.yaml    все 784 коэффициента, словарь 256
  mnist_block_dct.yaml     блоки 7×7, полная последовательность
  cifar10.yaml            RGB/Inception, требует больше ресурсов
  smoke.yaml              короткая сквозная проверка
src/
  config.py               наследование, валидация, создание модели
  datasets_loader.py      MNIST/CIFAR10/CelebA/ImageNet/NPZ
  decomposition.py        DCT, block DCT, DFT (Хартли), wavelet
  ordering.py             zigzag, radial, dyadic, block
  tokenizer.py            fit/encode/decode/save/load, grayscale и RGB
  model.py                causal Transformer, RoPE, BOS
  train.py                обучение, NLL, checkpoints best/latest
  generation.py           генерация
  evaluation.py           единый интерфейс оценки изображений
  metrics/                энкодеры и формулы метрик
  experiment.py           весь цикл эксперимента
tests/                    unit и сквозной offline-тест
reports/baseline/          результаты настоящего MNIST-бейзлайна
runs/<name>/<run-id>/      полные локальные артефакты, исключены из Git
```

Новый YAML наследует `base.yaml` через `extends: base.yaml`. Вложенные значения объединяются; `--set dotted.key=value` меняет существующее поле. Неизвестные ключи и некорректные основные параметры отклоняются. Параметры классов/преобразований допускают собственные аргументы.

| Раздел | Настройки |
| --- | --- |
| `experiment` | имя, seed, output_dir |
| `runtime` | auto/cpu/cuda/mps, CPU-потоки, workers |
| `data` | ds_name, root, размер, gray, доля val, лимиты split, download |
| `tokenizer` | decomposition/params, ordering/params, max_tokens, normalization, quantization |
| `model` | class_path, config_class, params |
| `training` | AdamW, batch, epochs, max_steps, интервалы логов/val, clip |
| `generation` | количество, batch, temperature, top_k |
| `metrics` | encoders, scores, knn_k, параметры MNIST-классификатора |
| `logging` | TensorBoard; JSONL записывается всегда |

Нормализация: `signed_log/none`. Квантование: `uniform/quantile`, `num_bins`, `per_position` (диапазон/квантили на каждой позиции либо общий). `max_tokens: null` сохраняет все коэффициенты; усечённые коэффициенты заполняются точными нулями до обратного преобразования. Для RGB коэффициенты каналов идут по очереди: усечение менее H×W сохранит только первый канал, поэтому RGB-конфиг использует полную последовательность. Размер словаря и длина модели автоматически выводятся из токенизатора.

Варианты Transformer задаются только `model.params`. Для подключения уже реализованной другой модели укажите её `class_path` и `config_class`: при смене класса конфига параметры предыдущей архитектуры сбрасываются. Контракт: конструктор принимает конфиг с `vocab_size/max_seq_len`; модель имеет `config`, `bos_token_id`, `forward(int64[B,L]) -> logits[B,L,V]`, `generate(num_samples, seq_len, device, temperature, top_k) -> int64[B,L]`. Реализация новой архитектуры требует кода, её подключение к runner — только конфига.

MNIST/CIFAR10 скачиваются torchvision. CelebA ожидается в `data/celeba` в формате torchvision, ImageNet — `data/imagenet/{train,val}/<class>/<image>`. Они автоматически не скачиваются. Размер/gray применяются ко всем наборам. Для собственного набора: `data.ds_name: npz`, `data.root: path/to/images.npz`; ключи `train_images/test_images` — uint8 NHW/NCHW, необязательные `train_labels/test_labels`. Validation выделяется из train; официальный test независим. Лимиты применяются после разбиения. Выбранные изображения/токены загружаются в RAM; полный ImageNet потребует потокового загрузчика.

## Логи, метрики и воспроизводимость

Выбраны **TensorBoard + JSONL**: локальные графики без аккаунта и передачи данных внешнему сервису. Скалярные имена общие для обучения/метрик; используется [SummaryWriter PyTorch](https://docs.pytorch.org/docs/stable/tensorboard.html).

Каждый запуск сохраняет `config.yaml`, `metadata.json` (Git revision/dirty, Python, пакеты, device, SHA256 данных), `tokenizer.json`, `best.pt`, `latest.pt`, `metrics.jsonl`, TensorBoard events, `summary.json`, `generated.npz`, `samples.png`, `real.png`, `reconstruction.png`, `loss.png`. Конфиг сохраняется после разрешения наследования и CLI. Лучший checkpoint выбирается по validation NLL; финальная итерация тоже оценивается. Checkpoints содержат состояние AdamW и номер шага; автоматическое продолжение прерванного запуска пока не реализовано.

Токенизатор подбирается только на train. MNIST-классификатор тоже обучается только на train, выбирается по val и сохраняется как `mnist_encoder.pt`. Test используется только для итоговых оценок. `evaluate` проверяет SHA256 данных и загружает сохранённые состояния. Seed фиксируется; повторяемость проверена на CPU. Побитовая идентичность между разными backend/версиями не гарантируется.

| Энкодер | Назначение |
| --- | --- |
| `mnist` | локальный CNN, 64 признака/10 классов, основной бейзлайн |
| `pixels` | изображение 8×8 без весов/вероятностей классов, smoke/диагностика |
| `inception` | torchvision Inception-v3 ImageNet, веса скачиваются при первом вызове |
| `clip` | CLIP ViT-B/32, требует requirements-metrics.txt и metrics.clip_prompts |

Доступны `metrics.scores`: `frechet`, `classifier_score`, `precision`, `recall`. Имена результатов включают энкодер. Fréchet устойчив к вырожденным ковариациям; kNN считается блоками, `knn_k` должен быть меньше real/fake sample count. `classifier_score_fake/real` = exp(mean KL(p(y|x)||p(y))); это Inception Score только с Inception. MNIST/CLIP используют аналогичную формулу своего классификатора. MNIST/пиксельный Fréchet **не является стандартным FID**. Torchvision Inception тоже отличается от специальных clean-fid/pytorch-fid: числа нельзя напрямую сравнивать с их публикациями. Малые выборки дают смещённые и шумные оценки.

Дополнительно: test NLL/perplexity/token accuracy/bits per token и MSE/PSNR восстановления токенизатора. Bits per token относится к квантованным коэффициентам, это не pixel bpd. NLL между разными токенизаторами не сравнивают напрямую.

## Проверки и результаты

```powershell
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/ruff.exe check .
```

Сквозной offline-тест: локальный NPZ → обучение → checkpoint → генерация → метрики → повторная оценка. Также проверяются разбиения, причинность attention, преобразования/порядки grayscale/RGB, сериализация, вырожденные признаки, обнаружение замены данных.

Результаты настоящего MNIST-запуска: [reports/baseline/README.md](reports/baseline/README.md).
