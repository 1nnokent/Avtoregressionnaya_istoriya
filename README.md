# Авторегрессионная генерация изображений

Объединены три авторские части: модель, датасеты/токенизация и метрики.
Общий запуск связывает их через YAML-конфиг. Алгоритмы компонентов сохранены.

```text
model/          # ветка model: модель, обучение, исходный конфиг и проверка
datasets/       # ветка data/baseline: загрузка, токенизация, исходные тесты/конфиги
metrics/        # ветка metrics: генерация, энкодеры, формулы метрик
experiments/    # связующий код: конфиги, запуск, состояние, логирование
configs/        # конфиги общих экспериментов с наследованием
scripts/        # установка окружения
tests/          # проверки интеграции без обучения
run_experiment.py
requirements.txt
```

Рабочая платформа — **локальный Python 3.12**; `device: auto` выбирает CUDA,
если она доступна, иначе CPU. Colab/Kaggle для запуска не требуются.

## Окружение и проверка без обучения

Из корня проекта в PowerShell:

```powershell
.\scripts\setup.ps1 -Dev
.\.venv\Scripts\python.exe run_experiment.py --config configs/mnist.yaml --dry-run
.\.venv\Scripts\python.exe -m pytest
```

Установка создаёт/использует `.venv`, устанавливает общие зависимости
и проверяет конфиг. Обучение и скачивание датасетов/весов она не запускает.
В Linux: `python3.12 -m venv .venv`, затем
`.venv/bin/python -m pip install -r requirements-dev.txt`.
Подбор сборки PyTorch под GPU описан в [официальной инструкции](https://pytorch.org/get-started/locally/).
Версии `torch` и `torchvision` нужно менять согласованно.
Авторские `*/requirements.txt` сохранены для истории; устанавливать нужно корневой файл.

## Эксперименты

`python run_experiment.py --config configs/mnist.yaml --dry-run` проверяет
конфиг и фабрики без создания модели, загрузки данных/весов и записи результатов.

Команды ниже предназначены для последующего запуска пользователем:

```powershell
# Обучение; после него — валидация и выбранные метрики.
python run_experiment.py --config configs/mnist.yaml

# Оценка готового checkpoint, без обучения.
python run_experiment.py --config configs/mnist.yaml --mode evaluate --checkpoint runs/<name>/<run>/latest.pt

# Продолжение обучения, с новым каталогом результатов.
python run_experiment.py --config configs/mnist.yaml --checkpoint runs/<name>/<run>/latest.pt
```

Для нового эксперимента скопируйте короткий конфиг с `extends: base.yaml`
и измените нужные поля. Примеры: `mnist_block_dct.yaml`, `cifar10.yaml`,
`mnist_clip.yaml`. Параметры модели — `model.params`, данные — `data.params`,
токенизация — `tokenizer.params`, метрики/энкодеры — `metrics`.
Размер словаря и длина последовательности определяются токенизатором.
Все поля с пояснениями — в [configs/base.yaml](configs/base.yaml).

`target: package.module:ClassOrFunction` выбирает другую уже реализованную
модель, загрузчик или токенизатор без изменения запуска. Интерфейсы должны
совпадать с текущими: модель принимает объект конфига и поддерживает `generate`,
загрузчик возвращает NCHW-массив, токенизатор предоставляет `fit/encode/decode`
и состояние текущих normalizer/quantizer. Новому алгоритму нужен его исходник;
для выбора существующих реализаций достаточно конфига.

Пути данных, результатов и checkpoint считаются от корня проекта;
`extends` — от файла конфига. Токенизатор подгоняется только по train.
При оценке/продолжении берётся `tokenizer.npz` рядом с checkpoint,
его параметры из текущего конфига не переобучаются. Архитектура модели должна
совпадать с checkpoint. Исходный trainer продолжает со следующей эпохи,
даже если checkpoint был сохранён посередине эпохи.

MNIST/CIFAR-10 скачиваются штатным загрузчиком при настоящем запуске.
CelebA/ImageNet нужно положить в `data/` по структуре,
описанной в `datasets/src/datasets_loader.py`. Для ImageNet `val` выделяется
из train, официальный каталог val используется как test.
Исходные энкодеры метрик работают только с grayscale: оставьте `gray: true`
или отключите `metrics.enabled` для RGB. CLIP использует авторские подсказки
цифр, поэтому пример с CLIP предназначен для MNIST.
Первый расчёт метрик скачивает предобученные веса Inception/CLIP.
200 изображений — исходный пример для проверки, не надёжная оценка качества.

## Логирование

Выбран локальный формат **JSONL + console.log**, без внешних сервисов и аккаунтов.
В `runs/<experiment.name>/<UTC-время>-<id>/` сохраняются:

- `config.yaml`, `metadata.json`: полный конфиг, seed, устройство, версии пакетов, Git commit;
- `console.log`, `metrics.jsonl`: вывод обучения, train loss, val loss/perplexity, события запуска;
- `tokenizer.npz`, `latest.pt`, при улучшении валидации — `best.pt`;
- `summary.json`: итоговые val loss/perplexity и выбранные FID, IS, precision/recall по энкодерам.

Используются функции автора из `metrics/`; связующий слой приводит форму
изображений и записывает результаты. Частота train/val-логов задаётся
существующими `log_interval` / `eval_interval`.

Минимальный отчёт об изменениях — [CHANGES.md](CHANGES.md).
