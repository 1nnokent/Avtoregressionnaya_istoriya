# Быстрый старт

Команды выполняются из корня проекта. Нужны Python 3.12 и `uv`.

## 1. Окружение

Если `.venv` уже настроено, пропустите установку. Иначе:

```powershell
./scripts/setup.ps1
```

## 2. Проверка без обучения

```powershell
.venv/Scripts/python.exe run_experiment.py validate --config configs/mnist_baseline.yaml
```

## 3. Как менять конфиг

Создайте `configs/my_experiment.yaml`; укажите только отличия от `base.yaml`:

```yaml
extends: base.yaml
experiment:
  name: my_experiment
data:
  ds_name: mnist
tokenizer:
  decomposition: dct
  max_tokens: 128
model:
  params:
    d_model: 128
    n_layers: 4
training:
  batch_size: 32
  learning_rate: 0.0003
  max_steps: 1000
```

`data` выбирает данные, `tokenizer` — представление, `model.params` — архитектуру, `training` — обучение, `metrics` — оценки. Остальные значения наследуются. Размер словаря и длина входа модели вычисляются автоматически.

Разовую настройку можно передать через `--set`:

```powershell
.venv/Scripts/python.exe run_experiment.py validate --config configs/my_experiment.yaml --set training.batch_size=16
```

## 4. Запуск обучения — когда потребуется

Эта команда действительно обучает модель, затем генерирует изображения и считает метрики:

```powershell
.venv/Scripts/python.exe run_experiment.py train --config configs/my_experiment.yaml
```

Результаты сохраняются в `runs/<имя>/<run-id>/`. Графики:

```powershell
.venv/Scripts/tensorboard.exe --logdir runs
```

`smoke.yaml` тоже выполняет обучение при команде `train`. Для проверки без обучения используйте `validate`.
