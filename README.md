# Battlesnake HardML SVM Safe

Этот вариант зафиксирован под runtime-модель **`svm`** и дополнительно усилен hard-guard защитой от out-of-bounds.
Его удобно залить в отдельную ветку `feature/hardml-svm-safe` и деплоить как отдельную змейку для сравнения в Battlesnake.

Внутри оставлен `/debug/compare`, поэтому при локальной отладке можно всё равно посмотреть, как на том же состоянии ходят остальные модели.


Идея проекта: не делать один «магический» ML-бот, а собрать **боевую архитектуру**:

1. **Safety Shield** — сначала отсекаем немедленную смерть: стены, тела, опасный head-to-head, маленькие карманы, тонкие коридоры.
2. **Feature Extractor** — для каждого возможного хода строим 32 признака: пространство, еда, хвост, head-to-head, predator pressure, ловушки.
3. **5 lightweight ML-моделей** — все веса экспортированы в `model_weights.py`, инференс идет в чистом Python без sklearn.
4. **Model Switcher** — можно запускать одну и ту же змейку в разных режимах и сравнивать на Battlesnake.
5. **Fallback Ladder** — если ML/логика падает, бот выбирает запасной безопасный ход.

## Модели внутри

| Runtime name | Что это | Зачем |
|---|---|---|
| `ridge` | Ridge regression по expert-score | стабильный линейный скорер |
| `logistic` | LogisticRegression: лучший ход / не лучший ход | ранжирование кандидатов |
| `svm` | Linear SVM через SGD hinge | агрессивный быстрый ранкер |
| `perceptron` | Perceptron | простой baseline для сравнения |
| `mlp` | маленькая MLP 32→10→1 | нелинейный ранкер |
| `ensemble` | взвешенная смесь 5 моделей | компромиссный режим |
| `expert` | эвристический эксперт без ML | запасной сильный режим |

По умолчанию в этом варианте стоит `svm`, потому что на синтетическом holdout он дал лучший top-1 выбор хода среди легких моделей.

## Результаты обучения

См. `artifacts/model_report.json`. Текущий датасет: 9000 синтетических позиций / 22165 candidate rows.

Ключевые holdout-метрики:

- `svm`: top1_acc ≈ 0.788
- `mlp`: top1_acc ≈ 0.776
- `logistic`: top1_acc ≈ 0.680
- `ensemble`: top1_acc ≈ 0.732
- `ridge`: top1_acc ≈ 0.563
- `perceptron`: top1_acc ≈ 0.581

Важно: это не «истинная вероятность победы», а качество имитации экспертной стратегии на synthetic holdout.

## Быстрый запуск

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python smoke_test.py
python backend.py
```

Проверить внешний вид:

```bash
curl http://localhost:8000/
```

Проверить сравнение моделей на одном состоянии:

```bash
curl -X POST http://localhost:8000/debug/compare \
  -H "Content-Type: application/json" \
  -d @sample_state.json
```

## Запуск разных моделей

Linux/macOS:

```bash
BATTLE_MODEL=svm python backend.py
BATTLE_MODEL=mlp python backend.py
BATTLE_MODEL=logistic python backend.py
BATTLE_MODEL=ensemble python backend.py
BATTLE_MODEL=expert python backend.py
```

Windows PowerShell:

```powershell
$env:BATTLE_MODEL="svm"; python backend.py
$env:BATTLE_MODEL="mlp"; python backend.py
$env:BATTLE_MODEL="ensemble"; python backend.py
```

## Как сравнить на Battlesnake

Самый простой вариант: создать на Render 5 сервисов из одной ветки, но с разными env var:

- `BATTLE_MODEL=svm`
- `BATTLE_MODEL=mlp`
- `BATTLE_MODEL=logistic`
- `BATTLE_MODEL=ensemble`
- `BATTLE_MODEL=expert`

Потом зарегистрировать их как 5 разных змей и прогнать custom games / турнир.

## Обучение заново

Для обучения нужны train-зависимости:

```bash
pip install -r requirements-train.txt
python training/train_models.py --states 12000 --seed 42
python smoke_test.py
```

Скрипт перезапишет:

- `model_weights.py`
- `artifacts/model_report.json`

## Почему synthetic dataset

Стабильного готового табличного датасета с `(game_state, chosen_move, outcome)` для Battlesnake я не нашла. Поэтому в проекте используется imitation-learning pipeline:

1. Генерируем много легальных-ish игровых позиций.
2. Прогоняем их через сильного эвристического эксперта.
3. Для каждого возможного хода считаем признаки.
4. Обучаем 5 маленьких моделей имитировать выбор эксперта.
5. Экспортируем веса в чистый Python.

Такой подход укладывается в 500 мс, потому что на боевом ходе нет sklearn, pandas, numpy и тяжелых моделей.

## Fallback Ladder

`choose_move()` никогда не должен падать. Если что-то идет не так:

1. Основная модель ранжирует safe candidates.
2. Если модель упала — ранжирует `expert`.
3. Если и это упало — пытается идти к хвосту.
4. Если хвост недостижим — выбирает легальный ход с максимальным flood-fill пространством.
5. Если вообще всё плохо — выбирает любой ход, который остается в границах поля. Слепого `up` больше нет, поэтому бот не должен умирать именно по причине `out of bounds`.

## Файлы

```text
backend.py                 Flask API: /, /start, /move, /end, /debug/compare
logic.py                   runtime logic + feature extractor + inference + fallbacks
model_weights.py           экспортированные веса 5 моделей
smoke_test.py              локальная проверка моделей и latency
requirements.txt           runtime-зависимости
requirements-train.txt     train-зависимости
training/train_models.py   генерация датасета + обучение + экспорт весов
artifacts/model_report.json
render.yaml
```


## Safe OOB Guard

В этой версии добавлен финальный защитный слой перед возвратом хода:

1. если выбранный моделью ход безопасен и остается внутри поля — возвращаем его;
2. если модель выбрала ход за границу или в тело — ремонтируем ход через legal fallback с максимальным flood-fill пространством;
3. если все legal-варианты недоступны, возвращаем любой in-bounds ход к центру поля.

Это не гарантирует бессмертие в полностью проигранной позиции, но убирает смерть по причине `out of bounds`.
