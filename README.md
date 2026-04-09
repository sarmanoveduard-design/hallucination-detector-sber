# Детектор галлюцинаций LLM (Sber Hackathon)

## Описание

Проект реализует систему детекции фактологических галлюцинаций в ответах LLM.

Детектор оценивает вероятность галлюцинации на основе:
- `prompt`
- `model_answer`
- optional `correct_answer`

Система поддерживает два режима:
1. Без эталонного ответа — работает по `prompt + model_answer`
2. С эталонным ответом — использует `correct_answer` как дополнительный сигнал

---

## Основная идея

Вместо внешней фактчекинг-верификации через API система оценивает:
- согласованность `prompt` и ответа
- структурные признаки ответа
- семантическую правдоподобность ответа

Это позволяет использовать детектор на произвольных доменах.

---

## Используемые признаки

### 1. Структурные признаки prompt и ответа
- длина `prompt`
- длина ответа
- отношение длин
- пересечение токенов
- наличие числовых расхождений
- наличие вопросительных конструкций

### 2. Признаки схожести ответа и correct_answer
Используются только если `correct_answer` существует:
- точное совпадение
- вхождение строки
- обратное вхождение
- разница длин

### 3. Семантические признаки
Используется модель:

`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

Сигналы:
- semantic similarity `prompt ↔ answer`
- semantic similarity `answer ↔ correct_answer`
- флаг наличия `correct_answer`

---

## Модель

Используется:
- `LogisticRegression`

Причины выбора:
- быстрая
- стабильная
- легко интерпретируемая
- подходит для lightweight production inference

---

## Итоговые результаты (preview dataset)

- **PR-AUC:** `0.907556`
- **Best threshold:** `0.45`
- **Best F1:** `0.8400`
- **Accuracy:** `0.8086`

---

## Latency

### Cold start
Около `5.5 секунд`

Причина:
- загрузка sentence embedding модели

Cold start происходит один раз при запуске сервиса.

### Warm inference
- `1 sample` ≈ `38 ms`
- `10 samples` ≈ `28 ms` на sample
- `50 samples` ≈ `23 ms` на sample

Вывод:
- warm inference укладывается в ограничение `500 ms`

---

## Структура проекта

```text
configs/
data/
models/
notebooks/
src/

README.md
requirements.txt
.gitignore
Dockerfile
.dockerignore
run_all.ps1

Основные файлы

src/
  predict_detector.py
  prompt_answer_features.py
  run_public_inference.py
  runtime.py
  semantic_features.py
  train_full_detector.py

Установка
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

Обучение модели
python src/train_full_detector.py

Инференс
python src/predict_detector.py

Прогон по публичному CSV
python src/run_public_inference.py

Входной файл
data/knowledge_bench_public.csv

Выходной файл
data/knowledge_bench_public_predictions.csv

В выходном CSV добавляются колонки:
hallucination_probability
prediction

Быстрый полный прогон
Для полного прогона основных этапов можно запустить:
.\run_all.ps1

Скрипт последовательно выполнит:

обучение модели
поиск лучшего threshold
полную оценку решения
benchmark latency
пример инференса

Запуск через Docker
Сборка образа
docker build -t hallucination-detector-sber .

Запуск контейнера
docker run --rm hallucination-detector-sber

После запуска результат сохраняется в контейнере в файл:
/app/data/knowledge_bench_public_predictions.csv

Архитектура решения
Pipeline:

prompt + model_answer (+ optional correct_answer)
→ feature extraction
→ LogisticRegression
→ hallucination probability
→ threshold prediction

Почему решение подходит для production
-не требует внешних API
-работает без correct_answer
-переносится на другие домены
-быстрый warm inference
-простое развертывание
-низкая сложность эксплуатации

Ограничения
-cold start из-за embedding модели
-возможны false positives при перефразированных ответах
-модель оценивает вероятность галлюцинации, а не абсолютную истину

Возможные улучшения
-alibration
-cross-encoder verification
-ансамбли моделей
-uncertainty estimation
-API wrapper