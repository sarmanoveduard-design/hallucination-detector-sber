# Детектор галлюцинаций LLM (Sber Hackathon)

## Описание

Проект реализует систему детекции фактологических галлюцинаций в ответах LLM.

Детектор оценивает вероятность галлюцинации на основе:

- prompt
- model_answer
- optional correct_answer

Система поддерживает два режима:

1 Режим без эталонного ответа  
Работает только по prompt + model_answer

2 Режим с эталонным ответом  
Использует correct_answer как дополнительный сигнал

---

## Основная идея

Вместо проверки знаний через внешние источники система оценивает:

- согласованность prompt и ответа
- структурные признаки ответа
- семантическую правдоподобность ответа

Это позволяет использовать детектор на произвольных доменах.

---

## Используемые признаки

### 1 Структурные признаки prompt и ответа

Примеры:

- длина prompt
- длина ответа
- отношение длин
- пересечение токенов
- наличие числовых расхождений
- наличие вопросительных конструкций

### 2 Признаки схожести ответа и correct_answer

Используются только если correct_answer существует:

- точное совпадение
- вхождение строки
- обратное вхождение
- разница длин

### 3 Семантические признаки

Используется модель:

sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

Сигналы:

- semantic similarity prompt ↔ answer
- semantic similarity answer ↔ correct_answer
- флаг наличия correct_answer

---

## Модель

Используется:

LogisticRegression

Причины выбора:

- быстрая
- стабильная
- легко интерпретируемая
- подходит для production inference

---

## Итоговые результаты (preview dataset)

PR-AUC:

0.907556

Best threshold:

0.45

Best F1:

0.8400

Accuracy:

0.8086

---

## Latency

### Cold start

Около:

5.5 секунд

Причина:

загрузка sentence embedding модели.

Cold start происходит один раз при запуске сервиса.

---

### Warm inference

1 sample:

≈ 38 ms

10 samples:

≈ 28 ms на sample

50 samples:

≈ 23 ms на sample

Вывод:

warm inference укладывается в ограничение 500 ms.

---

## Структура проекта
configs/
data/
models/
notebooks/
src/

README.md
requirements.txt
.gitignore


---

## Основные файлы


src/

benchmark_latency.py
evaluate_solution.py
find_best_threshold.py
predict_detector.py
prompt_answer_features.py
runtime.py
semantic_features.py
train_full_detector.py

experiments/

debug_sample_predictions.py
inspect_dataset.py
smoke_test.py
train_baseline_detector.py
train_similarity_detector.py


---

## Установка


python -m venv .venv
..venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt


---

## Обучение модели


python src/train_full_detector.py


---

## Поиск threshold


python src/find_best_threshold.py


---

## Инференс


python src/predict_detector.py


---

## Проверка latency


python src/benchmark_latency.py


---

## Полная оценка решения


python src/evaluate_solution.py


---

## Архитектура решения

Pipeline:

prompt + model_answer (+ optional correct_answer)

↓

feature extraction

↓

LogisticRegression

↓

hallucination probability

↓

threshold prediction

---

## Почему решение подходит для production

- не требует внешних API
- работает без correct_answer
- переносится на другие домены
- быстрый warm inference
- простое развертывание
- низкая сложность эксплуатации

---

## Ограничения

- cold start из-за embedding модели
- возможны false positives при перефразированных ответах
- модель оценивает вероятность галлюцинации, а не абсолютную истину

---

## Возможные улучшения

- calibration
- cross-encoder verification
- ансамбли моделей
- uncertainty estimation
- API сервис

## Быстрый полный прогон

Для полного прогона всех основных этапов можно запустить:

```powershell
.\run_all.ps1

Скрипт последовательно выполнит:

обучение модели
поиск лучшего threshold
полную оценку решения
benchmark latency
пример инференса
