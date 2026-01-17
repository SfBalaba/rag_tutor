# Research модуль для RAG системы

Инструменты для оценки и анализа качества RAG системы.

## 🚀 Установка

Используем uv для управления зависимостями:

```bash
# Основные research зависимости
uv sync --extra research

# Или через pip
pip install -e .[research]
```

## ⚡ Быстрый старт

После установки зависимостей запустите оценку:

```bash
# Оценка системы на тестовых данных
uv run python research/scripts/evaluate.py \
    --mode system \
    --golden-file research/data/test_dataset.csv \
    --output-file results.csv \
    --limit 5

# Сравнение моделей
uv run python research/scripts/evaluate_models.py

# Оценка качества ретривала
uv run python research/scripts/evaluate_retrieval.py \
    --eval-dataset eval_data.jsonl \
    --output-dir results/retrieval

# Генерация синтетических данных
uv run python research/scripts/generate_groundtruth.py \
    --questions-csv questions.csv \
    --output-file synthetic.csv
```

## 📊 Метрики

- **Cosine Similarity** - семантическая близость
- **RAG Triad** (DeepEval): Faithfulness, Answer Relevancy, Contextual Relevancy
- **Correctness** (G-Eval) - правильность ответа
- **BLEURT** - качество генерации текста

## 📁 Структура

```
research/
├── evals/                    # Система оценки
│   ├── __init__.py
│   └── evaluation.py        # Метрики и функции
├── scripts/                 # Скрипты
│   ├── evaluate.py         # Универсальная оценка
│   ├── evaluate_models.py  # Сравнение моделей
│   ├── evaluate_retrieval.py  # Оценка ретривала
│   └── generate_groundtruth.py  # Генерация данных
├── data/
│   └── test_dataset.csv    # Примеры данных
├── config.yaml             # Настройки research
└── README.md               # Документация
```

## 🔗 Интеграция

Research модуль полностью интегрирован с основной RAG системой:

- Использует `core.config` для основных настроек
- Работает с `core.vector_store` для доступа к данным
- Поддерживает `core.llm` модели из основного проекта
- Конфигурация в `research/config.yaml` дополняет основную

## 🔧 Использование

### 1. Оценка системы

```bash
uv run python research/scripts/evaluate.py \
    --mode system \
    --golden-file golden_data.csv \
    --output-file results.csv
```

### 2. Оценка ретривала

```bash
uv run python research/scripts/evaluate_retrieval.py \
    --eval-dataset eval_data.jsonl \
    --k-values 3 5 10 \
    --models bge-m3 jina-emb \
    --rerankers gte-base bge-v2-m3
```

### 3. Программный доступ

```python
from research.evals import evaluate_dataset

results = evaluate_dataset(
    dataset=test_data,
    model_name="google/gemini-2.5-flash-preview-05-20"
)
```

### 4. Форматы данных

**RAG оценка (CSV):**
```csv
question,answer,context
"Как открыть карту?","Обратитесь в банк","Инструкция по открытию..."
```

**Ретривал оценка (JSONL):**
```json
{"query": "вопрос", "chunk_id": 123, "match": 1.0}
```

**Результаты:**
- CSV/JSON с детальными оценками
- Сводные отчеты по метрикам

## ⚙️ Конфигурация

Модуль использует `research/config.yaml`:

```yaml
models_to_evaluate:
  - "google/gemini-2.5-flash-preview-05-20"
  - "qwen/qwen3-32b"

evaluation:
  eval_model: "google/gemini-2.5-flash-preview-05-20"
  generation_temperature: 0.0
  default_limit: 50

metrics:
  use_bleurt: true
  use_deepeval: true
```

## 📋 Требования

- Python 3.12+
- uv для управления зависимостями
- CUDA для GPU ускорения (опционально)

## ⚠️ Заметки

- DeepEval и BLEURT - опциональные зависимости (группа `research`)
- Настройки читаются из `research/config.yaml`
- BLEURT требует много GPU памяти
- Для работы с uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`
