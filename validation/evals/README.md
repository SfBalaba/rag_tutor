# Eval pipeline (draft)

Цель — использовать полноразмерный eval‑пайплайн (DeepEval/BLEURT, скрипты генерации/оценки) на текущих данных. Основной сценарий: запускаем скрипты из `evals/research/scripts` и подсовываем текущие данные/ответы.

## Быстрый старт
- Через uv из корня репозитория:  
  `uv venv && source .venv/bin/activate`  
  `uv pip install ".[validation]"`
- Оценка (только валидация Сони):  
  `python evals/research/scripts/evaluate.py --mode system --golden-file evals/golden_sets/validation_all.json --output-file evals/research/results/eval.csv`
  (зависит от `core.llm`, `core.vector_store` и DeepEval/BLEURT).
- Сравнение архитектур ретривера:  
  `python evals/research/scripts/compare_retrievers.py --include-llm`
  (LLM-режимы HyDE/Contextual/RAG-Fusion используют LLM и идут медленнее).
  Добавить соседние чанки:
  `python evals/research/scripts/compare_retrievers.py --use-neighbors --neighbor-window 3`

## Форматы данных
- Golden set / synthetic CSV: колонки `question,answer,context[,chunk_path]`.
- Ответы агента (CSV): `question,system_answer[,context_from_agent]`. Совпадение по колонке `question`.
- Результаты: CSV с метриками на каждый пример + JSON отчёт с усреднёнными показателями (создаётся рядом с CSV).

## Состав модулей
- `evals/golden_sets/math_golden_sample.csv` — стартовый golden set на университетских курсах.
- `evals/research/` — полноценный модуль с BLEURT/DeepEval метриками, скриптами `evaluate.py`, `evaluate_models.py`, `evaluate_retrieval.py`, `generate_groundtruth.py`, собственным README и config. Основной пайплайн опирается на него; нужен `core.*`.

## Где подключить своего агента
1) Сохраняйте ответы агента в CSV `question,system_answer,context_from_agent`.
2) Используйте `evals/research/scripts/evaluate.py` (режим system), подставив ваш CSV как golden и/или ответы агента.
3) Для тонкой настройки метрик/модели: редактируйте `evals/research/config.yaml`.

## Что можно улучшить позже
- Добавить/адаптировать `core.*`, чтобы скрипты `evals/research/scripts/*` работали из коробки.
- Привязать генерацию к готовому faiss-индексу `data-for-tutor/faiss_db` и обновить промпты под математику.
- Расширить golden set вопросами по школам (сейчас только university) и сценариями «chain-of-thought».
