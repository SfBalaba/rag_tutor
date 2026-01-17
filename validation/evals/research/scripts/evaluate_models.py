#!/usr/bin/env python
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EVALS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EVALS_ROOT))

from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from core.config import config
from core.llm import get_llm
from core.llm.chains import get_retrieval_chain
from core.ranking import get_doc_content, rerank_documents
from core.vector_store import get_vector_store
from research import research_config
from research.evals.evaluation import (
    cleanup_models,
    evaluate_batch,
    evaluate_dataset,
    generate_report,
    save_results,
)

os.makedirs("research/logs", exist_ok=True)

MODELS_TO_EVALUATE = research_config.get(
    "models_to_evaluate", ["google/gemini-2.5-flash-preview-05-20"]
)
TEMPERATURE = research_config.get("evaluation", {}).get("generation_temperature", 0.0)
EVAL_MODEL_NAME = research_config.get("evaluation", {}).get(
    "eval_model", "google/gemini-2.5-flash-preview-05-20"
)
TEST_DATASET_PATH = research_config.get("data", {}).get(
    "test_dataset", "research/data/test_dataset.csv"
)
LIMIT = research_config.get("evaluation", {}).get("default_limit", 20)
MAX_CONCURRENCY = research_config.get("evaluation", {}).get("max_concurrency", 3)

# Кэш документов для экономии времени
_document_cache: dict[str, Any] = {}


def precompute_documents_for_all_questions(
    dataset: pd.DataFrame, limit: int | None = None
) -> pd.DataFrame:
    """Предпосчитывает релевантные документы для всех вопросов один раз"""
    if limit is not None and limit < len(dataset):
        dataset = dataset.sample(limit, random_state=42).reset_index(drop=True)

    global _document_cache
    _document_cache.clear()

    print(f"🔍 Предпосчитываем документы для {len(dataset)} вопросов...")

    vector_store = get_vector_store()
    use_reranker = config.get("reranker", {}).get("enabled", False)
    search_top_k = config.get("database", {}).get("search_top_k", 10)
    rerank_top_k = config.get("reranker", {}).get("top_k", 5)

    for idx, (_, row) in enumerate(
        tqdm(dataset.iterrows(), total=len(dataset), desc="Получение и обработка документов")
    ):
        question = row["question"]

        # Получаем документы из векторной базы
        docs = vector_store.similarity_search(question, k=search_top_k)
        print(f"Вопрос {idx + 1}: Найдено {len(docs)} чанков")

        # Применяем реранкер если включен
        if use_reranker and docs:
            docs = rerank_documents(question, docs)[:rerank_top_k]
            print(f"Вопрос {idx + 1}: После реранжирования {len(docs)} документов")

        # Форматируем контекст
        docs_separator = config.get("docs_separator", "\n\n-----")
        formatted_context = docs_separator.join(get_doc_content(doc) for doc in docs)

        # Кэшируем результат
        _document_cache[question] = formatted_context

    print(f"✅ Предпосчёт завершён! Сохранено {len(_document_cache)} контекстов")
    return dataset


def create_model_specific_chain(model_name: str):
    """Создает специальную цепочку для конкретной модели с использованием предпосчитанных документов"""
    # Берем базовый промпт и адаптируем под модель
    qa_prompt = config.get("qa_prompt", "Контекст: {context}\n\nВопрос: {question}\n\nОтвет:")

    # Добавляем специальную обработку для Qwen моделей
    if "qwen" in model_name.lower():
        qa_prompt = "/no_think\n\n" + qa_prompt

    prompt = ChatPromptTemplate.from_template(qa_prompt)

    # Создаем новый экземпляр LLM для этой конкретной модели
    llm = get_llm(temperature=TEMPERATURE)
    # Примечание: в текущей архитектуре get_llm не принимает model_name
    # Возможно, потребуется модификация для поддержки разных моделей

    def get_cached_context(query):
        """Получает предпосчитанный контекст из кэша"""
        global _document_cache
        return _document_cache.get(query, "Контекст недоступен")

    rag_chain = (
        {"context": get_cached_context, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


async def generate_system_responses_async(
    dataset: pd.DataFrame,
    model_name: str,
    limit: int | None = None,
    max_concurrent_questions: int = 6,
) -> pd.DataFrame:
    """Асинхронно генерирует ответы системы для заданной модели"""
    if limit is not None and limit < len(dataset):
        dataset = dataset.sample(limit, random_state=42).reset_index(drop=True)

    # Создаем специальную цепочку для этой модели
    retrieval_chain = create_model_specific_chain(model_name)

    # Семафор для ограничения одновременных запросов к одной модели
    semaphore = asyncio.Semaphore(max_concurrent_questions)

    async def process_question(question: str, golden_answer: str, question_idx: int):
        """Обрабатывает один вопрос"""
        async with semaphore:
            print(f"\n[{model_name}] Вопрос {question_idx + 1}: {question}")

            try:
                # Генерируем ответ
                result = await retrieval_chain.ainvoke(question)

                return {
                    "question": question,
                    "system_answer": result,
                    "golden_answer": golden_answer,
                    "model": model_name,
                }
            except Exception as e:
                print(f"⚠️ [{model_name}] Ошибка при обработке вопроса {question_idx + 1}: {e}")
                return {
                    "question": question,
                    "system_answer": f"Ошибка: {str(e)}",
                    "golden_answer": golden_answer,
                    "model": model_name,
                }

    # Создаем задачи для всех вопросов
    tasks = [
        process_question(row["question"], row["answer"], idx)
        for idx, (_, row) in enumerate(dataset.iterrows())
    ]

    # Выполняем все задачи
    results = await asyncio.gather(*tasks)

    return pd.DataFrame(results)


async def evaluate_model(
    model_name: str, dataset: pd.DataFrame, output_dir: str, limit: int | None = None
):
    """Оценивает одну модель"""
    print(f"\n🔬 Начинаем оценку модели: {model_name}")
    start_time = time.time()

    try:
        # Генерируем ответы системы
        system_responses = await generate_system_responses_async(
            dataset,
            model_name,
            limit=limit,
            max_concurrent_questions=3,  # Ограничиваем для стабильности
        )

        print(f"✅ [{model_name}] Сгенерировано {len(system_responses)} ответов")

        # Сохраняем ответы системы
        responses_path = os.path.join(output_dir, f"{model_name.replace('/', '_')}_responses.csv")
        system_responses.to_csv(responses_path, index=False)

        # Создаем список актуальных контекстов
        actual_contexts = [
            _document_cache.get(row["question"], "") for _, row in dataset.iterrows()
        ]

        # Оценка с использованием eval модели
        evaluated_results = evaluate_dataset(
            dataset=dataset,
            system_responses=system_responses,
            model_name=EVAL_MODEL_NAME,  # Используем фиксированную модель для оценки
            temperature=TEMPERATURE,
            limit=limit,
            actual_contexts=actual_contexts,
        )

        # Генерируем отчет
        report = generate_report(evaluated_results)

        # Сохраняем результаты
        results_path = os.path.join(output_dir, f"{model_name.replace('/', '_')}_evaluation.csv")
        save_results(evaluated_results, results_path, report)

        elapsed_time = time.time() - start_time
        avg_score = report.get("summary", {}).get("overall_avg_score", 0.0)

        print(
            f"✅ [{model_name}] Оценка завершена за {elapsed_time:.1f}с. Средняя оценка: {avg_score:.4f}"
        )

        return {
            "model": model_name,
            "avg_score": avg_score,
            "evaluation_time": elapsed_time,
            "results_path": results_path,
            "responses_path": responses_path,
        }

    except Exception as e:
        print(f"❌ [{model_name}] Ошибка при оценке: {e}")
        return {"model": model_name, "error": str(e), "evaluation_time": time.time() - start_time}


async def run_evaluations(
    models: list, dataset: pd.DataFrame, output_dir: str, limit: int, concurrency: int
):
    """Запускает оценку всех моделей с ограничением параллелизма"""
    # Семафор для ограничения количества одновременно оцениваемых моделей
    semaphore = asyncio.Semaphore(concurrency)

    async def run_with_semaphore(model):
        async with semaphore:
            return await evaluate_model(model, dataset, output_dir, limit)

    # Создаем задачи для всех моделей
    tasks = [run_with_semaphore(model) for model in models]

    # Выполняем все задачи
    results = await asyncio.gather(*tasks, return_exceptions=True)

    return results


async def main():
    """Главная функция"""
    print("🚀 Запуск системы оценки моделей RAG")

    # Создаем директории
    os.makedirs("research/logs", exist_ok=True)
    os.makedirs("research/data", exist_ok=True)

    # Создаем директорию для результатов с временной меткой
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"research/results/evaluation_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Загружаем тестовый датасет
        if not os.path.exists(TEST_DATASET_PATH):
            print(f"❌ Тестовый датасет не найден: {TEST_DATASET_PATH}")
            print("Создайте файл с колонками: question, answer, context")
            return

        dataset = pd.read_csv(TEST_DATASET_PATH)
        print(f"📊 Загружен датасет: {len(dataset)} примеров")

        # Предпосчитываем документы для всех вопросов
        dataset = precompute_documents_for_all_questions(dataset, limit=LIMIT)

        # Запускаем оценку всех моделей
        print(f"\n🔬 Начинаем оценку {len(MODELS_TO_EVALUATE)} моделей")
        evaluation_results = await run_evaluations(
            MODELS_TO_EVALUATE, dataset, output_dir, LIMIT, MAX_CONCURRENCY
        )

        # Обрабатываем результаты
        successful_results = [
            r for r in evaluation_results if isinstance(r, dict) and "error" not in r
        ]
        failed_results = [r for r in evaluation_results if isinstance(r, dict) and "error" in r]

        print("\n📈 Результаты оценки:")
        print(f"✅ Успешно оценено: {len(successful_results)} моделей")
        print(f"❌ Ошибки: {len(failed_results)} моделей")

        if successful_results:
            # Сортируем по средней оценке
            successful_results.sort(key=lambda x: x.get("avg_score", 0), reverse=True)

            print("\n🏆 Рейтинг моделей:")
            for i, result in enumerate(successful_results, 1):
                print(f"{i}. {result['model']}: {result['avg_score']:.4f}")

        # Сохраняем сводный отчет
        summary_path = os.path.join(output_dir, "evaluation_summary.json")
        summary = {
            "timestamp": timestamp,
            "dataset_size": len(dataset),
            "models_evaluated": len(MODELS_TO_EVALUATE),
            "successful_evaluations": len(successful_results),
            "failed_evaluations": len(failed_results),
            "results": evaluation_results,
        }

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"\n📁 Все результаты сохранены в: {output_dir}")

    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        raise
    finally:
        # Очищаем модели из памяти
        cleanup_models()
        print("🧹 Память очищена")


def setup_model(model_name: str) -> bool:
    try:
        # Обновляем конфигурацию модели
        config["model"]["name"] = model_name
        get_llm(temperature=0.0)
        print(f"✅ Модель {model_name} настроена")
        return True
    except Exception as e:
        print(f"❌ Ошибка настройки модели {model_name}: {e}")
        return False


def evaluate_single_model(
    model_name: str,
    golden_data: list[dict[str, Any]],
    use_reranker: bool = True,
    top_search: int = 20,
    top_rerank: int = 5,
    limit: int | None = None,
) -> dict[str, Any]:
    print(f"\n🔄 Тестируем модель: {model_name}")

    if not setup_model(model_name):
        return {"model": model_name, "error": "Не удалось настроить модель"}

    try:
        rag_chain = get_retrieval_chain(
            top_search=top_search, top_rerank=top_rerank, use_reranker=use_reranker
        )

        eval_data = []
        questions_to_process = golden_data[:limit] if limit else golden_data

        print(f"📝 Генерируем ответы для {len(questions_to_process)} вопросов...")

        for i, item in enumerate(questions_to_process):
            try:
                question = item["question"]
                expected_answer = item.get("answer", "")

                # Используем кэш документов если доступен
                cache_key = question
                if cache_key in _document_cache:
                    contexts = _document_cache[cache_key]
                    answer = rag_chain.invoke(question)
                    if isinstance(answer, dict):
                        answer = answer.get("answer", "")
                else:
                    result = rag_chain.invoke(question)

                    if isinstance(result, dict):
                        contexts = result.get("context", [])
                        answer = result.get("answer", "")
                        # Кэшируем контекст
                        _document_cache[cache_key] = contexts
                    else:
                        contexts = []
                        answer = str(result)

                if isinstance(contexts, str):
                    contexts = [contexts]

                eval_item = {
                    "question": question,
                    "answer": answer,
                    "expected_answer": expected_answer,
                    "contexts": contexts,
                }

                eval_data.append(eval_item)

                if (i + 1) % 5 == 0:
                    print(f"📊 Обработано: {i + 1}/{len(questions_to_process)}")

            except Exception as e:
                print(f"❌ Ошибка при обработке вопроса {i}: {e}")
                continue

        if not eval_data:
            return {"model": model_name, "error": "Нет данных для оценки"}

        print(f"📊 Оценка {len(eval_data)} примеров...")
        results = evaluate_batch(eval_data, metrics=["rag_triad", "bleurt", "cosine_similarity"])

        # Агрегируем результаты
        aggregated = {}
        numeric_metrics = []

        for result in results:
            for key, value in result.items():
                if isinstance(value, int | float) and key not in ["question", "answer"]:
                    if key not in numeric_metrics:
                        numeric_metrics.append(key)

        for metric in numeric_metrics:
            values = [r.get(metric, 0) for r in results if isinstance(r.get(metric), int | float)]
            if values:
                aggregated[f"avg_{metric}"] = sum(values) / len(values)

        return {"model": model_name, "total_examples": len(eval_data), **aggregated}

    except Exception as e:
        print(f"❌ Ошибка при оценке модели {model_name}: {e}")
        return {"model": model_name, "error": str(e)}


if __name__ == "__main__":
    asyncio.run(main())
