#!/usr/bin/env python
import argparse
import csv
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EVALS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EVALS_ROOT))


from core.llm.chains import get_retrieval_chain
from core.ranking import get_doc_content, rerank_documents
from core.vector_store import get_vector_store
from research.evals.evaluation import evaluate_batch


def parse_arguments():
    parser = argparse.ArgumentParser(description="Оценка качества RAG системы")

    parser.add_argument(
        "--mode", choices=["synthetic", "system"], default="system", help="Режим оценки"
    )

    parser.add_argument("--golden-file", type=str, help="Файл с golden dataset (CSV)")

    parser.add_argument(
        "--output-file",
        type=str,
        default="evaluation_results.csv",
        help="Файл для сохранения результатов",
    )

    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["rag_triad", "cosine_similarity"],
        help="Список метрик для использования",
    )

    parser.add_argument(
        "--questions-file", type=str, help="Файл с вопросами (для synthetic режима)"
    )

    parser.add_argument(
        "--num-questions",
        type=int,
        default=50,
        help="Количество вопросов для генерации (synthetic режим)",
    )

    parser.add_argument("--use-reranker", action="store_true", help="Использовать reranker")

    parser.add_argument("--top-search", type=int, default=20, help="Top-K для поиска")

    parser.add_argument("--top-rerank", type=int, default=5, help="Top-K для reranker")

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Ограничить количество вопросов для оценки",
    )

    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Количество потоков для параллельной генерации ответов",
    )

    return parser.parse_args()


def load_golden_dataset(file_path: str) -> list[dict[str, Any]]:
    try:
        if file_path.endswith(".json"):
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            records = []
            for item in data:
                question = item.get("question") or item.get("query")
                answer = item.get("answer") or item.get("reference_answer", "")
                if not question:
                    continue
                record = {"question": question, "answer": answer}
                if "context" in item:
                    record["context"] = item.get("context")
                if "level" in item:
                    record["level"] = item.get("level")
                records.append(record)
            return records

        df = pd.read_csv(file_path)
        required_columns = ["question", "answer"]

        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Отсутствует обязательная колонка: {col}")

        # Приводим к правильному типу для mypy
        records = df.to_dict("records")
        return [{str(k): v for k, v in record.items()} for record in records]

    except Exception as e:
        print(f"Ошибка загрузки golden dataset: {e}")
        return []


def evaluate_system_mode(
    golden_data: list[dict[str, Any]],
    use_reranker: bool = True,
    top_search: int = 20,
    top_rerank: int = 5,
    top_k: int | None = None,
    limit: int | None = None,
    max_workers: int = 1,
) -> list[dict[str, Any]]:
    if top_k is None:
        top_k = top_rerank if use_reranker else top_search

    rag_chain = get_retrieval_chain(
        top_search=top_search, top_rerank=top_rerank, use_reranker=use_reranker
    )
    vector_store = get_vector_store()

    eval_data = []
    questions_to_process = golden_data[:limit] if limit else golden_data

    print(
        f"🔄 Генерируем ответы для {len(questions_to_process)} вопросов "
        f"(reranker={use_reranker}, top_search={top_search}, top_rerank={top_rerank}, "
        f"workers={max_workers})..."
    )

    def process_item(item: dict[str, Any]) -> dict[str, Any] | None:
        try:
            question = item["question"]
            expected_answer = item.get("answer", "")

            result = rag_chain.invoke(question)

            if isinstance(result, dict):
                contexts = result.get("context", [])
                if isinstance(contexts, str):
                    contexts = [contexts]
                answer = result.get("answer", "")
            else:
                answer = str(result)
                try:
                    docs = vector_store.similarity_search(question, k=top_search)
                    if use_reranker:
                        docs = rerank_documents(question, docs)
                    contexts = [get_doc_content(doc) for doc in docs[:top_k]]
                except Exception:
                    contexts = []

            return {
                "question": question,
                "answer": answer,
                "expected_answer": expected_answer,
                "contexts": contexts,
            }
        except Exception as e:
            print(f"❌ Ошибка при обработке вопроса: {e}")
            return None

    if max_workers <= 1:
        for i, item in enumerate(questions_to_process):
            eval_item = process_item(item)
            if eval_item:
                eval_data.append(eval_item)
            if (i + 1) % 10 == 0:
                print(f"📊 Обработано: {i + 1}/{len(questions_to_process)}")
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_item, item) for item in questions_to_process]
            for idx, future in enumerate(as_completed(futures), 1):
                result = future.result()
                if result:
                    eval_data.append(result)
                if idx % 10 == 0:
                    print(f"📊 Обработано: {idx}/{len(questions_to_process)}")

    return eval_data


def generate_synthetic_qa() -> str:
    output_file = "research/data/synthetic_qa_temp.csv"

    return output_file


def main():
    args = parse_arguments()

    if args.mode == "synthetic":
        print("🔄 Генерация синтетических данных...")
        synthetic_file = generate_synthetic_qa()
        eval_data = load_golden_dataset(synthetic_file)

        if not eval_data:
            print("❌ Не удалось сгенерировать синтетические данные")
            return

    elif args.mode == "system":
        if not args.golden_file:
            print("❌ Для system режима необходимо указать --golden-file")
            return

        golden_data = load_golden_dataset(args.golden_file)
        if not golden_data:
            print("❌ Не удалось загрузить golden dataset")
            return

        print("🔄 Режим system: генерация ответов и оценка...")
        eval_data = evaluate_system_mode(
            golden_data=golden_data,
            use_reranker=args.use_reranker,
            top_search=args.top_search,
            top_rerank=args.top_rerank,
            limit=args.limit,
            max_workers=args.max_workers,
        )

    if not eval_data:
        print("❌ Нет данных для оценки")
        return

    print(f"📊 Оценка {len(eval_data)} примеров...")
    results = evaluate_batch(eval_data, metrics=args.metrics)

    # Сохранение результатов
    output_path = Path(args.output_file)
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        if results:
            fieldnames = results[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    print(f"✅ Результаты сохранены в {output_path}")

    # Агрегированная статистика
    from research.evals.evaluation import aggregate_results

    aggregated = aggregate_results(results)
    if aggregated:
        print("\n📈 Агрегированные результаты:")
        for metric, value in aggregated.items():
            print(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    main()
