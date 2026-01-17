#!/usr/bin/env python
import argparse
import gc
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from sklearn.metrics import ndcg_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EVALS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EVALS_ROOT))

from core.vector_store import get_documents

EMBEDDING_MODELS = {
    "e5-large": "intfloat/multilingual-e5-large",
    "e5-base": "intfloat/multilingual-e5-base",
    "gte-large": "Alibaba-NLP/gte-multilingual-base",
    "labse": "sentence-transformers/LaBSE",
    "USER-bge-m3": "deepvk/USER-bge-m3",
    "jina-emb": "jinaai/jina-embeddings-v3",
    "KaLM": "HIT-TMG/KaLM-embedding-multilingual-mini-v1",
}

RERANKER_MODELS = {
    "gte-base": "Alibaba-NLP/gte-multilingual-reranker-base",
    "bge-v2-m3": "BAAI/bge-reranker-v2-m3",
    "jina-v2-base": "jinaai/jina-reranker-v2-base-multilingual",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Оценка качества ретривала")

    parser.add_argument("--eval-dataset", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="research/results/retrieval")
    parser.add_argument("--k-values", nargs="+", type=int, default=[3, 5, 10])
    parser.add_argument("--retrieval-k", type=int, default=100)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--models", nargs="+", type=str, choices=list(EMBEDDING_MODELS.keys()))
    parser.add_argument("--rerankers", nargs="+", type=str, choices=list(RERANKER_MODELS.keys()))

    return parser.parse_args()


def setup_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_evaluation_dataset(file_path):
    if file_path.endswith(".jsonl"):
        eval_df = pd.read_json(file_path, lines=True)
    else:
        eval_df = pd.read_csv(file_path)

    required_columns = ["question", "relevant_doc_ids"]
    for col in required_columns:
        if col not in eval_df.columns:
            raise ValueError(f"Отсутствует обязательная колонка: {col}")

    return eval_df


def create_embedder(model_name: str, device: str):
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name, model_kwargs={"device": device, "trust_remote_code": True}
    )

    return embeddings


def create_reranker(model_name: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, torch_dtype="auto", trust_remote_code=True
    ).to(device)
    model.eval()

    return tokenizer, model


def evaluate_retrieval_combination(
    eval_df: pd.DataFrame,
    documents: list[Any] | None,
    retrieval_model: str,
    reranker_model: str | None,
    device: str,
    k_list: list[int] | None = None,
    retrieval_k: int = 100,
) -> dict[str, Any]:
    """Оценивает одну комбинацию ретривер + реранкер"""
    if k_list is None:
        k_list = [1, 5, 10]

    print(f"🔄 Тестируем: {retrieval_model} + {reranker_model or 'без реранкера'}")

    if documents is None:
        raise ValueError("Не удалось получить документы из векторного хранилища")

    embedder = create_embedder(EMBEDDING_MODELS[retrieval_model], device)

    reranker_tokenizer = None
    reranker_model_obj = None
    if reranker_model:
        reranker_tokenizer, reranker_model_obj = create_reranker(
            RERANKER_MODELS[reranker_model], device
        )

    processed_queries = 0

    # Метрики для каждого k
    metrics: dict[int, dict[str, list[float]]] = {
        k: {"recall_scores": [], "precision_scores": [], "ndcg_scores": [], "acc_scores": []}
        for k in k_list
    }

    for idx, row in eval_df.iterrows():
        try:
            query = row["question"]
            relevant_doc_ids = row["relevant_doc_ids"]

            if isinstance(relevant_doc_ids, str):
                relevant_doc_ids = eval(relevant_doc_ids)
            relevant_doc_ids = set(map(str, relevant_doc_ids))

            # Поиск с эмбеддером
            doc_contents = [doc.page_content for doc in documents]
            query_embedding = embedder.embed_query(query)
            doc_embeddings = embedder.embed_documents(doc_contents[:retrieval_k])

            query_embedding = np.array(query_embedding).reshape(1, -1)
            doc_embeddings = np.array(doc_embeddings)

            similarities = np.dot(query_embedding, doc_embeddings.T).flatten()
            top_indices = np.argsort(similarities)[::-1][:retrieval_k]

            retrieved_docs = [documents[i] for i in top_indices]

            # Реранкинг
            if reranker_model and reranker_tokenizer and reranker_model_obj:
                inputs = [f"{query} [SEP] {doc.page_content}" for doc in retrieved_docs]

                with torch.no_grad():
                    tokenized = reranker_tokenizer(
                        inputs, padding=True, truncation=True, return_tensors="pt", max_length=2048
                    )
                    tokenized = {k: v.to(device) for k, v in tokenized.items()}
                    outputs = reranker_model_obj(**tokenized)
                    scores = outputs.logits.cpu().numpy().flatten()

                rerank_indices = np.argsort(scores)[::-1]
                retrieved_docs = [retrieved_docs[i] for i in rerank_indices]

            # Вычисляем метрики для каждого k
            for k in k_list:
                top_k_docs = retrieved_docs[:k]
                retrieved_doc_ids = set()

                for doc in top_k_docs:
                    if hasattr(doc, "metadata") and "doc_id" in doc.metadata:
                        retrieved_doc_ids.add(str(doc.metadata["doc_id"]))

                recall = (
                    len(relevant_doc_ids & retrieved_doc_ids) / len(relevant_doc_ids)
                    if relevant_doc_ids
                    else 0
                )
                precision = (
                    len(relevant_doc_ids & retrieved_doc_ids) / len(retrieved_doc_ids)
                    if retrieved_doc_ids
                    else 0
                )

                # NDCG
                relevance_scores = [
                    1 if str(doc.metadata.get("doc_id", "")) in relevant_doc_ids else 0
                    for doc in top_k_docs
                ]
                if len(relevance_scores) > 0:
                    ideal_scores = sorted(
                        [1] * len(relevant_doc_ids) + [0] * (k - len(relevant_doc_ids)),
                        reverse=True,
                    )[:k]
                    ndcg = (
                        ndcg_score([ideal_scores], [relevance_scores])
                        if sum(ideal_scores) > 0
                        else 0
                    )
                else:
                    ndcg = 0

                # Accuracy@k
                accuracy = 1 if any(score > 0 for score in relevance_scores) else 0

                # Сохраняем метрики
                metrics[k]["recall_scores"].append(recall)
                metrics[k]["precision_scores"].append(precision)
                metrics[k]["ndcg_scores"].append(ndcg)
                metrics[k]["acc_scores"].append(accuracy)

        except Exception as e:
            print(f"❌ Ошибка при обработке запроса {idx}: {e}")
            continue

        processed_queries += 1
        if processed_queries % 10 == 0:
            print(f"📊 Обработано: {processed_queries}")

    # Агрегируем результаты
    result: dict[str, str | int | float | None] = {
        "retriever": retrieval_model,
        "reranker": reranker_model,
        "processed_queries": processed_queries,
    }

    for k in k_list:
        if metrics[k]["recall_scores"]:  # Проверяем что есть данные
            result.update(
                {
                    f"recall@{k}": float(np.mean(metrics[k]["recall_scores"])),
                    f"precision@{k}": float(np.mean(metrics[k]["precision_scores"])),
                    f"ndcg@{k}": float(np.mean(metrics[k]["ndcg_scores"])),
                    f"accuracy@{k}": float(np.mean(metrics[k]["acc_scores"])),
                }
            )
        else:
            result.update(
                {f"recall@{k}": 0.0, f"precision@{k}": 0.0, f"ndcg@{k}": 0.0, f"accuracy@{k}": 0.0}
            )

    print(f"✅ Обработано запросов: {processed_queries}")

    # Очистка памяти
    del embedder
    if reranker_model_obj:
        del reranker_model_obj, reranker_tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    return result


def run_evaluation(args):
    """Запуск полной оценки"""

    # Настройка устройства
    device = args.device or setup_device()

    # Загрузка датасета
    eval_df = load_evaluation_dataset(args.eval_dataset)

    # Определяем модели для тестирования
    retrieval_models = args.models or list(EMBEDDING_MODELS.keys())
    reranker_models = args.rerankers or list(RERANKER_MODELS.keys())

    print(f"🧪 Тестируем {len(retrieval_models)} ретриверов × {len(reranker_models)} реранкеров")
    print(f"📊 Метрики: K = {args.k_values}")

    # Запуск оценки
    results = []
    total_combinations = len(retrieval_models) * len(reranker_models)

    for i, retrieval_model in enumerate(retrieval_models):
        for j, reranker_model in enumerate(reranker_models):
            combination_num = i * len(reranker_models) + j + 1
            print(f"\n{'=' * 60}")
            print(f"Комбинация {combination_num}/{total_combinations}")
            print(f"{'=' * 60}")

            try:
                result = evaluate_retrieval_combination(
                    eval_df=eval_df,
                    documents=get_documents(),
                    retrieval_model=retrieval_model,
                    reranker_model=reranker_model,
                    device=device,
                    k_list=args.k_values,
                    retrieval_k=args.retrieval_k,
                )
                results.append(result)

                # Промежуточное сохранение
                if results:
                    interim_df = pd.DataFrame(results)
                    os.makedirs(args.output_dir, exist_ok=True)
                    interim_path = os.path.join(args.output_dir, "interim_results.csv")
                    interim_df.to_csv(interim_path, index=False)

            except Exception as e:
                print(f"❌ Ошибка в комбинации {retrieval_model} + {reranker_model}: {e}")
                continue

    return results


def save_results(results: list, output_dir: str, k_values: list):
    """Сохранение результатов"""
    if not results:
        print("❌ Нет результатов для сохранения")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Создаем DataFrame
    results_df = pd.DataFrame(results)

    # Сохраняем полные результаты
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # CSV
    csv_path = os.path.join(output_dir, f"retrieval_evaluation_{timestamp}.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"📄 Результаты сохранены: {csv_path}")

    # JSON
    json_path = os.path.join(output_dir, f"retrieval_evaluation_{timestamp}.json")
    results_df.to_json(json_path, orient="records", indent=2, force_ascii=False)
    print(f"📄 Результаты сохранены: {json_path}")

    # Создаем сводную таблицу
    print("\n📊 Топ результаты по метрикам:")

    for k in k_values:
        print(f"\n🏆 Топ-3 по Recall@{k}:")
        top_recall = results_df.nlargest(3, f"recall@{k}")[["retriever", "reranker", f"recall@{k}"]]
        print(top_recall.to_string(index=False))

        print(f"\n🏆 Топ-3 по NDCG@{k}:")
        top_ndcg = results_df.nlargest(3, f"ndcg@{k}")[["retriever", "reranker", f"ndcg@{k}"]]
        print(top_ndcg.to_string(index=False))


def main():
    """Главная функция"""
    args = parse_args()

    print("🔬 Оценка качества ретривала")
    print(f"Датасет: {args.eval_dataset}")
    print(f"Выходная директория: {args.output_dir}")

    try:
        # Запуск оценки
        results = run_evaluation(args)

        # Сохранение результатов
        save_results(results, args.output_dir, args.k_values)

        print(f"\n✅ Оценка завершена! Результаты: {args.output_dir}")

    except Exception as e:
        print(f"❌ Ошибка: {e}")
        raise


if __name__ == "__main__":
    main()
