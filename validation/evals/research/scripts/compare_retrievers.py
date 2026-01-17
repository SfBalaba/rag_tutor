#!/usr/bin/env python
import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EVALS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EVALS_ROOT))

from core.config import config
from core.ranking import rerank_documents
from core.vector_store import get_vector_store
from core.retrieval.hybrid_search import HybridRetriever
from core.retrieval.hyde_enhanced import HyDERetriever
from core.retrieval.contextual_retrieval import ContextualRetriever
from core.retrieval.rag_fusion import RAGFusionRetriever
from core.retrieval.neighbor_enricher import NeighborEnricher


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description="Сравнение архитектур ретривера")
    parser.add_argument(
        "--validation-file",
        default=str(base_dir / "evals" / "golden_sets" / "validation_all.json"),
        help="JSON с вопросами и relevant_chunk_ids",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["classic", "hybrid"],
        choices=["classic", "hybrid", "hyde", "contextual", "rag_fusion"],
        help="Какие архитектуры сравнивать",
    )
    parser.add_argument(
        "--include-llm",
        action="store_true",
        help="Включить LLM-зависимые архитектуры (hyde/contextual/rag_fusion)",
    )
    parser.add_argument("--k-values", nargs="+", type=int, default=[5, 10, 20, 30, 50])
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=None,
        help="Количество кандидатов до реранка/фильтрации (по умолчанию max(k))",
    )
    parser.add_argument(
        "--use-reranker",
        action=argparse.BooleanOptionalAction,
        default=config.get("reranker", {}).get("enabled", False),
        help="Применять реранкер к результатам",
    )
    parser.add_argument(
        "--use-neighbors",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Расширять результаты соседними чанками",
    )
    parser.add_argument(
        "--neighbor-window",
        type=int,
        default=3,
        help="Размер окна соседей слева/справа",
    )
    parser.add_argument(
        "--filter-by-level",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Фильтровать результаты по уровню из валидации",
    )
    parser.add_argument(
        "--output-file",
        default=str(base_dir / "evals" / "research" / "results" / "retriever_compare.csv"),
    )
    parser.add_argument("--hybrid-alpha", type=float, default=0.5)
    parser.add_argument("--rag-fusion-queries", type=int, default=4)
    return parser.parse_args()


def load_validation_data(path: str) -> list[dict[str, Any]]:
    path_obj = Path(path)
    if not path_obj.is_absolute():
        base_dir = Path(__file__).resolve().parents[3]
        path_obj = base_dir / path_obj
    with open(path_obj, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def dcg_at_k(relevance_scores: list[int], k: int) -> float:
    return sum((rel / math.log2(i + 2)) for i, rel in enumerate(relevance_scores[:k]))


def ndcg_at_k(relevance_scores: list[int], k: int) -> float:
    if not any(relevance_scores):
        return 1.0
    dcg = dcg_at_k(relevance_scores, k)
    ideal_relevance = sorted(relevance_scores, reverse=True)
    idcg = dcg_at_k(ideal_relevance, k)
    return dcg / idcg if idcg > 0 else 0.0


def doc_id_from_metadata(meta: dict[str, Any]) -> tuple[Any, Any]:
    return meta.get("chunk_id"), meta.get("book_title")


def compute_metrics(
    retriever_fn: Callable[[str, int], list[Any]],
    val_set: list[dict[str, Any]],
    k_list: list[int],
    use_reranker: bool = False,
    filter_by_level: bool = True,
    neighbor_enricher: NeighborEnricher | None = None,
    neighbor_window: int = 3,
    candidate_k: int | None = None,
    progress_every: int = 10,
) -> list[dict[str, Any]]:
    results_val: list[dict[str, Any]] = []
    if not k_list:
        return results_val

    k_list_sorted = sorted(set(k_list))
    max_k = max(k_list_sorted)
    fetch_k = max_k
    if candidate_k and candidate_k > max_k:
        fetch_k = candidate_k
    metrics_sum_by_k: dict[int, dict[str, float]] = {}
    for k in k_list_sorted:
        metrics_sum_by_k[k] = {
            "recall@k": 0.0,
            "hit@k": 0.0,
            "mrr@k": 0.0,
            "precision@k": 0.0,
            "ndcg@k": 0.0,
            "level_consistency": 0.0,
            "avg_retrieved_count": 0.0,
        }

    n = len(val_set) if val_set else 1
    for idx, item in enumerate(val_set, 1):
        if progress_every and (idx == 1 or idx % progress_every == 0 or idx == n):
            print(f"  ⏳ Обрабатываем вопросы: {idx}/{n}")

        relevant_ids = set(
            zip(
                item.get("relevant_chunk_ids", []),
                [item.get("book_title")] * len(item.get("relevant_chunk_ids", [])),
            )
        )

        docs = retriever_fn(item["query"], fetch_k)

        if neighbor_enricher and docs:
            docs = neighbor_enricher.enrich(docs, window=neighbor_window)

        if use_reranker and docs:
            docs = rerank_documents(item["query"], docs)

        if filter_by_level and docs:
            level = item.get("level")
            filtered = [
                doc for doc in docs if hasattr(doc, "metadata") and doc.metadata.get("level") == level
            ]
            if filtered:
                docs = filtered

        for k in k_list_sorted:
            docs_k = docs[:k]
            metrics_sum_by_k[k]["avg_retrieved_count"] += len(docs_k)

            retrieved_ids = set()
            level_matches = 0

            for doc in docs_k:
                if not hasattr(doc, "metadata"):
                    continue
                meta = doc.metadata
                retrieved_ids.add(doc_id_from_metadata(meta))
                if meta.get("level") == item.get("level"):
                    level_matches += 1

            relevant_retrieved = len(relevant_ids & retrieved_ids)

            recall = relevant_retrieved / len(relevant_ids) if relevant_ids else 0.0
            precision = relevant_retrieved / len(retrieved_ids) if retrieved_ids else 0.0
            hit = 1.0 if relevant_retrieved else 0.0
            level_consistency = level_matches / len(docs_k) if docs_k else 0.0

            mrr = 0.0
            relevance_scores = []
            for rank, doc in enumerate(docs_k, 1):
                if not hasattr(doc, "metadata"):
                    relevance_scores.append(0)
                    continue
                meta = doc.metadata
                is_relevant = doc_id_from_metadata(meta) in relevant_ids
                relevance_scores.append(1 if is_relevant else 0)
                if is_relevant and mrr == 0.0:
                    mrr = 1.0 / rank

            ndcg = ndcg_at_k(relevance_scores, k)

            metrics_sum_by_k[k]["recall@k"] += recall
            metrics_sum_by_k[k]["precision@k"] += precision
            metrics_sum_by_k[k]["hit@k"] += hit
            metrics_sum_by_k[k]["mrr@k"] += mrr
            metrics_sum_by_k[k]["ndcg@k"] += ndcg
            metrics_sum_by_k[k]["level_consistency"] += level_consistency

    for k in k_list_sorted:
        metrics_sum = metrics_sum_by_k[k]
        avg_metrics = {key: value / n for key, value in metrics_sum.items()}
        avg_metrics["k"] = k
        results_val.append(avg_metrics)

    return results_val


def main() -> None:
    args = parse_args()

    if args.include_llm:
        for mode in ["hyde", "contextual", "rag_fusion"]:
            if mode not in args.modes:
                args.modes.append(mode)

    val_set = load_validation_data(args.validation_file)

    vector_store = get_vector_store()
    embed_model = getattr(vector_store, "embedding_model", None)
    neighbor_enricher = (
        NeighborEnricher(vector_store.documents)
        if hasattr(vector_store, "documents")
        else None
    )
    hybrid = HybridRetriever(alpha=args.hybrid_alpha)
    hyde = HyDERetriever(base_retriever=hybrid)
    contextual = ContextualRetriever(base_retriever=hybrid)
    rag_fusion = RAGFusionRetriever(base_retriever=hybrid)

    retrievers: dict[str, Callable[[str, int], list[Any]]] = {
        "classic": lambda q, k: vector_store.similarity_search(q, k=k),
        "hybrid": lambda q, k: hybrid.retrieve(q, k=k),
        "hyde": lambda q, k: hyde.retrieve(q, k=k, use_multiple_hypotheses=False),
        "contextual": lambda q, k: contextual.retrieve(q, k=k, expand_query=True),
        "rag_fusion": lambda q, k: rag_fusion.retrieve(
            q, k=k, num_queries=args.rag_fusion_queries, fusion_method="rrf"
        ),
    }

    results: list[dict[str, Any]] = []
    output_path = Path(args.output_file)
    if not output_path.is_absolute():
        base_dir = Path(__file__).resolve().parents[3]
        output_path = base_dir / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def save_results(rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        import csv

        fieldnames = [
            "retriever",
            "k",
            "recall@k",
            "hit@k",
            "mrr@k",
            "precision@k",
            "ndcg@k",
            "level_consistency",
            "avg_retrieved_count",
        ]
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"💾 Промежуточные результаты сохранены в {output_path}")
    for mode in args.modes:
        print(
            f"▶️ Режим {mode}: вопросы={len(val_set)}, k={sorted(set(args.k_values))}, "
            f"reranker={args.use_reranker}, neighbors={args.use_neighbors}, "
            f"candidates={args.candidate_k or 'max(k)'}"
        )
        retriever_fn = retrievers[mode]
        metrics = compute_metrics(
            retriever_fn,
            val_set,
            args.k_values,
            use_reranker=args.use_reranker,
            filter_by_level=args.filter_by_level,
            neighbor_enricher=neighbor_enricher if args.use_neighbors else None,
            neighbor_window=args.neighbor_window,
            candidate_k=args.candidate_k,
            progress_every=10,
        )
        for row in metrics:
            row["retriever"] = f"{mode}+neighbors" if args.use_neighbors else mode
            results.append(row)
        save_results(results)
        if embed_model is not None and hasattr(embed_model, "request_count"):
            cache_size = len(getattr(embed_model, "cache", {}) or {})
            print(
                f"ℹ️ Embeddings: requests={embed_model.request_count}, cache={cache_size}"
            )

    print(f"✅ Результаты сохранены в {output_path}")


if __name__ == "__main__":
    main()
