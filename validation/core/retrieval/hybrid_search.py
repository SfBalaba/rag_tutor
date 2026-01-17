from typing import Any

from rank_bm25 import BM25Okapi

from core.vector_store import get_embedding_model, get_vector_store


class HybridRetriever:
    def __init__(self, alpha: float = 0.5):
        """
        Гибридный ретривер, комбинирующий dense и sparse поиск

        Args:
            alpha: Вес для dense поиска (1-alpha для BM25)
        """
        self.alpha = alpha
        self.vector_store = get_vector_store()
        self.embedding_model = get_embedding_model()
        self.bm25 = None
        self.documents = None
        self._initialize_bm25()

    def _initialize_bm25(self):
        """Инициализирует BM25 индекс"""
        try:
            from core.vector_store import get_documents

            self.documents = get_documents()

            # Токенизируем документы для BM25
            tokenized_docs = [doc.page_content.lower().split() for doc in self.documents]
            self.bm25 = BM25Okapi(tokenized_docs)

            print(f"✅ BM25 инициализирован с {len(self.documents)} документами")
        except Exception as e:
            print(f"⚠️ Ошибка инициализации BM25: {e}")
            self.bm25 = None

    def retrieve(self, query: str, k: int = 5) -> list[Any]:
        """
        Гибридный поиск с комбинацией dense и sparse методов

        Args:
            query: Поисковый запрос
            k: Количество документов для возврата

        Returns:
            Список документов, ранжированных по гибридному скору
        """
        if not self.bm25 or not self.documents:
            # Fallback к обычному dense поиску
            return self.vector_store.similarity_search(query, k=k)

        # Dense поиск
        dense_results = self.vector_store.similarity_search_with_score(query, k=k * 2)

        # BM25 поиск
        query_tokens = query.lower().split()
        bm25_scores = self.bm25.get_scores(query_tokens)

        # Нормализуем скоры
        dense_scores = {}
        for doc, score in dense_results:
            # Конвертируем distance в similarity (чем меньше distance, тем больше similarity)
            similarity = 1 / (1 + score)
            dense_scores[doc.page_content] = similarity

        # Нормализуем BM25 скоры
        if len(bm25_scores) > 0:
            max_bm25 = max(bm25_scores)
            min_bm25 = min(bm25_scores)
            if max_bm25 > min_bm25:
                bm25_scores = [(score - min_bm25) / (max_bm25 - min_bm25) for score in bm25_scores]
            else:
                bm25_scores = [1.0] * len(bm25_scores)

        # Комбинируем скоры
        hybrid_scores = {}
        for i, doc in enumerate(self.documents):
            content = doc.page_content

            dense_score = dense_scores.get(content, 0.0)
            bm25_score = bm25_scores[i] if i < len(bm25_scores) else 0.0

            hybrid_score = self.alpha * dense_score + (1 - self.alpha) * bm25_score
            hybrid_scores[content] = (hybrid_score, doc)

        # Сортируем по гибридному скору
        sorted_results = sorted(hybrid_scores.values(), key=lambda x: x[0], reverse=True)

        return [doc for _, doc in sorted_results[:k]]

    def set_alpha(self, alpha: float):
        """Устанавливает новый вес для комбинации методов"""
        self.alpha = max(0.0, min(1.0, alpha))
