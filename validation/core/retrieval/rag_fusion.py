import asyncio
import re
from collections import defaultdict
from typing import Any

from core.llm import get_llm
from core.vector_store import get_embedding_model, get_vector_store


class RAGFusionRetriever:
    def __init__(self, base_retriever=None):
        """
        RAG-Fusion ретривер - генерирует множественные запросы и объединяет результаты

        Args:
            base_retriever: Базовый ретривер (по умолчанию vector store)
        """
        self.base_retriever = base_retriever or get_vector_store()
        self.llm = get_llm()
        self.embedding_model = get_embedding_model()

    def _generate_multiple_queries(self, original_query: str, num_queries: int = 4) -> list[str]:
        """Генерирует множественные варианты запроса"""
        prompt = f"""Сгенерируй {num_queries} различных поисковых запроса, которые помогут найти информацию для ответа на следующий вопрос:

Оригинальный вопрос: {original_query}

Сгенерированные запросы должны:
- Быть разными по формулировке, но искать ту же информацию
- Включать синонимы и альтернативные термины
- Покрывать разные аспекты вопроса
- Быть подходящими для поиска в базе знаний по математике

Запросы (по одному на строку):"""

        try:
            response = self.llm.invoke(prompt)
            if hasattr(response, "content"):
                content = response.content.strip()
            else:
                content = str(response).strip()

            # Парсим запросы
            queries = [q.strip() for q in content.split("\n") if q.strip()]

            # Убираем нумерацию/маркеры если есть
            cleaned_queries = []
            for query in queries:
                cleaned = re.sub(r"^[\\s\\-•*\\d]+[\\).:-]?\\s*", "", query)
                cleaned = cleaned.strip()
                if cleaned and len(cleaned) > 5:
                    cleaned_queries.append(cleaned)

            # Добавляем оригинальный запрос если его нет
            if original_query not in cleaned_queries:
                cleaned_queries.insert(0, original_query)

            return cleaned_queries[:num_queries] if cleaned_queries else [original_query]

        except Exception as e:
            print(f"⚠️ Ошибка генерации множественных запросов: {e}")
            return [original_query]

    def _reciprocal_rank_fusion(self, results_lists: list[list[Any]], k: int = 60) -> list[Any]:
        """
        Применяет Reciprocal Rank Fusion для объединения результатов

        Args:
            results_lists: Список списков результатов от разных запросов
            k: Параметр для RRF (обычно 60)

        Returns:
            Объединенный и ранжированный список результатов
        """
        # Словарь для накопления скоров документов
        doc_scores = defaultdict(float)
        doc_objects = {}  # Сохраняем объекты документов

        for results in results_lists:
            for rank, doc in enumerate(results):
                # Используем содержимое документа как ключ
                doc_key = doc.page_content if hasattr(doc, "page_content") else str(doc)

                # RRF формула: 1 / (k + rank)
                score = 1.0 / (k + rank + 1)
                doc_scores[doc_key] += score

                # Сохраняем объект документа
                if doc_key not in doc_objects:
                    doc_objects[doc_key] = doc

        # Сортируем по скору
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        # Возвращаем объекты документов в порядке убывания скора
        return [doc_objects[doc_key] for doc_key, _ in sorted_docs]

    def retrieve(
        self, query: str, k: int = 5, num_queries: int = 4, fusion_method: str = "rrf"
    ) -> list[Any]:
        """
        RAG-Fusion поиск

        Args:
            query: Оригинальный поисковый запрос
            k: Количество документов для возврата
            num_queries: Количество генерируемых запросов
            fusion_method: Метод объединения ("rrf" или "simple")

        Returns:
            Объединенный список документов
        """
        # Генерируем множественные запросы
        queries = self._generate_multiple_queries(query, num_queries)
        print(f"🔍 Сгенерировано {len(queries)} запросов для RAG-Fusion")

        # Выполняем поиск по каждому запросу
        all_results = []
        for i, q in enumerate(queries):
            print(f"  Запрос {i + 1}: {q}")

            if hasattr(self.base_retriever, "similarity_search"):
                results = self.base_retriever.similarity_search(q, k=k * 2)
            else:
                results = self.base_retriever.retrieve(q, k=k * 2)

            all_results.append(results)

        # Объединяем результаты
        if fusion_method == "rrf":
            fused_results = self._reciprocal_rank_fusion(all_results)
        else:
            # Простое объединение с удалением дубликатов
            fused_results = self._simple_fusion(all_results)

        return fused_results[:k]

    def _simple_fusion(self, results_lists: list[list[Any]]) -> list[Any]:
        """Простое объединение результатов с удалением дубликатов"""
        seen_contents = set()
        fused_results = []

        # Проходим по всем спискам результатов
        for results in results_lists:
            for doc in results:
                content = doc.page_content if hasattr(doc, "page_content") else str(doc)
                if content not in seen_contents:
                    fused_results.append(doc)
                    seen_contents.add(content)

        return fused_results

    async def retrieve_async(
        self, query: str, k: int = 5, num_queries: int = 4, max_concurrency: int = 3
    ) -> list[Any]:
        """
        Асинхронная версия RAG-Fusion поиска

        Args:
            query: Оригинальный поисковый запрос
            k: Количество документов для возврата
            num_queries: Количество генерируемых запросов
            max_concurrency: Максимальное количество одновременных запросов
        """
        # Генерируем множественные запросы
        queries = self._generate_multiple_queries(query, num_queries)
        print(f"🔍 Асинхронный RAG-Fusion с {len(queries)} запросами")

        # Семафор для ограничения concurrency
        semaphore = asyncio.Semaphore(max_concurrency)

        async def search_query(q: str) -> list[Any]:
            async with semaphore:
                # Выполняем поиск в executor (так как vector store синхронный)
                loop = asyncio.get_event_loop()
                if hasattr(self.base_retriever, "similarity_search"):
                    return await loop.run_in_executor(
                        None, self.base_retriever.similarity_search, q, k * 2
                    )
                else:
                    return await loop.run_in_executor(None, self.base_retriever.retrieve, q, k * 2)

        # Выполняем все поиски параллельно
        tasks = [search_query(q) for q in queries]
        all_results = await asyncio.gather(*tasks)

        # Объединяем результаты с помощью RRF
        fused_results = self._reciprocal_rank_fusion(all_results)

        return fused_results[:k]

    def retrieve_with_weights(
        self, query: str, k: int = 5, query_weights: dict[str, float] = None
    ) -> list[Any]:
        """
        RAG-Fusion с весами для разных типов запросов

        Args:
            query: Оригинальный запрос
            k: Количество документов
            query_weights: Веса для разных типов запросов
        """
        if query_weights is None:
            query_weights = {"original": 1.0, "synonyms": 0.8, "broader": 0.6, "specific": 0.9}

        # Генерируем специализированные запросы
        queries = self._generate_specialized_queries(query)

        # Выполняем поиск с весами
        weighted_results = []
        for query_type, q in queries.items():
            weight = query_weights.get(query_type, 1.0)

            if hasattr(self.base_retriever, "similarity_search"):
                results = self.base_retriever.similarity_search(q, k=k * 2)
            else:
                results = self.base_retriever.retrieve(q, k=k * 2)

            # Применяем веса к результатам
            weighted_results.append((results, weight))

        # Объединяем с учетом весов
        return self._weighted_fusion(weighted_results, k)

    def _generate_specialized_queries(self, query: str) -> dict[str, str]:
        """Генерирует специализированные типы запросов"""
        queries = {"original": query}

        # Запрос с синонимами
        synonym_prompt = f"Перефразируй запрос используя синонимы: {query}"
        try:
            response = self.llm.invoke(synonym_prompt)
            content = response.content if hasattr(response, "content") else str(response)
            queries["synonyms"] = content.strip()
        except Exception:
            queries["synonyms"] = query

        # Более широкий запрос
        broader_prompt = f"Сформулируй более общий запрос по теме: {query}"
        try:
            response = self.llm.invoke(broader_prompt)
            content = response.content if hasattr(response, "content") else str(response)
            queries["broader"] = content.strip()
        except Exception:
            queries["broader"] = query

        # Более специфичный запрос
        specific_prompt = f"Сформулируй более конкретный и детальный запрос: {query}"
        try:
            response = self.llm.invoke(specific_prompt)
            content = response.content if hasattr(response, "content") else str(response)
            queries["specific"] = content.strip()
        except Exception:
            queries["specific"] = query

        return queries

    def _weighted_fusion(
        self, weighted_results: list[tuple[list[Any], float]], k: int
    ) -> list[Any]:
        """Объединяет результаты с учетом весов"""
        doc_scores = defaultdict(float)
        doc_objects = {}

        for results, weight in weighted_results:
            for rank, doc in enumerate(results):
                doc_key = doc.page_content if hasattr(doc, "page_content") else str(doc)

                # Взвешенный RRF
                score = weight * (1.0 / (60 + rank + 1))
                doc_scores[doc_key] += score

                if doc_key not in doc_objects:
                    doc_objects[doc_key] = doc

        # Сортируем по взвешенному скору
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        return [doc_objects[doc_key] for doc_key, _ in sorted_docs[:k]]
