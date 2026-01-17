from typing import Any

from core.llm import get_llm
from core.vector_store import get_embedding_model, get_vector_store


class ContextualRetriever:
    def __init__(self, base_retriever=None):
        """
        Contextual Retrieval - добавляет контекст к документам для улучшения поиска

        Args:
            base_retriever: Базовый ретривер (по умолчанию vector store)
        """
        self.base_retriever = base_retriever or get_vector_store()
        self.llm = get_llm()
        self.embedding_model = get_embedding_model()
        self.context_cache = {}  # Кэш для контекстуализированных запросов

    def _generate_context_for_chunk(self, chunk_content: str, document_context: str = "") -> str:
        """Генерирует контекст для чанка документа"""
        prompt = f"""Дай краткий контекст (1-2 предложения) для следующего фрагмента текста, чтобы он был понятен без остального документа.

{f"Контекст документа: {document_context}" if document_context else ""}

Фрагмент текста:
{chunk_content}

Контекст должен:
- Объяснить, о чем этот фрагмент
- Добавить ключевые термины для поиска
- Быть кратким и информативным

Контекст:"""

        try:
            response = self.llm.invoke(prompt)
            if hasattr(response, "content"):
                return response.content.strip()
            return str(response).strip()
        except Exception as e:
            print(f"⚠️ Ошибка генерации контекста: {e}")
            return ""

    def _contextualize_query(self, query: str, conversation_history: list[str] = None) -> str:
        """Контекстуализирует запрос с учетом истории разговора"""
        if query in self.context_cache:
            return self.context_cache[query]

        if not conversation_history:
            return query

        history_text = "\n".join(conversation_history[-3:])  # Последние 3 сообщения

        prompt = f"""Перефразируй следующий запрос, добавив необходимый контекст из истории разговора, чтобы он был понятен без предыдущих сообщений.

История разговора:
{history_text}

Текущий запрос: {query}

Контекстуализированный запрос:"""

        try:
            response = self.llm.invoke(prompt)
            if hasattr(response, "content"):
                contextualized = response.content.strip()
            else:
                contextualized = str(response).strip()

            self.context_cache[query] = contextualized
            return contextualized
        except Exception as e:
            print(f"⚠️ Ошибка контекстуализации запроса: {e}")
            return query

    def _expand_query_with_context(self, query: str) -> str:
        """Расширяет запрос дополнительным контекстом"""
        prompt = f"""Расширь следующий поисковый запрос, добавив синонимы, связанные термины и возможные формулировки того же вопроса.

Оригинальный запрос: {query}

Расширенный запрос должен:
- Включать синонимы ключевых слов
- Добавлять связанные математические термины
- Сохранять смысл оригинального запроса
- Быть в формате поискового запроса

Расширенный запрос:"""

        try:
            response = self.llm.invoke(prompt)
            if hasattr(response, "content"):
                return response.content.strip()
            return str(response).strip()
        except Exception as e:
            print(f"⚠️ Ошибка расширения запроса: {e}")
            return query

    def retrieve(
        self,
        query: str,
        k: int = 5,
        conversation_history: list[str] = None,
        expand_query: bool = True,
    ) -> list[Any]:
        """
        Контекстуальный поиск

        Args:
            query: Поисковый запрос
            k: Количество документов для возврата
            conversation_history: История разговора для контекстуализации
            expand_query: Расширять ли запрос дополнительными терминами

        Returns:
            Список документов с учетом контекста
        """
        # Контекстуализируем запрос
        contextualized_query = self._contextualize_query(query, conversation_history)

        # Расширяем запрос если нужно
        if expand_query:
            expanded_query = self._expand_query_with_context(contextualized_query)
        else:
            expanded_query = contextualized_query

        # Выполняем поиск
        if hasattr(self.base_retriever, "similarity_search"):
            results = self.base_retriever.similarity_search(expanded_query, k=k * 2)
        else:
            results = self.base_retriever.retrieve(expanded_query, k=k * 2)

        # Дополнительная фильтрация по релевантности
        filtered_results = self._filter_by_relevance(results, query, k)

        return filtered_results

    def _filter_by_relevance(self, results: list[Any], original_query: str, k: int) -> list[Any]:
        """Фильтрует результаты по релевантности к оригинальному запросу"""
        if not results:
            return results

        # Простая фильтрация по ключевым словам
        query_words = set(original_query.lower().split())

        scored_results = []
        for doc in results:
            content = doc.page_content if hasattr(doc, "page_content") else str(doc)
            content_words = set(content.lower().split())

            # Подсчитываем пересечение ключевых слов
            overlap = len(query_words.intersection(content_words))
            relevance_score = overlap / len(query_words) if query_words else 0

            scored_results.append((relevance_score, doc))

        # Сортируем по релевантности
        scored_results.sort(key=lambda x: x[0], reverse=True)

        return [doc for _, doc in scored_results[:k]]

    def retrieve_with_context_injection(
        self, query: str, k: int = 5, inject_context: bool = True
    ) -> list[Any]:
        """
        Поиск с инъекцией контекста в найденные документы

        Args:
            query: Поисковый запрос
            k: Количество документов для возврата
            inject_context: Добавлять ли контекст к найденным документам
        """
        # Обычный поиск
        if hasattr(self.base_retriever, "similarity_search"):
            results = self.base_retriever.similarity_search(query, k=k)
        else:
            results = self.base_retriever.retrieve(query, k=k)

        if not inject_context:
            return results

        # Добавляем контекст к каждому документу
        contextualized_results = []
        for doc in results:
            if hasattr(doc, "page_content"):
                # Генерируем контекст для чанка
                context = self._generate_context_for_chunk(doc.page_content)

                # Создаем новый документ с контекстом
                if hasattr(doc, "metadata"):
                    enhanced_doc = type(doc)(
                        page_content=f"{context}\n\n{doc.page_content}", metadata=doc.metadata
                    )
                else:
                    enhanced_doc = doc
                    enhanced_doc.page_content = f"{context}\n\n{doc.page_content}"

                contextualized_results.append(enhanced_doc)
            else:
                contextualized_results.append(doc)

        return contextualized_results

    def clear_cache(self):
        """Очищает кэш контекстуализированных запросов"""
        self.context_cache.clear()
