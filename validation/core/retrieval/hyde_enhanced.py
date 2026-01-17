from typing import Any

from core.llm import get_llm
from core.vector_store import get_embedding_model, get_vector_store


class HyDERetriever:
    def __init__(self, base_retriever=None):
        """
        HyDE ретривер - генерирует гипотетические документы для улучшения поиска

        Args:
            base_retriever: Базовый ретривер (по умолчанию vector store)
        """
        self.base_retriever = base_retriever or get_vector_store()
        self.llm = get_llm()
        self.embedding_model = get_embedding_model()

    def _generate_hypothetical_document(self, query: str) -> str:
        """Генерирует гипотетический документ для запроса"""
        prompt = f"""Напиши короткий информативный текст, который мог бы содержать ответ на следующий вопрос:

Вопрос: {query}

Текст должен быть:
- Фактическим и информативным
- Содержать ключевые термины из вопроса
- Быть похожим на фрагмент учебника по математике
- Длиной 2-3 предложения

Текст:"""

        try:
            response = self.llm.invoke(prompt)
            if hasattr(response, "content"):
                return response.content.strip()
            return str(response).strip()
        except Exception as e:
            print(f"⚠️ Ошибка генерации гипотетического документа: {e}")
            return query  # Fallback к оригинальному запросу

    def _generate_multiple_hypotheses(self, query: str, num_hypotheses: int = 3) -> list[str]:
        """Генерирует несколько гипотетических документов"""
        hypotheses = []

        for i in range(num_hypotheses):
            # Варьируем промпт для разнообразия
            if i == 0:
                variation = "подробный экспертный"
            elif i == 1:
                variation = "практический с примерами"
            else:
                variation = "краткий и понятный"

            prompt = f"""Напиши {variation} текст, который мог бы содержать ответ на вопрос: {query}

Текст должен быть похож на фрагмент учебника по математике (2-3 предложения):"""

            try:
                response = self.llm.invoke(prompt)
                if hasattr(response, "content"):
                    hypothesis = response.content.strip()
                else:
                    hypothesis = str(response).strip()

                if hypothesis and len(hypothesis) > 20:
                    hypotheses.append(hypothesis)
            except Exception as e:
                print(f"⚠️ Ошибка генерации гипотезы {i + 1}: {e}")

        return hypotheses if hypotheses else [query]

    def retrieve(self, query: str, k: int = 5, use_multiple_hypotheses: bool = False) -> list[Any]:
        """
        Поиск с использованием HyDE

        Args:
            query: Поисковый запрос
            k: Количество документов для возврата
            use_multiple_hypotheses: Использовать несколько гипотез

        Returns:
            Список документов, найденных через гипотетические документы
        """
        if use_multiple_hypotheses:
            # Генерируем несколько гипотез
            hypotheses = self._generate_multiple_hypotheses(query, num_hypotheses=3)

            all_results = []
            seen_contents = set()

            for hypothesis in hypotheses:
                # Поиск по каждой гипотезе
                if hasattr(self.base_retriever, "similarity_search"):
                    results = self.base_retriever.similarity_search(hypothesis, k=k)
                else:
                    results = self.base_retriever.retrieve(hypothesis, k=k)

                # Добавляем уникальные результаты
                for doc in results:
                    content = doc.page_content if hasattr(doc, "page_content") else str(doc)
                    if content not in seen_contents:
                        all_results.append(doc)
                        seen_contents.add(content)

            return all_results[:k]
        else:
            # Одна гипотеза
            hypothetical_doc = self._generate_hypothetical_document(query)

            # Поиск по гипотетическому документу
            if hasattr(self.base_retriever, "similarity_search"):
                return self.base_retriever.similarity_search(hypothetical_doc, k=k)
            else:
                return self.base_retriever.retrieve(hypothetical_doc, k=k)

    def retrieve_with_original(self, query: str, k: int = 5, alpha: float = 0.7) -> list[Any]:
        """
        Комбинирует результаты HyDE и оригинального запроса

        Args:
            query: Поисковый запрос
            k: Количество документов для возврата
            alpha: Вес для HyDE результатов (1-alpha для оригинального запроса)
        """
        # HyDE результаты
        hyde_results = self.retrieve(query, k=k * 2)

        # Оригинальные результаты
        if hasattr(self.base_retriever, "similarity_search"):
            original_results = self.base_retriever.similarity_search(query, k=k * 2)
        else:
            original_results = self.base_retriever.retrieve(query, k=k * 2)

        # Комбинируем с весами
        combined_results = []
        seen_contents = set()

        # Сначала добавляем HyDE результаты с весом alpha
        hyde_count = int(k * alpha)
        for doc in hyde_results[:hyde_count]:
            content = doc.page_content if hasattr(doc, "page_content") else str(doc)
            if content not in seen_contents:
                combined_results.append(doc)
                seen_contents.add(content)

        # Затем добавляем оригинальные результаты
        for doc in original_results:
            if len(combined_results) >= k:
                break
            content = doc.page_content if hasattr(doc, "page_content") else str(doc)
            if content not in seen_contents:
                combined_results.append(doc)
                seen_contents.add(content)

        return combined_results[:k]
