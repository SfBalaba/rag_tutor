from typing import Any

from .contextual_retrieval import ContextualRetriever
from .hybrid_search import HybridRetriever
from .hyde_enhanced import HyDERetriever
from .rag_fusion import RAGFusionRetriever


class AdvancedRetriever:
    """
    Продвинутый ретривер, комбинирующий все техники:
    Classic RAG → Hybrid Search → HyDE → Contextual → RAG-Fusion
    """

    def __init__(self, config: dict[str, Any] = None):
        """
        Инициализирует продвинутый ретривер

        Args:
            config: Конфигурация для каждого компонента
        """
        if config is None:
            config = {
                "hybrid": {"alpha": 0.5},
                "hyde": {"use_multiple_hypotheses": False},
                "contextual": {"expand_query": True, "inject_context": True},
                "rag_fusion": {"num_queries": 4, "fusion_method": "rrf"},
            }

        self.config = config

        # Инициализируем компоненты последовательно
        self.hybrid_retriever = HybridRetriever(alpha=config["hybrid"]["alpha"])
        self.hyde_retriever = HyDERetriever(base_retriever=self.hybrid_retriever)
        self.contextual_retriever = ContextualRetriever(base_retriever=self.hyde_retriever)
        self.rag_fusion_retriever = RAGFusionRetriever(base_retriever=self.contextual_retriever)

    def retrieve_classic(self, query: str, k: int = 5) -> list[Any]:
        """Классический dense retrieval"""
        return self.hybrid_retriever.vector_store.similarity_search(query, k=k)

    def retrieve_hybrid(self, query: str, k: int = 5) -> list[Any]:
        """Classic RAG + Hybrid Search"""
        return self.hybrid_retriever.retrieve(query, k=k)

    def retrieve_hyde_enhanced(self, query: str, k: int = 5) -> list[Any]:
        """Hybrid + HyDE Enhanced"""
        use_multiple = self.config["hyde"]["use_multiple_hypotheses"]
        return self.hyde_retriever.retrieve(query, k=k, use_multiple_hypotheses=use_multiple)

    def retrieve_contextual(
        self, query: str, k: int = 5, conversation_history: list[str] = None
    ) -> list[Any]:
        """HyDE + Contextual Retrieval"""
        expand_query = self.config["contextual"]["expand_query"]
        return self.contextual_retriever.retrieve(
            query, k=k, conversation_history=conversation_history, expand_query=expand_query
        )

    def retrieve_full_stack(
        self, query: str, k: int = 5, conversation_history: list[str] = None
    ) -> list[Any]:
        """Полный стек: Contextual + RAG-Fusion (максимальное качество)"""
        num_queries = self.config["rag_fusion"]["num_queries"]
        fusion_method = self.config["rag_fusion"]["fusion_method"]

        # Используем contextual retriever как базу для RAG-Fusion
        return self.rag_fusion_retriever.retrieve(
            query, k=k, num_queries=num_queries, fusion_method=fusion_method
        )

    async def retrieve_full_stack_async(
        self,
        query: str,
        k: int = 5,
        conversation_history: list[str] = None,
        max_concurrency: int = 3,
    ) -> list[Any]:
        """Асинхронная версия полного стека"""
        num_queries = self.config["rag_fusion"]["num_queries"]

        return await self.rag_fusion_retriever.retrieve_async(
            query, k=k, num_queries=num_queries, max_concurrency=max_concurrency
        )

    def benchmark_all_methods(self, query: str, k: int = 5) -> dict[str, list[Any]]:
        """
        Сравнивает все методы на одном запросе

        Returns:
            Словарь с результатами каждого метода
        """
        print(f"🔬 Бенчмарк всех методов для запроса: {query}")

        results = {}

        # Classic RAG
        print("  📊 Classic RAG...")
        results["classic"] = self.retrieve_classic(query, k)

        # Hybrid Search
        print("  📊 + Hybrid Search...")
        results["hybrid"] = self.retrieve_hybrid(query, k)

        # HyDE Enhanced
        print("  📊 + HyDE Enhanced...")
        results["hyde"] = self.retrieve_hyde_enhanced(query, k)

        # Contextual Retrieval
        print("  📊 + Contextual Retrieval...")
        results["contextual"] = self.retrieve_contextual(query, k)

        # RAG-Fusion (Full Stack)
        print("  📊 + RAG-Fusion...")
        results["full_stack"] = self.retrieve_full_stack(query, k)

        return results

    def get_performance_config(self, mode: str = "optimal") -> dict[str, Any]:
        """
        Возвращает конфигурацию для разных режимов производительности

        Args:
            mode: "optimal", "production", "budget"
        """
        configs = {
            "optimal": {
                "hybrid": {"alpha": 0.5},
                "hyde": {"use_multiple_hypotheses": True},
                "contextual": {"expand_query": True, "inject_context": True},
                "rag_fusion": {"num_queries": 4, "fusion_method": "rrf"},
            },
            "production": {
                "hybrid": {"alpha": 0.6},
                "hyde": {"use_multiple_hypotheses": False},
                "contextual": {"expand_query": True, "inject_context": False},
                "rag_fusion": {"num_queries": 3, "fusion_method": "rrf"},
            },
            "budget": {
                "hybrid": {"alpha": 0.7},
                "hyde": {"use_multiple_hypotheses": False},
                "contextual": {"expand_query": False, "inject_context": False},
                "rag_fusion": {"num_queries": 2, "fusion_method": "simple"},
            },
        }

        return configs.get(mode, configs["production"])

    def update_config(self, new_config: dict[str, Any]):
        """Обновляет конфигурацию и пересоздает компоненты"""
        self.config.update(new_config)

        # Пересоздаем компоненты с новой конфигурацией
        self.hybrid_retriever.set_alpha(self.config["hybrid"]["alpha"])
        # Остальные компоненты используют обновленную конфигурацию автоматически

    def get_retrieval_stats(self, results: dict[str, list[Any]]) -> dict[str, dict[str, Any]]:
        """Анализирует статистику результатов разных методов"""
        stats = {}

        for method, docs in results.items():
            unique_docs = set()
            total_length = 0

            for doc in docs:
                content = doc.page_content if hasattr(doc, "page_content") else str(doc)
                unique_docs.add(content[:100])  # Первые 100 символов для уникальности
                total_length += len(content)

            stats[method] = {
                "total_docs": len(docs),
                "unique_docs": len(unique_docs),
                "avg_doc_length": total_length / len(docs) if docs else 0,
                "diversity_ratio": len(unique_docs) / len(docs) if docs else 0,
            }

        return stats


# Фабричная функция для создания ретривера
def create_advanced_retriever(mode: str = "production") -> AdvancedRetriever:
    """
    Создает продвинутый ретривер с предустановленной конфигурацией

    Args:
        mode: "optimal", "production", "budget"
    """
    retriever = AdvancedRetriever()
    config = retriever.get_performance_config(mode)
    retriever.update_config(config)

    print(f"✅ Создан AdvancedRetriever в режиме '{mode}'")
    return retriever
