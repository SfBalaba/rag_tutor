from core.config import config

try:
    from deepeval.models import DeepEvalBaseLLM

    DEEPEVAL_AVAILABLE = True
except ImportError:
    DEEPEVAL_AVAILABLE = False
    DeepEvalBaseLLM = object

from core.llm.models import ChatOpenRouter


class OpenRouterDeepEvalAdapter(DeepEvalBaseLLM):
    """Адаптер OpenRouter для DeepEval метрик"""

    def __init__(
        self, model_name: str | None = None, temperature: float = 0.0, max_tokens: int = 4096
    ):
        if not DEEPEVAL_AVAILABLE:
            raise ImportError("DeepEval не установлен. Выполните: uv sync --extra research")

        # Используем модель из конфига если не указана
        if model_name is None:
            model_name = config["model"]["name"]

        # Создаем экземпляр LangChain модели
        self.llm = ChatOpenRouter(
            model_name=model_name, temperature=temperature, max_tokens=max_tokens
        )
        self.model_name = model_name

    def load_model(self):
        """Загружает модель (требуется DeepEval)"""
        return self.llm

    def generate(self, prompt: str) -> str:
        """Синхронная генерация текста"""
        try:
            response = self.llm.invoke(prompt)
            return str(response.content)
        except Exception as e:
            print(f"⚠️ Ошибка генерации в DeepEval адаптере: {e}")
            return ""

    async def a_generate(self, prompt: str) -> str:
        """Асинхронная генерация текста"""
        try:
            response = await self.llm.ainvoke(prompt)
            return str(response.content)
        except Exception as e:
            print(f"⚠️ Ошибка асинхронной генерации в DeepEval адаптере: {e}")
            return ""

    def get_model_name(self) -> str:
        """Возвращает имя модели"""
        return f"OpenRouter-{self.model_name}"


def create_deepeval_adapter(
    model_name: str | None = None, temperature: float = 0.0, max_tokens: int = 4096
) -> OpenRouterDeepEvalAdapter | None:
    """Создает адаптер DeepEval для OpenRouter"""
    if not DEEPEVAL_AVAILABLE:
        print("⚠️ DeepEval не доступен. Установите: uv sync --extra research")
        return None

    try:
        return OpenRouterDeepEvalAdapter(
            model_name=model_name, temperature=temperature, max_tokens=max_tokens
        )
    except Exception as e:
        print(f"⚠️ Ошибка создания DeepEval адаптера: {e}")
        return None
