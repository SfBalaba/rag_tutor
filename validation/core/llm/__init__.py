from typing import Any

from core.config import config
from core.llm.models import ChatOpenRouter

try:
    from core.llm.vllm_client import ChatVLLM

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    ChatVLLM = None

_llm: Any = None


def get_llm(temperature=0.0, max_tokens=10000):
    global _llm

    if _llm is None:
        if config["model"]["inference"] == "local":
            if not VLLM_AVAILABLE:
                raise ImportError("vLLM не установлен. Установите: uv sync --extra local")
            _llm = ChatVLLM()
        else:
            _llm = ChatOpenRouter(temperature=temperature, max_tokens=max_tokens)

    # Для vLLM обновляем параметры на лету
    if config["model"]["inference"] == "local" and _llm is not None:
        _llm.temperature = temperature
        _llm.max_tokens = max_tokens

    return _llm
