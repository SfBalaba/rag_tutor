from langchain_openai import ChatOpenAI

from core.config import config


class ChatOpenRouter(ChatOpenAI):
    def __init__(self, model_name=config["model"]["name"], temperature=0.0, max_tokens=10000):
        provider = config.get("openrouter", {}).get("provider")
        extra_body = None
        if provider:
            extra_body = {"provider": {"order": [provider]}}

        super().__init__(
            model=model_name,
            api_key=config["openrouter"]["api_key"],
            base_url=config["openrouter"]["api_url"],
            temperature=temperature,
            max_completion_tokens=max_tokens,
            extra_body=extra_body,
        )
