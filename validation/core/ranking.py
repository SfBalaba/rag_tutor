import os
import time
import torch
import requests
from FlagEmbedding import FlagLLMReranker

from core.config import config

torch.classes.__path__ = []


def get_doc_content(doc):
    return doc.page_content if hasattr(doc, "page_content") else str(doc)


_reranker = None


class JinaReranker:
    """Jina API reranker wrapper with FlagEmbedding-compatible interface."""

    def __init__(self, model: str, api_key: str | None, base_url: str | None = None):
        if not api_key:
            raise ValueError("JINA_API_KEY is required for Jina reranker")
        self.model = model
        self.api_key = api_key
        self.base_url = (base_url or "https://api.jina.ai/v1").rstrip("/")
        self.session = requests.Session()
        self._last_request_ts: float | None = None
        try:
            self.timeout = int(os.getenv("JINA_RERANK_TIMEOUT", "60"))
        except ValueError:
            self.timeout = 60
        try:
            self.retries = int(os.getenv("JINA_RERANK_RETRIES", "3"))
        except ValueError:
            self.retries = 3
        try:
            self.backoff = float(os.getenv("JINA_RERANK_BACKOFF", "2.0"))
        except ValueError:
            self.backoff = 2.0
        try:
            self.min_interval = float(os.getenv("JINA_RERANK_MIN_INTERVAL", "0"))
        except ValueError:
            self.min_interval = 0.0
        try:
            self.max_doc_chars = int(os.getenv("JINA_RERANK_MAX_DOC_CHARS", "0"))
        except ValueError:
            self.max_doc_chars = 0

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _score_query(self, query: str, documents: list[str]) -> list[float]:
        if self.max_doc_chars and self.max_doc_chars > 0:
            documents = [doc[: self.max_doc_chars] for doc in documents]

        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": len(documents),
            "return_documents": False,
        }
        last_error = None
        for attempt in range(1, self.retries + 1):
            try:
                if self.min_interval and self._last_request_ts is not None:
                    elapsed = time.monotonic() - self._last_request_ts
                    sleep_for = self.min_interval - elapsed
                    if sleep_for > 0:
                        time.sleep(sleep_for)
                self._last_request_ts = time.monotonic()
                resp = self.session.post(
                    f"{self.base_url}/rerank",
                    headers=self._headers(),
                    json=payload,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                last_error = None
                break
            except requests.HTTPError as e:
                last_error = e
                status = e.response.status_code if e.response is not None else None
                if status == 429 and attempt < self.retries:
                    retry_after = e.response.headers.get("Retry-After") if e.response else None
                    wait = self.backoff * attempt
                    if retry_after:
                        try:
                            wait = max(wait, float(retry_after))
                        except ValueError:
                            pass
                    time.sleep(wait)
                    continue
                raise
            except Exception as e:
                last_error = e
                if attempt < self.retries:
                    time.sleep(self.backoff * attempt)
                    continue
                raise

        if last_error:
            raise last_error
        results = data.get("results", [])
        scores = [0.0] * len(documents)
        for item in results:
            idx = item.get("index")
            score = item.get("relevance_score", item.get("score"))
            if idx is None or score is None:
                continue
            if 0 <= idx < len(scores):
                scores[idx] = float(score)
        return scores

    def compute_score(self, pairs: list[list[str]]) -> list[float]:
        if not pairs:
            return []

        grouped: dict[str, list[tuple[int, str]]] = {}
        for idx, pair in enumerate(pairs):
            grouped.setdefault(pair[0], []).append((idx, pair[1]))

        scores = [0.0] * len(pairs)
        for query, items in grouped.items():
            docs = [doc for _, doc in items]
            group_scores = self._score_query(query, docs)
            for (idx, _), score in zip(items, group_scores, strict=False):
                scores[idx] = score

        return scores


def get_reranker():
    global _reranker
    if not config["reranker"]["enabled"] or _reranker is not None:
        return _reranker

    provider = config.get("reranker", {}).get("provider", "local")
    if provider == "jina":
        _reranker = JinaReranker(
            model=config["reranker"]["model"],
            api_key=config["reranker"].get("api_key") or os.getenv("JINA_API_KEY"),
            base_url=config["reranker"].get("api_url"),
        )
    else:
        _reranker = FlagLLMReranker(
            config["reranker"]["model"], use_fp16=torch.cuda.is_available()
        )
    return _reranker


def rerank_documents(query, docs, reranker=None):
    model = reranker or get_reranker()
    if not model:
        return docs

    pairs = [[query, get_doc_content(doc)] for doc in docs]
    scored_docs = list(zip(docs, model.compute_score(pairs), strict=False))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored_docs]
