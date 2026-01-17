from __future__ import annotations

from pathlib import Path
import pickle
from typing import Any

import numpy as np
import os
import time
import requests
import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from core.config import config

try:
    import faiss
except Exception:  # pragma: no cover - optional dependency at runtime
    faiss = None

torch.classes.__path__ = []
_embedding_model: HuggingFaceEmbeddings | None = None
_vector_store: Any | None = None
_BASE_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = _BASE_DIR.parent


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else _REPO_ROOT / path


def get_embedding_model(
    model_name: str = config["embedding"]["model"],
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> HuggingFaceEmbeddings:
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model

    provider = config.get("embedding", {}).get("provider", "local")
    if provider == "openrouter":
        embed_provider = (
            config.get("embedding", {}).get("provider_order")
            or os.getenv("OPENROUTER_EMBED_PROVIDER")
        )
        _embedding_model = OpenRouterEmbeddings(
            model_name=model_name,
            api_key=config.get("openrouter", {}).get("api_key") or os.getenv("OPENROUTER_API_KEY"),
            base_url=config.get("openrouter", {}).get("api_url", "https://openrouter.ai/api/v1/"),
            provider=embed_provider,
        )
    else:
        print(f"🔌 Local embeddings enabled: model={model_name}")
        _embedding_model = HuggingFaceEmbeddings(
            model_name=model_name, model_kwargs={"device": device, "trust_remote_code": True}
        )
    return _embedding_model


def _build_from_chunks(chunks_dir: Path, index_path: Path) -> FAISS:
    """Строит FAISS из .md чанков, если готового индекса нет."""
    docs: list[Document] = []
    for path in chunks_dir.rglob("*.md"):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if not text.strip():
            continue
        rel = path.relative_to(chunks_dir)
        docs.append(Document(page_content=text, metadata={"path": str(rel)}))

    if not docs:
        raise RuntimeError(f"Не найдены чанки в {chunks_dir}")

    vs = FAISS.from_documents(docs, get_embedding_model())
    index_path.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(index_path))
    return vs


class FaissIndexStore:
    """Lightweight FAISS-backed vector store using prebuilt index + metadata."""

    def __init__(self, index_path: Path, chunks_meta_path: Path, embedding_model: HuggingFaceEmbeddings):
        if faiss is None:
            raise ImportError("faiss is required to use FaissIndexStore")
        self.index_path = index_path
        self.chunks_meta_path = chunks_meta_path
        self.embedding_model = embedding_model
        self.index = faiss.read_index(str(index_path))
        self.documents = self._load_documents(chunks_meta_path)
        print(
            f"✅ FAISS index loaded: {index_path} (dim={self.index.d}, vectors={self.index.ntotal})"
        )
        print(f"📦 Documents loaded for retrieval: {len(self.documents)}")

    def _load_documents(self, chunks_meta_path: Path) -> list[Document]:
        with open(chunks_meta_path, "rb") as f:
            data = pickle.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Unsupported chunks metadata format: {type(data)}")

        docs: list[Document] = []
        for idx, item in enumerate(data):
            if not isinstance(item, dict):
                continue
            content = item.get("content", "")
            if not content:
                continue
            meta = dict(item)
            meta["doc_id"] = meta.get("chunk_id", idx)
            docs.append(Document(page_content=content, metadata=meta))
        return docs

    def _embed_query(self, query: str) -> np.ndarray:
        embedding = self.embedding_model.embed_query(query)
        if not embedding:
            raise ValueError(
                "Empty embedding returned from embeddings provider. "
                "Check OpenRouter availability or provider settings."
            )
        arr = np.array(embedding, dtype="float32").reshape(1, -1)
        if arr.shape[1] != self.index.d:
            raise ValueError(
                f"Embedding dimension mismatch: got {arr.shape[1]} expected {self.index.d}"
            )
        return arr

    def similarity_search_with_score(self, query: str, k: int = 5):
        if not self.documents:
            return []
        query_vec = self._embed_query(query)
        distances, indices = self.index.search(query_vec, k)
        results = []
        for distance, idx in zip(distances[0], indices[0], strict=False):
            if idx < 0 or idx >= len(self.documents):
                continue
            results.append((self.documents[idx], float(distance)))
        return results

    def similarity_search(self, query: str, k: int = 5):
        return [doc for doc, _ in self.similarity_search_with_score(query, k)]


class OpenRouterEmbeddings:
    """Minimal OpenRouter embeddings client (OpenAI-compatible embeddings endpoint)."""

    def __init__(self, model_name: str, api_key: str | None, base_url: str, provider: str | None = None):
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is required for OpenRouter embeddings")
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.provider = provider
        self.session = requests.Session()
        self.request_count = 0
        self.cache: dict[str, list[float]] = {}
        try:
            self.log_every = int(os.getenv("OPENROUTER_EMBED_LOG_EVERY", "20"))
        except ValueError:
            self.log_every = 20
        try:
            self.timeout = int(os.getenv("OPENROUTER_EMBED_TIMEOUT", "120"))
        except ValueError:
            self.timeout = 120
        try:
            self.retries = int(os.getenv("OPENROUTER_EMBED_RETRIES", "3"))
        except ValueError:
            self.retries = 3
        try:
            self.backoff = float(os.getenv("OPENROUTER_EMBED_BACKOFF", "2.0"))
        except ValueError:
            self.backoff = 2.0
        provider_label = f", provider={self.provider}" if self.provider else ", provider=auto"
        print(f"🔌 OpenRouter embeddings enabled: model={self.model_name}{provider_label}")

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/your-repo",
            "X-Title": "rag-tutor",
        }

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float] | None] = [None] * len(texts)
        missing_texts: list[str] = []
        missing_indices: list[int] = []

        for idx, text in enumerate(texts):
            cached = self.cache.get(text)
            if cached is not None:
                embeddings[idx] = cached
            else:
                missing_texts.append(text)
                missing_indices.append(idx)

        if missing_texts:
            self.request_count += 1
            if self.request_count == 1 or self.request_count % self.log_every == 0:
                print(
                    f"🔎 Embeddings request #{self.request_count} (batch_size={len(missing_texts)})"
                )

            payload = {"model": self.model_name, "input": missing_texts}
            if self.provider:
                payload["provider"] = {"order": [self.provider]}

            last_error = None
            for attempt in range(1, self.retries + 1):
                try:
                    resp = self.session.post(
                        f"{self.base_url}/embeddings",
                        headers=self._headers(),
                        json=payload,
                        timeout=self.timeout,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    items = data.get("data", [])
                    if not items:
                        raise ValueError("Empty embeddings response")
                    items = sorted(items, key=lambda x: x.get("index", 0))
                    new_embeddings = [item.get("embedding", []) for item in items]
                    for idx, emb in zip(missing_indices, new_embeddings, strict=False):
                        embeddings[idx] = emb
                        if emb:
                            self.cache[texts[idx]] = emb
                    last_error = None
                    break
                except Exception as e:
                    last_error = e
                    if attempt < self.retries:
                        sleep_for = self.backoff * attempt
                        print(
                            f"⚠️ Embeddings request failed (attempt {attempt}/{self.retries}): "
                            f"{e}. Retrying in {sleep_for:.1f}s..."
                        )
                        time.sleep(sleep_for)
                    else:
                        print(f"❌ Embeddings request failed after {self.retries} attempts: {e}")
                        raise

        return [emb if emb is not None else [] for emb in embeddings]

    def embed_query(self, text: str) -> list[float]:
        embeddings = self._embed_texts([text])
        return embeddings[0] if embeddings else []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed_texts(texts)


def _load_or_build_vector_store(index_path: Path, chunks_dir: Path) -> FAISS:
    """Пытается загрузить готовый FAISS, иначе строит заново из чанков."""
    if (index_path / "index.faiss").exists() and (index_path / "index.pkl").exists():
        return FAISS.load_local(
            str(index_path), get_embedding_model(), allow_dangerous_deserialization=True
        )
    return _build_from_chunks(chunks_dir, index_path)


def get_vector_store(
    index_path: str = config["database"]["index_path"],
    chunks_dir: str | None = config["database"].get("chunks_dir"),
) -> FAISS:
    global _vector_store
    if _vector_store is not None:
        return _vector_store

    faiss_index_path = config["database"].get("faiss_index_path")
    chunks_meta_path = config["database"].get("chunks_meta_path")

    if faiss_index_path and chunks_meta_path:
        faiss_index_p = _resolve_path(faiss_index_path)
        chunks_meta_p = _resolve_path(chunks_meta_path)
        if faiss_index_p.exists() and chunks_meta_p.exists():
            print(
                f"🧭 Using prebuilt FAISS index: {faiss_index_p} with metadata {chunks_meta_p}"
            )
            _vector_store = FaissIndexStore(
                index_path=faiss_index_p,
                chunks_meta_path=chunks_meta_p,
                embedding_model=get_embedding_model(),
            )
            return _vector_store

    index_path_p = _resolve_path(index_path)
    chunks_dir_p = _resolve_path(chunks_dir) if chunks_dir else _resolve_path("actual_data/parsed_chunks")

    _vector_store = _load_or_build_vector_store(index_path_p, chunks_dir_p)
    return _vector_store


def get_documents() -> list[Any] | None:
    """Получает все документы из векторного хранилища"""
    try:
        vector_store = get_vector_store()
        if hasattr(vector_store, "documents"):
            return list(vector_store.documents)
        if hasattr(vector_store, "docstore") and hasattr(vector_store.docstore, "_dict"):
            return list(vector_store.docstore._dict.values())
    except Exception as e:
        print(f"Ошибка получения документов: {e}")
    return None
