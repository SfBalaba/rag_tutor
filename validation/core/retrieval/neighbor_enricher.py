from __future__ import annotations

from typing import Iterable

from langchain_core.documents import Document


class NeighborEnricher:
    """Adds adjacent chunks from the same source_file using chunk_id ordering."""

    def __init__(self, documents: Iterable[Document]):
        self.documents = list(documents)
        self.id_to_index: dict[str, int] = {}
        for idx, doc in enumerate(self.documents):
            meta = getattr(doc, "metadata", {}) or {}
            chunk_id = meta.get("chunk_id")
            source_file = meta.get("source_file")
            if chunk_id is None or source_file is None:
                continue
            key = f"{chunk_id}::{source_file}"
            self.id_to_index[key] = idx

    def enrich(self, docs: list[Document], window: int = 3) -> list[Document]:
        if window <= 0 or not docs:
            return docs

        enriched: list[Document] = []
        seen_ids: set[str] = set()

        for doc in docs:
            meta = getattr(doc, "metadata", {}) or {}
            chunk_id = meta.get("chunk_id")
            source_file = meta.get("source_file")
            if chunk_id is None or source_file is None:
                if id(doc) not in seen_ids:
                    enriched.append(doc)
                    seen_ids.add(str(id(doc)))
                continue

            key = f"{chunk_id}::{source_file}"
            idx = self.id_to_index.get(key)
            if idx is None:
                if key not in seen_ids:
                    enriched.append(doc)
                    seen_ids.add(key)
                continue

            if key not in seen_ids:
                enriched.append(doc)
                seen_ids.add(key)

            for offset in range(-window, window + 1):
                if offset == 0:
                    continue
                neighbor_idx = idx + offset
                if neighbor_idx < 0 or neighbor_idx >= len(self.documents):
                    continue
                neighbor_doc = self.documents[neighbor_idx]
                neighbor_meta = getattr(neighbor_doc, "metadata", {}) or {}
                if neighbor_meta.get("source_file") != source_file:
                    continue
                neighbor_key = f"{neighbor_meta.get('chunk_id')}::{neighbor_meta.get('source_file')}"
                if neighbor_key in seen_ids:
                    continue
                enriched.append(neighbor_doc)
                seen_ids.add(neighbor_key)

        return enriched
