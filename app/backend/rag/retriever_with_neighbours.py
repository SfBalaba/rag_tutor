import os
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional

from .retriever import RAGRetriever

from dotenv import load_dotenv
load_dotenv()

levels = ["elementary", "middle_school", "high_school", "university"]


class RAGRetriverUpgrade(RAGRetriever):
    def __init__(self, faiss_db_path: Optional[str] = None,
            chunks_meta_path: Optional[str] = None,
            embedding_model_name: Optional[str] = None,
            top_k: int = 10,
            initial_retrieval_k: int = 10,
            use_reranker: bool = True,
            reranker_model_name: Optional[str] = None,
                use_context_enrichment: bool = True):
        super().__init__(faiss_db_path=faiss_db_path,
            chunks_meta_path=chunks_meta_path,
            embedding_model_name = embedding_model_name,
            top_k=top_k,
            initial_retrieval_k = initial_retrieval_k,
            use_reranker = use_reranker,
            reranker_model_name=reranker_model_name)
        self.use_context_enrichment = use_context_enrichment
        self.top_k = top_k

    def _enrich_with_neighbors(self, retrieved_chunks: List[Dict], window: int = 1) -> List[Dict]:
        """
        Расширяет список чанков соседними (слева и справа) из того же документа.
        
        Args:
            retrieved_chunks: список чанков из retrieve()
            window: сколько соседей брать слева и справа
        
        Returns:
            расширенный список чанков без дубликатов
        """
        if not self.chunks_data or window <= 0:
            return retrieved_chunks
    
        id_to_index = {}
        for idx, (text, meta) in enumerate(zip(self.chunks_data['texts'], self.chunks_data['metadata'])):
            chunk_id = meta.get("chunk_id") if isinstance(meta, dict) else str(idx) 
            chunk_id = f'{chunk_id}_{meta.get("source_file")}' 
            id_to_index[chunk_id] = idx
        self.retrived_chunks = retrieved_chunks
        
        self.id_to_index = id_to_index
        enriched = []
        seen_ids = set()
     
        for chunk in retrieved_chunks:
            orig_id = chunk["metadata"]["chunk_id"]
            orig_id = f'{orig_id}_{chunk["metadata"].get("source_file")}'
            if orig_id not in id_to_index:
                continue
    
            orig_idx = id_to_index[orig_id]
            source_file = chunk["metadata"].get("source_file", "")
    
            # Добавляем самого себя
            if orig_id not in seen_ids:
                enriched.append(chunk)
                seen_ids.add(orig_id)
    
            # Ищем соседей
            for offset in range(-window, window + 1):
                if offset == 0:
                    continue
                neighbor_idx = orig_idx + offset
                if neighbor_idx < 0 or neighbor_idx >= len(self.chunks_data['texts']):
                    continue
    
                neighbor_meta = self.chunks_data['metadata'][neighbor_idx]
                neighbor_source = neighbor_meta.get("source_file", "") if isinstance(neighbor_meta, dict) else ""
                
                # Только из того же файла
                if neighbor_source != source_file:
                    continue
    
                neighbor_id = neighbor_meta.get("chunk_id", str(neighbor_idx))
                if neighbor_id in seen_ids:
                    continue
    
                neighbor_chunk = {
                    "id": neighbor_id,
                    "text": self.chunks_data['texts'][neighbor_idx],
                    "metadata": neighbor_meta if isinstance(neighbor_meta, dict) else {"raw": str(neighbor_meta)},
                    "distance": float('inf'),  
                    "level": neighbor_meta.get("level", "unknown") if isinstance(neighbor_meta, dict) else "unknown"
                }
                enriched.append(neighbor_chunk)
                seen_ids.add(neighbor_id)
    
        return enriched


    def retrieve(self,
                 query: str,
            levels: Optional[List[str]] = None,
            top_k: Optional[int] = None,
            initial_retrieval_k: Optional[int] = None,
        ) -> List[Dict[str, Any]]:
        self.use_reranker = False
        
        if top_k is None:
            top_k = self.top_k
            
        results = super().retrieve(query,
            levels,
            top_k,
            initial_retrieval_k)
        self.use_reranker = True
        
        if self.use_context_enrichment and results:
            results = self._enrich_with_neighbors(results, window=3)
            if self.use_reranker and self.reranker:
                results = self._rerank_chunks(query, results)
        
        return results[:top_k]
    