"""RAG Retriever for finding relevant chunks from FAISS vector database.

Expected file structure:
- FAISS index directory: {faiss_db_path}/faiss.index
- Chunks metadata: {chunks_meta_path} (pickle file)

Example paths:
- Docker: /app/data/faiss_db/faiss.index and /app/data/all_chunks_with_meta_all.pickle
- Local: {project_root}/data/faiss_db/faiss.index and {project_root}/data/all_chunks_with_meta_all.pickle
"""

import os
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from dotenv import load_dotenv

load_dotenv()

# Уровни образования
LEVELS = ["elementary", "middle_school", "high_school", "university"]


class RAGRetriever:
    """Retriever for finding relevant chunks from FAISS vector database."""
    
    def __init__(
        self,
        faiss_db_path: Optional[str] = None,
        chunks_meta_path: Optional[str] = None,
        embedding_model_name: Optional[str] = None,
        top_k: int = 5,
        initial_retrieval_k: int = 30,
        use_reranker: bool = True,
        reranker_model_name: Optional[str] = None
    ):
        """
        Initialize RAG retriever with FAISS.
        
        Args:
            faiss_db_path: Path to FAISS index directory or file
            chunks_meta_path: Path to pickle file with chunks and metadata
            embedding_model_name: Name of embedding model
            top_k: Number of top chunks to return after reranking
            initial_retrieval_k: Number of chunks to retrieve initially (before reranking)
            use_reranker: Whether to use reranker to improve results
            reranker_model_name: Name of reranker model (default: cross-encoder/ms-marco-MiniLM-L-6-v2)
        """
        # Determine paths - strict paths only
        if faiss_db_path is None:
            # Check if we're in Docker
            is_docker = os.path.exists("/app")
            
            if is_docker:
                # Docker: strict path
                faiss_db_path = "/app/data/faiss_db"
            else:
                # Local: strict path relative to project root
                project_root = Path(__file__).parent.parent.parent.parent
                faiss_db_path = str(project_root / "data" / "faiss_db")
        
        if chunks_meta_path is None:
            # Check if we're in Docker
            is_docker = os.path.exists("/app")
            
            if is_docker:
                # Docker: strict path
                chunks_meta_path = "/app/data/all_chunks_with_meta_all.pickle"
            else:
                # Local: strict path relative to project root
                project_root = Path(__file__).parent.parent.parent.parent
                chunks_meta_path = str(project_root / "data" / "all_chunks_with_meta_all.pickle")
        
        self.faiss_db_path = faiss_db_path
        self.chunks_meta_path = chunks_meta_path
        self.top_k = top_k
        self.initial_retrieval_k = initial_retrieval_k
        self.use_reranker = use_reranker
        
        # Initialize embedding model
        if embedding_model_name is None:
            embedding_model_name = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
        else:
            embedding_model_name = os.getenv("EMBEDDING_MODEL", embedding_model_name)
        
        try:
            self.embedding_model = SentenceTransformer(embedding_model_name)
            print(f"✅ Loaded embedding model: {embedding_model_name}")
        except Exception as e:
            print(f"⚠️ Warning: Could not load embedding model: {e}")
            self.embedding_model = None
        
        # Initialize reranker
        self.reranker = None
        if use_reranker:
            if reranker_model_name is None:
                reranker_model_name = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
            try:
                self.reranker = CrossEncoder(reranker_model_name)
                print(f"✅ Loaded reranker model: {reranker_model_name}")
            except Exception as e:
                print(f"⚠️ Warning: Could not load reranker model: {e}")
                print(f"   Continuing without reranker...")
                self.use_reranker = False
        
        # Load FAISS index
        self.index = None
        self.chunks_data = None
        self._load_faiss_index()
        
        # Load chunks metadata
        self._load_chunks_metadata()
    
    def _load_faiss_index(self):
        """
        Load FAISS index from directory.
        
        Expected structure:
        - {faiss_db_path}/faiss.index - FAISS index file (required)
        - {faiss_db_path}/embeddings.npy - Embeddings array (optional)
        - {faiss_db_path}/metadata.jsonl - Metadata file (optional)
        """
        try:
            faiss_path = Path(self.faiss_db_path)
            
            # FAISS path must be a directory
            if not faiss_path.is_dir():
                print(f"⚠️ FAISS path must be a directory: {faiss_path}")
                return
            
            # Look for index file with strict name: faiss.index
            index_file = faiss_path / "faiss.index"
            
            if not index_file.exists():
                print(f"⚠️ FAISS index file not found: {index_file}")
                print(f"   Expected file: {faiss_path}/faiss.index")
                return
            
            # Load FAISS index
            self.index = faiss.read_index(str(index_file))
            index_dim = self.index.d
            print(f"✅ Loaded FAISS index from {index_file} (dimension: {index_dim}, vectors: {self.index.ntotal})")
            
            # Check if embedding model dimension matches index dimension
            if self.embedding_model is not None:
                test_embedding = self.embedding_model.encode("test", convert_to_numpy=True)
                model_dim = test_embedding.shape[0]
                if model_dim != index_dim:
                    print(f"⚠️ Warning: Embedding model dimension ({model_dim}) does not match FAISS index dimension ({index_dim})")
                    print(f"   This will cause search errors. Please ensure the index was created with intfloat/multilingual-e5-large")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not load FAISS index from {self.faiss_db_path}: {e}")
            self.index = None
    
    def _load_chunks_metadata(self):
        """Load chunks and metadata from pickle file."""
        try:
            if not os.path.exists(self.chunks_meta_path):
                print(f"⚠️ Chunks metadata file not found: {self.chunks_meta_path}")
                return
            
            with open(self.chunks_meta_path, 'rb') as f:
                data = pickle.load(f)
            
            # Handle different data formats
            if isinstance(data, dict):
                # If it's a dict, try to extract chunks and metadata
                if 'chunks' in data and 'metadata' in data:
                    self.chunks_data = {
                        'texts': data['chunks'],
                        'metadata': data['metadata']
                    }
                elif 'texts' in data and 'metadata' in data:
                    self.chunks_data = data
                else:
                    # Assume it's a list of dicts
                    self.chunks_data = {
                        'texts': [item.get('text', item.get('content', '')) for item in data],
                        'metadata': [item.get('metadata', item) for item in data]
                    }
            elif isinstance(data, list):
                # List of chunks with metadata
                self.chunks_data = {
                    'texts': [item.get('text', item.get('content', '')) for item in data],
                    'metadata': [item.get('metadata', item) for item in data]
                }
            else:
                print(f"⚠️ Unexpected data format in {self.chunks_meta_path}")
                return
            
            print(f"✅ Loaded {len(self.chunks_data['texts'])} chunks from {self.chunks_meta_path}")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not load chunks metadata from {self.chunks_meta_path}: {e}")
            self.chunks_data = None
    
    def _rerank_chunks(
        self,
        query: str,
        chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Rerank chunks using CrossEncoder.
        
        Args:
            query: User query
            chunks: List of chunks to rerank
            
        Returns:
            Reranked list of chunks
        """
        if not chunks or not self.reranker:
            return chunks
        
        try:
            # Prepare pairs for reranking: (query, chunk_text)
            pairs = [(query, chunk["text"]) for chunk in chunks]
            
            # Get reranking scores
            scores = self.reranker.predict(pairs)
            
            # Add scores to chunks and sort by score (descending)
            for i, chunk in enumerate(chunks):
                chunk["rerank_score"] = float(scores[i])
            
            # Sort by rerank score (higher is better)
            reranked = sorted(chunks, key=lambda x: x.get("rerank_score", 0), reverse=True)
            
            print(f"✅ Reranked {len(chunks)} chunks using CrossEncoder")
            return reranked
            
        except Exception as e:
            print(f"⚠️ Warning: Error during reranking: {e}")
            return chunks
    
    def retrieve(
        self,
        query: str,
        levels: Optional[List[str]] = None,
        top_k: Optional[int] = None,
        initial_retrieval_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks for a query with optional reranking.
        
        Args:
            query: User query
            levels: List of education levels to search (None = all levels)
            top_k: Number of top chunks to return after reranking (None = use self.top_k)
            initial_retrieval_k: Number of chunks to retrieve initially (None = use self.initial_retrieval_k)
            
        Returns:
            List of relevant chunks with metadata
        """
        if self.index is None or self.embedding_model is None or self.chunks_data is None:
            print("⚠️ RAG retriever not properly initialized. Returning empty results.")
            return []
        
        if top_k is None:
            top_k = self.top_k
        
        if initial_retrieval_k is None:
            initial_retrieval_k = self.initial_retrieval_k
        
        # Use larger k for initial retrieval if reranking is enabled
        retrieval_k = initial_retrieval_k if self.use_reranker else top_k
        
        # Encode query
        query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)
        query_embedding = query_embedding.astype('float32').reshape(1, -1)
        
        # Check dimension mismatch
        if query_embedding.shape[1] != self.index.d:
            error_msg = (
                f"Dimension mismatch: query embedding has dimension {query_embedding.shape[1]}, "
                f"but FAISS index expects dimension {self.index.d}. "
                f"Please use the same embedding model that was used to create the index."
            )
            print(f"⚠️ {error_msg}")
            raise ValueError(error_msg)
        
        # Search in FAISS index
        try:
            distances, indices = self.index.search(query_embedding, retrieval_k)
            
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < 0 or idx >= len(self.chunks_data['texts']):
                    continue
                
                text = self.chunks_data['texts'][idx]
                metadata = self.chunks_data['metadata'][idx]
                
                # Filter by level if specified
                if levels:
                    chunk_level = metadata.get('level', 'unknown') if isinstance(metadata, dict) else 'unknown'
                    if chunk_level not in levels:
                        continue
                
                chunk = {
                    "id": str(idx),
                    "text": text,
                    "metadata": metadata if isinstance(metadata, dict) else {"raw": str(metadata)},
                    "distance": float(distance),
                    "level": metadata.get('level', 'unknown') if isinstance(metadata, dict) else 'unknown'
                }
                results.append(chunk)
            
            # Rerank if enabled
            if self.use_reranker and self.reranker and results:
                results = self._rerank_chunks(query, results)
            
            # Return top_k results
            return results[:top_k]
            
        except Exception as e:
            error_msg = str(e) if str(e) else f"Unknown error (check dimension mismatch: query={query_embedding.shape[1]}, index={self.index.d if self.index else 'N/A'})"
            print(f"⚠️ Error during FAISS search: {error_msg}")
            import traceback
            print(f"   Traceback: {traceback.format_exc()}")
            return []
    
    def format_chunks_for_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Format retrieved chunks as context string for LLM.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Formatted context string
        """
        if not chunks:
            return ""
        
        context_parts = ["Релевантные материалы из учебников:\n"]
        
        for i, chunk in enumerate(chunks, 1):

            metadata = chunk.get("metadata", {})
            level = chunk.get("level", "unknown")
            source = metadata.get("chunk_file_path", "unknown") if isinstance(metadata, dict) else "unknown"
            filename = metadata.get("source_file", "unknown") if isinstance(metadata, dict) else "unknown"
            
            context_parts.append(f"\n--- Материал {i} ---")
            context_parts.append(f"Источник: {filename}")
            context_parts.append(f"Уровень: {level}")
            context_parts.append(f"Текст:\n{chunk['text']}\n")
        
        return "\n".join(context_parts)
