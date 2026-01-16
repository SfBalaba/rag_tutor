"""Math tutor agent with code interpreter capability."""

from typing import Optional, List, Dict, Any

from .base_agent import BaseAgent
from ..tools.code_interpreter import CodeInterpreter, CodeExecutionResult
# from ..rag.retriever import RAGRetriever
from ..rag.retriever_with_neighbours import RAGRetriverUpgrade

from ..prompts import MATH_TUTOR_AGENT


class MathTutorAgent(BaseAgent):
    """Math tutor agent that can execute Python code for calculations."""
    
    def __init__(
        self,
        openrouter_api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.7,
            use_rag: bool = True,
            faiss_db_path: Optional[str] = None,
            chunks_meta_path: Optional[str] = None,
    ):
        """
        Initialize the math tutor agent.
        
        Args:
            openrouter_api_key: OpenRouter API key (if None, uses OPENROUTER_API_KEY env var)
            model_name: Model to use (OpenRouter format, e.g., "openai/gpt-4o-mini")
            temperature: Model temperature
            use_rag: Whether to use RAG for retrieving relevant chunks
            faiss_db_path: Path to FAISS database directory
            chunks_meta_path: Path to pickle file with chunks metadata
        """
        # Initialize code interpreter
        self.code_interpreter = CodeInterpreter()
        # Store last code execution result for plots
        self.last_code_result = None
        
        # Initialize RAG retriever
        self.use_rag = use_rag
        if use_rag:
            try:
                # RAGRetriever now uses FAISS
                self.rag_retriever = RAGRetriverUpgrade(
                    faiss_db_path=faiss_db_path,
                    chunks_meta_path=chunks_meta_path,
                    top_k = 10,
                    initial_retrieval_k = 10,
                    use_reranker = True
                )
            except Exception as e:
                print(f"⚠️ Warning: Could not initialize RAG retriever: {e}")
                self.use_rag = False
                self.rag_retriever = None
        else:
            self.rag_retriever = None
        
        # Initialize base agent
        super().__init__(
            openrouter_api_key=openrouter_api_key,
            model_name=model_name,
            temperature=temperature,
            system_prompt=MATH_TUTOR_AGENT
        )
        
        # Create code execution tool with callback to store results
        def store_result(result: Dict[str, Any]):
            """Store execution result for later retrieval of plots."""
            self.last_code_result = result
        
        code_tool = self.code_interpreter.get_tool(result_callback=store_result)
        
        # Set tools
        self.tools = [code_tool]
        
        # Initialize agent
        self._initialize_agent()
    
    def chat(
        self,
        message: str,
        conversation_history: Optional[List[Any]] = None,
        levels: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Process a chat message with RAG retrieval.
        
        Args:
            message: User's message
            conversation_history: Previous conversation messages
            
        Returns:
            Dictionary with 'response', 'code_executed', 'code_result', and 'sources' keys
        """
        context = None
        sources = None
        
        # Retrieve relevant chunks if RAG is enabled
        if self.use_rag and self.rag_retriever:
            try:
                # Retrieve more chunks initially, reranker will select the best ones
                chunks = self.rag_retriever.retrieve(query=message, levels=levels)
                if chunks:
                    context = self.rag_retriever.format_chunks_for_context(chunks)
                    # Prepare sources for response
                    sources = []
                    for chunk in chunks:
                        metadata = chunk.get("metadata", {})
                        sources.append({
                            "filename": metadata.get("source_file", "unknown"), ## source_file
                            "level": chunk.get("level", "unknown"),
                            "source": metadata.get("chunk_file_path", "unknown"), ## chunk_file_path
                            "text_preview": chunk.get("text", "")[:200] + "..." if len(chunk.get("text", "")) > 200 else chunk.get("text", "") ## content
                        })
                    print(f"✅ Retrieved {len(chunks)} relevant chunks for query (after reranking)")
            except Exception as e:
                print(f"⚠️ Error during RAG retrieval: {e}")
        # Reset last code result before execution
        self.last_code_result = None
        
        # Call parent chat method with context
        result = super().chat(
            message=message,
            conversation_history=conversation_history,
            context=context
        )
        
        # Check if code was executed and extract plots
        # Note: last_code_result is set during code execution in callback
        if result.get("code_executed"):
            if self.last_code_result:
                result["code_result"] = CodeExecutionResult(
                    output=self.last_code_result.get("output", ""),
                    error=self.last_code_result.get("error"),
                    plots=self.last_code_result.get("plots")
                )
            else:
                # Code was executed but no result captured - create empty result
                result["code_result"] = CodeExecutionResult(
                    output="Code executed successfully",
                    error=None,
                    plots=None
                )
        
        # Add sources to result
        if sources:
            result["sources"] = sources
        
        return result

