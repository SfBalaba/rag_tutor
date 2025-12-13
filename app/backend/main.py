"""FastAPI server for Math Tutor Agent."""

import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "rag-tutor")

# Initialize LangSmith client (optional, but ensures connection)
try:
    from langsmith import Client
    langsmith_api_key = os.getenv("LANGCHAIN_API_KEY")
    if langsmith_api_key:
        langsmith_client = Client(api_key=langsmith_api_key)
        print("✅ LangSmith tracing enabled")
    else:
        print("⚠️ LANGCHAIN_API_KEY not set, tracing may not work")
except ImportError:
    print("⚠️ langsmith package not installed, tracing may not work")
except Exception as e:
    print(f"⚠️ LangSmith initialization warning: {e}")

# Add parent directory to path for imports
backend_dir = Path(__file__).parent
app_dir = backend_dir.parent
sys.path.insert(0, str(app_dir))

from backend.models import ChatRequest, ChatResponse, HealthResponse, SourceReference
from backend.agents.math_tutor_agent import MathTutorAgent


# Global agent instance
agent: MathTutorAgent = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    global agent
    # Startup
    try:
        # Determine paths - check if we're in Docker first
        if os.path.exists("/app"):
            # In Docker
            faiss_db_path = "/app/data/faiss_db"
            chunks_meta_path = "/app/data/all_chunks_with_meta_all.pickle"
        else:
            # Not in Docker, use relative paths
            project_root = Path(__file__).parent.parent.parent
            faiss_db_path = str(project_root / "data" / "faiss_db")
            chunks_meta_path = str(project_root / "data" / "all_chunks_with_meta_all.pickle")

        agent = MathTutorAgent(
            use_rag=os.getenv("USE_RAG", "true").lower() == "true",
            faiss_db_path=faiss_db_path,
            chunks_meta_path=chunks_meta_path
        )
        print("✅ Math Tutor Agent initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        raise
    yield
    # Shutdown
    agent = None
    print("👋 Math Tutor Agent shut down")


# Create FastAPI app
app = FastAPI(
    title="Math Tutor Agent API",
    description="API for math tutoring agent with code interpreter",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="healthy", version="1.0.0")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send a message to the math tutor agent.
    
    Args:
        request: Chat request with message and optional conversation history
        
    Returns:
        Chat response from the agent
    """
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        result = agent.chat(
            message=request.message,
            conversation_history=request.conversation_history
        )
        
        # Convert sources if present
        sources = None
        if result.get("sources"):
            sources = [
                SourceReference(**source) for source in result["sources"]
            ]
        
        return ChatResponse(
            response=result["response"],
            code_executed=result["code_executed"],
            code_result=result.get("code_result"),
            sources=sources
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing chat request: {str(e)}"
        )


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Math Tutor Agent API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "chat": "/chat (POST)"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

