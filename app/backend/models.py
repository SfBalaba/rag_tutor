"""Pydantic models for API requests and responses."""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

from backend.tools.code_interpreter import CodeExecutionResult


class ChatMessage(BaseModel):
    """Single chat message."""
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Request to send a message to the agent."""
    message: str = Field(..., description="User message")
    conversation_history: Optional[List[ChatMessage]] = Field(
        default=None,
        description="Previous conversation messages for context"
    )


class SourceReference(BaseModel):
    """Reference to a source document."""
    filename: str = Field(..., description="Source filename")
    level: str = Field(..., description="Education level")
    source: str = Field(..., description="Source path")
    text_preview: Optional[str] = Field(None, description="Preview of the text")


class ChatResponse(BaseModel):
    """Response from the agent."""
    response: str = Field(..., description="Agent's response")
    code_executed: bool = Field(default=False, description="Whether code was executed")
    code_result: Optional[CodeExecutionResult] = Field(
        None,
        description="Result of code execution if any"
    )
    sources: Optional[List[SourceReference]] = Field(
        None,
        description="Source references from RAG retrieval"
    )


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(default="healthy", description="Service status")
    version: str = Field(default="1.0.0", description="API version")


