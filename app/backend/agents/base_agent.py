"""Base agent class for all agents."""

import os
from typing import List, Optional, Dict, Any
from langchain.agents import create_agent
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from dotenv import load_dotenv

from ..prompts import DEFAULT_AGENT

# Load environment variables
load_dotenv()


class BaseAgent:
    """Base class for all agents."""
    
    def __init__(
        self,
        openrouter_api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.7,
        base_url: str = "https://openrouter.ai/api/v1",
        system_prompt: Optional[str] = None
    ):
        """
        Initialize the base agent.
        
        Args:
            openrouter_api_key: OpenRouter API key (if None, uses OPENROUTER_API_KEY env var)
            model_name: Model to use (OpenRouter format, e.g., "openai/gpt-4o-mini")
            temperature: Model temperature
            base_url: OpenRouter API base URL
            system_prompt: System prompt for the agent
        """
        # Get API key
        api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenRouter API key required. Set OPENROUTER_API_KEY environment variable "
                "or pass openrouter_api_key parameter."
            )
        
        # Get model name
        if model_name is None:
            model_name = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key,
            base_url=base_url,
            default_headers={
                "HTTP-Referer": "https://github.com/your-repo",
                "X-Title": "Math Tutor Agent"
            }
        )
        
        # System prompt
        self.system_prompt = system_prompt or self._get_default_system_prompt()
        
        # Tools (to be set by subclasses)
        self.tools: List[Tool] = []
        
        # Agent (to be initialized by subclasses)
        self.agent: Optional[Any] = None
    
    def _get_default_system_prompt(self) -> str:
        """Get default system prompt."""
        return DEFAULT_AGENT
    
    def _initialize_agent(self):
        """Initialize the agent. Must be called after tools are set."""
        if not self.tools:
            raise ValueError("Tools must be set before initializing agent")
        
        # Create agent using new LangChain 0.3+ API
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=self.system_prompt
        )
    
    def _convert_messages(self, messages: Optional[List[Any]]) -> List[BaseMessage]:
        """
        Convert message list to LangChain BaseMessage list.
        
        Args:
            messages: List of message objects with 'role' and 'content' attributes
            
        Returns:
            List of LangChain BaseMessage objects
        """
        if not messages:
            return []
        
        langchain_messages = []
        for msg in messages:
            if hasattr(msg, 'role') and hasattr(msg, 'content'):
                role = msg.role
                content = msg.content
            elif isinstance(msg, dict):
                role = msg.get("role")
                content = msg.get("content")
            else:
                continue
            
            if role == "user":
                langchain_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                langchain_messages.append(AIMessage(content=content))
        
        return langchain_messages
    
    def chat(
        self,
        message: str,
        conversation_history: Optional[List[Any]] = None,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a chat message and return response.
        
        Args:
            message: User's message
            conversation_history: Previous conversation messages
            context: Additional context to include (e.g., from RAG)
            
        Returns:
            Dictionary with 'response', 'code_executed', and 'code_result' keys
        """
        if self.agent is None:
            raise ValueError("Agent not initialized. Call _initialize_agent() first.")
        
        # Add context to message if provided
        if context:
            enhanced_message = f"{context}\n\nВопрос пользователя: {message}"
        else:
            enhanced_message = message
        
        # Convert conversation history to messages format
        messages = self._convert_messages(conversation_history)
        # Add current user message
        messages.append(HumanMessage(content=enhanced_message))
        
        # Prepare input for new API
        input_data = {"messages": messages}
        
        # Execute agent
        try:
            result = self.agent.invoke(input_data)
            
            # Extract response from result
            # In new API, result contains messages list
            if isinstance(result, dict) and "messages" in result:
                # Get last message (should be AI response)
                response_messages = result["messages"]
                if response_messages:
                    last_message = response_messages[-1]
                    if hasattr(last_message, 'content'):
                        response_text = last_message.content
                    elif isinstance(last_message, dict):
                        response_text = last_message.get("content", "I apologize, but I couldn't generate a response.")
                    else:
                        response_text = str(last_message)
                else:
                    response_text = "I apologize, but I couldn't generate a response."
            elif isinstance(result, dict) and "output" in result:
                response_text = result["output"]
            else:
                response_text = str(result)
            
            # Check if code was executed
            # In new API, check for tool calls in messages
            code_executed = False
            if isinstance(result, dict) and "messages" in result:
                for msg in result["messages"]:
                    # Check if message contains tool calls
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        for call in msg.tool_calls:
                            if isinstance(call, dict):
                                if call.get("name") == "python_code_interpreter":
                                    code_executed = True
                                    break
                            elif hasattr(call, 'name') and call.name == "python_code_interpreter":
                                code_executed = True
                                break
                    elif isinstance(msg, dict):
                        tool_calls = msg.get("tool_calls", [])
                        for call in tool_calls:
                            if isinstance(call, dict) and call.get("name") == "python_code_interpreter":
                                code_executed = True
                                break
                    if code_executed:
                        break
            
            return {
                "response": response_text,
                "code_executed": code_executed,
                "code_result": None
            }
        except Exception as e:
            return {
                "response": f"I encountered an error: {str(e)}",
                "code_executed": False,
                "code_result": None
            }

