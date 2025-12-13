"""Base tool interface for all tools."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Callable, Optional
from langchain_core.tools import Tool


class BaseTool(ABC):
    """Base abstract class for all tools."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Get the tool name."""
        pass
    
    @abstractmethod
    def execute(self, input: str) -> Dict[str, Any]:
        """
        Execute the tool with given input.
        
        Args:
            input: Input string for the tool
            
        Returns:
            Dictionary with execution results (must contain 'output' key)
        """
        pass
    
    def format_result(self, result: Dict[str, Any]) -> str:
        """
        Format execution result as a readable string.
        
        Args:
            result: Result dictionary from execute()
            
        Returns:
            Formatted string
        """
        parts = []
        
        if result.get('output'):
            parts.append(f"Output:\n{result['output']}")
        
        if result.get('error'):
            parts.append(f"Error:\n{result['error']}")
        
        if result.get('plots'):
            parts.append(f"Generated {len(result['plots'])} plot(s)")
        
        return '\n\n'.join(parts) if parts else 'No output'
    
    @abstractmethod
    def get_description(self) -> str:
        """
        Get the description for the tool.
        
        Returns:
            Tool description string
        """
        pass
    
    def _execute_with_callback(
        self, 
        input: str, 
        result_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> str:
        """
        Execute tool with optional callback to store results.
        
        Args:
            input: Input string for the tool
            result_callback: Optional callback to store execution result
            
        Returns:
            Formatted result string
        """
        result = self.execute(input)
        
        # Store result via callback if provided
        if result_callback:
            result_callback(result)
        
        return self.format_result(result)
    
    def get_tool(
        self, 
        result_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> Tool:
        """
        Get a LangChain Tool instance.
        
        Args:
            result_callback: Optional callback to store execution result
            
        Returns:
            LangChain Tool instance
        """
        return Tool(
            name=self.name,
            description=self.get_description(),
            func=lambda input_str: self._execute_with_callback(input_str, result_callback)
        )

