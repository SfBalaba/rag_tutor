"""Code interpreter using E2B Sandbox for safe Python code execution."""

import os
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from e2b_code_interpreter import Sandbox
from dotenv import load_dotenv
from .base_tool import BaseTool

load_dotenv()


class CodeExecutionResult(BaseModel):
    """Result of code execution."""
    output: str = Field(..., description="Output from code execution")
    error: Optional[str] = Field(None, description="Error message if execution failed")
    plots: Optional[List[str]] = Field(None, description="Base64 encoded plot images")


class CodeInterpreter(BaseTool):
    """Code interpreter using E2B Sandbox for safe execution."""
    
    def __init__(self, timeout: int = 60, max_output_length: int = 10000):
        """
        Initialize code interpreter with E2B.
        
        Args:
            timeout: Maximum execution time in seconds (E2B default is 60s)
            max_output_length: Maximum output length in characters
        """
        super().__init__()
        self.timeout = timeout
        self.max_output_length = max_output_length
        self.api_key = os.getenv("E2B_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "E2B_API_KEY not found in environment variables. "
                "Please set E2B_API_KEY in your .env file."
            )
    
    @property
    def name(self) -> str:
        """Get the tool name."""
        return "python_code_interpreter"
    
    def execute(self, input: str) -> Dict[str, Any]:
        """
        Execute Python code safely using E2B Sandbox.
        
        Args:
            input: Python code to execute
            
        Returns:
            Dictionary with 'output', 'error', and 'plots' keys
        """
        plots = []
        output = ""
        error = None
        
        try:
            with Sandbox(api_key=self.api_key) as sandbox:
                execution = sandbox.run_code(input)
                
                # Get text output
                output = execution.text or ""
                
                # Get logs (stdout/stderr)
                if execution.logs:
                    logs = []
                    if execution.logs.stdout:
                        logs.extend(execution.logs.stdout)
                    if execution.logs.stderr:
                        logs.extend(execution.logs.stderr)
                    
                    if logs:
                        logs_text = "\n".join(logs)
                        output = f"{output}\n{logs_text}" if output else logs_text
                
                # Extract plots from execution results
                # Matplotlib plots are returned in execution.results with 'png' attribute (base64)
                if execution.results:
                    for result_item in execution.results:
                        if hasattr(result_item, 'png') and result_item.png:
                            plots.append(result_item.png)
            
            # Limit output length
            if len(output) > self.max_output_length:
                output = output[:self.max_output_length] + "\n... (output truncated)"
            
            return {
                'output': output or 'Code executed successfully (no output)',
                'error': error,
                'plots': plots if plots else None
            }
            
        except Exception as e:
            error_msg = str(e)
            if len(error_msg) > self.max_output_length:
                error_msg = error_msg[:self.max_output_length] + "... (error truncated)"
            
            return {
                'output': output or '',
                'error': error_msg,
                'plots': plots if plots else None
            }
    
    def get_description(self) -> str:
        """
        Get the description for the code interpreter tool.
        
        Returns:
            Tool description string
        """
        return (
            "Execute Python code for mathematical computations, symbolic math, "
            "and plotting. Use this when you need to:\n"
            "- Calculate numerical results\n"
            "- Solve equations symbolically (use sympy)\n"
            "- Create plots and visualizations (use matplotlib)\n"
            "- Perform complex mathematical operations\n\n"
            "Available libraries: numpy (as np), sympy (as sp), matplotlib (as plt), "
            "math, cmath, fractions, decimal, statistics.\n\n"
            "Input should be valid Python code. The code will be executed and "
            "results (including plots) will be returned."
        )
