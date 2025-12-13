"""Tool for saving student preferences and learning profile to memory."""

from typing import Dict, Any, List, Optional
from .base_tool import BaseTool


class SaveInMemory(BaseTool):
    """Tool for saving student learning preferences and profile."""
    
    def __init__(self, agent_preferences: Optional[List[str]] = None):
        """
        Initialize the save in memory tool.
        
        Args:
            agent_preferences: Reference to agent's preferences list.
                              If None, creates a new empty list (will be set by agent).
        """
        super().__init__()
        
        # Reference to agent's preferences list
        # This allows the tool to directly modify the agent's memory
        if agent_preferences is None:
            self.preferences: List[str] = []
        else:
            self.preferences = agent_preferences
    
    @property
    def name(self) -> str:
        """Get the tool name."""
        return "save_in_memory"
    
    def execute(self, input: str) -> Dict[str, Any]:
        """
        Save student preferences as facts from input string.
        
        Args:
            input: String with a single fact to save (e.g., "Student is in 7th grade")
            
        Returns:
            Dictionary with 'output' and 'error' keys
        """
        try:
            # Treat input as a single fact string
            fact_str = input.strip()
            
            if not fact_str:
                return {
                    'output': '',
                    'error': 'Empty input. Please provide a fact to save.'
                }
            
            # Add fact if it's not already in preferences
            if fact_str not in self.preferences:
                self.preferences.append(fact_str)
                output = f"✅ Сохранено в память: {fact_str}"
            else:
                output = f"ℹ️ Факт уже был сохранен ранее: {fact_str}"
            
            return {
                'output': output,
                'error': None
            }
                
        except Exception as e:
            return {
                'output': '',
                'error': f"Error saving preferences: {str(e)}"
            }
    
    def get_preferences(self) -> List[str]:
        """
        Get current student preferences.
        
        Returns:
            List of facts (reference, not copy)
        """
        return self.preferences
    
    def get_description(self) -> str:
        """
        Get the description for the save in memory tool.
        
        Returns:
            Tool description string
        """
        return (
            "Save a fact about student preferences or profile to memory. "
            "Use this tool to save information about the student that you learned from their responses and behavior.\n\n"
            "IMPORTANT - Determining learning style:\n"
            "- DO NOT ask directly 'visual, analytical, or practical'\n"
            "- Analyze the student's responses and make inferences about their preferences\n"
            "- If the student asks for graphs, diagrams, visualizations - this is visual style\n"
            "- If the student asks for logical explanations, proofs, step-by-step analysis - this is analytical style\n"
            "- If the student asks for examples, problems, practical applications - this is practical style\n"
            "- If the student asks for detailed explanations with all nuances - this is 'thorough_understanding'\n"
            "- If the student asks to remember theorems in rhymes, mnemonics - this is 'theorems_in_verse'\n\n"
            "Input:\n"
            "- Always a single fact as a string (e.g., 'Student is in 7th grade')\n"
            "- To save multiple facts, call the tool multiple times with separate facts\n\n"
            "Example facts (determine based on student's responses):\n"
            "- 'Student is in 7th grade' (if student mentioned grade)\n"
            "- 'Prefers visual approach' (if student asks for graphs, diagrams)\n"
            "- 'Prefers analytical approach' (if student asks for proofs, logic)\n"
            "- 'Prefers practical approach' (if student asks for examples, problems)\n"
            "- 'Prefers thorough_understanding' (if student asks for detailed explanations)\n"
            "- 'Prefers theorems_in_verse' (if student asks to remember in rhymes)\n"
            "- 'Easier to remember theorems in verse form' (if student said so)\n\n"
            "The tool automatically saves facts and uses them to adapt explanations to the student."
        )

