"""LaTeX rendering utilities for Streamlit."""

import re
import streamlit as st
from typing import List, Tuple


def render_markdown_with_latex(text: str):
    """
    Render markdown text with LaTeX support.
    
    Detects and renders LaTeX formulas in various formats:
    - Square brackets: [ \formula ]
    - Dollar signs: $$ formula $$ or $ formula $
    - LaTeX format: \[ formula \] or \( formula \)
    - Standalone formulas containing LaTeX commands
    
    Args:
        text: Text to render with LaTeX formulas
    """
    # List of patterns to match LaTeX formulas in different formats
    patterns = [
        # 1. Formulas in square brackets: [ \formula ] or [ formula ]
        (r'\[\s*([^\]]+)\s*\]', 1),
        # 2. Already formatted: $$ formula $$ or $ formula $
        (r'\$\$([^$]+)\$\$', 1),
        (r'\$([^$]+)\$', 1),
        # 3. LaTeX format: \[ formula \] or \( formula \)
        (r'\\\[([^\]]+)\\\]', 1),
        (r'\\\(([^\)]+)\\\)', 1),
    ]
    
    # Collect all matches with their positions
    matches = []
    
    # Find matches from all patterns
    for pattern, group_num in patterns:
        for match in re.finditer(pattern, text):
            formula = match.group(group_num).strip()
            if formula:
                matches.append((match.start(), match.end(), formula))
    
    # Also detect standalone LaTeX formulas (lines or segments containing LaTeX commands)
    # Look for text segments that contain LaTeX commands like \sin, \cos, \frac, \left, \right
    # and are not already matched
    lines = text.split('\n')
    line_start = 0
    for line in lines:
        line_end = line_start + len(line)
        # Check if line contains LaTeX commands and is not already matched
        if '\\' in line and any(cmd in line for cmd in ['\\sin', '\\cos', '\\tan', '\\frac', '\\left', '\\right', '\\sqrt']):
            # Check if this line is not already covered by a match
            is_covered = any(line_start >= m_start and line_end <= m_end for m_start, m_end, _ in matches)
            if not is_covered:
                # Check if line looks like a formula (contains math operators and LaTeX)
                if re.search(r'[=+\-*/()]', line) or '\\' in line:
                    formula = line.strip()
                    if formula and not formula.startswith('#') and not formula.startswith('```'):
                        matches.append((line_start, line_end, formula))
        line_start = line_end + 1  # +1 for the newline
    
    # Sort matches by position
    matches.sort(key=lambda x: x[0])
    
    # Remove overlapping matches (keep the first/longest one)
    filtered_matches = []
    for start, end, formula in matches:
        overlaps = False
        for f_start, f_end, _ in filtered_matches:
            if not (end <= f_start or start >= f_end):
                overlaps = True
                break
        if not overlaps:
            filtered_matches.append((start, end, formula))
    
    # Split text by formulas
    parts = []
    last_end = 0
    
    for start, end, formula in filtered_matches:
        # Add text before formula
        if start > last_end:
            text_part = text[last_end:start]
            if text_part.strip():
                parts.append(('text', text_part))
        
        # Add formula
        parts.append(('latex', formula))
        last_end = end
    
    # Add remaining text
    if last_end < len(text):
        remaining = text[last_end:]
        if remaining.strip():
            parts.append(('text', remaining))
    
    # If no formulas found, render as regular markdown
    if not parts:
        st.markdown(text)
        return
    
    # Render parts: text as markdown, formulas as LaTeX
    for part_type, content in parts:
        if part_type == 'text':
            st.markdown(content)
        elif part_type == 'latex':
            st.latex(content)

