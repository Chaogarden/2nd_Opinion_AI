# core/json_llm_utils.py
# ==============================
# JSON extraction utilities for LLM outputs
# Robustly handles ```json fences and extracts balanced JSON objects
# ==============================

from typing import Optional


def extract_json_block(text: str) -> Optional[str]:
    """
    Return the first balanced {...} block in `text`.
    Handles optional ```json ... ``` wrappers and ignores stray braces.
    
    Args:
        text: Raw LLM output that may contain JSON wrapped in markdown fences.
    
    Returns:
        The extracted JSON string, or None if no valid block found.
    """
    if not text:
        return None

    # Strip code fences if present
    t = text.strip()
    if t.startswith("```"):
        # e.g. ```json\n{...}\n```
        parts = t.split("```")
        # Collect anything between the first and second fence
        if len(parts) >= 3:
            t = parts[1] if "{" in parts[1] else parts[2]
        # Also handle case where there's just opening fence
        elif len(parts) >= 2 and "{" in parts[1]:
            t = parts[1]

    # Remove language tag if present (e.g., "json\n{...")
    if t.startswith("json"):
        t = t[4:].lstrip()

    # Scan for balanced braces
    start = None
    depth = 0
    for i, ch in enumerate(t):
        if ch == "{":
            if start is None:
                start = i
                depth = 1
            else:
                depth += 1
        elif ch == "}":
            if start is not None:
                depth -= 1
                if depth == 0:
                    return t[start:i+1]
    return None


def extract_json_array(text: str) -> Optional[str]:
    """
    Return the first balanced [...] array in `text`.
    Similar to extract_json_block but for arrays.
    
    Args:
        text: Raw LLM output that may contain a JSON array.
    
    Returns:
        The extracted JSON array string, or None if no valid array found.
    """
    if not text:
        return None

    t = text.strip()
    if t.startswith("```"):
        parts = t.split("```")
        if len(parts) >= 3:
            t = parts[1] if "[" in parts[1] else parts[2]
        elif len(parts) >= 2 and "[" in parts[1]:
            t = parts[1]

    if t.startswith("json"):
        t = t[4:].lstrip()

    # Scan for balanced brackets
    start = None
    depth = 0
    for i, ch in enumerate(t):
        if ch == "[":
            if start is None:
                start = i
                depth = 1
            else:
                depth += 1
        elif ch == "]":
            if start is not None:
                depth -= 1
                if depth == 0:
                    return t[start:i+1]
    return None


def clean_json_string(text: str) -> str:
    """
    Clean common JSON issues in LLM outputs.
    
    - Removes trailing commas before } or ]
    - Handles control characters that break JSON parsing
    - Fixes common escape issues
    
    Args:
        text: Raw JSON string that may have minor issues.
    
    Returns:
        Cleaned JSON string.
    """
    import re
    
    # Remove trailing commas before closing braces/brackets
    text = re.sub(r',\s*}', '}', text)
    text = re.sub(r',\s*]', ']', text)
    
    # Escape control characters inside JSON strings
    # This handles newlines, tabs, etc. that LLMs sometimes put in string values
    result = []
    in_string = False
    escape_next = False
    
    for char in text:
        if escape_next:
            result.append(char)
            escape_next = False
            continue
        
        if char == '\\' and in_string:
            escape_next = True
            result.append(char)
            continue
            
        if char == '"':
            in_string = not in_string
            result.append(char)
            continue
        
        if in_string:
            # Check for control characters (0x00-0x1F)
            code = ord(char)
            if code < 32:
                if code == 9:  # Tab
                    result.append('\\t')
                elif code == 10:  # Newline (LF)
                    result.append('\\n')
                elif code == 13:  # Carriage return (CR)
                    result.append('\\r')
                else:
                    # Other control chars - use unicode escape
                    result.append(f'\\u{code:04x}')
            else:
                result.append(char)
        else:
            result.append(char)
    
    return ''.join(result)

