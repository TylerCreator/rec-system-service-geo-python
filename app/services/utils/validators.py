"""
Shared validation utilities
"""
from typing import Any, List, Dict


def is_hashable(value: Any) -> bool:
    """
    Check if value is hashable (can be used as dict key)
    
    Args:
        value: Value to check
    
    Returns:
        True if hashable, False otherwise
    """
    return isinstance(value, (str, int, float, bool, type(None)))


def categorize_params(params: List[Dict], input_widget_types: List[str] = None, 
                      output_widget_types: List[str] = None) -> Dict[str, Dict]:
    """
    Categorize parameters into internal (file/dataset) and external
    
    Args:
        params: List of parameter dictionaries
        input_widget_types: Widget types to consider as internal inputs
        output_widget_types: Widget types to consider as internal outputs
    
    Returns:
        Dictionary with 'internal' and 'external' categorized parameters
    """
    from .constants import WIDGET_FILE, WIDGET_FILE_SAVE
    
    if input_widget_types is None:
        input_widget_types = [WIDGET_FILE]
    if output_widget_types is None:
        output_widget_types = [WIDGET_FILE_SAVE]
    
    external = {}
    internal = {}
    
    for param in params:
        widget_name = param.get('widget', {}).get('name') if isinstance(param.get('widget'), dict) else None
        fieldname = param.get('fieldname')
        
        if widget_name in input_widget_types or widget_name in output_widget_types:
            internal[fieldname] = widget_name
        else:
            external[fieldname] = param.get('type')
    
    return {"internal": internal, "external": external}

