"""
Shared parsing utilities
"""
import json
from datetime import datetime
from typing import Any, Dict, List, Optional


def safe_json_parse(json_string: Any, default_value: Any = None) -> Any:
    """
    Safely parse JSON string with error handling
    
    Args:
        json_string: String to parse or already parsed object
        default_value: Value to return on error (defaults to {})
    
    Returns:
        Parsed JSON or default value
    """
    if default_value is None:
        default_value = {}
    
    try:
        return json.loads(json_string) if isinstance(json_string, str) else json_string
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        return default_value


def parse_datetime(date_string: Any) -> Optional[datetime]:
    """
    Parse datetime string to datetime object (timezone-naive for PostgreSQL)
    
    Args:
        date_string: String or datetime object to parse
    
    Returns:
        Parsed datetime or None if parsing fails
    """
    if not date_string:
        return None
    
    if isinstance(date_string, datetime):
        # Remove timezone info if present (PostgreSQL expects naive datetime)
        return date_string.replace(tzinfo=None) if date_string.tzinfo else date_string
    
    try:
        # Try parsing ISO format: '2025-10-10 12:52:38'
        return datetime.strptime(date_string, '%Y-%m-%d %H:%M:%S')
    except (ValueError, TypeError):
        try:
            # Try parsing with microseconds
            return datetime.strptime(date_string, '%Y-%m-%d %H:%M:%S.%f')
        except (ValueError, TypeError):
            try:
                # Try ISO format with T and timezone
                dt = datetime.fromisoformat(date_string.replace('Z', '+00:00'))
                # Remove timezone info for PostgreSQL
                return dt.replace(tzinfo=None)
            except (ValueError, TypeError, AttributeError):
                print(f"Warning: Could not parse datetime: {date_string}")
                return None


def parse_service_params(params_string: Any) -> List[Dict]:
    """
    Parse service parameters from JSON string
    
    Args:
        params_string: JSON string or list of parameters
    
    Returns:
        List of parameter dictionaries
    """
    try:
        return json.loads(params_string) if isinstance(params_string, str) else params_string or []
    except:
        return []


def to_string(value: Any) -> Optional[str]:
    """
    Convert value to string for VARCHAR fields
    
    Args:
        value: Value to convert
    
    Returns:
        String representation or None
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value
    return str(value)

