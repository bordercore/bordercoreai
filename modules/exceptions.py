"""
This module defines custom exception classes for JSON parsing failures and improper LLM response formats.
"""

class JsonParsingError(Exception):
    """Custom exception for JSON parsing errors."""


class LLMResponseError(Exception):
    """Custom exception for errors in the LLM response format."""
