class JsonParsingError(Exception):
    """Custom exception for JSON parsing errors."""
    pass


class LLMResponseError(Exception):
    """Custom exception for errors in the LLM response format."""
    pass
