import re
import json
from utils.logger import get_logger

logger = get_logger("Conversions")

def extract_json_from_markdown(llm_string: str):
    """
    Regex-based utility to extract and parse JSON blocks from LLM strings.
    """
    logger.info("Attempting to extract JSON from LLM markdown string.")
    cleaned = re.sub(
        r"```(?:json)?\s*|\s*```", "", llm_string, flags=re.IGNORECASE
    ).strip()

    if not cleaned:
        logger.error("Cleaning failed: No JSON content found after removing markdown blocks.")
        raise ValueError("No JSON content found after cleaning the markdown.")

    try:
        data = json.loads(cleaned)
        logger.info("Successfully extracted and parsed JSON data.")
        return data
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing failed: {e}")
        raise ValueError(f"Failed to parse JSON: {e}")
