"""
Canonical model-name handling.

A model is written as "provider/model", e.g. "gemini/gemini-2.5-flash". The
prefix decides which provider handles the call, so a model that does not exist
yet still routes correctly without any change to assets/model_pricing.csv.

This lives apart from router.py because cost_tracker.py needs the same parsing
and is imported by the router.
"""

from typing import Optional, Tuple

# Accepted prefixes mapped onto the internal provider key.
PROVIDER_ALIASES = {
    "gemini": "google",
    "google": "google",
    "openai": "openai",
    "anthropic": "anthropic",
    "claude": "anthropic",
    "perplexity": "perplexity",
    "sonar": "perplexity",
    "ollama": "ollama",
    "vllm": "vllm",
}


def split_model_name(model_name: str) -> Tuple[Optional[str], str]:
    """
    Splits "provider/model" into (canonical_provider, model).

    Only a known provider alias counts as a prefix, so a bare repository-style
    name such as "meta-llama/Llama-3.1-8B" is left intact. Splitting on the
    first slash keeps the rest of the path, so "vllm/meta-llama/Llama-3.1-8B"
    yields ("vllm", "meta-llama/Llama-3.1-8B").
    """
    prefix, separator, remainder = model_name.partition("/")
    if separator and remainder:
        provider = PROVIDER_ALIASES.get(prefix.strip().lower())
        if provider:
            return provider, remainder
    return None, model_name


def strip_provider_prefix(model_name: str) -> str:
    """Returns the bare model id the SDK and the pricing table expect."""
    return split_model_name(model_name)[1]
