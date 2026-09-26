"""
The one place that decides which LLM we talk to (same seam as class 3).

Gemini is reached through its OpenAI-compatible endpoint, so the `openai` SDK works
unchanged. Swapping provider = set LLM_BASE_URL / LLM_MODEL / the key. Nothing else moves.
"""

import os

from openai import OpenAI

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
# Pinned, and a "lite" model on purpose: the free tier's daily cap on bigger models is
# smaller than one evaluation run (see the class 3 README).
DEFAULT_MODEL = "gemini-3.5-flash-lite"
# Gemini spends part of this budget "thinking"; too small = silently truncated answers.
DEFAULT_MAX_TOKENS = 2000


def get_api_key() -> str | None:
    return os.getenv("GEMINI_API_KEY")


def get_model() -> str:
    return os.getenv("LLM_MODEL", DEFAULT_MODEL)


def get_max_tokens() -> int:
    return int(os.getenv("LLM_MAX_TOKENS", DEFAULT_MAX_TOKENS))


def is_configured() -> bool:
    return bool(get_api_key())


def build_client() -> OpenAI | None:
    """A ready client, or None when there is no key (the app still boots and says why)."""
    if not is_configured():
        return None
    return OpenAI(
        api_key=get_api_key(),
        base_url=os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL),
        max_retries=int(os.getenv("LLM_MAX_RETRIES", 5)),
    )


def describe() -> dict:
    """Config summary for /health and MLflow params. Never includes the key."""
    return {
        "base_url": os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL),
        "model": get_model(),
        "max_tokens": get_max_tokens(),
        "api_key_configured": is_configured(),
    }
