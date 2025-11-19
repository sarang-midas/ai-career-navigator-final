"""Centralized LLM client helpers.

Provides get_groq_client() and get_openai_client() helpers that safely
initialize and return clients only when credentials and SDKs are available.
This prevents import-time exceptions when running in environments without
API keys (e.g., local dev or Streamlit Cloud preview without secrets).
"""
from typing import Optional
import os

try:
    from groq import Groq  # type: ignore
except Exception:
    Groq = None

try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None


def get_groq_client() -> Optional[object]:
    """Return an initialized Groq client or None if not available."""
    api_key = os.getenv("GROQ_API_KEY")
    if Groq is None or not api_key:
        return None
    try:
        return Groq(api_key=api_key)
    except Exception:
        return None


def get_openai_client() -> Optional[object]:
    """Return an initialized OpenAI client or None if not available.

    The OpenAI SDK will read OPENAI_API_KEY when creating the client.
    We prefer to pass the key explicitly when available to avoid surprises.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if OpenAI is None or not api_key:
        return None
    try:
        return OpenAI(api_key=api_key)
    except Exception:
        # Last resort: attempt default constructor (some env setups use other vars)
        try:
            return OpenAI()
        except Exception:
            return None
