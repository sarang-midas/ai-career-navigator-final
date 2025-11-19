"""
Centralized Groq LLM client helper.

Provides get_groq_client() that safely initializes the Groq client
ONLY using the GROQ API key. OpenAI support removed completely.
"""

from typing import Optional
import os

try:
    from groq import Groq  # type: ignore
except Exception:
    Groq = None


def get_groq_client() -> Optional[object]:
    """
    Return an initialized Groq client or None if not available.
    Reads API key from environment variable or Streamlit secrets.
    """

    # First try environment variable
    api_key = st.secrets("GROQ_API_KEY")

    # If running inside Streamlit, fallback to secrets
    if not api_key:
        try:
            import streamlit as st  # type: ignore
            api_key = st.secrets.get("GROQ_API_KEY") if hasattr(st, "secrets") else None
        except Exception:
            pass

    # If Groq SDK not installed OR no key available → return None
    if Groq is None or not api_key:
        return None

    # Initialize Groq client
    try:
        return Groq(api_key=api_key)
    except Exception:
        return None
