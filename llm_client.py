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


def get_groq_client():
    """
    Initialize Groq client using Streamlit secrets (primary)
    or environment variables (fallback).
    """

    # --- Try Streamlit secrets ---
    st = None
    try:
        import streamlit as st
        api_key = st.secrets.get("GROQ_API_KEY", None)
    except Exception:
        api_key = None

    # --- Fallback to environment variable ---
    if not api_key:
        api_key = os.getenv("GROQ_API_KEY")

    # --- No client or no key ---
    if Groq is None or not api_key:
        return None

    # --- Create Groq client ---
    try:
        return Groq(api_key=api_key)
    except Exception:
        return None
