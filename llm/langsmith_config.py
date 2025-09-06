"""
LangSmith configuration utilities for the travel assistant.
"""
import os
from typing import Optional
from langsmith import Client
from langchain_core.tracers import LangChainTracer


def get_langsmith_client() -> Optional[Client]:
    """Initialize LangSmith client if API key is available."""
    api_key = os.getenv("LANGCHAIN_API_KEY")
    if api_key:
        return Client(api_key=api_key)
    return None


def get_langsmith_tracer() -> Optional[LangChainTracer]:
    """Get LangSmith tracer for tracing."""
    client = get_langsmith_client()
    if client:
        return LangChainTracer(
            client=client,
            project_name=os.getenv("LANGCHAIN_PROJECT", "travel-assistant")
        )
    return None


def is_langsmith_enabled() -> bool:
    """Check if LangSmith tracing is enabled."""
    return (
        os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true" and
        os.getenv("LANGCHAIN_API_KEY") is not None
    )


def setup_langsmith_environment():
    """Set up LangSmith environment variables with defaults."""
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", "travel-assistant")
    os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
