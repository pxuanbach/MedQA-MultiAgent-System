"""Shared pytest fixtures for the tests suite.

Requires LM Studio running locally with a model loaded.
"""
import pytest
from langchain_openai import ChatOpenAI


@pytest.fixture(scope="module")
def lmstudio_model():
    """Connect to LM Studio (qwen3-8b). Skip all tests in this module if unavailable."""
    model = ChatOpenAI(
        base_url="http://localhost:1234/v1",
        api_key="lm-studio",
        model="qwen/qwen3-8b",
        temperature=0,
        max_tokens=512,
    )
    # Smoke-check the connection
    try:
        model.invoke("hello")
    except Exception as e:
        pytest.skip(f"LM Studio not available: {e}")
    return model
