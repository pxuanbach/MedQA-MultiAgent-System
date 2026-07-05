"""Tests for the retriever agent.

Requires LM Studio running locally with a model loaded.
Run with:  uv run pytest tests/test_retriever.py -v
Or individually:  uv run pytest tests/test_retriever.py::test_retriever_direct -v
"""
import sys
sys.path.insert(0, 'src')

import pytest
from langchain_openai import ChatOpenAI

from medqa_multi_agents.agents.retriever import (
    RetrievedContext,
    _load_retriever_prompt,
    create_retriever_agent,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def lmstudio_model():
    """Connect to LM Studio (qwen3-8b). Skip all tests in this module if unavailable."""
    model = ChatOpenAI(
        base_url="http://localhost:1234/v1",
        api_key="lm-studio",
        model="qwen3-8b",
        temperature=0,
        max_tokens=512,
    )
    # Smoke-check the connection
    try:
        model.invoke("hello")
    except Exception as e:
        pytest.skip(f"LM Studio not available: {e}")
    return model


@pytest.fixture
def retriever_tool(lmstudio_model):
    return create_retriever_agent(model=lmstudio_model)


# ---------------------------------------------------------------------------
# Schema / unit tests  (no LM Studio required)
# ---------------------------------------------------------------------------

class TestRetrieverSchema:
    def test_retrieved_context_has_context_field(self):
        rc = RetrievedContext(context="test context")
        assert rc.context == "test context"

    def test_retrieved_context_serialises_to_dict(self):
        rc = RetrievedContext(context="mechanism of aspirin")
        d = rc.model_dump()
        assert d == {"context": "mechanism of aspirin"}

    def test_load_retriever_prompt_returns_non_empty_string(self):
        prompt = _load_retriever_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "Retriever Agent" in prompt


class TestRetrieverToolSignature:
    def test_tool_name_is_retrieve_context(self, lmstudio_model):
        tool = create_retriever_agent(model=lmstudio_model)
        assert tool.name == "retrieve_context"

    def test_tool_description_mentions_retrieve(self, lmstudio_model):
        tool = create_retriever_agent(model=lmstudio_model)
        assert "retrieve" in tool.description.lower()


# ---------------------------------------------------------------------------
# Integration tests  (require LM Studio + Chroma)
# ---------------------------------------------------------------------------

class TestRetrieverIntegration:
    def test_retriever_direct(self, lmstudio_model):
        """End-to-end: retriever formats retrieved Chroma context as JSON."""
        tool = create_retriever_agent(model=lmstudio_model, top_k=3)
        result = tool.invoke("What is the mechanism of action of aspirin?")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_retriever_invocation_returns_string(self, retriever_tool):
        """Tool invoke returns a string context (not a raw object)."""
        result = retriever_tool.invoke(
            "What are the side effects of metformin?"
        )
        assert isinstance(result, str)
        assert len(result) > 10
