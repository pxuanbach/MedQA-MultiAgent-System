"""Tests for the rewriter agent.

Requires LM Studio running locally with a model loaded.
Run with:  uv run pytest tests/test_rewriter.py -v
Or individually:  uv run pytest tests/test_rewriter.py::test_rewriter_direct -v
"""
import sys
sys.path.insert(0, 'src')

import pytest
from langchain_openai import ChatOpenAI

from medqa_multi_agents.agents.rewriter import (
    RewrittenQuery,
    _load_rewriter_prompt,
    create_rewriter_agent,
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
        max_tokens=256,
    )
    # Smoke-check the connection
    try:
        model.invoke("hello")
    except Exception as e:
        pytest.skip(f"LM Studio not available: {e}")
    return model


@pytest.fixture
def rewriter_tool(lmstudio_model):
    return create_rewriter_agent(model=lmstudio_model)


# ---------------------------------------------------------------------------
# Schema / unit tests  (no LM Studio required)
# ---------------------------------------------------------------------------

class TestRewriterSchema:
    def test_rewritten_query_has_query_field(self):
        rq = RewrittenQuery(query="test query")
        assert rq.query == "test query"

    def test_rewritten_query_serialises_to_dict(self):
        rq = RewrittenQuery(query="mechanism of aspirin")
        d = rq.model_dump()
        assert d == {"query": "mechanism of aspirin"}

    def test_load_rewriter_prompt_returns_non_empty_string(self):
        prompt = _load_rewriter_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "Rewriter Agent" in prompt


class TestRewriterToolSignature:
    def test_tool_name_is_retrieve_documents(self, lmstudio_model):
        tool = create_rewriter_agent(model=lmstudio_model)
        assert tool.name == "retrieve_documents"

    def test_tool_description_mentions_rewrite(self, lmstudio_model):
        tool = create_rewriter_agent(model=lmstudio_model)
        assert "rewrite" in tool.description.lower()


# ---------------------------------------------------------------------------
# Integration tests  (require LM Studio)
# ---------------------------------------------------------------------------

class TestRewriterAgentDirect:
    """Test model.with_structured_output directly (no create_agent wrapper)."""

    def test_with_structured_output_returns_rewritten_query(self, lmstudio_model):
        question = "What is the mechanism of action of warfarin?"
        system_prompt = _load_rewriter_prompt()
        response = lmstudio_model.with_structured_output(RewrittenQuery).invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ])
        assert isinstance(response, RewrittenQuery)
        assert isinstance(response.query, str)
        assert len(response.query) > 5

    @pytest.mark.parametrize("question", [
        "A 65-year-old man with hypertension and diabetes presents with crushing chest pain. "
        "ECG shows ST-elevation in leads II, III, and aVF. "
        "What is the most appropriate initial management?",
        "A patient with severe rheumatoid arthritis is started on methotrexate. "
        "Which of the following is the most appropriate monitoring test to detect "
        "the most serious potential adverse effect of this medication?",
        "Describe the pathophysiology of diabetic ketoacidosis.",
    ])
    def test_rewritten_queries_are_nontrivial(self, lmstudio_model, question):
        system_prompt = _load_rewriter_prompt()
        response = lmstudio_model.with_structured_output(RewrittenQuery).invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ])
        # Query should be shorter than the original question but not empty
        assert len(response.query) > 5
        assert len(response.query) <= len(question)


class TestRewriterAgentTool:
    """Test the @tool retrieve_documents end-to-end."""

    def test_retrieve_documents_returns_string(self, rewriter_tool):
        result = rewriter_tool.invoke({
            "question": "What is the mechanism of action of aspirin?"
        })
        assert isinstance(result, str)
        assert len(result) > 5

    def test_retrieve_documents_pharmacology(self, rewriter_tool):
        result = rewriter_tool.invoke({
            "question": (
                "A patient with severe rheumatoid arthritis is started on methotrexate. "
                "Which of the following is the most appropriate monitoring test to detect "
                "the most serious potential adverse effect of this medication?"
            )
        })
        assert isinstance(result, str)
        # Should mention something related to methotrexate monitoring
        assert len(result) > 10
