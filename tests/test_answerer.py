"""Tests for the answerer agent.

Requires LM Studio running locally with a model loaded.
Run with:  uv run pytest tests/test_answerer.py -v
Or individually:  uv run pytest tests/test_answerer.py::test_answer_question_direct -v
"""
import sys
sys.path.insert(0, 'src')

import pytest

from medqa_multi_agents.agents.answerer import (
    Answer,
    _load_answerer_prompt,
    create_answerer_agent,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def answerer_tool(lmstudio_model):
    return create_answerer_agent(model=lmstudio_model)


# ---------------------------------------------------------------------------
# Schema / unit tests  (no LM Studio required)
# ---------------------------------------------------------------------------

class TestAnswererSchema:
    def test_answer_has_answer_field(self):
        a = Answer(answer="Aspirin inhibits platelet aggregation.")
        assert a.answer == "Aspirin inhibits platelet aggregation."

    def test_answer_serialises_to_dict(self):
        a = Answer(answer="Warfarin is a vitamin K antagonist.")
        d = a.model_dump()
        assert d == {"answer": "Warfarin is a vitamin K antagonist."}

    def test_load_answerer_prompt_returns_non_empty_string(self):
        prompt = _load_answerer_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "Answerer Agent" in prompt


class TestAnswererToolSignature:
    def test_tool_name_is_answer_question(self, lmstudio_model):
        tool = create_answerer_agent(model=lmstudio_model)
        assert tool.name == "answer_question"

    def test_tool_description_mentions_answer(self, lmstudio_model):
        tool = create_answerer_agent(model=lmstudio_model)
        assert "answer" in tool.description.lower()


# ---------------------------------------------------------------------------
# Integration tests  (require LM Studio + Chroma)
# ---------------------------------------------------------------------------

class TestAnswererIntegration:
    def test_answer_question_direct(self, lmstudio_model):
        """End-to-end: answerer synthesises answer from context and question."""
        tool = create_answerer_agent(model=lmstudio_model)
        context = (
            "Aspirin irreversibly inhibits cyclooxygenase (COX-1 and COX-2), "
            "which prevents the conversion of arachidonic acid to prostaglandins "
            "and thromboxane A2. This reduces platelet aggregation and inflammation."
        )
        question = "What is the mechanism of action of aspirin?"
        result = tool.invoke({"context": context, "question": question})
        assert isinstance(result, str)
        assert len(result) > 0

    def test_answer_question_invocation_returns_string(self, answerer_tool):
        """Tool invoke with context+question returns a string answer."""
        context = (
            "Metformin is a biguanide that decreases hepatic glucose production "
            "and increases insulin sensitivity in peripheral tissues."
        )
        question = "What are the side effects of metformin?"
        result = answerer_tool.invoke({"context": context, "question": question})
        assert isinstance(result, str)
        assert len(result) > 5
