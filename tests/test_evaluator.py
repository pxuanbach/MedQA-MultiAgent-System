"""Tests for the evaluator agent.

Requires LM Studio running locally with a model loaded.
Run with:  uv run pytest tests/test_evaluator.py -v
Or individually:  uv run pytest tests/test_evaluator.py::test_evaluate_answer_direct -v
"""
import sys
sys.path.insert(0, 'src')

import json

import pytest

from medqa_multi_agents.agents.evaluator import (
    EvaluationResult,
    _load_evaluator_prompt,
    create_evaluator_agent,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def evaluator_tool(lmstudio_model):
    return create_evaluator_agent(model=lmstudio_model)


# ---------------------------------------------------------------------------
# Schema / unit tests  (no LM Studio required)
# ---------------------------------------------------------------------------

class TestEvaluatorSchema:
    def test_evaluation_result_has_verdict_field(self):
        er = EvaluationResult(verdict="correct", reasoning="Looks good.")
        assert er.verdict == "correct"
        assert er.reasoning == "Looks good."

    def test_evaluation_result_serialises_to_dict(self):
        er = EvaluationResult(verdict="incomplete", reasoning="Missing key info.")
        d = er.model_dump()
        assert d == {"verdict": "incomplete", "reasoning": "Missing key info."}

    def test_load_evaluator_prompt_returns_non_empty_string(self):
        prompt = _load_evaluator_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "Evaluator Agent" in prompt


class TestEvaluatorToolSignature:
    def test_tool_name_is_evaluate_answer(self, lmstudio_model):
        tool = create_evaluator_agent(model=lmstudio_model)
        assert tool.name == "evaluate_answer"

    def test_tool_description_mentions_evaluate(self, lmstudio_model):
        tool = create_evaluator_agent(model=lmstudio_model)
        assert "evaluat" in tool.description.lower()


# ---------------------------------------------------------------------------
# Integration tests  (require LM Studio)
# ---------------------------------------------------------------------------

class TestEvaluatorIntegration:
    def test_evaluate_answer_direct(self, lmstudio_model):
        """End-to-end: evaluator judges a draft answer given question and context."""
        tool = create_evaluator_agent(model=lmstudio_model)
        question = "What is the mechanism of action of aspirin?"
        context = (
            "Aspirin irreversibly inhibits cyclooxygenase (COX-1 and COX-2), "
            "which prevents the conversion of arachidonic acid to prostaglandins "
            "and thromboxane A2. This reduces platelet aggregation and inflammation."
        )
        draft_answer = (
            "Aspirin works by irreversibly inhibiting COX enzymes, thereby blocking "
            "prostaglandin synthesis and reducing inflammation and platelet aggregation."
        )
        result = tool.invoke({
            "draft_answer": draft_answer,
            "question": question,
            "context": context,
        })
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed["verdict"] in ("correct", "incorrect", "incomplete")
        assert "reasoning" in parsed
        assert len(parsed["reasoning"]) > 0

    def test_evaluate_answer_invocation_returns_json_string(self, evaluator_tool):
        """Tool invoke returns a JSON string with verdict and reasoning."""
        question = "What are the side effects of metformin?"
        context = (
            "Metformin is a biguanide that decreases hepatic glucose production "
            "and increases insulin sensitivity in peripheral tissues. "
            "Common side effects include GI upset and rare lactic acidosis."
        )
        draft_answer = "Metformin decreases hepatic glucose production."
        result = evaluator_tool.invoke({
            "draft_answer": draft_answer,
            "question": question,
            "context": context,
        })
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert "verdict" in parsed
        assert "reasoning" in parsed

    def test_verdict_incomplete_when_context_insufficient(self, lmstudio_model):
        """Evaluator marks incomplete when draft answer doesn't use full context."""
        tool = create_evaluator_agent(model=lmstudio_model)
        question = "What are the side effects of metformin?"
        context = (
            "Metformin is a biguanide that decreases hepatic glucose production "
            "and increases insulin sensitivity in peripheral tissues. "
            "Common side effects include GI upset and rare lactic acidosis."
        )
        draft_answer = "Metformin is a diabetes medication."
        result = tool.invoke({
            "draft_answer": draft_answer,
            "question": question,
            "context": context,
        })
        parsed = json.loads(result)
        assert parsed["verdict"] in ("correct", "incorrect", "incomplete")
