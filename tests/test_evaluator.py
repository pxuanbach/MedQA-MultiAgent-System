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
    def test_evaluation_result_has_all_rubric_fields(self):
        er = EvaluationResult(
            correctness=3,
            completeness=2,
            evidence_alignment=2,
            distractor_elimination=1,
            verdict="correct",
            reasoning="Mostly correct but didn't rule out distractors well.",
            evaluator_confidence=0.75,
        )
        assert er.verdict == "correct"
        assert er.correctness == 3
        assert er.completeness == 2
        assert er.evidence_alignment == 2
        assert er.distractor_elimination == 1
        assert er.evaluator_confidence == 0.75

    def test_evaluation_result_serialises_to_dict(self):
        er = EvaluationResult(
            correctness=1,
            completeness=1,
            evidence_alignment=0,
            distractor_elimination=0,
            verdict="incorrect",
            reasoning="Medical errors and no vignette connection.",
            evaluator_confidence=0.9,
        )
        d = er.model_dump()
        assert d["verdict"] == "incorrect"
        assert d["correctness"] == 1
        assert d["evidence_alignment"] == 0
        assert d["evaluator_confidence"] == 0.9

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
    def test_evaluate_answer_returns_rubric_scores(self, lmstudio_model):
        """End-to-end: evaluator returns per-criterion rubric scores."""
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
        # Rubric fields
        assert "correctness" in parsed
        assert "completeness" in parsed
        assert "evidence_alignment" in parsed
        assert "distractor_elimination" in parsed
        assert "evaluator_confidence" in parsed
        # Each score should be 0-3
        for key in ("correctness", "completeness", "evidence_alignment", "distractor_elimination"):
            assert 0 <= parsed[key] <= 3, f"{key} out of range: {parsed[key]}"
        assert 0.0 <= parsed["evaluator_confidence"] <= 1.0
        assert parsed["verdict"] in ("correct", "incorrect", "incomplete")
        assert len(parsed["reasoning"]) > 0

    def test_evaluate_answer_with_minimal_input(self, evaluator_tool):
        """Evaluator judges based only on question, context, and draft answer."""
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
        parsed = json.loads(result)
        assert "correctness" in parsed
        assert "completeness" in parsed
        # completeness should be low since draft didn't mention side effects
        assert parsed["completeness"] <= 2

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
        # completeness should be low for such a minimal answer
        assert parsed["completeness"] <= 1
