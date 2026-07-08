"""Tests for the supervisor workflow (StateGraph with revision loop)."""
from __future__ import annotations

import sys
sys.path.insert(0, 'src')

import json
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_tools():
    """Patch all four agent tools so nodes can be tested without LM Studio."""
    with patch("medqa_multi_agents._get_tools") as mock_get_tools:
        mock_get_tools.return_value = (
            _MockTool("hypertension treatment protocol"),  # retrieve_documents
            _MockTool("ACE inhibitors relax blood vessels..."),  # retrieve_context
            _MockTool("ACE inhibitors are first-line..."),  # answer_question
            _MockTool(json.dumps({"verdict": "correct", "reasoning": "Accurate answer."})),  # evaluate
        )
        yield mock_get_tools


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MockTool:
    """Stand-in for a langchain @tool that returns a fixed string."""

    def __init__(self, response: str):
        self._response = response

    def invoke(self, input_: dict) -> str:  # noqa: D401
        return self._response


class _MockModel:
    """Stand-in that never calls out to LM Studio."""

    def __init__(self):
        pass


# ---------------------------------------------------------------------------
# State schema tests  (no LM Studio required)
# ---------------------------------------------------------------------------


def test_state_has_required_fields():
    """State TypedDict exposes all required workflow fields."""
    from medqa_multi_agents import State

    state = State(
        question="What causes hypertension?",
        rewritten_query="",
        context="",
        draft_answer="",
        evaluation_result="",
        revision_count=0,
        evaluation_reasoning="",
        final_answer="",
    )
    assert state["question"] == "What causes hypertension?"
    assert state["revision_count"] == 0


def test_max_revision_loops_default():
    """MAX_REVISION_LOOPS is 2 unless overridden by env var."""
    import medqa_multi_agents as pkg

    assert pkg.MAX_REVISION_LOOPS == 2


def test_route_finalize_when_correct():
    """Conditional edge routes to 'finalize' when verdict is 'correct'."""
    from medqa_multi_agents import _route_after_evaluate

    state = {
        "question": "What is ACE inhibitor mechanism?",
        "rewritten_query": "ACE inhibitor mechanism",
        "context": "ACE inhibitors block angiotensin-converting enzyme...",
        "draft_answer": "They block ACE.",
        "evaluation_result": json.dumps({"verdict": "correct", "reasoning": "OK"}),
        "revision_count": 0,
        "evaluation_reasoning": "OK",
        "final_answer": "",
    }
    assert _route_after_evaluate(state) == "finalize"


def test_route_answer_when_incorrect():
    """Conditional edge routes back to 'answer' when verdict is 'incorrect'."""
    from medqa_multi_agents import _route_after_evaluate

    state = {
        "question": "What is ACE inhibitor mechanism?",
        "rewritten_query": "ACE inhibitor mechanism",
        "context": "ACE inhibitors block angiotensin-converting enzyme...",
        "draft_answer": "They block ACE.",
        "evaluation_result": json.dumps({"verdict": "incorrect", "reasoning": "Wrong."}),
        "revision_count": 0,
        "evaluation_reasoning": "Wrong.",
        "final_answer": "",
    }
    assert _route_after_evaluate(state) == "answer"


def test_route_answer_when_incomplete():
    """Conditional edge routes back to 'answer' when verdict is 'incomplete'."""
    from medqa_multi_agents import _route_after_evaluate

    state = {
        "question": "What is ACE inhibitor mechanism?",
        "rewritten_query": "ACE inhibitor mechanism",
        "context": "ACE inhibitors block angiotensin-converting enzyme...",
        "draft_answer": "They block ACE.",
        "evaluation_result": json.dumps({"verdict": "incomplete", "reasoning": "Missing detail."}),
        "revision_count": 1,
        "evaluation_reasoning": "Missing detail.",
        "final_answer": "",
    }
    assert _route_after_evaluate(state) == "answer"


def test_route_finalize_at_max_revision_loops():
    """Conditional edge goes to 'finalize' when revision_count >= MAX_REVISION_LOOPS."""
    from medqa_multi_agents import MAX_REVISION_LOOPS, _route_after_evaluate

    state = {
        "question": "What is ACE inhibitor mechanism?",
        "rewritten_query": "ACE inhibitor mechanism",
        "context": "ACE inhibitors block angiotensin-converting enzyme...",
        "draft_answer": "They block ACE.",
        "evaluation_result": json.dumps({"verdict": "incorrect", "reasoning": "Wrong."}),
        "revision_count": MAX_REVISION_LOOPS,
        "evaluation_reasoning": "Wrong.",
        "final_answer": "",
    }
    assert _route_after_evaluate(state) == "finalize"


def test_route_finalize_correct_at_max_revision_loops():
    """Even 'correct' at max revision loops goes to finalize."""
    from medqa_multi_agents import MAX_REVISION_LOOPS, _route_after_evaluate

    state = {
        "question": "What is ACE inhibitor mechanism?",
        "rewritten_query": "ACE inhibitor mechanism",
        "context": "ACE inhibitors block angiotensin-converting enzyme...",
        "draft_answer": "ACE inhibitors block the angiotensin-converting enzyme.",
        "evaluation_result": json.dumps({"verdict": "correct", "reasoning": "Good."}),
        "revision_count": MAX_REVISION_LOOPS,
        "evaluation_reasoning": "Good.",
        "final_answer": "",
    }
    assert _route_after_evaluate(state) == "finalize"


# ---------------------------------------------------------------------------
# Node function tests  (no LM Studio required)
# ---------------------------------------------------------------------------


def test_node_rewrite_retrieve(mock_tools):
    """_node_rewrite_retrieve returns rewritten_query and context."""
    from medqa_multi_agents import _node_rewrite_retrieve

    state = {"question": "How do ACE inhibitors work?"}
    result = _node_rewrite_retrieve(state)

    assert "rewritten_query" in result
    assert "context" in result
    assert isinstance(result["rewritten_query"], str)
    assert isinstance(result["context"], str)


def test_node_answer(mock_tools):
    """_node_answer returns draft_answer."""
    from medqa_multi_agents import _node_answer

    state = {
        "question": "How do ACE inhibitors work?",
        "context": "ACE inhibitors block the angiotensin-converting enzyme...",
    }
    result = _node_answer(state)

    assert "draft_answer" in result
    assert isinstance(result["draft_answer"], str)


def test_node_evaluate(mock_tools):
    """_node_evaluate returns evaluation_result and evaluation_reasoning."""
    from medqa_multi_agents import _node_evaluate

    state = {
        "question": "How do ACE inhibitors work?",
        "context": "ACE inhibitors block the angiotensin-converting enzyme...",
        "draft_answer": "ACE inhibitors block the angiotensin-converting enzyme.",
    }
    result = _node_evaluate(state)

    assert "evaluation_result" in result
    assert "evaluation_reasoning" in result
    assert isinstance(result["evaluation_result"], str)
    assert isinstance(result["evaluation_reasoning"], str)
    # Verify it is valid JSON
    parsed = json.loads(result["evaluation_result"])
    assert "verdict" in parsed
    assert "reasoning" in parsed


def test_node_finalize():
    """_node_finalize copies draft_answer to final_answer."""
    from medqa_multi_agents import _node_finalize

    state = {
        "question": "How do ACE inhibitors work?",
        "rewritten_query": "ACE inhibitor mechanism",
        "context": "...",
        "draft_answer": "ACE inhibitors block ACE.",
        "evaluation_result": '{"verdict": "correct", "reasoning": "OK"}',
        "revision_count": 0,
        "evaluation_reasoning": "OK",
        "final_answer": "",
    }
    result = _node_finalize(state)

    assert result["final_answer"] == "ACE inhibitors block ACE."


# ---------------------------------------------------------------------------
# Workflow graph structure tests  (no LM Studio required)
# ---------------------------------------------------------------------------


def test_workflow_compiles_without_error():
    """The compiled graph is a StateGraph and has the expected nodes."""
    from medqa_multi_agents import workflow

    # Should be a compiled StateGraph
    assert hasattr(workflow, "nodes")
    # Nodes: rewrite_retrieve, answer, evaluate, finalize
    node_names = set(workflow.nodes.keys())
    assert "rewrite_retrieve" in node_names
    assert "answer" in node_names
    assert "evaluate" in node_names
    assert "finalize" in node_names


def test_workflow_invoke_end_to_end_with_mock_tools(mock_tools):
    """Full workflow.invoke returns final_answer when evaluation passes."""
    from medqa_multi_agents import invoke

    result = invoke("What is the mechanism of action of ACE inhibitors?")
    assert isinstance(result, str)
    assert len(result) > 0


def test_workflow_invoke_with_revision_loop(mock_tools):
    """Full workflow falls through to finalize after MAX_REVISION_LOOPS exhausted."""
    # Override eval tool to always return 'incorrect' so we test the loop boundary
    with patch("medqa_multi_agents._get_tools") as mock_get_tools:
        mock_get_tools.return_value = (
            _MockTool("hypertension"),  # retrieve_documents
            _MockTool("context..."),  # retrieve_context
            _MockTool("draft answer"),  # answer_question
            _MockTool(json.dumps({"verdict": "incorrect", "reasoning": "Bad."})),  # evaluate — always wrong
        )
        from medqa_multi_agents import invoke

        result = invoke("What is the mechanism of action of ACE inhibitors?")
        # Should still return a result (finalize was eventually reached)
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Public API tests
# ---------------------------------------------------------------------------


def test_invoke_exported():
    """'invoke' is in __all__."""
    import medqa_multi_agents as pkg

    assert "invoke" in pkg.__all__


def test_workflow_exported():
    """'workflow' is in __all__."""
    import medqa_multi_agents as pkg

    assert "workflow" in pkg.__all__
