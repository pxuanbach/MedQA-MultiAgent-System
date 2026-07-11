"""Tests for the supervisor workflow (StateGraph with revision loop).

Covers V3 graph structure, node functions, routing logic, and public API.
All tests run without LM Studio (tools are mocked).
"""
from __future__ import annotations

import sys
sys.path.insert(0, 'src')

import json
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MockTool:
    """Stand-in for a langchain @tool that returns a fixed string."""

    def __init__(self, response: str):
        self._response = response

    def invoke(self, input_) -> str:  # noqa: D401
        return self._response


# Default empty memory state for tests that don't need memory rules
_EMPTY_MEMORY = {
    "global_memory": [],
    "retrieval_memory": [],
    "reasoner_memory": [],
    "verifier_memory": [],
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_tools():
    """Patch _get_tools so nodes can be tested without LM Studio."""
    with patch("medqa_multi_agents._get_tools") as mock_get_tools:
        mock_get_tools.return_value = (
            _MockTool("hypertension treatment protocol"),           # retrieve_documents
            _MockTool("ACE inhibitors relax blood vessels..."),     # retrieve_context
            _MockTool("ACE inhibitors are first-line..."),          # answer_question
            _MockTool(json.dumps({"verdict": "correct", "reasoning": "Accurate answer."})),  # evaluate
        )
        yield mock_get_tools


# ---------------------------------------------------------------------------
# State schema tests  (no LM Studio required)
# ---------------------------------------------------------------------------


def test_state_has_required_fields():
    """State TypedDict exposes all required workflow fields including memory."""
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
        global_memory=[],
        retrieval_memory=[],
        reasoner_memory=[],
        verifier_memory=[],
    )
    assert state["question"] == "What causes hypertension?"
    assert state["revision_count"] == 0
    assert isinstance(state["global_memory"], list)
    assert isinstance(state["retrieval_memory"], list)
    assert isinstance(state["reasoner_memory"], list)
    assert isinstance(state["verifier_memory"], list)


def test_state_no_past_sessions_field():
    """State no longer has 'past_sessions' — long-term memory uses structured lists."""
    from medqa_multi_agents import State
    import typing
    hints = typing.get_type_hints(State)
    assert "past_sessions" not in hints, (
        "'past_sessions' must not be in State; use global_memory/retrieval_memory etc."
    )


def test_max_revision_loops_default():
    """MAX_REVISION_LOOPS is 2 unless overridden by env var."""
    import medqa_multi_agents as pkg
    assert pkg.MAX_REVISION_LOOPS == 2


# ---------------------------------------------------------------------------
# Routing tests  (no LM Studio required)
# ---------------------------------------------------------------------------


def _make_route_state(**overrides) -> dict:
    base = {
        "question": "What is ACE inhibitor mechanism?",
        "rewritten_query": "ACE inhibitor mechanism",
        "context": "ACE inhibitors block angiotensin-converting enzyme...",
        "draft_answer": "They block ACE.",
        "evaluation_result": json.dumps({"verdict": "correct", "reasoning": "OK"}),
        "revision_count": 0,
        "evaluation_reasoning": "OK",
        "final_answer": "",
        **_EMPTY_MEMORY,
    }
    base.update(overrides)
    return base


def test_route_finalize_when_correct():
    """Conditional edge routes to 'finalize' when verdict is 'correct'."""
    from medqa_multi_agents import _route_after_evaluate
    state = _make_route_state(evaluation_result=json.dumps({"verdict": "correct", "reasoning": "OK"}))
    assert _route_after_evaluate(state) == "finalize"


def test_route_answer_when_incorrect():
    """Conditional edge routes back to 'answer' when verdict is 'incorrect'."""
    from medqa_multi_agents import _route_after_evaluate
    state = _make_route_state(evaluation_result=json.dumps({"verdict": "incorrect", "reasoning": "Wrong."}))
    assert _route_after_evaluate(state) == "answer"


def test_route_answer_when_incomplete():
    """Conditional edge routes back to 'answer' when verdict is 'incomplete'."""
    from medqa_multi_agents import _route_after_evaluate
    state = _make_route_state(
        evaluation_result=json.dumps({"verdict": "incomplete", "reasoning": "Missing detail."}),
        revision_count=1,
    )
    assert _route_after_evaluate(state) == "answer"


def test_route_finalize_at_max_revision_loops():
    """Conditional edge goes to 'finalize' when revision_count >= MAX_REVISION_LOOPS."""
    from medqa_multi_agents import MAX_REVISION_LOOPS, _route_after_evaluate
    state = _make_route_state(
        evaluation_result=json.dumps({"verdict": "incorrect", "reasoning": "Wrong."}),
        revision_count=MAX_REVISION_LOOPS,
    )
    assert _route_after_evaluate(state) == "finalize"


def test_route_finalize_correct_at_max_revision_loops():
    """Even 'correct' at max revision loops goes to finalize."""
    from medqa_multi_agents import MAX_REVISION_LOOPS, _route_after_evaluate
    state = _make_route_state(
        revision_count=MAX_REVISION_LOOPS,
        draft_answer="ACE inhibitors block the angiotensin-converting enzyme.",
        evaluation_result=json.dumps({"verdict": "correct", "reasoning": "Good."}),
    )
    assert _route_after_evaluate(state) == "finalize"


# ---------------------------------------------------------------------------
# Node function tests  (no LM Studio required)
# ---------------------------------------------------------------------------


def test_node_rewrite_retrieve(mock_tools):
    """_node_rewrite_retrieve returns rewritten_query and context."""
    from medqa_multi_agents import _node_rewrite_retrieve

    state = {"question": "How do ACE inhibitors work?", **_EMPTY_MEMORY}
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
        "evaluation_result": "",
        "revision_count": 0,
        **_EMPTY_MEMORY,
    }
    result = _node_answer(state)

    assert "draft_answer" in result
    assert isinstance(result["draft_answer"], str)


def test_node_answer_increments_revision_on_incorrect(mock_tools):
    """_node_answer increments revision_count when previous verdict was incorrect."""
    from medqa_multi_agents import _node_answer

    state = {
        "question": "How do ACE inhibitors work?",
        "context": "ACE inhibitors block the angiotensin-converting enzyme...",
        "evaluation_result": json.dumps({"verdict": "incorrect", "reasoning": "Bad."}),
        "revision_count": 1,
        **_EMPTY_MEMORY,
    }
    result = _node_answer(state)
    assert result["revision_count"] == 2


def test_node_evaluate(mock_tools):
    """_node_evaluate returns evaluation_result and evaluation_reasoning."""
    from medqa_multi_agents import _node_evaluate

    state = {
        "question": "How do ACE inhibitors work?",
        "context": "ACE inhibitors block the angiotensin-converting enzyme...",
        "draft_answer": "ACE inhibitors block the angiotensin-converting enzyme.",
        **_EMPTY_MEMORY,
    }
    result = _node_evaluate(state)

    assert "evaluation_result" in result
    assert "evaluation_reasoning" in result
    parsed = json.loads(result["evaluation_result"])
    assert "verdict" in parsed
    assert "reasoning" in parsed


def test_node_finalize():
    """_node_finalize copies draft_answer to final_answer (no save_session call)."""
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
        **_EMPTY_MEMORY,
    }
    result = _node_finalize(state)
    assert result["final_answer"] == "ACE inhibitors block ACE."


def test_node_load_memory_returns_lists():
    """_node_load_memory always returns lists for all four memory fields."""
    from medqa_multi_agents import _node_load_memory

    state = {"question": "What causes dry cough after lisinopril?", **_EMPTY_MEMORY}
    result = _node_load_memory(state)

    for field in ("global_memory", "retrieval_memory", "reasoner_memory", "verifier_memory"):
        assert field in result
        assert isinstance(result[field], list)


# ---------------------------------------------------------------------------
# Workflow graph structure tests  (no LM Studio required)
# ---------------------------------------------------------------------------


def test_workflow_compiles_without_error():
    """The workflow exposes the expected graph nodes."""
    from medqa_multi_agents import workflow

    assert hasattr(workflow, "nodes")
    node_names = set(workflow.nodes.keys())
    # V3 nodes
    assert "load_memory" in node_names
    assert "rewrite_retrieve" in node_names
    assert "answer" in node_names
    assert "evaluate" in node_names
    assert "finalize" in node_names
    # Old 'recall' node must be gone
    assert "recall" not in node_names, "'recall' (Q&A session memory) must not exist in V3"


def test_workflow_invoke_end_to_end_with_mock_tools(mock_tools):
    """Full workflow.invoke returns final_answer when evaluation passes."""
    from medqa_multi_agents import invoke

    result = invoke("What is the mechanism of action of ACE inhibitors?")
    assert isinstance(result, str)
    assert len(result) > 0


def test_workflow_direct_invoke_with_mock_tools(mock_tools):
    """workflow.invoke keeps the public {'question': ...} call shape."""
    from medqa_multi_agents import workflow

    result = workflow.invoke({"question": "What is the mechanism of action of ACE inhibitors?"})

    assert isinstance(result, dict)
    assert result["final_answer"] == "ACE inhibitors are first-line..."
    assert result["revision_count"] == 0
    # V3 memory fields must be present
    assert "global_memory" in result
    assert "retrieval_memory" in result
    assert "reasoner_memory" in result
    assert "verifier_memory" in result


def test_workflow_invoke_with_revision_loop(mock_tools):
    """Full workflow falls through to finalize after MAX_REVISION_LOOPS exhausted."""
    with patch("medqa_multi_agents._get_tools") as mock_get_tools:
        mock_get_tools.return_value = (
            _MockTool("hypertension"),    # retrieve_documents
            _MockTool("context..."),      # retrieve_context
            _MockTool("draft answer"),    # answer_question
            _MockTool(json.dumps({"verdict": "incorrect", "reasoning": "Bad."})),  # always wrong
        )
        from medqa_multi_agents import invoke
        result = invoke("What is the mechanism of action of ACE inhibitors?")
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


def test_enable_memory_exported():
    """ENABLE_MEMORY is the public switch for optional memory."""
    import medqa_multi_agents as pkg
    assert "ENABLE_MEMORY" in pkg.__all__
    assert isinstance(pkg.ENABLE_MEMORY, bool)


def test_save_session_no_longer_exported():
    """save_session must not be in the V3 public API (long-term memory is read-only)."""
    import medqa_multi_agents as pkg
    assert not hasattr(pkg, "save_session"), (
        "save_session must not exist in V3; official eval must not write to memory"
    )


def test_build_thread_config_preserves_existing_thread_id():
    """Short-term memory config does not overwrite caller-supplied thread_id."""
    from medqa_multi_agents.memory.short_term import ENABLE_MEMORY, build_thread_config

    config = {"configurable": {"thread_id": "case-123", "checkpoint_ns": "medqa"}}
    result = build_thread_config(config)

    if not ENABLE_MEMORY:
        assert result is config
        return
    assert result is not config
    assert result["configurable"]["thread_id"] == "case-123"
    assert result["configurable"]["checkpoint_ns"] == "medqa"


def test_build_thread_config_generates_thread_id_when_enabled():
    """Memory-enabled graph invocations get a unique thread_id by default."""
    from medqa_multi_agents.memory.short_term import ENABLE_MEMORY, build_thread_config

    if not ENABLE_MEMORY:
        pytest.skip("ENABLE_MEMORY is disabled")

    result = build_thread_config()
    assert result["configurable"]["thread_id"].startswith("medqa-")
