"""MedQA Multi-Agent System package.

Public API
----------
``workflow``    : supervisor workflow facade
``invoke(...)`` : convenience wrapper around ``workflow.invoke``
"""
from __future__ import annotations

import json
import os
from typing import Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from medqa_multi_agents.agents.answerer import create_answerer_agent
from medqa_multi_agents.agents.evaluator import create_evaluator_agent
from medqa_multi_agents.agents.retriever import create_retriever_agent
from medqa_multi_agents.agents.rewriter import create_rewriter_agent
from medqa_multi_agents.memory import (
    ENABLE_MEMORY,
    build_thread_config,
    get_checkpointer,
    recall_memory as _recall_memory_tool,
    save_session,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LM_STUDIO_URL = os.environ.get("LM_STUDIO_URL", "http://localhost:1234/v1")
LM_STUDIO_MODEL = os.environ.get("LM_STUDIO_MODEL", "qwen/qwen3-8b")
MAX_REVISION_LOOPS = int(os.environ.get("MAX_REVISION_LOOPS", "2"))

# ---------------------------------------------------------------------------
# Shared model (one per process to avoid GPU OOM on re-load)
# ---------------------------------------------------------------------------
_model: BaseChatModel | None = None


def _get_model() -> BaseChatModel:
    global _model
    if _model is None:
        _model = ChatOpenAI(
            base_url=LM_STUDIO_URL,
            api_key="lm-studio",
            model=LM_STUDIO_MODEL,
            temperature=0,
            max_tokens=1024,
        )
    return _model


# ---------------------------------------------------------------------------
# Shared tools (one per process)
# ---------------------------------------------------------------------------
_retrieve_documents = None
_retrieve_context = None
_answer_question = None
_evaluate_answer = None


def _get_tools():
    global _retrieve_documents, _retrieve_context, _answer_question, _evaluate_answer
    if _retrieve_documents is None:
        model = _get_model()
        _retrieve_documents = create_rewriter_agent(model)
        _retrieve_context = create_retriever_agent(model)
        _answer_question = create_answerer_agent(model)
        _evaluate_answer = create_evaluator_agent(model)
    return _retrieve_documents, _retrieve_context, _answer_question, _evaluate_answer


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class State(TypedDict):
    question: str
    rewritten_query: str
    context: str
    draft_answer: str
    evaluation_result: str
    revision_count: int
    evaluation_reasoning: str
    final_answer: str
    past_sessions: str  # long-term memory recall result (empty string when disabled)


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------


def _node_recall(state: State) -> dict:
    """Call recall_memory to surface relevant past sessions (only when ENABLE_MEMORY)."""
    if not ENABLE_MEMORY:
        return {"past_sessions": ""}
    past = _recall_memory_tool.invoke({"question": state["question"]})
    return {"past_sessions": past}


def _node_rewrite_retrieve(state: State) -> dict:
    """Rewrite the question and retrieve relevant chunks from Chroma."""
    retrieve_documents, retrieve_context, _, _ = _get_tools()
    rewritten = retrieve_documents.invoke(state["question"])
    context = retrieve_context.invoke(rewritten)
    return {
        "rewritten_query": rewritten,
        "context": context,
    }


def _node_answer(state: State) -> dict:
    """Generate a draft answer from the retrieved context.

    Increments revision_count if this is a revision loop (previous verdict
    was 'incorrect' or 'incomplete').
    """
    _, _, answer_question_tool, _ = _get_tools()
    draft = answer_question_tool.invoke({
        "context": state["context"],
        "question": state["question"],
    })
    update = {"draft_answer": draft}
    # Increment revision_count if coming from a failed evaluation
    verdict_raw = state.get("evaluation_result", "")
    try:
        parsed = json.loads(verdict_raw)
        verdict = parsed.get("verdict", "").lower()
    except Exception:
        verdict = ""
    if verdict in ("incorrect", "incomplete"):
        update["revision_count"] = state["revision_count"] + 1
    return update


def _node_evaluate(state: State) -> dict:
    """Judge the draft answer; extract verdict and reasoning."""
    _, _, _, evaluate_tool = _get_tools()
    raw = evaluate_tool.invoke({
        "draft_answer": state["draft_answer"],
        "question": state["question"],
        "context": state["context"],
    })
    parsed = json.loads(raw)
    return {
        "evaluation_result": raw,
        "evaluation_reasoning": parsed.get("reasoning", ""),
    }


def _node_finalize(state: State) -> dict:
    """Promote draft_answer to final_answer and persist to long-term memory."""
    final = state["draft_answer"]
    if ENABLE_MEMORY:
        save_session(state["question"], final)
    return {"final_answer": final}


# ---------------------------------------------------------------------------
# Conditional edge — decide whether to revise or finalize
# ---------------------------------------------------------------------------


def _route_after_evaluate(state: State) -> str:
    """Conditional edge: loop back to answer if revision slots remain."""
    verdict_raw = state.get("evaluation_result", "")
    try:
        parsed = json.loads(verdict_raw)
        verdict = parsed.get("verdict", "").lower()
    except Exception:
        verdict = ""

    if verdict in ("incorrect", "incomplete") and state["revision_count"] < MAX_REVISION_LOOPS:
        return "answer"
    return "finalize"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def _build_graph():
    builder = StateGraph(State)

    # Nodes
    builder.add_node("recall", _node_recall)
    builder.add_node("rewrite_retrieve", _node_rewrite_retrieve)
    builder.add_node("answer", _node_answer)
    builder.add_node("evaluate", _node_evaluate)
    builder.add_node("finalize", _node_finalize)

    # Edges
    builder.add_edge(START, "recall")
    builder.add_edge("recall", "rewrite_retrieve")
    builder.add_edge("rewrite_retrieve", "answer")
    builder.add_edge("answer", "evaluate")
    builder.add_conditional_edges("evaluate", _route_after_evaluate)
    builder.add_edge("finalize", END)

    checkpointer = get_checkpointer()
    if checkpointer is None:
        return builder.compile()
    return builder.compile(checkpointer=checkpointer)


def _prepare_initial_state(input_: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    initial_state = dict(input_)
    thread_id = initial_state.pop("thread_id", None)
    initial_state.setdefault("revision_count", 0)
    return initial_state, thread_id


class Workflow:
    """Thin facade that preserves workflow.invoke({...}) with memory enabled."""

    def __init__(self, graph):
        self._graph = graph

    @property
    def nodes(self):
        return self._graph.nodes

    def invoke(self, input_: dict[str, Any], config: dict[str, Any] | None = None, **kwargs):
        initial_state, thread_id = _prepare_initial_state(input_)
        next_config = build_thread_config(config, thread_id=thread_id)
        return self._graph.invoke(initial_state, config=next_config, **kwargs)

    def __getattr__(self, name: str):
        return getattr(self._graph, name)


workflow = Workflow(_build_graph())

# ---------------------------------------------------------------------------
# Public invoke helper
# ---------------------------------------------------------------------------


def invoke(question: str, *, thread_id: str | None = None) -> str:
    """Answer a MedQA question end-to-end through the supervisor workflow.

    Parameters
    ----------
    question : str
        The MedQA-style question to answer.

    Returns
    -------
    str
        The final answer text.
    """
    initial_state = {
        "question": question,
        "revision_count": 0,
    }
    if thread_id is not None:
        initial_state["thread_id"] = thread_id
    result = workflow.invoke(initial_state)
    return result["final_answer"]


__all__ = ["ENABLE_MEMORY", "workflow", "invoke", "save_session"]
