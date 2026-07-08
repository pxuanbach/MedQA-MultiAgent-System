"""MedQA Multi-Agent System package.

Public API
----------
``workflow``    : compiled LangGraph StateGraph (supervisor orchestrator)
``invoke({...})``: convenience wrapper around ``workflow.invoke``
"""
from __future__ import annotations

import json
import os
from typing import Annotated, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from medqa_multi_agents.agents.answerer import create_answerer_agent
from medqa_multi_agents.agents.evaluator import create_evaluator_agent
from medqa_multi_agents.agents.retriever import create_retriever_agent
from medqa_multi_agents.agents.rewriter import create_rewriter_agent

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


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------


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
    """Promote draft_answer to final_answer."""
    return {"final_answer": state["draft_answer"]}


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
builder = StateGraph(State)

# Nodes
builder.add_node("rewrite_retrieve", _node_rewrite_retrieve)
builder.add_node("answer", _node_answer)
builder.add_node("evaluate", _node_evaluate)
builder.add_node("finalize", _node_finalize)

# Edges
builder.add_edge(START, "rewrite_retrieve")
builder.add_edge("rewrite_retrieve", "answer")
builder.add_edge("answer", "evaluate")
builder.add_conditional_edges("evaluate", _route_after_evaluate)
builder.add_edge("finalize", END)

workflow = builder.compile()

# ---------------------------------------------------------------------------
# Public invoke helper
# ---------------------------------------------------------------------------


def invoke(question: str) -> str:
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
    result = workflow.invoke({
        "question": question,
        "revision_count": 0,
    })
    return result["final_answer"]


__all__ = ["workflow", "invoke"]
