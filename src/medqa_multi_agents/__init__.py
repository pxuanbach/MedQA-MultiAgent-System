"""MedQA Multi-Agent System — V3 supervisor workflow.

Variant behaviour
-----------------
V3 (ENABLE_MEMORY=true, default):
    Uses short-term memory (LangGraph checkpointer) and long-term memory
    (frozen role-specific rule store).  A dedicated ``load_memory`` node
    runs after the parser/recall step and injects rules into graph state.
    Agents receive memory via prompt context.

V2 (ENABLE_MEMORY=false):
    Multi-agent workflow without memory injection.  The ``load_memory`` node
    is a no-op that stores empty lists.  No rules are injected into prompts.

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
    long_term_memory,
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
    # Core question fields
    question: str
    rewritten_query: str
    context: str
    draft_answer: str
    evaluation_result: str
    revision_count: int
    evaluation_reasoning: str
    final_answer: str
    # Long-term memory fields (V3 only; empty lists in V2)
    global_memory: list[dict]       # shared rules for all agents
    retrieval_memory: list[dict]    # retrieval_planner rules
    reasoner_memory: list[dict]     # reasoner rules
    verifier_memory: list[dict]     # verifier rules


# ---------------------------------------------------------------------------
# Node: load_memory
# ---------------------------------------------------------------------------


def _node_load_memory(state: State) -> dict:
    """Load role-specific rules from the frozen long-term memory store.

    V3 (ENABLE_MEMORY=true): retrieves top-k rules for each agent role and
    stores them in state.  These are injected into agent prompts downstream.

    V2 (ENABLE_MEMORY=false): returns empty lists — no memory injection.

    Note: memory rules are auxiliary guidance only.  They do NOT constitute
    medical evidence and must not override the retrieved textbook context.
    """
    if not ENABLE_MEMORY:
        return {
            "global_memory": [],
            "retrieval_memory": [],
            "reasoner_memory": [],
            "verifier_memory": [],
        }

    # Build a query from the question (or an empty string if not yet set)
    query = state.get("question", "")

    global_rules = long_term_memory.get_rules_for_agent("global", query=query)
    retrieval_rules = long_term_memory.get_rules_for_agent("retrieval_planner", query=query)
    reasoner_rules = long_term_memory.get_rules_for_agent("reasoner", query=query)
    verifier_rules = long_term_memory.get_rules_for_agent("verifier", query=query)

    return {
        "global_memory": global_rules,
        "retrieval_memory": retrieval_rules,
        "reasoner_memory": reasoner_rules,
        "verifier_memory": verifier_rules,
    }


# ---------------------------------------------------------------------------
# Node: rewrite_retrieve
# ---------------------------------------------------------------------------


def _format_memory_section(rules: list[dict], label: str) -> str:
    """Format memory rules as a prompt section string."""
    if not rules:
        return ""
    lines = [f"\n## Long-Term Memory Rules: {label} (Auxiliary Guidance Only)"]
    lines.append(
        "These are system-derived auxiliary rules. "
        "Do NOT treat them as medical evidence. "
        "The primary evidence is the current question and retrieved textbook context."
    )
    for r in rules:
        lines.append(f"- [{r['id']}] {r['rule']}")
    return "\n".join(lines)


def _node_rewrite_retrieve(state: State) -> dict:
    """Rewrite the question and retrieve relevant chunks from Chroma.

    When ENABLE_MEMORY=true, injects retrieval_memory rules (and global_memory)
    into the rewriter prompt to guide query construction.
    """
    retrieve_documents, retrieve_context, _, _ = _get_tools()

    # Build augmented question string for the rewriter
    question = state["question"]
    memory_hint = ""
    if ENABLE_MEMORY:
        global_rules = state.get("global_memory", [])
        retrieval_rules = state.get("retrieval_memory", [])
        combined = global_rules + retrieval_rules
        memory_hint = _format_memory_section(combined, "Retrieval Planning")

    rewriter_input = question if not memory_hint else f"{question}\n{memory_hint}"
    rewritten = retrieve_documents.invoke(rewriter_input)
    context = retrieve_context.invoke(rewritten)
    return {
        "rewritten_query": rewritten,
        "context": context,
    }


# ---------------------------------------------------------------------------
# Node: answer
# ---------------------------------------------------------------------------


def _node_answer(state: State) -> dict:
    """Generate a draft answer from the retrieved context.

    When ENABLE_MEMORY=true, injects reasoner_memory rules into the prompt.
    Increments revision_count if this is a revision loop.
    """
    _, _, answer_question_tool, _ = _get_tools()

    question = state["question"]
    context = state["context"]
    memory_hint = ""
    if ENABLE_MEMORY:
        global_rules = state.get("global_memory", [])
        reasoner_rules = state.get("reasoner_memory", [])
        combined = global_rules + reasoner_rules
        memory_hint = _format_memory_section(combined, "Reasoning")

    augmented_question = question if not memory_hint else f"{question}\n{memory_hint}"
    draft = answer_question_tool.invoke({
        "context": context,
        "question": augmented_question,
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


# ---------------------------------------------------------------------------
# Node: evaluate
# ---------------------------------------------------------------------------


def _node_evaluate(state: State) -> dict:
    """Judge the draft answer; extract verdict and reasoning.

    When ENABLE_MEMORY=true, injects verifier_memory rules into the prompt.
    """
    _, _, _, evaluate_tool = _get_tools()

    question = state["question"]
    memory_hint = ""
    if ENABLE_MEMORY:
        global_rules = state.get("global_memory", [])
        verifier_rules = state.get("verifier_memory", [])
        combined = global_rules + verifier_rules
        memory_hint = _format_memory_section(combined, "Verification")

    augmented_question = question if not memory_hint else f"{question}\n{memory_hint}"
    raw = evaluate_tool.invoke({
        "draft_answer": state["draft_answer"],
        "question": augmented_question,
        "context": state["context"],
    })
    parsed = json.loads(raw)
    return {
        "evaluation_result": raw,
        "evaluation_reasoning": parsed.get("reasoning", ""),
    }


# ---------------------------------------------------------------------------
# Node: finalize
# ---------------------------------------------------------------------------


def _node_finalize(state: State) -> dict:
    """Promote draft_answer to final_answer.

    Long-term memory is read-only — we do NOT write predictions back to memory.
    """
    final = state["draft_answer"]
    return {"final_answer": final}


# ---------------------------------------------------------------------------
# Conditional edge
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

    # Nodes (V3 full system)
    builder.add_node("load_memory", _node_load_memory)
    builder.add_node("rewrite_retrieve", _node_rewrite_retrieve)
    builder.add_node("answer", _node_answer)
    builder.add_node("evaluate", _node_evaluate)
    builder.add_node("finalize", _node_finalize)

    # Edges
    # START -> load_memory -> rewrite_retrieve -> answer -> evaluate -> finalize -> END
    builder.add_edge(START, "load_memory")
    builder.add_edge("load_memory", "rewrite_retrieve")
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
    # Ensure memory fields are present (default empty)
    initial_state.setdefault("global_memory", [])
    initial_state.setdefault("retrieval_memory", [])
    initial_state.setdefault("reasoner_memory", [])
    initial_state.setdefault("verifier_memory", [])
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


__all__ = ["ENABLE_MEMORY", "workflow", "invoke", "State"]
