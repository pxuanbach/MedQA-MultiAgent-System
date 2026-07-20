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
import json
import os
from pathlib import Path
from typing import Any, TypedDict

from dotenv import load_dotenv

# Load .env file from project root
_env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(_env_path, override=True)

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
import sys

from medqa_multi_agents.trace import _Colours, get_trace_logger


def _truncate(s: str, max_len: int) -> str:
    """Truncate string for logging, adding ellipsis if needed."""
    s = s.replace("\n", " ").strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."

# ---------------------------------------------------------------------------
# Configuration (OpenAI-compatible)
# ---------------------------------------------------------------------------
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://localhost:1234/v1")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "not-needed")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "qwen2.5-7b-instruct-1m")
MAX_REVISION_LOOPS = int(os.environ.get("MAX_REVISION_LOOPS", "2"))
MAX_RETRIEVAL_LOOPS = int(os.environ.get("MAX_RETRIEVAL_LOOPS", "2"))

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph


# ---------------------------------------------------------------------------
# Shared model (one per process to avoid GPU OOM on re-load)
# ---------------------------------------------------------------------------
_model: BaseChatModel | None = None


def _get_model() -> BaseChatModel:
    global _model
    if _model is None:
        _model = ChatOpenAI(
            base_url=OPENAI_BASE_URL,
            api_key=OPENAI_API_KEY,
            model=OPENAI_MODEL,
            temperature=0,
            max_tokens=8192,
        )
    return _model


# ---------------------------------------------------------------------------
# Shared tools (one per process)
# ---------------------------------------------------------------------------
_tools: tuple | None = None


def _get_tools():
    """Return cached (retrieve_documents, retrieve_context, answer_question, evaluate_answer).

    Tools are created once per process to avoid redundant model re-wrapping.
    """
    global _tools
    if _tools is None:
        model = _get_model()
        retrieve_documents = create_rewriter_agent(model)
        retrieve_context = create_retriever_agent(model, top_k=3)
        answer_question = create_answerer_agent(model)
        evaluate_answer = create_evaluator_agent(model)
        _tools = (retrieve_documents, retrieve_context, answer_question, evaluate_answer)
    return _tools


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
    retrieval_count: int           # how many retrieval rounds (initial + enrichments)
    retrieved_chunk_ids: list[str]  # "source:page" of chunks already retrieved (dedup)
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
    logger = get_trace_logger()

    # Build augmented question string for the rewriter
    question = state["question"]
    memory_hint = ""
    if ENABLE_MEMORY:
        global_rules = state.get("global_memory", [])
        retrieval_rules = state.get("retrieval_memory", [])
        combined = global_rules + retrieval_rules
        memory_hint = _format_memory_section(combined, "Retrieval Planning")

    rewriter_input = question if not memory_hint else f"{question}\n{memory_hint}"

    # Log rewriter start
    logger.log_agent_start("rewriter", rewriter_input)
    rewritten_raw = retrieve_documents.invoke(rewriter_input)
    # Parse JSON response: {"query": ..., "reasoning": ...}
    try:
        rewritten_parsed = json.loads(rewritten_raw)
        rewritten = rewritten_parsed.get("query", rewritten_raw)
        rewriter_reasoning = rewritten_parsed.get("reasoning", "")
        rewriter_keywords = rewritten_parsed.get("keywords", [])
    except (json.JSONDecodeError, TypeError, AttributeError) as exc:
        import sys
        print(f"[WARNING] Rewriter returned unparseable JSON ({exc}): {_truncate(rewritten_raw, 80)}", file=sys.stderr)
        rewritten = rewritten_raw
        rewriter_reasoning = ""
        rewriter_keywords = []
    logger.log_agent_end("rewriter", output=rewritten, reasoning=rewriter_reasoning, keywords=rewriter_keywords)

    # Log retriever start
    logger.log_agent_start("retriever", rewritten)
    retriever_raw = retrieve_context.invoke(rewritten)

    # Parse JSON response: {"context": ..., "chunks": [...], "sources": [...], "pages": [...]}
    try:
        retriever_parsed = json.loads(retriever_raw)
        context = retriever_parsed.get("context", retriever_raw)
        sources = retriever_parsed.get("sources", [])
        pages = retriever_parsed.get("pages", [])
        chunks_info = retriever_parsed.get("chunks", [])
        chunk_ids = [f"{s}:{p}" for s, p in zip(sources, pages)]
    except Exception:
        context = retriever_raw
        chunks_info = []
        chunk_ids = []

    logger.log_agent_end("retriever", output=context, chunks=chunks_info)

    return {
        "rewritten_query": rewritten,
        "context": context,
        "retrieved_chunk_ids": chunk_ids,
    }


def _node_retrieve_enrich(state: State) -> dict:
    """Retrieve additional context when the current context was insufficient.

    Uses skip_ids to avoid re-retrieving chunks already seen in prior rounds.
    Appends new chunks to existing context to address gaps identified
    by the evaluator. Increments retrieval_count to track enrichment rounds.
    """
    _, retrieve_context, _, _ = _get_tools()
    logger = get_trace_logger()

    evaluation_reasoning = state.get("evaluation_reasoning", "")
    retrieval_count = state.get("retrieval_count", 1)
    skip_ids = state.get("retrieved_chunk_ids", [])

    # Build a supplemental query from what the evaluator found missing.
    # Include already-retrieved chunk IDs so the retriever agent can skip them.
    already_retrieved = ", ".join(skip_ids) if skip_ids else "none"
    enrich_query = (
        f"{state['rewritten_query']}\n\n"
        f"Evaluator noted (address these gaps): {evaluation_reasoning}\n\n"
        f"Already retrieved — do NOT retrieve these chunks again: {already_retrieved}"
    )

    logger.log_agent_start("retriever_enrich", f"Enriching with skip_ids={len(skip_ids)}: {enrich_query[:200]}...")

    # Pass skip_ids to avoid re-retrieving the same chunks
    retriever_raw = retrieve_context.invoke(enrich_query)

    try:
        retriever_parsed = json.loads(retriever_raw)
        new_context = retriever_parsed.get("context", retriever_raw)
        sources = retriever_parsed.get("sources", [])
        pages = retriever_parsed.get("pages", [])
        chunks_info = retriever_parsed.get("chunks", [])
        new_chunk_ids = [f"{s}:{p}" for s, p in zip(sources, pages)]
    except Exception:
        new_context = retriever_raw
        chunks_info = []
        new_chunk_ids = []

    # Append new context to existing
    existing_context = state.get("context", "")
    combined_context = (
        f"{existing_context}\n\n"
        f"--- Additional context (retrieval round {retrieval_count + 1}) ---\n\n"
        f"{new_context}"
    )

    logger.log_agent_end(
        "retriever_enrich",
        output=f"Added {len(new_chunk_ids)} new chunks (total {len(skip_ids) + len(new_chunk_ids)}), context length: {len(combined_context)}",
        chunks=chunks_info,
    )

    return {
        "context": combined_context,
        "retrieval_count": retrieval_count + 1,
        "retrieved_chunk_ids": skip_ids + new_chunk_ids,
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
    logger = get_trace_logger()

    question = state["question"]
    context = state["context"]
    memory_hint = ""
    if ENABLE_MEMORY:
        global_rules = state.get("global_memory", [])
        reasoner_rules = state.get("reasoner_memory", [])
        combined = global_rules + reasoner_rules
        memory_hint = _format_memory_section(combined, "Reasoning")

    augmented_question = question if not memory_hint else f"{question}\n{memory_hint}"

    logger.log_agent_start("answerer", f"Context: {context[:300]}...\nQuestion: {question}")
    answer_raw = answer_question_tool.invoke({
        "context": context,
        "question": augmented_question,
    })
    # Parse JSON response: {"answer": ..., "reasoning": ...}
    try:
        answer_parsed = json.loads(answer_raw)
        draft = answer_parsed.get("answer", answer_raw)
        answerer_reasoning = answer_parsed.get("reasoning", "")
    except Exception:
        draft = answer_raw
        answerer_reasoning = ""
    logger.log_agent_end("answerer", output=draft, reasoning=answerer_reasoning)

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
    logger = get_trace_logger()

    question = state["question"]
    memory_hint = ""
    if ENABLE_MEMORY:
        global_rules = state.get("global_memory", [])
        verifier_rules = state.get("verifier_memory", [])
        combined = global_rules + verifier_rules
        memory_hint = _format_memory_section(combined, "Verification")

    augmented_question = question if not memory_hint else f"{question}\n{memory_hint}"

    logger.log_agent_start("evaluator", f"Draft: {state['draft_answer'][:200]}...")
    raw = evaluate_tool.invoke({
        "draft_answer": state["draft_answer"],
        "question": augmented_question,
        "context": state["context"],
    })
    parsed = json.loads(raw)
    verdict = parsed.get("verdict", "")
    evaluator_reasoning = parsed.get("reasoning", "")
    # Extract rubric scores for trace logging
    rubric_scores = {
        k: parsed[k]
        for k in ("correctness", "completeness", "evidence_alignment", "distractor_elimination", "evaluator_confidence")
        if k in parsed
    }
    logger.log_agent_end(
        "evaluator",
        output=f"verdict={verdict}",
        reasoning=evaluator_reasoning,
        rubric=rubric_scores,
    )

    # Log revision loop if needed
    if verdict in ("incorrect", "incomplete") and state["revision_count"] < MAX_REVISION_LOOPS:
        logger.log_revision(state["revision_count"] + 1, verdict)

    return {
        "evaluation_result": raw,
        "evaluation_reasoning": evaluator_reasoning,
    }


# ---------------------------------------------------------------------------
# Node: finalize
# ---------------------------------------------------------------------------


def _node_finalize(state: State) -> dict:
    """Promote draft_answer to final_answer.

    Long-term memory is read-only — we do NOT write predictions back to memory.
    """
    logger = get_trace_logger()
    final = state["draft_answer"]
    revision_count = state.get("revision_count", 0)
    if revision_count > 0:
        logger.log_revision(revision_count, "finalized")
    print(f"\n{_Colours.BOLD}{_Colours.FINAL}[FINAL ANSWER]{_Colours.RESET} {final}\n",
          file=sys.stdout)
    return {"final_answer": final}


# ---------------------------------------------------------------------------
# Conditional edge
# ---------------------------------------------------------------------------


def _route_after_evaluate(state: State) -> str:
    """Conditional routing after evaluation.

    Flow:
      verdict=incomplete + retrieval_count < MAX_RETRIEVAL_LOOPS
        → retrieve_enrich (get more/better context)
      verdict=incorrect + revision_count < MAX_REVISION_LOOPS
        → answer (re-answer with same context, different reasoning attempt)
      verdict=incomplete but retrieval exhausted
        → answer (make the best of what we have)
      otherwise → finalize
    """
    verdict_raw = state.get("evaluation_result", "")
    verdict = ""
    if verdict_raw:
        try:
            parsed = json.loads(verdict_raw)
            verdict = parsed.get("verdict", "").lower()
        except (json.JSONDecodeError, TypeError, AttributeError) as exc:
            import sys
            print(
                f"[WARNING] Evaluator returned unparseable evaluation_result "
                f"({exc}): {_truncate(verdict_raw, 80)}",
                file=sys.stderr,
            )

    retrieval_count = state.get("retrieval_count", 1)
    revision_count = state.get("revision_count", 0)

    if verdict == "incomplete":
        if retrieval_count < MAX_RETRIEVAL_LOOPS:
            return "retrieve_enrich"
        if revision_count < MAX_REVISION_LOOPS:
            return "answer"

    if verdict == "incorrect":
        if revision_count < MAX_REVISION_LOOPS:
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
    builder.add_node("retrieve_enrich", _node_retrieve_enrich)
    builder.add_node("finalize", _node_finalize)

    # Edges
    #
    #  START → load_memory → rewrite_retrieve → answer
    #                                           ↓
    #                                      evaluate
    #                                           ↓
    #                         ┌────────────────┴────────────────┐
    #                         ↓                                 ↓
    #                  [incomplete]?                    [incorrect]?
    #                         ↓                                 ↓
    #              retrieve_enrich ──→ answer          answer
    #                         ↓                    (same context, new reasoning)
    #                   (enriched)                         ↓
    #                         └──────── evaluate ◄─────────┘
    #                                           ↓
    #                                    [correct / done]
    #                                           ↓
    #                                       finalize
    #
    builder.add_edge(START, "load_memory")
    builder.add_edge("load_memory", "rewrite_retrieve")
    builder.add_edge("rewrite_retrieve", "answer")
    builder.add_edge("answer", "evaluate")
    builder.add_conditional_edges("evaluate", _route_after_evaluate)
    builder.add_edge("retrieve_enrich", "answer")
    builder.add_edge("finalize", END)

    checkpointer = get_checkpointer()
    if checkpointer is None:
        return builder.compile()
    return builder.compile(checkpointer=checkpointer)


def _prepare_initial_state(input_: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    initial_state = dict(input_)
    thread_id = initial_state.pop("thread_id", None)
    initial_state.setdefault("revision_count", 0)
    initial_state.setdefault("retrieval_count", 1)  # first retrieval = round 1
    initial_state.setdefault("retrieved_chunk_ids", [])  # track seen chunks to avoid re-retrieval
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
        logger = get_trace_logger()
        question = input_.get("question", "")
        trace_id = logger.start_trace(question)

        initial_state, thread_id = _prepare_initial_state(input_)
        next_config = build_thread_config(config, thread_id=thread_id)

        result = self._graph.invoke(initial_state, config=next_config, **kwargs)

        final_answer = result.get("final_answer", "")
        revision_count = result.get("revision_count", 0)
        logger.end_trace(final_answer, revision_count)

        return result

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
        "retrieval_count": 1,
    }
    if thread_id is not None:
        initial_state["thread_id"] = thread_id
    result = workflow.invoke(initial_state)
    return result["final_answer"]


__all__ = ["ENABLE_MEMORY", "workflow", "invoke", "State", "get_trace_logger"]
