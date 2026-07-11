"""Long-term memory (store) for the MedQA supervisor workflow.

Uses LangGraph's InMemoryStore for dev/testing. When ENABLE_MEMORY is true,
the supervisor gains a ``recall_memory`` tool that searches past Q&A sessions
by keyword-derived namespace and a ``save_session`` helper to persist them.

Namespace strategy
------------------
Keywords are extracted from the question at runtime by lowercasing and
filtering stop-words / short tokens.  A namespace tuple is built as::

    ("medqa", <keyword1>, <keyword2>, ...)

This makes cross-session recall specific enough to avoid noise while still
generalising across semantically related questions.
"""
from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.tools import tool

from medqa_multi_agents.memory.short_term import ENABLE_MEMORY

# ---------------------------------------------------------------------------
# Module-level store singleton
# ---------------------------------------------------------------------------
_store = None


def get_store():
    """Return the process-wide InMemoryStore when memory is enabled."""
    global _store
    if not ENABLE_MEMORY:
        return None
    if _store is None:
        try:
            from langgraph.store.memory import InMemoryStore
        except ImportError:
            # Older langgraph versions
            from langgraph.checkpoint.memory import InMemorySaver as InMemoryStore  # noqa: F401
            _store = None
            return None
        _store = InMemoryStore()
    return _store


# ---------------------------------------------------------------------------
# Keyword extraction
# ---------------------------------------------------------------------------
_STOP_WORDS = frozenset(
    {
        "a", "an", "the", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "shall", "can",
        "of", "in", "on", "at", "to", "for", "with", "from", "by",
        "and", "or", "but", "not", "if", "as", "it", "its", "this",
        "that", "these", "those", "what", "which", "who", "whom",
        "how", "when", "where", "why", "all", "each", "every",
        "most", "more", "other", "some", "such", "no", "nor", "so",
        "yet", "both", "either", "neither", "one", "two", "three",
        "patient", "presents", "following", "appropriate", "most",
        "likely", "diagnosis", "treatment", "management", "test",
    }
)
_MIN_KEYWORD_LEN = 4
_MAX_KEYWORDS = 3


def extract_keywords(text: str) -> tuple[str, ...]:
    """Return up to MAX_KEYWORDS lowercase tokens from *text*.

    Tokens are alpha-only, longer than MIN_KEYWORD_LEN, and not stop-words.
    """
    tokens = re.findall(r"[a-z]+", text.lower())
    keywords = [
        t for t in tokens
        if len(t) >= _MIN_KEYWORD_LEN and t not in _STOP_WORDS
    ]
    # Deduplicate while preserving first-occurrence order
    seen: set[str] = set()
    unique: list[str] = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            unique.append(kw)
        if len(unique) == _MAX_KEYWORDS:
            break
    return tuple(unique)


def _make_namespace(question: str) -> tuple[str, ...]:
    keywords = extract_keywords(question)
    return ("medqa",) + (keywords if keywords else ("general",))


# ---------------------------------------------------------------------------
# Store helpers
# ---------------------------------------------------------------------------

def save_session(
    question: str,
    final_answer: str,
    *,
    extra: dict[str, Any] | None = None,
    session_id: str | None = None,
) -> None:
    """Persist a completed Q&A session to the long-term store.

    Does nothing when ENABLE_MEMORY is false or the store is unavailable.
    """
    store = get_store()
    if store is None:
        return

    import uuid
    ns = _make_namespace(question)
    key = session_id or str(uuid.uuid4())
    value = {
        "question": question,
        "answer": final_answer,
        **(extra or {}),
    }
    store.put(ns, key, value)


def _format_sessions(items: list) -> str:
    """Format store search results into a readable string for the LLM."""
    if not items:
        return "No relevant past sessions found."
    parts: list[str] = []
    for item in items:
        val = item.value if hasattr(item, "value") else item
        q = val.get("question", "?")
        a = val.get("answer", "?")
        parts.append(f"Q: {q}\nA: {a}")
    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# recall_memory tool
# ---------------------------------------------------------------------------

@tool
def recall_memory(question: str) -> str:
    """Search long-term memory for past Q&A sessions relevant to the question.

    Returns a formatted string of past sessions, or a message that none exist.
    Only available when ENABLE_MEMORY is true.
    """
    store = get_store()
    if store is None:
        return "Long-term memory is disabled."
    ns = _make_namespace(question)
    try:
        results = store.search(ns)
    except Exception:
        results = []
    return _format_sessions(list(results))


__all__ = [
    "ENABLE_MEMORY",
    "get_store",
    "save_session",
    "recall_memory",
    "extract_keywords",
]
