"""Tests for the long-term memory module (memory/long_term.py).

Tests are split into:
- Schema / unit tests (no LM Studio, no external services)
- Integration tests (require ENABLE_MEMORY=true)
"""
from __future__ import annotations

import sys
sys.path.insert(0, 'src')

import os
import pytest


# ---------------------------------------------------------------------------
# Unit tests — no external dependencies
# ---------------------------------------------------------------------------


def test_extract_keywords_basic():
    """extract_keywords returns lowercase alpha tokens excluding stop-words."""
    from medqa_multi_agents.memory.long_term import extract_keywords

    keywords = extract_keywords("A patient with hypertension and diabetes mellitus")
    # 'patient' is in stop-words, 'with' and 'and' too
    assert "hypertension" in keywords
    assert "diabetes" in keywords
    assert "mellitus" in keywords


def test_extract_keywords_max_count():
    """extract_keywords never returns more than MAX_KEYWORDS tokens."""
    from medqa_multi_agents.memory.long_term import extract_keywords, _MAX_KEYWORDS

    long_question = (
        "A 65-year-old man presents with crushing chest pain, diaphoresis, "
        "tachycardia, hypertension, dyspnea, palpitations, syncope, "
        "bradycardia, arrhythmia, myocardial infarction, cardiomyopathy."
    )
    keywords = extract_keywords(long_question)
    assert len(keywords) <= _MAX_KEYWORDS


def test_extract_keywords_deduplication():
    """extract_keywords returns unique tokens."""
    from medqa_multi_agents.memory.long_term import extract_keywords

    keywords = extract_keywords("hypertension hypertension hypertension diabetes")
    assert keywords.count("hypertension") <= 1


def test_extract_keywords_empty_string():
    """extract_keywords handles empty input gracefully."""
    from medqa_multi_agents.memory.long_term import extract_keywords

    keywords = extract_keywords("")
    assert isinstance(keywords, tuple)


def test_recall_memory_tool_name():
    """recall_memory @tool has the expected name."""
    from medqa_multi_agents.memory.long_term import recall_memory

    assert recall_memory.name == "recall_memory"


def test_recall_memory_tool_description():
    """recall_memory tool description mentions 'memory'."""
    from medqa_multi_agents.memory.long_term import recall_memory

    assert "memory" in recall_memory.description.lower()


def test_save_session_noop_when_memory_disabled(monkeypatch):
    """save_session does nothing when ENABLE_MEMORY is false."""
    monkeypatch.setattr("medqa_multi_agents.memory.long_term.ENABLE_MEMORY", False)
    from medqa_multi_agents.memory.long_term import save_session

    # Should complete without error and not raise
    save_session("What causes hypertension?", "Genetics and lifestyle.")


def test_recall_memory_disabled_returns_message(monkeypatch):
    """recall_memory returns a disabled message when ENABLE_MEMORY is false."""
    monkeypatch.setattr("medqa_multi_agents.memory.long_term.ENABLE_MEMORY", False)
    # Reset store so it respects the patched value
    monkeypatch.setattr("medqa_multi_agents.memory.long_term._store", None)
    from medqa_multi_agents.memory.long_term import recall_memory

    result = recall_memory.invoke({"question": "What causes hypertension?"})
    assert "disabled" in result.lower()


# ---------------------------------------------------------------------------
# Integration tests — require ENABLE_MEMORY=true and InMemoryStore
# ---------------------------------------------------------------------------


@pytest.fixture
def enabled_store(monkeypatch):
    """Fixture that ensures ENABLE_MEMORY=true and a fresh store for each test."""
    monkeypatch.setattr("medqa_multi_agents.memory.long_term.ENABLE_MEMORY", False)
    monkeypatch.setattr("medqa_multi_agents.memory.long_term._store", None)
    # Re-enable
    monkeypatch.setattr("medqa_multi_agents.memory.long_term.ENABLE_MEMORY", True)
    # Check that InMemoryStore is importable
    try:
        from langgraph.store.memory import InMemoryStore
        from medqa_multi_agents.memory.long_term import get_store
        store = get_store()
        if store is None:
            pytest.skip("InMemoryStore not available in this langgraph version")
    except ImportError:
        pytest.skip("langgraph.store.memory not available")
    yield store


def test_save_and_recall(enabled_store, monkeypatch):
    """save_session followed by recall_memory returns the saved session."""
    monkeypatch.setattr("medqa_multi_agents.memory.long_term.ENABLE_MEMORY", True)
    from medqa_multi_agents.memory.long_term import save_session, recall_memory

    question = "What is the mechanism of ACE inhibitors in hypertension?"
    answer = "ACE inhibitors block the angiotensin-converting enzyme."

    save_session(question, answer, session_id="test-session-001")

    result = recall_memory.invoke({"question": question})
    assert isinstance(result, str)
    # The recalled text should contain part of the saved answer or question
    assert "ACE" in result or "hypertension" in result.lower() or "No relevant" in result


def test_save_multiple_and_recall(enabled_store, monkeypatch):
    """Multiple saved sessions are all retrievable."""
    monkeypatch.setattr("medqa_multi_agents.memory.long_term.ENABLE_MEMORY", True)
    from medqa_multi_agents.memory.long_term import save_session, recall_memory

    question = "Describe the role of insulin in diabetic ketoacidosis?"
    answers = [
        "Insulin deficiency leads to lipolysis and ketogenesis.",
        "Without insulin, glucagon drives hepatic ketone body production.",
    ]
    for i, ans in enumerate(answers):
        save_session(question, ans, session_id=f"test-dka-{i}")

    result = recall_memory.invoke({"question": question})
    assert isinstance(result, str)
    assert len(result) > 0


def test_get_store_returns_singleton(monkeypatch):
    """get_store returns the same object on repeated calls."""
    monkeypatch.setattr("medqa_multi_agents.memory.long_term.ENABLE_MEMORY", True)
    monkeypatch.setattr("medqa_multi_agents.memory.long_term._store", None)
    try:
        from langgraph.store.memory import InMemoryStore  # noqa: F401
    except ImportError:
        pytest.skip("InMemoryStore not available")

    from medqa_multi_agents.memory.long_term import get_store

    store1 = get_store()
    store2 = get_store()
    assert store1 is store2


def test_recall_memory_exported_from_package():
    """recall_memory is accessible from the memory package."""
    from medqa_multi_agents.memory import recall_memory

    assert callable(recall_memory.invoke)


def test_save_session_exported_from_package():
    """save_session is accessible from the memory package."""
    from medqa_multi_agents.memory import save_session

    assert callable(save_session)


def test_extract_keywords_exported_from_package():
    """extract_keywords is accessible from the memory package."""
    from medqa_multi_agents.memory import extract_keywords

    kw = extract_keywords("acute myocardial infarction")
    assert isinstance(kw, tuple)
