"""Memory helpers for the MedQA supervisor workflow.

Three memory concepts
---------------------
Short-term memory:
    LangGraph state (TypedDict fields) — lives only during one MedQA question.
    Implemented in ``short_term.py``.

Long-term memory:
    Frozen role-specific rule store loaded from ``memory/long_term_rules.json``.
    Read-only during official evaluation.  Does NOT store Q&A pairs or gold answers.
    Implemented in ``long_term.py`` as ``LongTermMemory``.

RAG retrieval:
    Textbook evidence fetched from the Chroma vectorstore at query time.
    Completely separate from memory — implemented in ``vectorstore/``.
"""

from medqa_multi_agents.memory.short_term import (
    ENABLE_MEMORY,
    build_thread_config,
    get_checkpointer,
)
from medqa_multi_agents.memory.long_term import (
    LongTermMemory,
    long_term_memory,
    extract_keywords,
    REQUIRED_FIELDS,
)

__all__ = [
    # Short-term / checkpointer
    "ENABLE_MEMORY",
    "build_thread_config",
    "get_checkpointer",
    # Long-term / rule store
    "LongTermMemory",
    "long_term_memory",
    "extract_keywords",
    "REQUIRED_FIELDS",
]
