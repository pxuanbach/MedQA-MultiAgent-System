"""Short-term LangGraph checkpoint memory.

The project uses a single ENABLE_MEMORY flag. For now this module only owns the
short-term checkpointer; long-term memory is added by a later feature.
"""
from __future__ import annotations

import os
from copy import deepcopy
from typing import Any
from uuid import uuid4

TRUE_VALUES = {"1", "true", "yes", "on"}
FALSE_VALUES = {"0", "false", "no", "off"}


def _read_enable_memory() -> bool:
    """Read ENABLE_MEMORY, defaulting to enabled for the memory features."""
    raw = os.environ.get("ENABLE_MEMORY", "true").strip().lower()
    if raw in FALSE_VALUES:
        return False
    if raw in TRUE_VALUES:
        return True
    # Unknown value — default to disabled (safe fallback)
    import sys
    print(
        f"[WARNING] Unknown ENABLE_MEMORY value {raw!r}; defaulting to false. "
        f"Valid values are: {TRUE_VALUES | FALSE_VALUES}",
        file=sys.stderr,
    )
    return False


ENABLE_MEMORY = _read_enable_memory()
_checkpointer = None


def get_checkpointer():
    """Return the process-wide checkpointer when memory is enabled."""
    global _checkpointer
    if not ENABLE_MEMORY:
        return None
    if _checkpointer is None:
        try:
            from langgraph.checkpoint.memory import InMemorySaver
        except ImportError:
            from langgraph.checkpoint.memory import MemorySaver as InMemorySaver

        _checkpointer = InMemorySaver()
    return _checkpointer


def build_thread_config(
    config: dict[str, Any] | None = None,
    *,
    thread_id: str | None = None,
) -> dict[str, Any] | None:
    """Ensure LangGraph checkpointer config contains a thread_id.

    If memory is disabled, the original config is returned unchanged.
    """
    if not ENABLE_MEMORY:
        return config

    next_config = deepcopy(config) if config is not None else {}
    configurable = dict(next_config.get("configurable") or {})
    configurable.setdefault("thread_id", thread_id or f"medqa-{uuid4()}")
    next_config["configurable"] = configurable
    return next_config
