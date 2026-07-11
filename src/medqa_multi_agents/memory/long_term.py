"""Long-term memory for the MedQA Multi-Agent System.

Design principles
-----------------
Long-term memory is a **frozen, role-specific rule store**.  It is completely
separate from:

* **Short-term memory** — the LangGraph state for one question (``short_term.py``).
* **RAG retrieval** — textbook evidence fetched from Chroma at query time.

The memory store lives in ``memory/long_term_rules.json`` (repo root) and
contains only manually-designed rules and checklists derived from development-set
error analysis.  It does NOT contain:

* MedQA question text, answer options, or gold answers
* Official test predictions or evaluation verdicts
* Session histories or chat trajectories

During official evaluation the store is read-only (default).

Search API
----------
``LongTermMemory.search()`` scores rules by keyword overlap and applies
optional metadata filters (agent / topic / memory_type / tags).  The
retrieval is fully deterministic — no stochastic embeddings by default.

Usage
-----
::

    from medqa_multi_agents.memory.long_term import LongTermMemory

    mem = LongTermMemory("memory/long_term_rules.json", read_only=True)
    rules = mem.get_rules_for_agent("reasoner", query="patient dry cough medication")
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Required schema fields
# ---------------------------------------------------------------------------
REQUIRED_FIELDS = frozenset(
    {"id", "agent", "memory_type", "topic", "rule", "source", "tags", "confidence", "created_at"}
)

# ---------------------------------------------------------------------------
# Token helpers for keyword retrieval
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
        "patient", "presents", "following", "appropriate",
        "likely", "diagnosis", "treatment", "management", "test",
        "the", "use", "used", "include", "includes", "question",
    }
)
_MIN_TOKEN_LEN = 3


def _tokenize(text: str) -> set[str]:
    """Return a set of lowercase alpha tokens from *text*, excluding stop-words."""
    raw_tokens = re.findall(r"[a-z]+", text.lower())
    return {
        t for t in raw_tokens
        if len(t) >= _MIN_TOKEN_LEN and t not in _STOP_WORDS
    }


def _record_tokens(record: dict) -> set[str]:
    """Build a searchable token set from all text fields of a record."""
    parts: list[str] = [
        record.get("rule", ""),
        record.get("topic", ""),
        record.get("agent", ""),
        record.get("memory_type", ""),
        " ".join(record.get("tags", [])),
    ]
    return _tokenize(" ".join(parts))


# ---------------------------------------------------------------------------
# LongTermMemory class
# ---------------------------------------------------------------------------


class LongTermMemory:
    """Frozen, role-specific rule store for the MedQA Multi-Agent System.

    Parameters
    ----------
    memory_path:
        Path to the JSON file containing memory rules.
    read_only:
        If ``True`` (default), ``add_rule()`` and ``save()`` raise
        ``RuntimeError``.  Official evaluation must use ``read_only=True``.
    retrieval_mode:
        ``"keyword"`` (default) uses deterministic token-overlap scoring.
        ``"embedding"`` is reserved for future semantic search.
    top_k:
        Default number of rules to return per search call.
    """

    def __init__(
        self,
        memory_path: str,
        read_only: bool = True,
        retrieval_mode: str = "keyword",
        top_k: int = 3,
    ) -> None:
        self.memory_path = Path(memory_path)
        self.read_only = read_only
        self.retrieval_mode = retrieval_mode
        self.default_top_k = top_k
        self._rules: list[dict] = []
        self._loaded = False

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self) -> list[dict]:
        """Load rules from *memory_path* and return them.

        Validates that every record contains the required schema fields.
        Raises ``FileNotFoundError`` if the file is missing.
        Raises ``ValueError`` if any record is missing required fields.
        """
        if not self.memory_path.exists():
            raise FileNotFoundError(
                f"Long-term memory file not found: {self.memory_path}"
            )
        raw: list[dict] = json.loads(self.memory_path.read_text(encoding="utf-8-sig"))
        for i, record in enumerate(raw):
            missing = REQUIRED_FIELDS - record.keys()
            if missing:
                raise ValueError(
                    f"Memory record at index {i} (id={record.get('id', '?')!r}) "
                    f"is missing required fields: {missing}"
                )
        self._rules = raw
        self._loaded = True
        return list(self._rules)

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        agent: str | None = None,
        topic: str | None = None,
        memory_type: str | None = None,
        tags: list[str] | None = None,
        top_k: int | None = None,
    ) -> list[dict]:
        """Return the top-k rules most relevant to *query*.

        Filters
        -------
        *agent*, *topic*, *memory_type*: exact match (case-insensitive).
        *tags*: rule must contain **all** provided tags.

        Scoring
        -------
        Token overlap between *query* tokens and the rule's searchable text.
        Rules with score 0 are included only if no filtered rules have score > 0,
        as a fallback to surface any rule for the requested agent.
        """
        self._ensure_loaded()
        k = top_k if top_k is not None else self.default_top_k

        query_tokens = _tokenize(query)
        candidates = self._apply_filters(agent, topic, memory_type, tags)

        if not candidates:
            return []

        # Score by token overlap
        scored: list[tuple[int, dict]] = []
        for record in candidates:
            rtokens = _record_tokens(record)
            score = len(query_tokens & rtokens)
            scored.append((score, record))

        # Sort descending by score, then by id for determinism
        scored.sort(key=lambda x: (-x[0], x[1]["id"]))

        # If all scores are 0, still return top-k (generic fallback)
        return [r for _, r in scored[:k]]

    def _apply_filters(
        self,
        agent: str | None,
        topic: str | None,
        memory_type: str | None,
        tags: list[str] | None,
    ) -> list[dict]:
        """Return records that match all provided metadata filters."""
        result: list[dict] = []
        for record in self._rules:
            if agent is not None and record.get("agent", "").lower() != agent.lower():
                continue
            if topic is not None and record.get("topic", "").lower() != topic.lower():
                continue
            if memory_type is not None and record.get("memory_type", "").lower() != memory_type.lower():
                continue
            if tags is not None:
                record_tags = {t.lower() for t in record.get("tags", [])}
                if not all(t.lower() in record_tags for t in tags):
                    continue
            result.append(record)
        return result

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def get_rules_for_agent(
        self,
        agent: str,
        query: str,
        topic: str | None = None,
        memory_type: str | None = None,
        top_k: int | None = None,
    ) -> list[dict]:
        """Return relevant rules for a specific agent role.

        Combines agent-filtered keyword search.  Always includes at least the
        top-k rules for this agent even if query overlap is zero.

        Parameters
        ----------
        agent:
            One of ``"global"``, ``"retrieval_planner"``, ``"retriever"``,
            ``"reasoner"``, ``"verifier"``, ``"finalizer"``.
        query:
            Text derived from the current question / case summary.
        topic:
            Optional topic filter (e.g. ``"pharmacology"``).
        memory_type:
            Optional type filter (e.g. ``"verifier_checklist"``).
        top_k:
            How many rules to return (defaults to ``self.default_top_k``).
        """
        return self.search(
            query=query,
            agent=agent,
            topic=topic,
            memory_type=memory_type,
            top_k=top_k,
        )

    # ------------------------------------------------------------------
    # Write operations (disabled in read-only mode)
    # ------------------------------------------------------------------

    def add_rule(self, rule: dict) -> None:
        """Add a new rule to the in-memory store.

        Raises
        ------
        RuntimeError
            Always raised when ``read_only=True``.
        ValueError
            If the rule is missing required schema fields.
        """
        if self.read_only:
            raise RuntimeError(
                "LongTermMemory is in read-only mode. "
                "add_rule() is not permitted during official evaluation. "
                "To add rules offline, set read_only=False and call save()."
            )
        self._ensure_loaded()
        missing = REQUIRED_FIELDS - rule.keys()
        if missing:
            raise ValueError(f"Rule is missing required fields: {missing}")
        self._rules.append(rule)

    def save(self) -> None:
        """Persist in-memory rules back to *memory_path*.

        Raises
        ------
        RuntimeError
            Always raised when ``read_only=True``.
        """
        if self.read_only:
            raise RuntimeError(
                "LongTermMemory is in read-only mode. "
                "save() is not permitted during official evaluation."
            )
        self.memory_path.write_text(
            json.dumps(self._rules, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def format_rules_for_prompt(self, rules: list[dict]) -> str:
        """Format a list of rules into a readable string for prompt injection."""
        if not rules:
            return "(No relevant long-term memory rules found.)"
        lines: list[str] = []
        for r in rules:
            lines.append(f"- [{r['id']}] {r['rule']}")
        return "\n".join(lines)

    def __len__(self) -> int:
        self._ensure_loaded()
        return len(self._rules)

    def __repr__(self) -> str:
        return (
            f"LongTermMemory(path={self.memory_path!r}, "
            f"read_only={self.read_only}, "
            f"loaded={self._loaded}, "
            f"rules={len(self._rules)})"
        )


# ---------------------------------------------------------------------------
# Module-level default instance (used by the supervisor workflow)
# ---------------------------------------------------------------------------

# Resolve path relative to the repository root (3 levels up from this file)
_DEFAULT_RULES_PATH = Path(__file__).parent.parent.parent.parent / "memory" / "long_term_rules.json"

long_term_memory = LongTermMemory(
    memory_path=str(_DEFAULT_RULES_PATH),
    read_only=True,
    retrieval_mode="keyword",
    top_k=3,
)


# ---------------------------------------------------------------------------
# Backwards-compat keyword helper (kept for existing tests that import it)
# ---------------------------------------------------------------------------

def extract_keywords(text: str) -> tuple[str, ...]:
    """Return up to 3 non-stop-word tokens from *text* (order preserved)."""
    tokens = re.findall(r"[a-z]+", text.lower())
    seen: set[str] = set()
    unique: list[str] = []
    for t in tokens:
        if len(t) >= 4 and t not in _STOP_WORDS and t not in seen:
            seen.add(t)
            unique.append(t)
        if len(unique) == 3:
            break
    return tuple(unique)


__all__ = [
    "LongTermMemory",
    "long_term_memory",
    "extract_keywords",
    "REQUIRED_FIELDS",
]
