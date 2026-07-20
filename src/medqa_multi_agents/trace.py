"""Trace logging for agent reasoning.

Provides structured logging of each agent's reasoning process during
the MedQA multi-agent workflow. Traces are saved to JSON files in the
logs directory and also printed to console for real-time visibility.

Usage
-----
    from medqa_multi_agents.trace import get_trace_logger

    logger = get_trace_logger()
    logger.log_agent_start("rewriter", question)
    # ... agent runs ...
    logger.log_agent_end("rewriter", output="...", reasoning="...")
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
LOGS_DIR = Path(__file__).parent.parent.parent / "logs"
DEV_TRACES_DIR = LOGS_DIR / "dev_traces"
OFFICIAL_TRACES_DIR = LOGS_DIR / "official_traces"


def _ensure_logs_dir():
    DEV_TRACES_DIR.mkdir(parents=True, exist_ok=True)
    OFFICIAL_TRACES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Console colours (Windows-safe)
# ---------------------------------------------------------------------------
class _Colours:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    # Agent colours
    REWRITER = "\033[94m"   # blue
    RETRIEVER = "\033[96m" # cyan
    ANSWERER = "\033[92m"  # green
    EVALUATOR = "\033[93m" # yellow
    MEMORY = "\033[95m"    # magenta
    FINAL = "\033[97m"     # white


def _agent_colour(name: str) -> str:
    return getattr(_Colours, name.upper(), _Colours.RESET)


# ---------------------------------------------------------------------------
# Trace structure
# ---------------------------------------------------------------------------


class AgentStep:
    """Records a single agent invocation."""

    def __init__(
        self,
        agent_name: str,
        input_: str = "",
        output: str = "",
        reasoning: str = "",
        chunks: list[dict] | None = None,
        timestamp: str | None = None,
        rubric: dict | None = None,
        keywords: list[str] | None = None,
    ):
        self.agent_name = agent_name
        self.input = input_
        self.output = output
        self.reasoning = reasoning
        self.chunks = chunks or []
        self.timestamp = timestamp or datetime.utcnow().isoformat() + "Z"
        self.rubric = rubric
        self.keywords = keywords or []

    def to_dict(self) -> dict[str, Any]:
        d = {
            "agent": self.agent_name,
            "input": self.input,
            "output": self.output,
            "reasoning": self.reasoning,
            "chunks": self.chunks,
            "timestamp": self.timestamp,
        }
        if self.rubric is not None:
            d["rubric"] = self.rubric
        if self.keywords:
            d["keywords"] = self.keywords
        return d


class Trace:
    """Full trace for one question invocation."""

    def __init__(self, question: str, trace_id: str | None = None):
        self.question = question
        self.trace_id = trace_id or str(uuid.uuid4())[:8]
        self.question_id: str = ""
        self.steps: list[AgentStep] = []
        self.revision_count: int = 0
        self.final_answer: str = ""
        self.started_at: str = datetime.utcnow().isoformat() + "Z"
        self.ended_at: str = ""

    def add_step(self, step: AgentStep):
        self.steps.append(step)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "question_id": self.question_id,
            "question": self.question,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "revision_count": self.revision_count,
            "final_answer": self.final_answer,
            "steps": [s.to_dict() for s in self.steps],
        }


# ---------------------------------------------------------------------------
# TraceLogger
# ---------------------------------------------------------------------------


class TraceLogger:
    """Thread-safe logger that writes traces to files and prints to console."""

    def __init__(self, *, verbose: bool = True, trace_dir: Path | None = None):
        _ensure_logs_dir()
        self._verbose = verbose
        self._trace_dir = trace_dir or DEV_TRACES_DIR
        self._trace: Trace | None = None
        self._lock = Lock()

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def start_trace(self, question: str, trace_id: str | None = None) -> str:
        """Begin a new trace for a question. Returns the trace_id."""
        with self._lock:
            self._trace = Trace(question, trace_id)
            return self._trace.trace_id

    def end_trace(self, final_answer: str, revision_count: int = 0):
        """Finalise and persist the current trace."""
        with self._lock:
            if self._trace is None:
                return
            self._trace.final_answer = final_answer
            self._trace.revision_count = revision_count
            self._trace.ended_at = datetime.utcnow().isoformat() + "Z"
            self._persist()

    def set_question_id(self, question_id: str):
        """Set the dataset question ID if known."""
        with self._lock:
            if self._trace:
                self._trace.question_id = question_id

    # ------------------------------------------------------------------
    # Agent step logging
    # ------------------------------------------------------------------

    def log_agent_start(
        self,
        agent_name: str,
        input_: str,
        *,
        colour: str | None = None,
    ):
        """Log the start of an agent invocation (console only)."""
        if not self._verbose:
            return
        c = colour or _agent_colour(agent_name)
        print(
            f"{c}{_Colours.BOLD}[{agent_name.upper()} START]{_Colours.RESET} "
            f"{_truncate(input_, 200)}",
            file=sys.stdout,
        )

    def log_agent_end(
        self,
        agent_name: str,
        output: str,
        *,
        reasoning: str = "",
        chunks: list[dict] | None = None,
        colour: str | None = None,
        rubric: dict | None = None,
        keywords: list[str] | None = None,
    ):
        """Log the end of an agent invocation (console + memory for trace)."""
        c = colour or _agent_colour(agent_name)
        if self._verbose:
            print(
                f"{c}{_Colours.BOLD}[{agent_name.upper()} END]{_Colours.RESET} "
                f"{_truncate(output, 200)}",
                file=sys.stdout,
            )
            if rubric:
                scores = [f"{k}={v}" for k, v in rubric.items() if k != "reasoning"]
                print(f"{c}  Rubric: {', '.join(scores)}{_Colours.RESET}")
            if reasoning:
                print(
                    f"{c}  Reasoning: {_truncate(reasoning, 500)}{_Colours.RESET}",
                    file=sys.stdout,
                )
            if chunks:
                print(f"{c}  Chunks retrieved: {len(chunks)}")
                for chunk in chunks:
                    score_str = f" score={chunk['score']}" if "score" in chunk else ""
                    print(f"{c}    - [{chunk['source']}] p.{chunk['page']}{score_str}: "
                          f"{chunk['content_preview'][:100]}...{_Colours.RESET}")
            if keywords:
                print(f"{c}  Keywords: {', '.join(keywords)}{_Colours.RESET}")
        with self._lock:
            if self._trace is not None:
                self._trace.add_step(
                    AgentStep(
                        agent_name=agent_name,
                        input_="",  # input logged at start
                        output=output,
                        reasoning=reasoning,
                        rubric=rubric,
                        keywords=keywords,
                    )
                )

    def log_revision(self, revision_count: int, verdict: str):
        """Log a revision loop iteration."""
        if not self._verbose:
            return
        print(
            f"{_Colours.BOLD}{_Colours.REWRITER}[REVISION #{revision_count} — "
            f"verdict: {verdict}]{_Colours.RESET}",
            file=sys.stdout,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist(self):
        if self._trace is None:
            return
        filename = f"trace_{self._trace.trace_id}.json"
        path = self._trace_dir / filename
        try:
            path.write_text(json.dumps(self._trace.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as exc:
            print(f"[TRACE ERROR] Could not write {path}: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------
_logger: TraceLogger | None = None
_init_lock = Lock()


def get_trace_logger(
    *,
    verbose: bool = True,
    trace_dir: Path | None = None,
) -> TraceLogger:
    """Get the global TraceLogger singleton.

    If a second call requests a different *trace_dir* than the existing
    singleton's, a warning is printed but the existing singleton is returned.
    """
    global _logger
    with _init_lock:
        if _logger is None:
            _logger = TraceLogger(verbose=verbose, trace_dir=trace_dir)
        elif trace_dir is not None and _logger._trace_dir != trace_dir:
            import sys
            print(
                f"[WARNING] TraceLogger already initialised with trace_dir="
                f"{_logger._trace_dir!r}; ignoring requested trace_dir={trace_dir!r}",
                file=sys.stderr,
            )
        return _logger


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _truncate(s: str, max_len: int) -> str:
    """Truncate string for display, adding ellipsis if needed."""
    s = s.replace("\n", " ").strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."
