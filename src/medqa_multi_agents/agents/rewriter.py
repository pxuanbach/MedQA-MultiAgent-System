"""Rewriter agent — reformulates MedQA questions for optimal retrieval."""

from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel

REWRITER_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "rewriter.md"


def _load_rewriter_prompt() -> str:
    return REWRITER_PROMPT_PATH.read_text(encoding="utf-8")


class RewrittenQuery(BaseModel):
    """Structured output schema for the rewriter agent."""

    query: str


def create_rewriter_agent(
    model: BaseChatModel,
    tools: list[BaseTool] | None = None,
) -> BaseTool:
    """Create the rewriter agent tool.

    Args:
        model: The shared chat model to use for all agents.
        tools: Ignored (present for API compatibility with future multi-tool agents).

    Returns:
        A ``retrieve_documents`` tool that rewrites MedQA questions.
    """

    @tool
    def retrieve_documents(question: str) -> str:
        """Rewrite a medical exam question for optimal textbook retrieval.

        Args:
            question: The original MedQA-style question (multiple choice or free response).

        Returns:
            The rewritten query string — clean, search-friendly, clinically precise.
        """
        system_prompt = _load_rewriter_prompt()
        response = model.with_structured_output(RewrittenQuery).invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ])
        return response.query

    return retrieve_documents
