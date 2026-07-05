"""Rewriter agent — reformulates MedQA questions for optimal retrieval."""

from pathlib import Path

from langchain.agents import create_agent
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
        tools: Optional list of tools to pass to create_agent.
               If None, an empty list is used.

    Returns:
        A ``retrieve_documents`` tool wrapped via ``create_agent``.
    """
    if tools is None:
        tools = []

    agent = create_agent(
        model=model,
        system_prompt=_load_rewriter_prompt(),
        tools=tools,
    )

    @tool
    def retrieve_documents(question: str) -> str:
        """Rewrite a medical exam question for optimal textbook retrieval."""
        result = agent.invoke({"messages": [{"role": "user", "content": question}]})
        # Return only the text of the final agent message
        return result["messages"][-1].content

    return retrieve_documents
