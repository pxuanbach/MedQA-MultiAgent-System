"""Retriever agent — searches ChromaDB for relevant textbook context."""

from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel

from medqa_multi_agents.vectorstore.db import load_vectorstore

RETRIEVER_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "retriever.md"


def _load_retriever_prompt() -> str:
    return RETRIEVER_PROMPT_PATH.read_text(encoding="utf-8")


# Cached vectorstore — initialised once per process to avoid GPU OOM on re-load
_vectorstore = None


def _get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = load_vectorstore()
    return _vectorstore


class RetrievedContext(BaseModel):
    """Structured output schema for the retriever agent."""

    context: str


def create_retriever_agent(
    model: BaseChatModel,
    tools: list[BaseTool] | None = None,
    top_k: int = 5,
) -> BaseTool:
    """Create the retriever agent tool.

    Args:
        model: The shared chat model to use for all agents.
        tools: Ignored (present for API compatibility with future multi-tool agents).
        top_k: Number of top relevant chunks to retrieve from Chroma.

    Returns:
        A ``retrieve_context`` tool that searches Chroma and returns context.
    """

    @tool
    def retrieve_context(rewritten_query: str) -> str:
        """Retrieve relevant medical textbook context for a rewritten query.

        Args:
            rewritten_query: The rewritten search query from the rewriter agent.

        Returns:
            A JSON string with a "context" field containing combined textbook
            passages relevant to answering the question.
        """
        vectorstore = _get_vectorstore()
        results = vectorstore.similarity_search(rewritten_query, k=top_k)
        retrieved_texts = [doc.page_content for doc in results]

        # Build context string for the model to format
        combined_context = "\n\n---\n\n".join(retrieved_texts)

        # Use the model to format the final structured response
        system_prompt = _load_retriever_prompt()
        response = model.with_structured_output(RetrievedContext).invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": rewritten_query},
        ])
        return response.context

    return retrieve_context
