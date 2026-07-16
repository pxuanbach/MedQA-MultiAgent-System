"""Retriever agent — decides between vectorstore and long-term memory tools."""

from pathlib import Path
import json

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel

from langchain_core.tools import BaseTool, tool

from medqa_multi_agents.memory import ENABLE_MEMORY
from medqa_multi_agents.memory.long_term import long_term_memory
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


# ---------------------------------------------------------------------------
# Tool 1 — search_vectorstore
# ---------------------------------------------------------------------------

class RetrievedVectorChunks(BaseModel):
    """Structured output for search_vectorstore tool."""

    sources: list[str]
    pages: list[str]
    scores: list[float]
    content_previews: list[str]
    combined_context: str


@tool
def search_vectorstore(query: str, top_k: int = 3) -> str:
    """Search the ChromaDB vector store for relevant medical textbook chunks.

    Args:
        query: The rewritten search query from the rewriter agent.
        top_k: Number of top chunks to retrieve.

    Returns:
        JSON string with source, page, score, content_preview for each chunk,
        plus a combined_context string.
    """
    vectorstore = _get_vectorstore()
    results = vectorstore.similarity_search_with_score(query, k=top_k)

    sources, pages, scores, previews = [], [], [], []
    texts = []
    for doc, score in results:
        meta = doc.metadata or {}
        sources.append(meta.get("source", "unknown"))
        pages.append(meta.get("page", "unknown"))
        scores.append(round(score, 4))
        previews.append(doc.page_content[:120].replace("\n", " ") + "...")
        texts.append(doc.page_content)

    combined = "\n\n---\n\n".join(texts)

    structured = RetrievedVectorChunks(
        sources=sources,
        pages=pages,
        scores=scores,
        content_previews=previews,
        combined_context=combined,
    )
    return json.dumps(structured.model_dump(), ensure_ascii=False)


# ---------------------------------------------------------------------------
# Tool 2 — search_memory
# ---------------------------------------------------------------------------

class RetrievedMemoryRules(BaseModel):
    """Structured output for search_memory tool."""

    rules: list[dict]  # Each rule: id, agent, topic, rule, source, tags, confidence


@tool
def search_memory(query: str, agent: str = "retriever", top_k: int = 3) -> str:
    """Search the long-term memory rule store for relevant retrieval/planning rules.

    Args:
        query: The rewritten search query from the rewriter agent.
        agent: Which agent role to retrieve rules for (default: "retriever").
        top_k: Number of top rules to retrieve.

    Returns:
        JSON string with a "rules" list containing rule dictionaries.
    """
    rules = long_term_memory.get_rules_for_agent(
        agent=agent,
        query=query,
        top_k=top_k,
    )
    return json.dumps({"rules": rules}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Retriever agent — decides which tool(s) to use, then synthesises a response
# ---------------------------------------------------------------------------

class RetrievedContext(BaseModel):
    """Structured output schema for the retriever agent."""

    context: str
    chunks: list[dict] = []  # Chunk metadata for tracing


def create_retriever_agent(
    model: BaseChatModel,
    tools: list[BaseTool] | None = None,
    top_k: int = 5,
) -> Runnable:
    """Create the retriever agent runnable.

    The agent uses a ReAct-style loop: it receives a rewritten query,
    decides whether to call ``search_vectorstore`` and/or ``search_memory``,
    then synthesises a final ``RetrievedContext`` response.

    Args:
        model: The shared chat model to use for all agents.
        tools: Additional tools to expose alongside the built-in ones.
        top_k: Default number of top chunks to retrieve from Chroma.

    Returns:
        A ``Runnable`` that accepts a query string and returns a JSON string
        with the retrieved context and chunk metadata.
    """
    # Always include the built-in tools; optionally extend with extras
    built_in = [search_vectorstore, search_memory]
    if tools:
        built_in = built_in + list(tools)

    system_prompt = _load_retriever_prompt()

    def _run(query: str) -> str:
        """ReAct loop: model decides tool calls until it returns a final answer."""
        messages: list = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query),
        ]
        max_steps = 6
        for _ in range(max_steps):
            response = model.bind_tools(built_in, tool_choice="auto", parallel_tool_calls=False).invoke(messages)
            messages.append(response)

            # If model returned a tool call, execute it and append result
            if response.tool_calls:
                for tc in response.tool_calls:
                    tool_name = tc["name"]
                    tool_args = tc["args"]
                    tool = next(t for t in built_in if t.name == tool_name)
                    result = tool.invoke(tool_args)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "name": tool_name,
                        "content": result,
                    })
            else:
                # No tool call — model returned a direct structured response
                return response.content

        # Fallback: model didn't return structured output after max steps
        # Force a final structured call
        final = model.with_structured_output(RetrievedContext).invoke(messages)
        return json.dumps({
            "context": final.context,
            "chunks": final.chunks,
        }, ensure_ascii=False)

    return RunnableLambda(_run)
