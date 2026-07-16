"""Answerer agent — synthesises a final answer from retrieved context."""

from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel

ANSWERER_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "answerer.md"


def _load_answerer_prompt() -> str:
    return ANSWERER_PROMPT_PATH.read_text(encoding="utf-8")


class Answer(BaseModel):
    """Structured output schema for the answerer agent."""

    answer: str
    reasoning: str = ""


def create_answerer_agent(
    model: BaseChatModel,
    tools: list[BaseTool] | None = None,
) -> BaseTool:
    """Create the answerer agent tool.

    Args:
        model: The shared chat model to use for all agents.
        tools: Ignored (present for API compatibility with future multi-tool agents).

    Returns:
        An ``answer_question`` tool that synthesises a final answer
        from retrieved context and the original question.
    """

    @tool
    def answer_question(context: str, question: str) -> str:
        """Synthesise a final answer to a MedQA-style question using textbook context.

        Args:
            context: Retrieved textbook passages relevant to the question.
            question: The original MedQA-style question (multiple choice or free response).

        Returns:
            The generated answer text.
        """
        system_prompt = _load_answerer_prompt()
        user_message = (
            f"Context:\n{context}\n\nQuestion:\n{question}"
        )
        response = model.with_structured_output(Answer).invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ])
        import json
        return json.dumps({"answer": response.answer, "reasoning": response.reasoning})

    return answer_question
