# Goal

Build a multi-agent system that answers medical exam questions (MedQA / USMLE) by searching through medical textbooks.

## Target

Evolve into a multi-agent system where separate agents handle rewriting, retrieval, answering, and self-evaluation, orchestrated by a supervisor that decides the flow and can revise answers through an evaluator feedback loop.

## Requirements

- Source code: `src/medqa_multi_agents/`
- Vector store: persistent Chroma (not in-memory)
- Prompts: markdown files in `prompts/` directory (not hardcoded in Python)
- Memory: optional, controlled by a single `ENABLE_MEMORY` flag
- Interface: `workflow.invoke({'question': '...'})` — same as the original `__init__.py`

## Dependency Order

```
deps-langchain → dir-structure → prompts-markdown → vectorstore-chroma
    → agents-rewriter → agents-retriever → agents-answerer → evaluator-agent
    → supervisor-workflow-core
    → ingest-script
    → memory-short-term
    → memory-long-term
    → supervisor-workflow-memory
```

`supervisor-workflow-core` is independently testable. `supervisor-workflow-memory` requires core + both memory modules.
