# Goal

Build a multi-agent system that answers medical exam questions (MedQA / USMLE) by searching through medical textbooks.

## Target

A complete V3 multi-agent system with:
- **Role-specialized agents**: Retrieval Planner (Rewriter), Textbook Retriever, Medical Reasoner (Answerer), Verifier (Evaluator), Finalizer
- **Orchestration**: LangGraph StateGraph with revision loop (verifier feedback → re-answer if incorrect/incomplete, up to MAX_REVISION_LOOPS)
- **RAG**: Persistent Chroma vectorstore over 18 English MedQA-USMLE textbooks
- **Short-term memory**: LangGraph InMemorySaver checkpointer (per question session, gated by ENABLE_MEMORY)
- **Long-term memory**: Frozen role-specific rule store (memory/long_term_rules.json), read-only during eval, injected into agent prompts

## System Variants

| Variant | Memory | RAG | Verifier |
|---|---|---|---|
| V0 | None | None | None — direct LLM |
| V1 | None | Yes | None |
| V2 | None | Yes | Yes |
| V3 | Short-term + Long-term | Yes | Yes |
| V4 | Short-term + Long-term | Yes | None |

**Current implementation: V3** (ENABLE_MEMORY=true)

## Requirements

- Source code: `src/medqa_multi_agents/`
- Vector store: persistent Chroma (not in-memory), in `src/medqa_multi_agents/vectorstore/.chroma_db/`
- Prompts: markdown files in `src/medqa_multi_agents/prompts/` (not hardcoded in Python)
- Long-term memory: `memory/long_term_rules.json` — frozen JSON rule store (role-specific, read-only in eval)
- Short-term memory: LangGraph checkpointer, controlled by `ENABLE_MEMORY` env var
- Interface: `workflow.invoke({'question': '...'})` — same as original `__init__.py`
- Config: `config.yaml` for memory and eval settings

## V3 Graph Flow

```
START
  → load_memory        (inject role-specific rules from frozen JSON into State)
  → rewrite_retrieve   (rewriter rewrites query; retriever fetches Chroma chunks)
  → answer             (answerer generates draft answer using context + reasoner rules)
  → evaluate           (evaluator judges answer using verifier rules)
  → [conditional]
      if incorrect/incomplete and loops remain → answer (revision loop)
      else → finalize
  → finalize           (promotes draft_answer to final_answer; no write to memory)
  → END
```

## Memory Design (critical for benchmark safety)

- **Long-term memory ≠ RAG**: rules store ≠ textbook corpus
- **Long-term memory ≠ chat history**: no Q&A pairs, no gold answers stored
- **Frozen at eval**: `LongTermMemory(read_only=True)` raises RuntimeError on write
- **Dev traces** go to `logs/dev_traces/` — distilled manually into rules, never auto-injected
- **Official traces** go to `logs/official_traces/` — logged for reproducibility only

## Dependency Order

```
deps-langchain → dir-structure → prompts-markdown → vectorstore-chroma
    → agents-rewriter → agents-retriever → agents-answerer → evaluator-agent
    → supervisor-workflow-core
    → ingest-script
    → memory-short-term
    → memory-long-term          ← V3 frozen rule store (DONE)
    → supervisor-workflow-memory ← V3 full integration (DONE)
    → memory-integration        ← DONE
```

All features currently **passing**. Next optional work: V0/V1/V4 variants, benchmark runner, PostgreSQL persistence.
