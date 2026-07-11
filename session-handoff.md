# Session Handoff — Session 010 (2026-07-11)

## What Was Done This Session

Refactored the long-term memory module from an InMemoryStore Q&A session cache to a **frozen role-specific rule store** aligned with the MedQA benchmark safety requirements.

## Current Verified State

```
uv run pytest -v  →  72 passed, 21 skipped (LM Studio), 0 failed
```

Working tree: **clean** (2 commits ahead of origin).

## V3 Architecture (as of this session)

```
State fields:
  question, rewritten_query, context, draft_answer,
  evaluation_result, revision_count, evaluation_reasoning, final_answer,
  global_memory, retrieval_memory, reasoner_memory, verifier_memory

Graph:
  START → load_memory → rewrite_retrieve → answer → evaluate
       ↑__ (revision loop: incorrect/incomplete + loops < MAX) __|
  → finalize → END
```

`ENABLE_MEMORY=true`  → V3 (checkpointer + rule injection)  
`ENABLE_MEMORY=false` → V2 (no memory)

## Key Files Changed

| File | What It Does Now |
|---|---|
| `memory/long_term_rules.json` | Frozen JSON, 9 rules, 5 agent roles |
| `src/medqa_multi_agents/memory/long_term.py` | `LongTermMemory` class — keyword search, metadata filters, read-only enforcement |
| `src/medqa_multi_agents/__init__.py` | V3 graph — `_node_load_memory` replaces old `_node_recall` |
| `config.yaml` | Memory settings, official/dev modes |
| `tests/test_long_term_memory.py` | 36 tests (7 required spec scenarios) |
| `tests/test_supervisor.py` | 24 tests for new State fields |
| `README.md` | Memory design docs |

## What Must NOT Be Done

- Do NOT add `save_session` / `recall_memory @tool` back — memory is frozen
- Do NOT store Q&A pairs, gold answers, or answer_idx in `long_term_rules.json`
- Do NOT call `build_rules_from_dev_logs.py` during evaluation
- Do NOT update long-term memory at inference time

## Next Candidate Work (all optional)

1. **V0 variant** — direct LLM, no RAG, no memory
2. **V1 variant** — RAG-only, no agents
3. **V4 variant** — V3 without verifier node
4. **Benchmark runner** — script to evaluate on MedQA-USMLE dev/test set, output prediction JSON
5. **PostgreSQL persistence** — replace InMemorySaver with PostgresCheckpointer for production

## Startup for Next Session

```powershell
$env:PATH = "C:\Users\abc\.local\bin;" + $env:PATH
pwsh -File init.ps1   # syncs deps + runs pytest
```

If pytest baseline fails, fix it before starting new work.
