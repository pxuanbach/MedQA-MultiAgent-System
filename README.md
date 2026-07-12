# MedQA Multi-Agent System

A LangGraph-orchestrated multi-agent RAG workflow for evaluating factual accuracy on the **MedQA-USMLE** benchmark. The goal is reproducible benchmark evaluation, not clinical deployment.

---

## System Variants

| Variant | Description |
|---|---|
| V0 | Direct LLM — no RAG, no memory |
| V1 | RAG-only — textbook retrieval, no agents |
| V2 | Multi-agent workflow — no long-term memory (`ENABLE_MEMORY=false`) |
| V3 | Full system — multi-agent + RAG + short-term memory + long-term rule memory |
| V4 | V3 without verifier (optional ablation) |

---

## Memory Design

The system uses three distinct memory concepts. They are **not interchangeable**.

### 1. Short-Term Memory

- Implemented as **LangGraph state** (`TypedDict`).
- Exists only during one MedQA question.
- Stores intermediate outputs: `question`, `rewritten_query`, `context`, `draft_answer`, `evaluation_result`, `revision_count`, `final_answer`.
- Also stores in-question memory results: `global_memory`, `retrieval_memory`, `reasoner_memory`, `verifier_memory`.
- Reset after each question.
- Enabled by `ENABLE_MEMORY=true` (checkpointer for thread-level persistence within a session).

### 2. Long-Term Memory

- Implemented as a **frozen, role-specific rule store** (`memory/long_term_rules.json`).
- Persists across questions and runs.
- Built **only** from manual design or development-set error analysis.
- **Read-only during official evaluation** — cannot be updated at inference time.
- Stores general rules, checklists, and pitfalls — **not** full question-answer examples.
- Accessed via `LongTermMemory` class (`src/medqa_multi_agents/memory/long_term.py`).

#### Why is long-term memory frozen?

Benchmark integrity requires that the evaluation system does not learn from test questions. Allowing memory writes during official evaluation would constitute data leakage. The frozen design ensures reproducibility: every run uses the same static rule set.

#### Long-term memory is NOT:
- Chat history or session history
- A cache of past Q&A answers
- A repository of case trajectories
- A second RAG corpus

### 3. RAG Textbook Retrieval

- Retrieves textbook chunks from the **MedQA-USMLE textbook corpus** (18 English textbooks).
- Stored in a persistent **Chroma** vector database.
- Provides the primary medical evidence for reasoning.
- Must NOT contain MedQA QA pairs, answer_idx, gold answers, or official test questions.
- Completely separate from the long-term memory rule store.

#### RAG vs Long-Term Memory

| | RAG Retrieval | Long-Term Memory |
|---|---|---|
| Content | Textbook passages | Role-specific rules and checklists |
| Purpose | Medical evidence | Auxiliary system guidance |
| Scope | Per question | Across all questions |
| Mutability | Read-only | Read-only (official eval) |
| Format | Unstructured text chunks | Structured JSON records |

---

## Long-Term Memory Schema

Each memory record must have these required fields:

```json
{
  "id": "reasoner_001",
  "agent": "reasoner",
  "memory_type": "procedural_rule",
  "topic": "pharmacology",
  "rule": "When symptoms appear after starting a new medication, consider adverse drug effects before primary disease progression.",
  "source": "manual_design",
  "tags": ["pharmacology", "adverse_effects", "medication_timeline"],
  "confidence": 0.85,
  "created_at": "2026-07-01"
}
```

**Valid `agent` values:** `global`, `retrieval_planner`, `retriever`, `reasoner`, `verifier`, `finalizer`

**Valid `memory_type` values:** `semantic_rule`, `procedural_rule`, `verifier_checklist`, `retrieval_strategy`, `error_pattern`

---

## Dev Traces vs Long-Term Memory

Dev traces (in `logs/dev_traces/`) may contain full question-level data for analysis:

```json
{
  "id": "dev_001",
  "question": "...",
  "rewritten_query": "...",
  "retrieved_sources": ["chunk_001"],
  "prediction": "C",
  "gold_answer": "B",
  "evaluation_verdict": "wrong",
  "error_type": "retrieval_irrelevant"
}
```

**Dev traces must never be directly injected as long-term memory.** The correct workflow is:

```
dev run
  → dev trace logs (logs/dev_traces/)
  → manual error analysis
  → distill general rules (python src/medqa_multi_agents/memory/build_rules_from_dev_logs.py)
  → add rules to memory/long_term_rules.json (manual review required)
  → freeze memory
  → official evaluation
```

Official traces (`logs/official_traces/`) are logged for reproducibility but must never be used to update memory before or during official evaluation.

---

## Which Variants Use Memory

| Variant | Short-term (checkpointer) | Long-term (rule store) | RAG |
|---|---|---|---|
| V0 | ✗ | ✗ | ✗ |
| V1 | ✗ | ✗ | ✓ |
| V2 | ✗ | ✗ | ✓ |
| V3 | ✓ | ✓ | ✓ |
| V4 | ✓ | ✓ (no verifier rules) | ✓ |

Control with `ENABLE_MEMORY` environment variable:
- `ENABLE_MEMORY=true` (default) → V3 behaviour
- `ENABLE_MEMORY=false` → V2 behaviour

---

## Leakage Prevention

The system is designed to prevent data leakage from official test sets:

1. **`long_term_rules.json` contains no MedQA test questions, gold answers, or `answer_idx` values.**
2. **`LongTermMemory(read_only=True)` raises `RuntimeError` on any write attempt.**
3. **`_node_finalize` does not write predictions back to the rule store.**
4. **`build_rules_from_dev_logs.py` is an offline-only script — not called during evaluation.**
5. **Official traces are logged separately and never fed back into memory.**

---

## Project Structure

```
memory/
  long_term_rules.json       ← frozen rule store (read-only in eval)
logs/
  dev_traces/                ← dev Q&A trace logs (not memory)
  official_traces/           ← official reproducibility logs
src/
  medqa_multi_agents/
    __init__.py              ← V3 supervisor workflow (LangGraph StateGraph)
    agents/                  ← rewriter, retriever, answerer, evaluator
    memory/
      __init__.py
      short_term.py          ← checkpointer, ENABLE_MEMORY flag
      long_term.py           ← LongTermMemory class (frozen JSON rule store)
      build_rules_from_dev_logs.py  ← offline analysis script
    prompts/                 ← markdown prompts for each agent
    vectorstore/             ← Chroma DB for RAG
    scripts/                 ← ingest_textbooks.py
config.yaml                  ← memory and evaluation configuration
```

---

## Quick Start

```bash
# Install dependencies
uv sync --all-groups

# Ingest textbooks into Chroma (one-time)
make ingest

# Run tests
uv run pytest -v

# Answer a question (V3 with memory)
python main.py
```

---

## Configuration

See `config.yaml` for memory settings:

```yaml
memory:
  long_term:
    enabled: true
    path: "memory/long_term_rules.json"
    read_only: true
    retrieval_mode: "keyword"
    top_k_per_agent: 3
```
