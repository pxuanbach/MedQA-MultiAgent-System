# MedQA Multi-Agent System — Goal

**Working Directory:** `src/medqa-multi-agents/`

---

## Expected Output

A production-ready multi-agent MedQA system that answers USMLE-style medical questions using RAG over 17 English medical textbooks, with cross-session memory persistence and multi-source answer verification — all running on **local LM Studio models**.

**Deliverable:** A Python package `src/medqa-multi-agents/` with:
- `state.py` — Extended DeepAgentState with vector_store and memory fields
- `memory.py` — Markdown-based MemoryManager for cross-session persistence
- `vector_store.py` — Chroma vector store initialized with LM Studio embeddings
- `tools/retrieve.py` — `retrieve_medical_context` tool for RAG retrieval
- `agents/query_processor.py` — Agent that decomposes MedQA questions into search tasks
- `agents/rag_searcher.py` — Agent that (1) retrieves top-k context via vector DB, then (2) reads context and generates a medical answer
- `agents/evaluator.py` — Agent that verifies answers against multiple textbook sources
- `prompts.py` — All agent system prompts
- `indexing/run_indexing.py` — One-time script to index 17 English textbooks

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    MedQA Multi-Agent System                              │
│                    Working Dir: src/medqa-multi-agents/                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                     DeepAgentState (state.py)                    │   │
│  │  ├── todos: list[Todo]                                          │   │
│  │  ├── files: dict[str, str]  (virtual FS)                        │   │
│  │  ├── vector_store: Chroma  (NEW)                                │   │
│  │  ├── memory: MemoryManager  (NEW)                               │   │
│  │  └── messages: list[BaseMessage]                                │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                       3 Agents                                    │   │
│  │                                                                  │   │
│  │  ┌─────────────────┐    ┌──────────────────┐   ┌────────────┐ │   │
│  │  │ query_processor │───▶│  rag_searcher    │──▶│ evaluator  │ │   │
│  │  │    (Agent 1)    │    │  (Agent 2)       │   │ (Agent 3) │ │   │
│  │  │                 │    │  retrieve + ans  │   │            │ │   │
│  │  └─────────────────┘    └──────────────────┘   └────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌─────────────────┐    ┌──────────────────┐   ┌────────────────┐    │
│  │  MemoryManager   │    │   Vector Store   │   │   Textbooks    │    │
│  │  (memory.py)    │    │  (Chroma DB)     │   │  (EN, 17 txt) │    │
│  │                  │    │                  │   │                │    │
│  │  memory/        │    │  ./vector_store/ │   │ datasets/      │    │
│  │  ├── session/   │    │                  │   │ textbooks/en/  │    │
│  │  ├── concepts/  │    │                  │   │                │    │
│  │  └── agents/    │    │                  │   │                │    │
│  └─────────────────┘    └──────────────────┘   └────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

1. **Question In** → `query_processor` decomposes MedQA question into search tasks
2. **Retrieval + Synthesis** → `rag_searcher` calls `retrieve_medical_context` tool (top-k vector similarity), then reads retrieved context and generates a medical answer
3. **Verification** → `evaluator` checks answer against multiple textbook sources, outputs confidence score
4. **Memory** → `MemoryManager` saves Q&A, reasoning, and consulted sources to markdown files

---

## Key Components

| File | Purpose |
|------|---------|
| `state.py` | Extended AgentState with vector_store + memory fields |
| `memory.py` | Markdown-based cross-session memory (session/, concepts/, agents/) |
| `vector_store.py` | Chroma + LM Studio embeddings initialization |
| `tools/retrieve.py` | `retrieve_medical_context` tool |
| `agents/*.py` | 3 agent definitions with system prompts |
| `prompts.py` | All agent system prompts + evaluation prompt |
| `indexing/run_indexing.py` | One-time textbook indexing script |

---

## Dependencies

```toml
# pyproject.toml additions
langchain>=0.3.0
langchain-chroma>=0.1.0
langchain-text-splitters>=0.3.0
```

**LM Studio endpoints used:**
- Chat: `POST http://localhost:1234/v1/chat/completions`
- Embeddings: `POST http://localhost:1234/v1/embeddings`

---

## Success Criteria

- [ ] All 3 agents defined and registered
- [ ] Vector store populated with 17 English textbooks
- [ ] `retrieve_medical_context` tool returns relevant chunks with source attribution
- [ ] Evaluator checks answer against >=2 sources and outputs confidence score
- [ ] Memory persists Q&A across sessions via markdown files
- [ ] End-to-end MedQA question answering works via single CLI command

---

## Running the System

```bash
# 1. Index textbooks (one-time)
uv run python -m src.medqa_multi_agents.indexing.run_indexing

# 2. Start LM Studio (ensure localhost:1234 is running)

# 3. Run MedQA Q&A
uv run python -m src.medqa_multi_agents.main "What are the symptoms of myocardial infarction?"
```
