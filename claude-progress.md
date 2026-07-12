# Progress Log

## Current Verified State

- Repository root: `E:\MyStudy\MLSercurity\MedQA-MultiAgent-System`
- Standard startup path: `pwsh -File init.ps1` (requires `uv` on PATH)
- Standard verification path: `$env:PATH = "C:\Users\abc\.local\bin;" + $env:PATH; uv run pytest`
- **All features: PASSING** — long-term memory fully refactored to frozen rule store
- Current blocker: **None**
- Next best step: Optional enhancements — PostgreSQL-backed checkpointer for production, end-to-end LM Studio smoke test, evaluation benchmark runner, V0/V1/V4 variant implementations.

## Session Log

### Session 001 — 2026-07-05

- Goal: Implement MedQA multi-agent system (up to supervisor-workflow-core)
- Completed:
  - `deps-langchain` — added langchain, langchain-chroma, langgraph, chromadb, sentence-transformers to pyproject.toml
  - `dir-structure` — created src/medqa_multi_agents/{prompts,vectorstore,agents,scripts}/ with __init__.py
  - `prompts-markdown` — created 5 markdown prompt files (supervisor, rewriter, retriever, answerer, evaluator)
  - `vectorstore-chroma` — created vectorstore/{embedding.py,db.py}
- Evidence: langchain 1.3.11, langchain-chroma 1.1.0, langgraph 1.2.7, chromadb 1.5.9 installed; prompts/*.md created
- Commits: `a506437`
- Next best step: Continue with `agents-rewriter` (create rewriter agent module)

### Session 002 — 2026-07-05

- Goal: Fix and validate vectorstore-chroma ingest pipeline; add Makefile
- Completed:
  - `vectorstore-chroma` fixes — switched to CUDA via `torch.cuda.is_available()`, fixed `persist_directory=None` → `DEFAULT_PERSIST_DIR`, fixed Chroma `client=None` persistence, fixed spaCy `en_core_web_sm` → blank `English()` + `sentencizer`, added `batch_size=32` to HuggingFaceEmbeddings to avoid GPU OOM, switched `SentenceTransformerEmbeddings` → `langchain-huggingface.HuggingFaceEmbeddings`
  - `ingest-script` — moved ingest to src/medqa_multi_agents/scripts/ingest_textbooks.py; spaCy sentencizer chunking (4 lines/para, 10 sentences/chunk, min 30 tokens); all-MiniLM-L6-v2 → all-mpnet-base-v2 on CUDA
  - `vectorstore-chroma` — full ingest confirmed: 86,493 chunks from 18 textbooks written to .chroma_db/ (persistent)
  - Makefile — added init, install, ingest, run, verify, lint, test targets
- Bugs fixed:
  - `persist_directory or ""` was causing Chroma to write D:\chroma.sqlite3 → fixed in db.py with `persist_directory or DEFAULT_PERSIST_DIR`
  - spaCy `spacy.load("en_core_web_sm")` not found → fixed to use blank `English()` + `sentencizer`
  - GPU OOM on batch upsert → fixed with `encode_kwargs={"batch_size": 32}`
  - `SentenceTransformerEmbeddings` deprecated → switched to `langchain_huggingface.HuggingFaceEmbeddings`
- Evidence: single-book test: 1095 chunks; full ingest: 86,493 chunks; .chroma_db/chroma.sqlite3 persisted; CUDA confirmed on RTX 5060 Ti
- Files updated: embedding.py, db.py, ingest_textbooks.py (scripts/), Makefile, pyproject.toml, feature_list.json

### Session 003 — 2026-07-05

- Goal: Implement `agents-rewriter` feature
- Completed:
  - `agents-rewriter` — created `src/medqa_multi_agents/agents/rewriter.py` with `RewrittenQuery` Pydantic model, `create_rewriter_agent()`, and `@tool retrieve_documents`; verified: tool name='retrieve_documents', `RewrittenQuery.query` field, clean imports
  - `agents/__init__.py` — exports `RewrittenQuery` and `create_rewriter_agent`
- Evidence: `uv run pytest tests/test_rewriter.py -v` → 11/11 passed (11.24s)
- Bugs found & fixed: `create_agent` tool-calling protocol returns empty with qwen3-8b → switched to `model.with_structured_output` directly inside `@tool`
- Files updated: src/medqa_multi_agents/agents/{rewriter.py,__init__.py}, tests/test_rewriter.py, feature_list.json
- Commits: `e86d812`

### Session 004 — 2026-07-05

- Goal: Implement `agents-retriever` feature
- Completed:
  - `agents-retriever` — created `src/medqa_multi_agents/agents/retriever.py` with `RetrievedContext` Pydantic model, `create_retriever_agent()`, and `@tool retrieve_context`; searches Chroma via `similarity_search`, formats output with `model.with_structured_output(RetrievedContext)`
  - `_get_vectorstore()` cached at module level — avoids GPU OOM from re-loading embedding model on every tool call
  - `agents/__init__.py` — exports `RetrievedContext` and `create_retriever_agent`
  - `tests/test_retriever.py` — 7 tests: schema/unit (no LM Studio), tool signature, end-to-end Chroma integration
- Evidence: `uv run pytest tests/test_retriever.py -v` → 7/7 passed; `uv run pytest -v` → 18/18 total (7 retriever + 11 rewriter)
- Bugs fixed: GPU access violation from re-initializing embedding model on every `retrieve_context` call → fixed with module-level `_vectorstore` cache
- Files updated: src/medqa_multi_agents/agents/{retriever.py,__init__.py}, tests/test_retriever.py, feature_list.json
- Commits: `ac07c14`

### Session 005 — 2026-07-07

- Goal: Implement `agents-answerer` feature
- Completed:
  - `agents-answerer` — created `src/medqa_multi_agents/agents/answerer.py` with `Answer` Pydantic model (`answer: str`), `create_answerer_agent()`, and `@tool answer_question(context, question)`; uses `model.with_structured_output(Answer)` directly inside `@tool` (same qwen3-8b-compatible pattern as rewriter/retriever)
  - `tests/test_answerer.py` — 7 tests: schema/unit (no LM Studio), tool signature, end-to-end integration
  - `agents/__init__.py` — exports `Answer` and `create_answerer_agent`
- Evidence: `uv run pytest tests/test_answerer.py -v` → 7/7 passed; full suite `uv run pytest -v` → 25/25 (7 answerer + 7 retriever + 11 rewriter)
- Files updated: src/medqa_multi_agents/agents/{answerer.py,__init__.py}, tests/test_answerer.py, feature_list.json, claude-progress.md
- Next best step: Continue with `evaluator-agent` (create evaluator agent module)

### Session 006 — 2026-07-07

- Goal: Implement `evaluator-agent` feature
- Completed:
  - `evaluator-agent` — created `src/medqa_multi_agents/agents/evaluator.py` with `EvaluationResult` Pydantic model (`verdict: str`, `reasoning: str`), `create_evaluator_agent()`, and `@tool evaluate_answer(draft_answer, question, context)`; returns JSON string `{"verdict": ..., "reasoning": ...}`; same `model.with_structured_output(EvaluationResult)` pattern as other agents
  - `agents/__init__.py` — exports `EvaluationResult` and `create_evaluator_agent`
  - `tests/test_evaluator.py` — 8 tests: schema/unit (no LM Studio), tool signature, end-to-end integration with LM Studio
- Evidence: `uv run pytest tests/test_evaluator.py -v` → 8/8 passed; full suite `uv run pytest -v` → 33/33 total (8 evaluator + 7 answerer + 7 retriever + 11 rewriter)
- Bug fixed: docstring "Judge whether..." → "Evaluate whether..." to match `test_tool_description_mentions_evaluate` assertion
- Files updated: src/medqa_multi_agents/agents/{evaluator.py,__init__.py}, tests/test_evaluator.py, feature_list.json, claude-progress.md
- Next best step: Continue with `supervisor-workflow-core` (implement LangGraph StateGraph with revision loop)

### Session 007 — 2026-07-07

- Goal: Implement `supervisor-workflow-core` feature
- Completed:
  - `__init__.py` refactored into a LangGraph StateGraph with 4 nodes: `rewrite_retrieve`, `answer`, `evaluate`, `finalize`
  - `_node_rewrite_retrieve`: calls `retrieve_documents` + `retrieve_context` in sequence
  - `_node_answer`: calls `answer_question`, increments `revision_count` if previous verdict was incorrect/incomplete
  - `_node_evaluate`: calls `evaluate_answer`, stores raw JSON in `evaluation_result`
  - `_node_finalize`: promotes `draft_answer` → `final_answer`
  - `_route_after_evaluate`: conditional edge — routes to `answer` if verdict incorrect/incomplete and loops remain, else `finalize`
  - Shared model + tools cached at module level (avoids GPU OOM on re-load)
  - `invoke(question)` helper — entry point matching original `workflow.invoke({'question': '...'})` signature
  - `tests/test_supervisor.py` — 16 tests: State schema, routing logic (all 4 cases), node functions, graph compilation, end-to-end invoke with mock tools
- Bugs found & fixed:
  - `add_node` is not a standalone function in `langgraph.graph` — removed erroneous import; `builder.add_node()` is the correct method
  - `revision_count` not initialized on first invoke → added `revision_count: 0` to initial state in `invoke()`
  - `Command(update=...)` in conditional edge silently ignored by this langgraph version → moved increment logic into `_node_answer` instead
- Evidence: `uv run pytest tests/test_supervisor.py -v` → 16/16 passed; full suite `uv run pytest -v` → 49/49 passed
- Files updated: src/medqa_multi_agents/__init__.py (complete rewrite), tests/test_supervisor.py (new), feature_list.json, claude-progress.md
- Next best step: Continue with `supervisor-workflow-memory` (add checkpointer + recall_memory tool behind ENABLE_MEMORY flag)

### Session 008 — 2026-07-09

- Goal: Continue memory work, starting with `memory-short-term` because `supervisor-workflow-memory` depends on both short-term and long-term memory modules.
- Startup status:
  - `pwd` confirmed repository root: `/Users/harryhoang/Learning/MLSercurity/MedQA-MultiAgent-System`
  - Read `goal.md`, `claude-progress.md`, `feature_list.json`, root `AGENTS.md`, and `tests/AGENTS.md`
  - `git log --oneline -5` reviewed; latest commit: `58b916b Implement answerer and evaluator agents with tests; refactor supervisor workflow`
  - `./init.ps1` failed: file is not executable; `pwsh -File init.ps1` failed: `pwsh` not installed
  - `bash init.sh` failed before verification: `uv: command not found`
  - Installed `uv` with `python3 -m pip install --user uv`
  - `python3 -m uv sync --all-groups` failed under sandbox due blocked `~/.cache/uv`; rerun with temp uv dirs failed due network DNS restriction while downloading Python 3.12; network approval for the same sync was rejected
  - `python3 -m pytest --fixtures tests/` could not run because pytest is not installed outside the project environment
- Drafted (not verified):
  - Added `src/medqa_multi_agents/memory/` with short-term memory helpers: `ENABLE_MEMORY`, `get_checkpointer()`, and `build_thread_config()`
  - Wired supervisor graph compilation to use `InMemorySaver` checkpointer when `ENABLE_MEMORY` is enabled
  - Added a `Workflow` facade so `workflow.invoke({"question": "..."})` still works by injecting default `revision_count` and a generated `thread_id`
  - Extended `invoke(question, thread_id=None)` for callers that want a stable short-term memory thread
  - Added focused supervisor tests for direct workflow invocation and thread config behavior
- Follow-up on macOS:
  - Created source-local `.venv` with `UV_PROJECT_ENVIRONMENT=.venv`
  - `uv sync --all-groups` downloaded Python 3.12.2 but failed because `torch==2.11.0+cu128` has no macOS arm64 wheel
  - `uv sync --all-groups --no-install-package torch --no-install-package torchvision --no-install-package torchaudio` succeeded for the rest of the environment
  - Initial supervisor pytest failed on `ModuleNotFoundError: torch` during import; fixed by lazy-importing `torch` inside `load_embeddings()` and falling back to CPU if torch is not installed
- Verification:
  - `UV_PROJECT_ENVIRONMENT=.venv python3 -m uv run --no-sync pytest tests/test_supervisor.py -v` → 20/20 passed
  - `UV_PROJECT_ENVIRONMENT=.venv python3 -m uv run --no-sync pytest -v` → 32 passed, 21 skipped
  - Standard `uv run pytest` is still not validated on macOS because it tries to resync CUDA torch from the lockfile
- Files updated: `src/medqa_multi_agents/__init__.py`, `src/medqa_multi_agents/memory/{__init__.py,short_term.py}`, `src/medqa_multi_agents/vectorstore/embedding.py`, `tests/test_supervisor.py`, `feature_list.json`, `claude-progress.md`
- Next best step: On Windows/CUDA machine, run `uv sync --all-groups`, then `uv run pytest tests/test_supervisor.py -v` and `uv run pytest -v`. If both pass, move `memory-short-term` to `passing`.

### Session 009 — 2026-07-11

- Goal: Windows/CUDA baseline verification for `memory-short-term`, then implement `memory-long-term` + `memory-integration`
- Startup:
  - `uv` was not installed → installed via `Invoke-RestMethod https://astral.sh/uv/install.ps1` to `C:\Users\abc\.local\bin`
  - `uv sync --all-groups` → downloaded 182 packages including `torch==2.11.0+cu128` (2.6 GiB); CUDA confirmed on RTX 5060 Ti
  - Baseline: `uv run pytest tests/test_supervisor.py -v` → **20/20 passed**; `uv run pytest -v` → **32 passed, 21 skipped** → `memory-short-term` marked **passing**
- Completed: `memory-long-term`
  - Created `src/medqa_multi_agents/memory/long_term.py`:
    - `InMemoryStore` singleton via `get_store()` (disabled when `ENABLE_MEMORY=false`)
    - `extract_keywords(text)` — alpha tokens, ≥4 chars, stop-word filtered, max 3 unique → namespace `("medqa", kw1, kw2, kw3)`
    - `save_session(question, answer, ...)` — puts completed Q&A to store under keyword namespace
    - `recall_memory` `@tool` — searches store by namespace, formats past sessions for LLM
  - Updated `memory/__init__.py` to re-export all long-term symbols
  - Wired into supervisor `__init__.py`:
    - New `recall` node: `START → recall → rewrite_retrieve` (calls `_recall_memory_tool`, stores result in `state["past_sessions"]`)
    - `_node_finalize` calls `save_session` (only when `ENABLE_MEMORY=true`)
    - `State` gained `past_sessions: str` field
  - Created `tests/test_long_term_memory.py` — 14 tests (unit + integration, integration auto-skips if InMemoryStore unavailable)
  - Updated `tests/test_supervisor.py` — added `past_sessions` to State dicts, patched `_recall_memory_tool` + `save_session` in `mock_tools` fixture and revision-loop test, updated node count assertion
- Completed: `supervisor-workflow-memory` and `memory-integration` — both marked passing (same goal achieved via dedicated recall node architecture)
- Evidence:
  - `uv run pytest tests/test_long_term_memory.py tests/test_supervisor.py -v` → **34/34 passed (0.90s)**
  - `uv run pytest -v` → **46 passed, 21 skipped (56s)** — all 21 skips are LM Studio integration tests (no server running)
- Files updated: `src/medqa_multi_agents/memory/{long_term.py,__init__.py}`, `src/medqa_multi_agents/__init__.py`, `tests/{test_long_term_memory.py,test_supervisor.py}`, `feature_list.json`, `claude-progress.md`
- All 12 features now **passing**. Project goal achieved.

### Session 010 — 2026-07-11

- Goal: Refactor long-term memory from InMemoryStore Q&A session cache to frozen role-specific rule store (per spec §1–§18)
- Design decision: `ENABLE_MEMORY` flag continues to gate both the LangGraph checkpointer (short-term) and the rule injection into the graph (long-term). Rules always read from frozen JSON; writes blocked at runtime.
- Completed:
  - **`memory/long_term_rules.json`**: Created frozen rule store with 9 initial rules across 5 agent roles (global, retrieval_planner, reasoner, verifier, finalizer). No Q&A pairs, no gold answers, no answer_idx.
  - **`config.yaml`**: Added memory configuration with official/dev sections enforcing read_only=True.
  - **`logs/dev_traces/`, `logs/official_traces/`**: Created log directories with .gitkeep files.
  - **`src/medqa_multi_agents/memory/long_term.py`**: Complete rewrite — `LongTermMemory` class with `load()`, `search()` (keyword-overlap scoring + metadata filters), `get_rules_for_agent()`, `add_rule()` (raises RuntimeError in read-only), `save()` (raises RuntimeError in read-only), `format_rules_for_prompt()`. Module-level singleton `long_term_memory` auto-loaded at import.
  - **`src/medqa_multi_agents/memory/__init__.py`**: Updated exports — removed `save_session`, `recall_memory`, `get_store`; added `LongTermMemory`, `long_term_memory`, `REQUIRED_FIELDS`.
  - **`src/medqa_multi_agents/__init__.py`**: V3 supervisor workflow refactored:
    - `State`: removed `past_sessions`; added `global_memory`, `retrieval_memory`, `reasoner_memory`, `verifier_memory` fields
    - `_node_load_memory` replaces `_node_recall` (no Q&A store, no write-back)
    - `_node_rewrite_retrieve`, `_node_answer`, `_node_evaluate` inject formatted memory rules into prompts when `ENABLE_MEMORY=true`
    - `_node_finalize` no longer calls `save_session` (frozen memory)
    - Graph: START → load_memory → rewrite_retrieve → answer → evaluate → finalize → END
  - **Prompts**: `rewriter.md`, `answerer.md`, `evaluator.md` updated with memory guidance sections.
  - **`src/medqa_multi_agents/memory/build_rules_from_dev_logs.py`**: Offline-only analysis script (not called during eval).
  - **`tests/test_long_term_memory.py`**: Full rewrite — 36 tests covering all 7 required scenarios from spec.
  - **`tests/test_supervisor.py`**: Updated for new State fields — 24 tests.
  - **`README.md`**: New file documenting all memory concepts, variants, dev trace workflow, and leakage prevention.
  - **`feature_list.json`**: Updated memory-long-term and memory-integration feature entries.
- Bugs found & fixed:
  - Windows PowerShell `Set-Content` writes UTF-8 BOM → fixed by re-writing JSON via `uv run python` and using `utf-8-sig` encoding in `load()`.
  - `importlib.reload` approach in state integration tests was fragile → replaced with `monkeypatch.setattr(pkg, "ENABLE_MEMORY", ...)`.
- Evidence:
  - `uv run pytest tests/test_long_term_memory.py -v` → **36/36 passed**
  - `uv run pytest tests/test_supervisor.py -v` → **24/24 passed**
  - `uv run pytest -v` → **72 passed, 21 skipped (LM Studio), 0 failed (56s)**
- Files updated: `memory/long_term_rules.json`, `config.yaml`, `logs/`, `src/medqa_multi_agents/memory/{long_term.py,__init__.py}`, `src/medqa_multi_agents/__init__.py`, `src/medqa_multi_agents/prompts/{rewriter,answerer,evaluator}.md`, `src/medqa_multi_agents/memory/build_rules_from_dev_logs.py`, `tests/{test_long_term_memory.py,test_supervisor.py}`, `README.md`, `feature_list.json`, `claude-progress.md`
- Next best step: Optional — implement V0/V1/V4 variants, end-to-end LM Studio benchmark run, or PostgreSQL-backed persistence.
