# Progress Log

## Current Verified State

- Repository root: `/Users/harryhoang/Learning/MLSercurity/MedQA-MultiAgent-System`
- Standard startup path: `./init.ps1` (not executable on this macOS checkout; `pwsh` unavailable; `bash init.sh` is the local equivalent but depends on `uv`)
- Standard verification path: `uv run pytest`
- Current highest-priority unfinished feature: `memory-short-term`
- Current blocker: macOS local verification can run from source `.venv` only with `uv run --no-sync` because the lockfile wants CUDA torch (`torch==2.11.0+cu128`), which has no macOS arm64 wheel. Windows verification is still needed before marking `memory-short-term` passing.

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
