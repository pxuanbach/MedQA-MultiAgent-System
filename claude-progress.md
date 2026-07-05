# Progress Log

## Current Verified State

- Repository root: `D:\Dev\MedQA-MultiAgent-System`
- Standard startup path: `.\init.ps1`
- Standard verification path: `uv run pytest`
- Current highest-priority unfinished feature: `agents-retriever`
- Current blocker: none

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
- Commits: `e86d812`, `<pending>`
- Next best step: Continue with `agents-answerer` (create answerer agent module)