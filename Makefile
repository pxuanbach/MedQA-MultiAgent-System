# MedQA Multi-Agent System — Makefile
# Requires: uv (https://github.com/astral-sh/uv)

# ── Default target ─────────────────────────────────────────────────────────────
.PHONY: help
help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ── Initialization ─────────────────────────────────────────────────────────────
.PHONY: init
init: ## Run full init: install deps, verify harness, run baseline tests
	@echo "==> [1/4] Syncing dependencies"
	uv sync --all-groups
	@echo "==> [2/4] Verifying harness files"
	@for f in AGENTS.md CLAUDE.md feature_list.json clean-state-checklist.md; do \
		if [ -f "$$f" ]; then echo "  OK: $$f"; \
		else echo "  MISSING: $$f"; exit 1; fi; \
	done
	@echo "==> [3/4] Running baseline verification"
	uv run pytest -q || true
	@echo "==> [4/4] Init complete. Run 'make run' to start."

.PHONY: install
install: ## Sync all dependencies
	uv sync --all-groups

.PHONY: verify
verify: ## Run baseline verification (pytest)
	uv run pytest

# ── Vector store ───────────────────────────────────────────────────────────────
.PHONY: ingest
ingest: ## Ingest textbooks into Chroma vector store
	uv run python -c "import sys; sys.path.insert(0, 'src'); from medqa_multi_agents.scripts.ingest_textbooks import ingest_textbooks; ingest_textbooks()"

# ── Workflow ───────────────────────────────────────────────────────────────────
.PHONY: run
run: ## Run the main workflow
	uv run python -m main

# ── Development ───────────────────────────────────────────────────────────────
.PHONY: lint
lint: ## Run linters
	uv run ruff check src/

.PHONY: test
test: ## Run all tests
	uv run pytest
