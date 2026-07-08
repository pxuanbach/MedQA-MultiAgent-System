# AGENTS.md — Tests Directory

## Fixture-First Testing Workflow

When authoring a new test in this directory:

1. **List available fixtures first** — run `pytest --fixtures tests/` to see what shared fixtures already exist before writing any new code.

2. **Reuse existing fixtures** — design your tests around fixtures from `conftest.py`. Do not create duplicate fixtures for capabilities that already exist.

3. **If a needed fixture is missing** — prefer adding it to `conftest.py` as a shared resource rather than embedding it in a single test file.

## Adding New Fixtures

- Put module-scoped fixtures in `conftest.py` so they are shared across all test files in this directory.
- Use `scope="module"` for expensive resources (LLM connections, database clients).
- Use `scope="function"` (the default) for cheap per-test fixtures.
- Always call `pytest.skip()` inside the fixture (not in the test) if a resource is unavailable, so the whole module is skipped cleanly.

## Adding New Test Files

- Follow the naming convention `test_<module>.py`.
- Import fixtures via `pytest` — no need to import `conftest.py` explicitly; pytest auto-discovers it.
