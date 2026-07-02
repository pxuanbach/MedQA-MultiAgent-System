#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Replace these commands with the correct commands for your repository.
INSTALL_CMD=(uv sync --all-groups)
VERIFY_CMD=(uv run pytest)
START_CMD=(uv run python -m main)

echo "==> [1/5] Working directory: $PWD"
echo "==> [2/5] Syncing dependencies"
"${INSTALL_CMD[@]}"

echo "==> [3/5] Verifying harness files..."
FILES_OK=true
for file in AGENTS.md CLAUDE.md feature_list.json clean-state-checklist.md; do
  if [ ! -f "$file" ]; then
    echo "  MISSING: $file"
    FILES_OK=false
  else
    echo "  OK: $file"
  fi
done

if [ "$FILES_OK" != true ]; then
  echo "=== Init complete with warnings. Some harness files are missing. ==="
  exit 1
fi

echo "==> [4/5] Running baseline verification"
set +e  # Allow pytest to fail without stopping the script
"${VERIFY_CMD[@]}"
set -e

echo "==> [5/5] Startup command"
printf ' %q' "${START_CMD[@]}"
printf '\n'

if [ "${RUN_START_COMMAND:-0}" = "1" ]; then
  echo "==> Starting the app"
  exec "${START_CMD[@]}"
fi

echo "Set RUN_START_COMMAND=1 if you want init.sh to launch the app directly."