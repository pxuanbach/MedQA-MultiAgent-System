"""build_rules_from_dev_logs.py — OFFLINE SCRIPT ONLY.

This script is NOT part of official evaluation.
Run it manually to analyse dev-trace logs and generate candidate long-term memory rules.

Usage
-----
::

    python src/medqa_multi_agents/memory/build_rules_from_dev_logs.py \\
        --traces-dir logs/dev_traces \\
        --output candidate_rules.json

The script reads dev trace JSON files from logs/dev_traces/, identifies
common error patterns, and prints candidate rule records to stdout (or a
file).  Candidate rules must be reviewed and manually added to
memory/long_term_rules.json before they become part of the frozen store.

IMPORTANT
---------
* This script MUST NOT be called during official evaluation.
* Candidate output is not automatically written to long_term_rules.json.
* Do not store question text, gold answers, or answer_idx in rule records.
* Only distill general patterns (e.g. "retrieval misses drug timeline") into
  procedural rules.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Dev trace schema (for reference only — not enforced at runtime)
# ---------------------------------------------------------------------------
# {
#   "id": "dev_001",
#   "question": "...",          ← dev only, not stored in memory
#   "rewritten_query": "...",   ← dev only
#   "retrieved_sources": [...], ← dev only
#   "prediction": "C",          ← dev only
#   "gold_answer": "B",         ← dev only
#   "evaluation_verdict": "wrong",
#   "error_type": "retrieval_irrelevant"
# }

KNOWN_ERROR_TYPES = {
    "retrieval_irrelevant": (
        "retrieval_planner",
        "retrieval_strategy",
        "Include more specific keywords when the retrieved context does not match "
        "the clinical scenario. Prefer symptom + organ-system + mechanism combinations.",
    ),
    "reasoning_mechanism": (
        "reasoner",
        "procedural_rule",
        "Identify the core clinical mechanism before selecting an option. "
        "Avoid selecting options that are only tangentially related to the disease area.",
    ),
    "verification_missed_distractor": (
        "verifier",
        "verifier_checklist",
        "Verify that the selected option is specifically better than distractor options, "
        "not just medically plausible in isolation.",
    ),
    "format_multiple_answers": (
        "finalizer",
        "procedural_rule",
        "Return exactly one answer option label. Do not output multiple candidates.",
    ),
}


def load_traces(traces_dir: Path) -> list[dict]:
    traces: list[dict] = []
    for p in sorted(traces_dir.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, list):
                traces.extend(data)
            elif isinstance(data, dict):
                traces.append(data)
        except Exception as e:
            print(f"[WARN] Could not parse {p}: {e}", file=sys.stderr)
    return traces


def analyse_traces(traces: list[dict]) -> list[dict]:
    """Identify frequent error types and generate candidate rules."""
    error_counts: Counter[str] = Counter()
    for t in traces:
        et = t.get("error_type", "").strip()
        if et:
            error_counts[et] += 1

    candidate_rules: list[dict] = []
    for error_type, count in error_counts.most_common():
        if error_type not in KNOWN_ERROR_TYPES:
            print(
                f"[INFO] Unknown error_type={error_type!r} ({count} occurrences) — "
                "no candidate rule generated; review manually.",
                file=sys.stderr,
            )
            continue
        agent, memory_type, rule_text = KNOWN_ERROR_TYPES[error_type]
        candidate_id = f"{agent[:4]}_{error_type[:12]}_candidate"
        candidate_rules.append({
            "id": candidate_id,
            "agent": agent,
            "memory_type": memory_type,
            "topic": "general",
            "rule": rule_text,
            "source": "dev_error_analysis",
            "tags": [error_type, "candidate"],
            "confidence": round(min(0.5 + count * 0.05, 0.9), 2),
            "created_at": "REPLACE_WITH_DATE",
            "_dev_note": (
                f"Generated from {count} dev traces with error_type={error_type!r}. "
                "Review and edit before adding to long_term_rules.json."
            ),
        })

    return candidate_rules


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate candidate long-term memory rules from dev trace logs.",
        epilog="OFFLINE USE ONLY — do not run during official evaluation.",
    )
    parser.add_argument(
        "--traces-dir",
        type=Path,
        default=Path("logs/dev_traces"),
        help="Directory containing dev trace JSON files (default: logs/dev_traces)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file for candidate rules (default: stdout)",
    )
    args = parser.parse_args()

    if not args.traces_dir.exists():
        print(f"[ERROR] Traces directory does not exist: {args.traces_dir}", file=sys.stderr)
        sys.exit(1)

    traces = load_traces(args.traces_dir)
    print(f"[INFO] Loaded {len(traces)} dev trace records.", file=sys.stderr)

    if not traces:
        print("[INFO] No traces found. Nothing to analyse.", file=sys.stderr)
        return

    candidates = analyse_traces(traces)
    print(f"[INFO] Generated {len(candidates)} candidate rule(s).", file=sys.stderr)
    print("[WARN] Candidate rules are NOT automatically written to long_term_rules.json.", file=sys.stderr)
    print("[WARN] Review each rule and add manually after verification.", file=sys.stderr)

    output_json = json.dumps(candidates, indent=2, ensure_ascii=False)
    if args.output is not None:
        args.output.write_text(output_json, encoding="utf-8")
        print(f"[INFO] Candidate rules written to {args.output}", file=sys.stderr)
    else:
        print(output_json)


if __name__ == "__main__":
    main()
