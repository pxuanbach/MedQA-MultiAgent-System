# CLAUDE.md

You are working in a repository designed for long-running implementation work.
Prioritize reliable completion, continuity across sessions, and explicit
verification over speed.

## Operating Loop

At the start of every session:

1. Run `pwd` and confirm you are in the expected repository root.
2. Read `goal.md` to understand the current project goal and expected output.
3. Read `claude-progress.md`.
4. Read `feature_list.json`.
5. Review recent commits with `git log --oneline -5`.
6. Run `./init.ps1`.
7. Check whether the baseline smoke or end-to-end path is already broken.

Then select exactly one unfinished feature and work only on that feature until
you either verify it or document why it is blocked.

## Rules

- One active feature at a time.
- Do not claim completion without runnable evidence.
- Do not rewrite the feature list to hide unfinished work.
- Do not remove or weaken tests just to make the task look complete.
- Use repository artifacts as the system of record.

## Required Files

- `goal.md`: project goal and expected output
- `feature_list.json`
- `claude-progress.md`
- `init.ps1`
- `session-handoff.md` when a compact handoff is useful

## Completion Gate

A feature can move to `passing` only after the required verification succeeds
and the result is recorded.

## Before You Stop

1. Update the progress log.
2. Update the feature state.
3. Record what is still broken or unverified.
4. Commit once the repository is safe to resume.
5. Leave a clean restart path for the next session.