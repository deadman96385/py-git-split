# git-split Pipeline and Architecture

This document captures the logic pipeline used by `git_split.py`.

## 1) Entry and Environment

1. CLI starts through Click (`split`, `rebase`, `check`).
2. `.env` values are loaded from:
   - current working directory
   - script directory
3. API key resolution order:
   - `--api-key`
   - `ANTHROPIC_API_KEY` environment variable

## 2) Diff Acquisition

`split` mode:
- staged: `git diff --cached --unified=3`
- unstaged: `git diff --unified=3`

`rebase` mode:
- target commit diff: `git show --unified=3 <commit>`
- original commit message used as context

## 3) Parse to Units

1. Raw patch is parsed into:
   - per-file headers
   - hunk objects (`id`, file path, header, line payload, index)
2. Summary stats are computed (files, hunks, add/remove counts, add ratio).

## 4) Split Strategy Selection

Two paths:

- Context split + AI grouping (default):
  - optional tree-sitter-assisted sub-hunk splitting (`adaptive` or `fine`)
  - in `rebase` mode, tree-sitter reads file content from the target commit (not the working tree)
  - then AI decides grouping
- Bulk mode (`--force-bulk` or auto trigger on very large addition-heavy diffs):
  - deterministic grouping by file/hunk volume limits

## 5) Message Generation

1. Group count may be capped (`--max-generated-commits`).
2. Commit messages are generated:
   - AI generation in normal flow
   - hybrid deterministic + AI in bulk flow
3. Optional commit body behavior:
   - `off`, `auto`, `always`

## 6) Interactive Review Loop

Controlled by `GIT_SPLIT_AUTO_ACCEPT` env var (default: `false`):
- When `false` (default): interactive review loop is shown
- When `true`: groups are auto-accepted without prompting (integrity still verified)

When interactive, user can:
- accept
- move a hunk between groups
- edit commit message
- re-run AI grouping
- abort

Integrity is rechecked before acceptance.

## 7) Execution

### Split execution

1. Reset index (`git reset HEAD`).
2. For each group:
   - apply patch hunks to index, or stage files for bulk groups
   - commit with message/body
3. Verify resulting tree matches expected tree.

### Rebase execution

1. Build a script that:
   - resets target commit (`git reset HEAD^`)
   - replays each planned group as commit(s)
   - continues rebase
2. Optionally auto-launch interactive rebase edit + script execution.
   - If the rebase setup step fails, execution stops and the manual run command is printed.
3. Verify resulting tree matches expected target tree.

## 8) Safety and Failure Handling

- Tree verification guards against silent loss/regression.
- Group integrity checks catch assignment bugs.
- Dry-run mode allows preview without changing git state.
- Explicit process exits on patch/apply failures.
