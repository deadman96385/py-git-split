# git-split

`git-split` breaks a large diff into smaller, logical commits. Designed for vendor BSP and kernel drops, it uses a deterministic structural pipeline to partition and bundle changes before calling the LLM — so the model only arbitrates what the structure couldn't resolve.

## Requirements

- Python 3.10+
- Git
- `ANTHROPIC_API_KEY`
- `VOYAGE_API_KEY` (optional — enables embedding refinement for ambiguous bundles)

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows PowerShell activation:

```powershell
.venv\Scripts\Activate.ps1
```

Set API key:

```bash
export ANTHROPIC_API_KEY=your_key_here
```

You can also place it in `.env` (see `.env.example`).

Automation defaults:

- `GIT_SPLIT_AUTO_ACCEPT=true` skips the interactive review loop.

## Usage

```bash
python git_split.py split
python git_split.py split --unstaged
python git_split.py rebase HEAD~2
python git_split.py check
```

Common flags:

```
--dry-run
--message-body off|auto|always
--max-generated-commits N
--export-bundles            # inspect structural output before any LLM call
--export-format md|json
--kernel-root <path>        # manual kernel root override
--no-release-context        # skip HEAD commit version/date extraction
--skip-components <list>    # comma-separated component IDs to exclude
--attachment-threshold N    # affinity score to join a bundle (default: 4)
--max-anchor-breadth N      # max files per anchor before sub-splitting (default: 20)
```

## How it works

The structural pipeline runs fully deterministically before any model call:

1. **Repo analysis** — detects `FULL_BSP` vs `KERNEL_ONLY` layout; extracts release version/date from HEAD commit if present
2. **Structural partitioning** — hard-separates the diff by component (`kernel`, `hardware`, `vendor`, `bootloader`, etc.) using path prefix rules; classifies DTS, defconfig, Kconfig, Makefile, HAL interfaces
3. **Feature extraction** — lexical symbol extraction with rarity weighting; board/SoC token tiering; path token extraction
4. **Candidate bundling** — greedy anchor-seeded bundling with scored affinity signals, co-change locking, negative board-target affinity, wiring-to-payload asymmetry, and hard vetoes; preliminary commit labels assigned before LLM
5. **LLM arbitration** — model sees structured bundle summaries, not raw hunks; generates commit messages and resolves ambiguous splits

Use `--export-bundles` to inspect the structural output and exit before any LLM call.

## Safety Model

- Validates group integrity (no missing/duplicate hunks)
- Supports dry-run for split and rebase flows
- Verifies final Git tree to prevent data loss during split/rewrite

## Architecture

See `PIPELINE.md` for the full pipeline and execution model.
