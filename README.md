# git-split

`git-split` breaks a large diff into smaller, logical commits with AI-assisted grouping and message generation.

## Requirements

- Python 3.10+
- Git
- `ANTHROPIC_API_KEY`

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

PowerShell:

```powershell
$env:ANTHROPIC_API_KEY="your_key_here"
```

You can also place it in `.env` (see `.env.example`).

Automation defaults:

- `GIT_SPLIT_AUTO_ACCEPT=true` skips the interactive review loop.
- Set `GIT_SPLIT_AUTO_ACCEPT=false` to force manual interactive review.

## Usage

```bash
python git_split.py split
python git_split.py split --unstaged
python git_split.py rebase HEAD~2
python git_split.py check
```

Common flags:

- `--dry-run`
- `--granularity adaptive|fine`
- `--line-safety conservative|balanced|aggressive`
- `--message-body off|auto|always`
- `--max-generated-commits N`
- `--force-bulk`

## Safety Model

- Validates group integrity (no missing/duplicate hunks)
- Supports dry-run for split and rebase flows
- Verifies final Git tree to prevent data loss during split/rewrite

## Architecture

See `PIPELINE.md` for the full pipeline and execution model.
