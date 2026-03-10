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

## 4) Structural Pipeline (Stages 1–4)

The structural pipeline runs before any LLM calls. It is fully deterministic.

### Stage 1 — Repo Analysis (`repo_analysis.py`)

Detects the repo layout from the diff's file path list:

- **`FULL_BSP`** — kernel lives in a subdirectory (e.g. `kernel/`); Android BSP layers present
- **`KERNEL_ONLY`** — entire repo is a kernel (GKI-style); no kernel subdirectory
- **`UNKNOWN`** — detection score below threshold; structural pipeline still runs best-effort

Kernel root detection scores candidate roots at depths 0–4 using `KERNEL_FINGERPRINTS` (weighted path patterns: `arch/`, `include/linux/`, `drivers/`, etc.) with a depth penalty to prevent nested false positives. Candidate roots containing `ARTIFACT_PATH_SEGMENTS` (`tmp-ws`, `goldfish`, `out`, etc.) are suppressed before scoring.

If running inside a git repo, optionally extracts release version and date from the HEAD commit subject (e.g. `firetvcube-6.2.9.4-20221228`) for use as context in the LLM stage. Skipped silently if unavailable.

CLI flags: `--kernel-root`, `--no-release-context`

### Stage 2 — Structural Partitioning (`structural_partitioning.py`)

Hard-separates the diff into independent component streams by path prefix. No model calls.

In `FULL_BSP` mode, files are routed to components by first-prefix-wins matching:

| Prefix | Component | Pipeline |
|--------|-----------|----------|
| `bootable/` | bootloader | simple |
| `prebuilts/` | prebuilts | skip |
| `external/` | external | skip |
| `tmp-ws/` | artifacts | skip |
| `hardware/` | hardware_hal | simple |
| `vendor/` | vendor | simple |
| `frameworks/` | frameworks | simple |
| `fireos/` | oem_layer | simple |
| `device/` | device | simple |
| `packages/` | packages | simple |
| `system/` | system | simple |
| *(kernel root)* | kernel | full |

Pipeline meanings:
- `full` — kernel stream; all stages run with subsystem-aware logic
- `simple` — non-kernel stream; structural bundling with directory locality
- `skip` — excluded; logged in export output

In `KERNEL_ONLY` mode, all files receive the `full` kernel pipeline (including `vendor/`, which contains kernel vendor modules in this layout).

Within the kernel stream, files are further classified: new-file additions, DTS/DTSI, defconfig, Kconfig, Makefile/Kbuild, HAL interfaces. Kconfig and Makefile hunks are tagged for pairing using CONFIG symbol extraction:

```python
extract_config_symbols_defined()   # r'^\+config\s+([\w]+)'
extract_config_symbols_referenced() # r'\$\(CONFIG_([\w]+)\)'
```

CLI flags: `--skip-components`

### Stage 3 — Feature Extraction (`feature_extraction.py`)

Extracts lightweight signals from each hunk without AST parsing.

**Lexical symbol extraction** from added lines using targeted regexes for kernel C: function definitions, type definitions (`struct`/`enum`/`typedef`), macro definitions (`#define`), and call sites. A stoplist (`COMMON_KERNEL_SYMBOLS`) suppresses pervasive symbols (`dev_err`, `kzalloc`, `mutex_lock`, etc.) that would create false bridges. Remaining symbols are weighted by inverse log frequency — a symbol shared between 2 hunks out of 500 scores much higher than one shared between 200.

**Board ID extraction** from file stems and parent directory names. Tokens are assigned to specificity tiers: `board` (e.g. `p212`), `soc` (e.g. `s905x`, `g12a`), `vendor` (e.g. `amlogic`), `family` (e.g. `meson`). Higher-tier tokens provide stronger grouping signal. A stoplist suppresses generic tokens (`arm64`, `generic`, `v1`, etc.).

**Path tokens** — directory components below the component root, used as a soft grouping signal.

### Stage 4 — Candidate Bundling (`candidate_bundling.py`)

Builds candidate commit bundles using greedy anchor-seeded scoring. No model calls.

**Anchor seeds** are created first from high-confidence structural groups:
1. New-file subsystem groups (by `kernel_subsystem` for kernel, by parent dir for simple streams)
2. Kconfig/Makefile-paired sets matched via shared CONFIG symbols
3. DTS board groups (by board prefix from filename stem)
4. Defconfig groups (by board name)
5. HAL interface groups (by package path)

Large anchors (>20 files) are sub-split by filename prefix to prevent over-merging when a vendor imports multiple unrelated drivers into one directory.

**Greedy attachment** iterates unassigned hunks (build-wiring first, modifications last) and scores each against all existing bundles. Hunks scoring above `ATTACHMENT_THRESHOLD` (default: 4) are attached; those that find no home become singleton bundles flagged low-confidence.

**Scoring signals:**

| Signal | Weight |
|--------|--------|
| Shared subsystem | +6 |
| Shared board token (tier1) | +5 per token |
| Shared symbols (rarity-weighted) | +4 (up to ×5) |
| Shared directory | +3 |
| Kconfig/Makefile pairing | +5 |
| Shared DTS board prefix | +4 |
| Shared defconfig board | +4 |
| Shared HAL package | +4 |
| Shared component baseline | +2 |

**Co-change anchor locking** — files that appear together in the same directory within the diff are treated as likely partners. When a hunk joins a bundle already containing its directory partners, a bonus of up to +6 is applied, specificity-weighted by `1/log(dir_size)`. Small tight clusters (driver + header pair) score much higher than large noisy directories. A +2 dir-level bonus applies when the hunk's parent directory is a sibling of a bundle directory (same grandparent).

**Negative board-target affinity** — when a hunk's board tokens conflict with a bundle's board tokens (non-empty, non-overlapping) and there is no shared driver subsystem, a `-12` penalty is applied. If they conflict but do share a subsystem (driver core is common across boards), a softer `-3` penalty applies instead. This prevents the common bad merge of "same vendor, same subsystem, different product."

**Wiring-to-payload asymmetry** — wiring files (Kconfig, Makefile, Kbuild, defconfig, Android.mk, Android.bp) can attach to payload bundles but must not bridge unrelated payload groups:
- Wiring hunk → bundle with matching payload (same dir or subsystem): `+6`
- Wiring hunk → bundle with mismatched payload: `-3`
- Wiring hunk → wiring-only bundle: `-3`

**Hard vetoes** (return `-inf`, no positive signals can override):
- Component mismatch
- DTS hunk joining non-DTS bundle (unless Kconfig-paired)
- Defconfig joining bundle with different board
- Generated file bridge (`.pb.h`, `.autogen.*`) used as connector
- Binary blob bridge (`.bin`, `.fw`, `.dtb`) used as connector

**Preliminary taxonomy labels** are assigned deterministically before the LLM sees anything:
`feat: add`, `feat: DTS board support`, `feat: defconfig`, `feat: HAL interface`, `feat: Kconfig`, `feat: import`, `fix: remove`, `chore: update`.

**Export mode** — pass `--export-bundles` to exit after Stage 4 and write a structured report (markdown or JSON) without making any LLM calls or modifying git state. The report includes a diff-level confidence score, per-component bundle summaries, and a multi-release warning if scope is unusually broad (>47 subsystems or >89 bundles).

CLI flags: `--attachment-threshold`, `--max-anchor-breadth`, `--export-bundles`, `--export-format`, `--export-file`

## 5) Stage 5 — Embedding Refinement (`embedding_refinement.py`) [optional]

For bundles flagged low-confidence by Stage 4, uses `voyage-code-3` (Voyage AI) to embed bundle summaries and compute cosine similarity. Bundles in the same component stream that exceed a merge threshold (default: 0.75) are proposed for merging and flagged for LLM confirmation in Stage 6.

Skipped silently if `voyageai` is not installed. Requires `VOYAGE_API_KEY`.

CLI flags: `--embed on|off|auto`, `--merge-threshold`

## 6) Stage 6 — LLM Arbitration

The LLM operates on structured bundle summaries, not raw diff hunks.

- **High-confidence bundles**: LLM generates commit message only. No grouping decision.
- **Low-confidence / singleton bundles**: LLM makes split-or-merge decision and generates commit message.
- **Embedding-proposed merges**: LLM confirms or rejects proposed bundle merges from Stage 5.

If release context was extracted in Stage 1, it is prepended to all prompts: *"This diff represents vendor release 7.6.2.4 dated 2023-03-03."*

Non-kernel (`simple`) streams use a two-pass approach: Pass 1 classifies bundles into broad groups (cheap); Pass 2 generates commit messages per group.

Group count may be capped with `--max-generated-commits`. Optional commit body behavior: `off`, `auto`, `always`.

## 7) Interactive Review Loop

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

## 8) Execution

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

## 9) Safety and Failure Handling

- Tree verification guards against silent loss/regression.
- Group integrity checks catch assignment bugs.
- Dry-run mode allows preview without changing git state.
- Explicit process exits on patch/apply failures.
- Structural engine import failures degrade gracefully (`--no-structural` falls back to tree-sitter + AI grouping pipeline).
