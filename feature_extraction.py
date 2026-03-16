#!/usr/bin/env python3
"""Stage 3: Feature extraction — lexical symbols, path tokens, board IDs."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from structural_partitioning import PartitionedHunk


# ---------------------------------------------------------------------------
# Symbol extraction regexes
# ---------------------------------------------------------------------------

FUNC_DEF = re.compile(
    r'^\+\s*(?:static\s+|inline\s+|extern\s+)*'
    r'(?:(?:unsigned\s+|signed\s+|const\s+|volatile\s+|struct\s+\w+|enum\s+\w+|\w+)\s+[*\s]*)'
    r'(\w+)\s*\([^)]*\)\s*(?:\{|$)',
    re.MULTILINE,
)

TYPE_DEF = re.compile(
    r'^\+\s*(?:typedef\s+)?(?:struct|enum|union)\s+(\w+)\s*[\{;]',
    re.MULTILINE,
)

MACRO_DEF = re.compile(
    r'^\+\s*#\s*define\s+(\w+)',
    re.MULTILINE,
)

FUNC_CALL = re.compile(
    r'^\+.*\b(\w+)\s*\(',
    re.MULTILINE,
)

# ---------------------------------------------------------------------------
# Common kernel symbols to suppress (high-frequency / low-signal)
# ---------------------------------------------------------------------------

COMMON_KERNEL_SYMBOLS = {
    "static", "int", "void", "char", "long", "unsigned", "signed",
    "const", "struct", "enum", "union", "typedef", "return", "if",
    "else", "for", "while", "do", "switch", "case", "break", "continue",
    "goto", "NULL", "true", "false", "bool", "size_t", "u8", "u16",
    "u32", "u64", "s8", "s16", "s32", "s64", "uint8_t", "uint16_t",
    "uint32_t", "uint64_t", "int8_t", "int16_t", "int32_t", "int64_t",
    "printk", "pr_info", "pr_err", "pr_warn", "pr_debug", "dev_info",
    "dev_err", "dev_warn", "dev_dbg", "EXPORT_SYMBOL", "EXPORT_SYMBOL_GPL",
    "MODULE_LICENSE", "MODULE_AUTHOR", "MODULE_DESCRIPTION",
    "module_init", "module_exit", "MODULE_DEVICE_TABLE",
    "IS_ERR", "PTR_ERR", "ERR_PTR", "container_of", "ARRAY_SIZE",
    "BIT", "BITS_PER_LONG", "likely", "unlikely", "barrier",
    "min", "max", "clamp", "min_t", "max_t", "clamp_t",
    "kmalloc", "kfree", "kzalloc", "vmalloc", "vfree",
    "memset", "memcpy", "memmove", "memcmp", "strlen", "strcpy",
    "snprintf", "sprintf", "sscanf",
    "spin_lock", "spin_unlock", "mutex_lock", "mutex_unlock",
    "init_completion", "complete", "wait_for_completion",
    "of_device_id", "platform_device", "platform_driver",
    "device", "driver", "module",
}


def symbol_weight(symbol: str, occurrence_count: int) -> float:
    """Compute symbol weight. Returns 0.0 for common/stop-list symbols.

    Key insight: symbols appearing in 2-3 hunks are the strongest co-change signal
    and should be weighted *highest*, not lowest. A symbol shared by multiple hunks
    indicates those hunks should be grouped together.

    Weight curve:
    - 1 occurrence: 1.0 (unique, good signal)
    - 2-3 occurrences: 2.0 (prime co-grouping signal)
    - 4+ occurrences: 1.0 / log(1 + count) (dilutes as count grows)
    """
    if symbol in COMMON_KERNEL_SYMBOLS:
        return 0.0
    if len(symbol) < 3:
        return 0.0
    # Reward symbols appearing in 2-3 hunks (strong co-change signal)
    if occurrence_count in (2, 3):
        return 2.0
    # For rare symbols (1 occurrence) or common ones (4+), use log scaling
    return 1.0 / math.log(1 + occurrence_count)


# ---------------------------------------------------------------------------
# Board ID extraction
# ---------------------------------------------------------------------------

BOARD_ID_STOPLIST = {
    "common", "generic", "default", "base", "main", "core", "init",
    "config", "setup", "board", "device", "platform", "chip", "soc",
    "linux", "android", "kernel", "driver", "module", "system",
    "test", "debug", "release", "build", "src", "lib", "include",
    "arch", "arm", "arm64", "x86", "mips", "riscv",
    "samsung", "google", "qualcomm", "mediatek", "rockchip",
}

BOARD_ID_TIER = {
    # Tier 1: very high signal — specific chip/board identifiers
    "tier1": 3.0,
    # Tier 2: family identifiers
    "tier2": 2.0,
    # Tier 3: vendor/platform identifiers
    "tier3": 1.0,
}

# Heuristics for tier classification
_TIER1_RE = re.compile(
    r'^[a-z][a-z0-9]*(?:[_-][a-z0-9]+){2,}$|'   # e.g. sm8550-hdk, mt6983-v2
    r'^[a-z]{2,6}[0-9]{3,}[a-z]?$',              # e.g. sm8550, mt6983, g12a
    re.IGNORECASE,
)
_TIER2_RE = re.compile(
    r'^[a-z]{2,8}[0-9]{2,3}[a-z]{0,2}$',         # e.g. sdm845, mt6765
    re.IGNORECASE,
)


def _classify_board_token_tier(token: str) -> str:
    if _TIER1_RE.match(token):
        return "tier1"
    if _TIER2_RE.match(token):
        return "tier2"
    return "tier3"


@dataclass
class BoardToken:
    token: str
    tier: str
    weight: float


def extract_board_ids(file_path: str) -> list[BoardToken]:
    """Extract board ID tokens from a file path."""
    tokens: list[BoardToken] = []
    # Split path into parts and tokenize each part
    parts = file_path.replace("\\", "/").split("/")
    stem = Path(file_path).stem

    # Collect candidate tokens from path parts + stem segments
    candidates: set[str] = set()
    for part in parts:
        # Split on dash and underscore
        for tok in re.split(r'[-_]', part):
            tok_lower = tok.lower().strip()
            if tok_lower and len(tok_lower) >= 3 and tok_lower not in BOARD_ID_STOPLIST:
                candidates.add(tok_lower)

    # Also add full stem and full dir-name segments as candidates
    for part in parts:
        part_lower = part.lower()
        if (
            len(part_lower) >= 4
            and part_lower not in BOARD_ID_STOPLIST
            and not part_lower.endswith(('.c', '.h', '.S', '.s'))
        ):
            candidates.add(part_lower)

    for tok in candidates:
        # Filter: must be alphanumeric (with dashes/underscores), not pure numbers
        if not re.match(r'^[a-z][a-z0-9_-]*$', tok):
            continue
        if re.match(r'^\d+$', tok):
            continue
        tier = _classify_board_token_tier(tok)
        weight = BOARD_ID_TIER[tier]
        tokens.append(BoardToken(token=tok, tier=tier, weight=weight))

    return tokens


def extract_path_tokens(file_path: str, component_root: str) -> list[str]:
    """Extract meaningful path tokens relative to component_root."""
    fp = file_path.replace("\\", "/")
    if component_root and fp.startswith(component_root.rstrip("/") + "/"):
        fp = fp[len(component_root.rstrip("/")) + 1:]

    parts = fp.split("/")
    tokens: list[str] = []
    for part in parts:
        # Strip extensions
        stem = Path(part).stem if "." in part else part
        if stem and len(stem) >= 2:
            tokens.append(stem.lower())
        # Also add directory components as-is
        if part != parts[-1] and part:
            tokens.append(part.lower())

    return list(dict.fromkeys(tokens))  # deduplicate while preserving order


# ---------------------------------------------------------------------------
# HunkFeatures dataclass
# ---------------------------------------------------------------------------

@dataclass
class HunkFeatures:
    hunk_id: str
    file_path: str
    component: str
    pipeline: str
    is_new_file: bool
    is_dts: bool
    is_kconfig: bool
    is_makefile: bool
    is_defconfig: bool
    is_hal_interface: bool
    kernel_subsystem: Optional[str]
    dts_board_prefix: Optional[str]
    defconfig_board: Optional[str]
    # Extracted feature data
    weighted_symbols: dict[str, float]   # symbol -> weight
    path_tokens: list[str]
    board_ids: list[BoardToken]
    added_lines: int
    removed_lines: int
    net_lines: int
    raw_lines: list[str] = field(default_factory=list)
    bsp_subsystem: Optional[str] = None
    hunk_header: str = ""


def _count_occurrences(partitioned_hunks: list[PartitionedHunk]) -> dict[str, int]:
    """Count how many times each symbol appears across all hunks."""
    counts: dict[str, int] = {}
    for ph in partitioned_hunks:
        body = "\n".join(ph.lines)
        for sym in set(
            FUNC_DEF.findall(body)
            + TYPE_DEF.findall(body)
            + MACRO_DEF.findall(body)
        ):
            counts[sym] = counts.get(sym, 0) + 1
    return counts


def _extract_symbols_from_hunk(lines: list[str]) -> list[str]:
    """Extract symbol names from added lines of a hunk.

    Includes function definitions, type definitions, macro definitions, and function calls.
    Call-site symbols are weighted lower but provide strong co-change signals.
    """
    added_body = "\n".join(l for l in lines if l.startswith("+"))
    syms: list[str] = []
    syms.extend(FUNC_DEF.findall(added_body))
    syms.extend(TYPE_DEF.findall(added_body))
    syms.extend(MACRO_DEF.findall(added_body))
    # Add function calls (with marker so they can be weighted differently if needed)
    syms.extend(FUNC_CALL.findall(added_body))
    return syms


def extract_features(
    partitioned_hunks: list[PartitionedHunk],
    symbol_occurrence_counts: Optional[dict[str, int]] = None,
) -> list[HunkFeatures]:
    """Extract features from partitioned hunks.

    If symbol_occurrence_counts is None, performs a first pass to count.
    """
    if symbol_occurrence_counts is None:
        symbol_occurrence_counts = _count_occurrences(partitioned_hunks)

    results: list[HunkFeatures] = []

    for ph in partitioned_hunks:
        # Count added/removed lines
        added_lines = sum(1 for l in ph.lines if l.startswith("+"))
        removed_lines = sum(1 for l in ph.lines if l.startswith("-"))
        net_lines = added_lines - removed_lines

        # Extract and weight symbols
        raw_syms = _extract_symbols_from_hunk(ph.lines)
        weighted: dict[str, float] = {}
        for sym in raw_syms:
            count = symbol_occurrence_counts.get(sym, 1)
            w = symbol_weight(sym, count)
            if w > 0.0:
                weighted[sym] = max(weighted.get(sym, 0.0), w)

        # Path tokens
        component_root = ""
        if ph.component != "kernel":
            # Use the first matching BSP prefix as component root
            from structural_partitioning import BSP_COMPONENTS
            for prefix, comp, _ in BSP_COMPONENTS:
                if comp == ph.component:
                    component_root = prefix.rstrip("/")
                    break
        path_tokens = extract_path_tokens(ph.file_path, component_root)

        # Board IDs
        board_ids = extract_board_ids(ph.file_path)

        results.append(
            HunkFeatures(
                hunk_id=ph.hunk_id,
                file_path=ph.file_path,
                component=ph.component,
                pipeline=ph.pipeline,
                is_new_file=ph.is_new_file,
                is_dts=ph.is_dts,
                is_kconfig=ph.is_kconfig,
                is_makefile=ph.is_makefile,
                is_defconfig=ph.is_defconfig,
                is_hal_interface=ph.is_hal_interface,
                kernel_subsystem=ph.kernel_subsystem,
                dts_board_prefix=ph.dts_board_prefix,
                defconfig_board=ph.defconfig_board,
                weighted_symbols=weighted,
                path_tokens=path_tokens,
                board_ids=board_ids,
                added_lines=added_lines,
                removed_lines=removed_lines,
                net_lines=net_lines,
                raw_lines=list(ph.lines),
                bsp_subsystem=ph.bsp_subsystem,
                hunk_header=ph.hunk_header,
            )
        )

    return results
