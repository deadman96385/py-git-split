#!/usr/bin/env python3
"""Stage 4: Candidate bundling — greedy anchor-seeded bundling with hard vetoes."""
from __future__ import annotations

import json
import math
import re
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from repo_analysis import ReleaseContext
from feature_extraction import HunkFeatures


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BUNDLE_SCORE_WEIGHTS = {
    "shared_subsystem":          6.0,
    "shared_board_tier1":        5.0,
    "shared_symbols":            4.0,
    "shared_path_prefix":        3.0,
    "shared_board_tier2":        3.0,
    "kconfig_makefile_pair":     5.0,
    "shared_dts_board":          4.0,
    "shared_defconfig":          4.0,
    "shared_hal_package":        4.0,
    "shared_component":          2.0,
    "shared_path_tokens":        1.5,
    "same_new_file_group":       2.0,
    # Co-change anchor locking
    "cochange_file_bonus":       4.0,   # file is a known partner of files already in bundle
    "cochange_dir_bonus":        2.0,   # parent dir is a sibling dir of a bundle dir
    # Board target conflict
    "cross_board_penalty":       -3.0,  # conflicting boards but shared driver core
    "conflicting_board_penalty": -12.0, # conflicting boards with no shared driver core
    # Wiring-to-payload asymmetry
    "wiring_payload_match_bonus":  6.0, # wiring hunk attaches to bundle with matching payload
    "wiring_no_payload_penalty":  -3.0, # wiring hunk, no matching payload or wiring-only bundle
    # Hard cross-component
    "cross_component_penalty":   -8.0,
}

ATTACHMENT_THRESHOLD = 4
MAX_ANCHOR_BREADTH = 20

# Hard-veto file-extension patterns
_GENERATED_RE = re.compile(r'\.(pb\.h|pb\.cc|autogen\.[^/]+)$')
_BINARY_BLOB_RE = re.compile(r'\.(bin|fw|dtb)$')


# ---------------------------------------------------------------------------
# Bundle dataclass
# ---------------------------------------------------------------------------

@dataclass
class Bundle:
    bundle_id: str
    hunks: list[HunkFeatures]
    confidence: str                         # "high" | "low"
    preliminary_label: str
    component: str
    release_context: Optional[ReleaseContext] = None
    confidence_score: float = 0.0

    def add(self, hunk: HunkFeatures, score: float) -> None:
        self.hunks.append(hunk)
        # Update running average confidence score
        n = len(self.hunks)
        self.confidence_score = (self.confidence_score * (n - 1) + score) / n

    def summary(self) -> str:
        files = sorted({h.file_path for h in self.hunks})
        file_names = [Path(f).name for f in files]
        net_lines = sum(h.net_lines for h in self.hunks)
        added_lines = sum(h.added_lines for h in self.hunks)

        # Subsystem
        subsystems = {h.kernel_subsystem for h in self.hunks if h.kernel_subsystem}
        subsystem_str = ", ".join(sorted(subsystems)) if subsystems else self.component

        # Parent dir for display
        dirs = {str(Path(f).parent) for f in files}
        primary_dir = sorted(dirs, key=lambda d: -sum(1 for f in files if str(Path(f).parent) == d))[0] if dirs else ""

        # Symbols
        all_syms: dict[str, float] = {}
        for h in self.hunks:
            for sym, w in h.weighted_symbols.items():
                all_syms[sym] = all_syms.get(sym, 0.0) + w
        top_syms = sorted(all_syms, key=lambda s: -all_syms[s])[:5]
        extra_syms = max(0, len(all_syms) - 5)
        sym_str = ", ".join(top_syms)
        if extra_syms:
            sym_str += f" ({extra_syms} more)"

        # Board IDs
        board_tokens: dict[str, float] = {}
        for h in self.hunks:
            for bt in h.board_ids:
                board_tokens[bt.token] = board_tokens.get(bt.token, 0.0) + bt.weight
        top_boards = sorted(board_tokens, key=lambda t: -board_tokens[t])[:5]
        board_str = ", ".join(top_boards) if top_boards else ""

        # Kconfig pair
        kconfig_hunks = [h for h in self.hunks if h.is_kconfig or h.is_makefile]
        kconfig_syms: list[str] = []
        for h in kconfig_hunks:
            kconfig_syms.extend(h.weighted_symbols.keys())
        kconfig_str = ", ".join(f"CONFIG_{s}" for s in kconfig_syms[:3]) if kconfig_syms else ""

        lines = [
            f"Bundle {self.bundle_id} [{self.preliminary_label}] {primary_dir}/  confidence: {self.confidence}",
            f"  Files ({len(files)}): {', '.join(file_names[:6])}{'...' if len(file_names) > 6 else ''}",
        ]
        if subsystem_str:
            lines.append(f"  Subsystem: {subsystem_str}")
        if sym_str:
            lines.append(f"  Symbols defined: {sym_str}")
        if board_str:
            lines.append(f"  Board IDs: {board_str}")
        lines.append(f"  Net lines: +{added_lines} | Hunk count: {len(self.hunks)}")
        if kconfig_str:
            lines.append(f"  Kconfig pair: {kconfig_str}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Preliminary label assignment
# ---------------------------------------------------------------------------

def assign_preliminary_label(bundle: Bundle) -> str:
    """Assign a preliminary commit label from bundle heuristics."""
    hunks = bundle.hunks
    if not hunks:
        return "chore: update"

    total_added = sum(h.added_lines for h in hunks)
    total_removed = sum(h.removed_lines for h in hunks)
    new_files = sum(1 for h in hunks if h.is_new_file)
    all_new = new_files == len(hunks)
    has_dts = any(h.is_dts for h in hunks)
    has_defconfig = any(h.is_defconfig for h in hunks)
    has_kconfig = any(h.is_kconfig for h in hunks)
    has_hal = any(h.is_hal_interface for h in hunks)
    component = bundle.component

    subsystems = {h.kernel_subsystem for h in hunks if h.kernel_subsystem}
    primary_subsystem = sorted(subsystems, key=lambda s: -sum(1 for h in hunks if h.kernel_subsystem == s))[0] if subsystems else None

    # Determine scope string
    if primary_subsystem:
        scope = primary_subsystem.replace("/", "-")
    elif component:
        scope = component.replace("_", "-")
    else:
        scope = "repo"

    if has_dts:
        board_prefixes = {h.dts_board_prefix for h in hunks if h.dts_board_prefix}
        bp = sorted(board_prefixes)[0] if board_prefixes else scope
        if all_new:
            return f"feat({bp}): add DTS board support"
        return f"feat({bp}): update DTS"

    if has_defconfig:
        boards = {h.defconfig_board for h in hunks if h.defconfig_board}
        board_str = sorted(boards)[0] if boards else scope
        if all_new:
            return f"feat({board_str}): add defconfig"
        return f"feat({board_str}): update defconfig"

    if has_hal:
        if all_new:
            return f"feat({scope}): add HAL interface"
        return f"feat({scope}): update HAL interface"

    if has_kconfig:
        if all_new:
            return f"feat({scope}): add Kconfig entries"
        return f"feat({scope}): update Kconfig"

    if all_new and total_added > 0:
        if component in ("prebuilts", "external"):
            return f"feat({scope}): import"
        return f"feat({scope}): add"

    if total_removed == 0 and total_added > 0:
        return f"feat({scope}): add"

    if total_added == 0 and total_removed > 0:
        return f"fix({scope}): remove"

    return f"chore({scope}): update"


# ---------------------------------------------------------------------------
# Co-change index
# ---------------------------------------------------------------------------

def build_cochange_index(features: list[HunkFeatures]) -> dict[str, set[str]]:
    """Build file → set of co-present files in the same directory across the diff.

    Used to give a bonus when a candidate hunk joins a bundle that already contains
    its common partners (e.g. driver source + local header, DTS + defconfig).
    """
    by_dir: dict[str, list[str]] = defaultdict(list)
    for h in features:
        by_dir[str(Path(h.file_path).parent)].append(h.file_path)
    file_partners: dict[str, set[str]] = {}
    for files in by_dir.values():
        s = set(files)
        for f in files:
            file_partners[f] = s - {f}
    return file_partners


def _is_wiring(hunk: HunkFeatures) -> bool:
    """True if the hunk is a wiring file: Kconfig, Makefile, defconfig, Android mk/bp."""
    fp = hunk.file_path.lower()
    return (
        hunk.is_kconfig
        or hunk.is_makefile
        or hunk.is_defconfig
        or fp.endswith("android.mk")
        or fp.endswith("android.bp")
        or (fp.endswith(".mk") and "android" in fp)
    )


# ---------------------------------------------------------------------------
# Affinity scoring
# ---------------------------------------------------------------------------

def score_affinity(
    hunk: HunkFeatures,
    bundle: Bundle,
    weights: dict,
    cochange_index: Optional[dict] = None,
) -> float:
    """Compute affinity score between a hunk and an existing bundle.

    Applies hard vetoes (returns -inf) and soft penalties.
    """
    if not bundle.hunks:
        return 0.0

    # --- Hard vetoes ---

    # 1. Component mismatch
    if hunk.component != bundle.component:
        return float("-inf")

    # 2. DTS non-board merge: DTS hunk must share board prefix with bundle
    if hunk.is_dts:
        bundle_boards = {h.dts_board_prefix for h in bundle.hunks if h.dts_board_prefix}
        if bundle_boards and hunk.dts_board_prefix not in bundle_boards:
            return float("-inf")

    # 3. Defconfig unrelated: defconfig board must match
    if hunk.is_defconfig:
        bundle_defconfigs = {h.defconfig_board for h in bundle.hunks if h.defconfig_board}
        if bundle_defconfigs and hunk.defconfig_board not in bundle_defconfigs:
            return float("-inf")

    # 4. Generated file bridge: don't bridge .pb.h/.autogen files with non-generated
    hunk_is_generated = bool(_GENERATED_RE.search(hunk.file_path))
    bundle_has_generated = any(_GENERATED_RE.search(h.file_path) for h in bundle.hunks)
    bundle_has_normal = any(not _GENERATED_RE.search(h.file_path) for h in bundle.hunks)
    if hunk_is_generated and bundle_has_normal and not bundle_has_generated:
        return float("-inf")
    if not hunk_is_generated and bundle_has_generated and not bundle_has_normal:
        return float("-inf")

    # 5. Binary blob bridge
    hunk_is_binary = bool(_BINARY_BLOB_RE.search(hunk.file_path))
    bundle_has_binary = any(_BINARY_BLOB_RE.search(h.file_path) for h in bundle.hunks)
    bundle_has_nonbinary = any(not _BINARY_BLOB_RE.search(h.file_path) for h in bundle.hunks)
    if hunk_is_binary and bundle_has_nonbinary and not bundle_has_binary:
        return float("-inf")
    if not hunk_is_binary and bundle_has_binary and not bundle_has_nonbinary:
        return float("-inf")

    # --- Positive scoring ---
    score = 0.0

    # Shared subsystem
    if hunk.kernel_subsystem:
        bundle_subsystems = {h.kernel_subsystem for h in bundle.hunks if h.kernel_subsystem}
        if hunk.kernel_subsystem in bundle_subsystems:
            score += weights.get("shared_subsystem", BUNDLE_SCORE_WEIGHTS["shared_subsystem"])

    # Shared DTS board
    if hunk.is_dts and hunk.dts_board_prefix:
        bundle_dts = {h.dts_board_prefix for h in bundle.hunks if h.dts_board_prefix}
        if hunk.dts_board_prefix in bundle_dts:
            score += weights.get("shared_dts_board", BUNDLE_SCORE_WEIGHTS["shared_dts_board"])

    # Shared defconfig board
    if hunk.is_defconfig and hunk.defconfig_board:
        bundle_dc = {h.defconfig_board for h in bundle.hunks if h.defconfig_board}
        if hunk.defconfig_board in bundle_dc:
            score += weights.get("shared_defconfig", BUNDLE_SCORE_WEIGHTS["shared_defconfig"])

    # Shared board IDs (tier1)
    hunk_tier1 = {bt.token for bt in hunk.board_ids if bt.tier == "tier1"}
    bundle_tier1 = {bt.token for h in bundle.hunks for bt in h.board_ids if bt.tier == "tier1"}
    shared_t1 = hunk_tier1 & bundle_tier1
    if shared_t1:
        score += weights.get("shared_board_tier1", BUNDLE_SCORE_WEIGHTS["shared_board_tier1"]) * len(shared_t1)

    # Shared board IDs (tier2)
    hunk_tier2 = {bt.token for bt in hunk.board_ids if bt.tier == "tier2"}
    bundle_tier2 = {bt.token for h in bundle.hunks for bt in h.board_ids if bt.tier == "tier2"}
    shared_t2 = hunk_tier2 & bundle_tier2
    if shared_t2:
        score += weights.get("shared_board_tier2", BUNDLE_SCORE_WEIGHTS["shared_board_tier2"]) * len(shared_t2)

    # Shared symbols
    hunk_syms = set(hunk.weighted_symbols.keys())
    bundle_syms = {sym for h in bundle.hunks for sym in h.weighted_symbols}
    shared_syms = hunk_syms & bundle_syms
    if shared_syms:
        score += weights.get("shared_symbols", BUNDLE_SCORE_WEIGHTS["shared_symbols"]) * min(len(shared_syms), 5)

    # Kconfig/Makefile pair
    if hunk.is_kconfig or hunk.is_makefile:
        bundle_has_kconfig = any(h.is_kconfig for h in bundle.hunks)
        bundle_has_makefile = any(h.is_makefile for h in bundle.hunks)
        if (hunk.is_kconfig and bundle_has_makefile) or (hunk.is_makefile and bundle_has_kconfig):
            score += weights.get("kconfig_makefile_pair", BUNDLE_SCORE_WEIGHTS["kconfig_makefile_pair"])

    # Shared path prefix
    hunk_dir = str(Path(hunk.file_path).parent)
    bundle_dirs = {str(Path(h.file_path).parent) for h in bundle.hunks}
    if hunk_dir in bundle_dirs:
        score += weights.get("shared_path_prefix", BUNDLE_SCORE_WEIGHTS["shared_path_prefix"])

    # Shared path tokens
    hunk_ptoks = set(hunk.path_tokens)
    bundle_ptoks = {tok for h in bundle.hunks for tok in h.path_tokens}
    shared_ptoks = hunk_ptoks & bundle_ptoks
    if shared_ptoks:
        score += weights.get("shared_path_tokens", BUNDLE_SCORE_WEIGHTS["shared_path_tokens"]) * min(len(shared_ptoks), 4)

    # HAL interface shared package
    if hunk.is_hal_interface:
        hunk_hal_pkg = str(Path(hunk.file_path).parent)
        bundle_hal_pkgs = {str(Path(h.file_path).parent) for h in bundle.hunks if h.is_hal_interface}
        if hunk_hal_pkg in bundle_hal_pkgs:
            score += weights.get("shared_hal_package", BUNDLE_SCORE_WEIGHTS["shared_hal_package"])

    # Shared component (baseline)
    score += weights.get("shared_component", BUNDLE_SCORE_WEIGHTS["shared_component"])

    # --- Feature 1: Co-change anchor locking ---
    # Bonus when a candidate hunk joins a bundle already containing its common partners.
    # Specificity-weighted: small directory = stronger signal.
    if cochange_index:
        bundle_files = {h.file_path for h in bundle.hunks}
        hunk_partners = cochange_index.get(hunk.file_path, set())
        partner_overlap = hunk_partners & bundle_files
        if partner_overlap:
            dir_size = len(hunk_partners) + 1  # total files in this dir in the diff
            specificity = 1.0 / math.log1p(dir_size)
            cochange_bonus = weights.get("cochange_file_bonus", BUNDLE_SCORE_WEIGHTS["cochange_file_bonus"])
            score += min(cochange_bonus * len(partner_overlap) * specificity, 6.0)

        # Dir co-change: hunk's parent dir is a sibling of a bundle dir (same grandparent)
        hunk_dir = str(Path(hunk.file_path).parent)
        hunk_grandparent = str(Path(hunk_dir).parent)
        bundle_dirs_set = {str(Path(h.file_path).parent) for h in bundle.hunks}
        for bd in bundle_dirs_set:
            if str(Path(bd).parent) == hunk_grandparent and bd != hunk_dir:
                score += weights.get("cochange_dir_bonus", BUNDLE_SCORE_WEIGHTS["cochange_dir_bonus"])
                break

    # --- Feature 2: Negative board-target affinity ---
    # Conflicting board tokens (non-empty, non-overlapping) → strong penalty unless
    # there is explicit shared-driver evidence (shared subsystem).
    if hunk_tier1 and bundle_tier1 and not shared_t1:
        bundle_subsystems_set = {h.kernel_subsystem for h in bundle.hunks if h.kernel_subsystem}
        has_shared_driver = bool(hunk.kernel_subsystem and hunk.kernel_subsystem in bundle_subsystems_set)
        if has_shared_driver:
            score += weights.get("cross_board_penalty", BUNDLE_SCORE_WEIGHTS["cross_board_penalty"])
        else:
            score += weights.get("conflicting_board_penalty", BUNDLE_SCORE_WEIGHTS["conflicting_board_penalty"])

    # --- Feature 3: Wiring-to-payload asymmetry ---
    # Wiring hunks (Kconfig, Makefile, defconfig, Android mk/bp) should attach to
    # payload bundles, but must not bridge unrelated payload groups.
    if _is_wiring(hunk):
        bundle_payload_hunks = [h for h in bundle.hunks if not _is_wiring(h)]
        if bundle_payload_hunks:
            # Bundle has real payload: check if it's in the same dir/subsystem
            bundle_payload_dirs = {str(Path(h.file_path).parent) for h in bundle_payload_hunks}
            hunk_dir = str(Path(hunk.file_path).parent)
            bundle_payload_subsystems = {h.kernel_subsystem for h in bundle_payload_hunks if h.kernel_subsystem}
            matching_payload = hunk_dir in bundle_payload_dirs or (
                hunk.kernel_subsystem and hunk.kernel_subsystem in bundle_payload_subsystems
            )
            if matching_payload:
                score += weights.get("wiring_payload_match_bonus", BUNDLE_SCORE_WEIGHTS["wiring_payload_match_bonus"])
            else:
                score += weights.get("wiring_no_payload_penalty", BUNDLE_SCORE_WEIGHTS["wiring_no_payload_penalty"])
        else:
            # Wiring-only bundle: discourage using wiring as a bridge
            score += weights.get("wiring_no_payload_penalty", BUNDLE_SCORE_WEIGHTS["wiring_no_payload_penalty"])

    return score


# ---------------------------------------------------------------------------
# Anchor creation
# ---------------------------------------------------------------------------

def _make_bundle_id() -> str:
    return str(uuid.uuid4())[:8]


def create_anchor_bundles(
    features: list[HunkFeatures],
    partition_result,
    max_anchor_breadth: int = MAX_ANCHOR_BREADTH,
) -> list[Bundle]:
    """Create anchor seed bundles from structural groups."""
    anchors: list[Bundle] = []

    # Group new-file hunks by subsystem (kernel) or parent dir (simple streams)
    new_file_hunks = [h for h in features if h.is_new_file]

    # a. New-file subsystem groups
    kernel_new: dict[str, list[HunkFeatures]] = defaultdict(list)
    simple_new: dict[str, list[HunkFeatures]] = defaultdict(list)
    for h in new_file_hunks:
        if h.component == "kernel" or h.pipeline == "full":
            key = h.kernel_subsystem or str(Path(h.file_path).parent)
            kernel_new[key].append(h)
        else:
            key = str(Path(h.file_path).parent)
            simple_new[key].append(h)

    for group_dict in (kernel_new, simple_new):
        for key, hunks in group_dict.items():
            if not hunks:
                continue
            component = hunks[0].component
            # Cap anchor breadth by splitting large groups on filename prefix
            if len(hunks) > max_anchor_breadth:
                subgroups: dict[str, list[HunkFeatures]] = defaultdict(list)
                for h in hunks:
                    stem = Path(h.file_path).stem
                    # Use first 3 chars of stem as sub-prefix
                    subkey = stem[:3] if len(stem) >= 3 else stem
                    subgroups[subkey].append(h)
                for subhunks in subgroups.values():
                    b = Bundle(
                        bundle_id=_make_bundle_id(),
                        hunks=list(subhunks),
                        confidence="high",
                        preliminary_label="",
                        component=component,
                    )
                    anchors.append(b)
            else:
                b = Bundle(
                    bundle_id=_make_bundle_id(),
                    hunks=list(hunks),
                    confidence="high",
                    preliminary_label="",
                    component=component,
                )
                anchors.append(b)

    # b. Kconfig/Makefile-paired sets
    # Use the CONFIG symbol dicts extracted by structural_partitioning (partition_result).
    # kconfig_symbols: hunk_id → [SYMBOL_NAME, ...]  (from "+config FOO" lines)
    # makefile_symbols: hunk_id → [SYMBOL_NAME, ...]  (from "$(CONFIG_FOO)" references)
    hunk_by_id = {h.hunk_id: h for h in features}

    # Build symbol → kconfig hunk and symbol → makefile hunk maps
    sym_to_kconfig: dict[str, HunkFeatures] = {}
    sym_to_makefile: dict[str, HunkFeatures] = {}

    if partition_result is not None:
        for hunk_id, syms in (partition_result.kconfig_symbols or {}).items():
            h = hunk_by_id.get(hunk_id)
            if h:
                for sym in syms:
                    sym_to_kconfig[sym] = h
        for hunk_id, syms in (partition_result.makefile_symbols or {}).items():
            h = hunk_by_id.get(hunk_id)
            if h:
                for sym in syms:
                    sym_to_makefile[sym] = h

    # Pair kconfig + makefile hunks that share a CONFIG symbol
    paired_kconfig_ids: set[str] = set()
    for sym, kh in sym_to_kconfig.items():
        mh = sym_to_makefile.get(sym)
        if mh is None or kh.hunk_id == mh.hunk_id:
            continue
        # These two belong together; create or extend an anchor
        combined = [kh, mh]
        component = kh.component
        b = Bundle(
            bundle_id=_make_bundle_id(),
            hunks=combined,
            confidence="high",
            preliminary_label="",
            component=component,
        )
        anchors.append(b)
        paired_kconfig_ids.add(kh.hunk_id)
        paired_kconfig_ids.add(mh.hunk_id)

    # c. DTS board groups
    dts_by_board: dict[str, list[HunkFeatures]] = defaultdict(list)
    for h in features:
        if h.is_dts and h.dts_board_prefix:
            dts_by_board[h.dts_board_prefix].append(h)
    for board, hunks in dts_by_board.items():
        component = hunks[0].component
        b = Bundle(
            bundle_id=_make_bundle_id(),
            hunks=list(hunks),
            confidence="high",
            preliminary_label="",
            component=component,
        )
        anchors.append(b)

    # d. Defconfig groups
    defconfig_by_board: dict[str, list[HunkFeatures]] = defaultdict(list)
    for h in features:
        if h.is_defconfig and h.defconfig_board:
            defconfig_by_board[h.defconfig_board].append(h)
    for board, hunks in defconfig_by_board.items():
        component = hunks[0].component
        b = Bundle(
            bundle_id=_make_bundle_id(),
            hunks=list(hunks),
            confidence="high",
            preliminary_label="",
            component=component,
        )
        anchors.append(b)

    # e. HAL interface groups
    hal_by_pkg: dict[str, list[HunkFeatures]] = defaultdict(list)
    for h in features:
        if h.is_hal_interface:
            pkg = str(Path(h.file_path).parent)
            hal_by_pkg[pkg].append(h)
    for pkg, hunks in hal_by_pkg.items():
        component = hunks[0].component
        b = Bundle(
            bundle_id=_make_bundle_id(),
            hunks=list(hunks),
            confidence="high",
            preliminary_label="",
            component=component,
        )
        anchors.append(b)

    return anchors


# ---------------------------------------------------------------------------
# Bundle building (greedy algorithm)
# ---------------------------------------------------------------------------

def build_bundles(
    features: list[HunkFeatures],
    partition_result,
    weights: Optional[dict] = None,
    attachment_threshold: float = ATTACHMENT_THRESHOLD,
    max_anchor_breadth: int = MAX_ANCHOR_BREADTH,
) -> list[Bundle]:
    """Full greedy anchor-seeded bundling algorithm."""
    if weights is None:
        weights = BUNDLE_SCORE_WEIGHTS

    if not features:
        return []

    # Create initial anchor bundles
    anchors = create_anchor_bundles(features, partition_result, max_anchor_breadth=max_anchor_breadth)

    # Dedup anchor seeds: a hunk (e.g. a DTS new-file) can be claimed by multiple
    # anchor sections (new-file group AND dts-board group). Keep each hunk_id in
    # only the first anchor that claimed it.
    _seen_in_anchor: set[str] = set()
    for bundle in anchors:
        bundle.hunks = [h for h in bundle.hunks if h.hunk_id not in _seen_in_anchor]
        for h in bundle.hunks:
            _seen_in_anchor.add(h.hunk_id)

    # Build co-change index once for the whole diff
    cochange_index = build_cochange_index(features)

    # Track which hunk IDs are already assigned
    assigned: set[str] = set()
    for bundle in anchors:
        for h in bundle.hunks:
            assigned.add(h.hunk_id)

    # Greedy attachment: for each unassigned hunk, find best bundle
    unassigned = [h for h in features if h.hunk_id not in assigned]

    # Iterate until no more attachments possible
    changed = True
    while changed and unassigned:
        changed = False
        still_unassigned: list[HunkFeatures] = []

        for hunk in unassigned:
            best_score = attachment_threshold
            best_bundle: Optional[Bundle] = None

            for bundle in anchors:
                sc = score_affinity(hunk, bundle, weights, cochange_index=cochange_index)
                if sc > best_score:
                    best_score = sc
                    best_bundle = bundle

            if best_bundle is not None:
                best_bundle.add(hunk, best_score)
                assigned.add(hunk.hunk_id)
                changed = True
            else:
                still_unassigned.append(hunk)

        unassigned = still_unassigned

    # Remaining unassigned hunks → singleton bundles
    for hunk in unassigned:
        b = Bundle(
            bundle_id=_make_bundle_id(),
            hunks=[hunk],
            confidence="low",
            preliminary_label="",
            component=hunk.component,
        )
        b.confidence_score = 0.0
        anchors.append(b)

    # Remove empty bundles
    bundles = [b for b in anchors if b.hunks]

    # Assign preliminary labels
    for bundle in bundles:
        bundle.preliminary_label = assign_preliminary_label(bundle)

    # Mark low-confidence bundles: singletons or score below threshold
    for bundle in bundles:
        if len(bundle.hunks) == 1 or bundle.confidence_score < attachment_threshold:
            bundle.confidence = "low"

    return bundles


# ---------------------------------------------------------------------------
# Export report generation
# ---------------------------------------------------------------------------

def generate_export_report(
    bundles: list[Bundle],
    analysis,
    export_format: str = "md",
) -> str:
    """Generate a human-readable or JSON export report for all bundles."""
    total_hunks = sum(len(b.hunks) for b in bundles)
    high_conf = sum(1 for b in bundles if b.confidence == "high")
    low_conf = len(bundles) - high_conf

    # Compute diff-level confidence
    if total_hunks > 0:
        weighted_conf = sum(
            (1.0 if b.confidence == "high" else 0.0) * len(b.hunks)
            for b in bundles
        ) / total_hunks
    else:
        weighted_conf = 0.0

    # Count distinct subsystems
    subsystems = {h.kernel_subsystem for b in bundles for h in b.hunks if h.kernel_subsystem}
    n_subsystems = len(subsystems)

    multi_release_warning = (n_subsystems > 47) or (len(bundles) > 89)

    release_ctx = analysis.release_context if hasattr(analysis, "release_context") else None

    if export_format == "json":
        data = {
            "summary": {
                "total_bundles": len(bundles),
                "high_confidence": high_conf,
                "low_confidence": low_conf,
                "diff_confidence": round(weighted_conf, 3),
                "total_hunks": total_hunks,
                "multi_release_warning": multi_release_warning,
                "release_context": {
                    "version": release_ctx.version if release_ctx else None,
                    "date": release_ctx.date if release_ctx else None,
                    "raw_subject": release_ctx.raw_subject if release_ctx else None,
                } if release_ctx else None,
            },
            "bundles": [
                {
                    "bundle_id": b.bundle_id,
                    "component": b.component,
                    "preliminary_label": b.preliminary_label,
                    "confidence": b.confidence,
                    "confidence_score": round(b.confidence_score, 3),
                    "hunk_count": len(b.hunks),
                    "files": sorted({h.file_path for h in b.hunks}),
                    "net_lines": sum(h.net_lines for h in b.hunks),
                    "added_lines": sum(h.added_lines for h in b.hunks),
                    "subsystems": sorted({h.kernel_subsystem for h in b.hunks if h.kernel_subsystem}),
                }
                for b in bundles
            ],
        }
        return json.dumps(data, indent=2)

    # Markdown format
    lines: list[str] = []
    lines.append("# py-git-split Bundle Report")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Total bundles**: {len(bundles)}")
    lines.append(f"- **High confidence**: {high_conf}")
    lines.append(f"- **Low confidence**: {low_conf}")
    lines.append(f"- **Diff-level confidence**: {weighted_conf:.1%}")
    lines.append(f"- **Total hunks**: {total_hunks}")

    if release_ctx:
        v_str = release_ctx.version or "unknown"
        d_str = release_ctx.date or "unknown"
        lines.append(f"- **Release context**: version={v_str}, date={d_str}")
        lines.append(f"  - Subject: `{release_ctx.raw_subject}`")

    if multi_release_warning:
        lines.append("")
        lines.append("> **WARNING**: Multi-release diff detected "
                     f"({n_subsystems} subsystems, {len(bundles)} bundles). "
                     "This diff may span multiple release cycles.")

    # Group by component
    by_component: dict[str, list[Bundle]] = defaultdict(list)
    for b in bundles:
        by_component[b.component].append(b)

    for component in sorted(by_component.keys()):
        comp_bundles = by_component[component]
        lines.append("")
        lines.append(f"## Component: `{component}`")
        lines.append("")
        for b in comp_bundles:
            lines.append(b.summary())
            lines.append("")

    return "\n".join(lines)
