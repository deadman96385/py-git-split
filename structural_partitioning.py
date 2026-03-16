#!/usr/bin/env python3
"""Stage 2: Structural partitioning — hard-separate diff into component streams."""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from bundling_constants import DEFAULT_MAX_ANCHOR_BREADTH, DEFAULT_DEEP_ROOT_THRESHOLD
from repo_analysis import RepoAnalysis, RepoMode


# Ordered list of (prefix, component_name, pipeline)
# pipeline: "full" | "simple" | "skip"
BSP_COMPONENTS = [
    ("bootable/",   "bootable",      "simple"),
    ("bootloader/", "bootloader",    "simple"),
    ("prebuilts/",  "prebuilts",     "skip"),
    ("external/",   "external",      "skip"),
    ("tmp-ws/",     "artifacts",     "skip"),
    ("hardware/",   "hardware_hal",  "simple"),
    ("vendor/",     "vendor",        "simple"),
    ("frameworks/", "frameworks",    "simple"),
    ("fireos/",     "oem_layer",     "simple"),
    ("device/",     "device",        "simple"),
    ("packages/",   "packages",      "simple"),
    ("system/",     "system",        "simple"),
]

# Density threshold: a depth-2 subsystem with more files than this is a candidate
# for deeper splitting.
_DEEP_ROOT_THRESHOLD = DEFAULT_DEEP_ROOT_THRESHOLD
_DEEP_ROOT_MAX_DEPTH = 6

_DTS_RE = re.compile(r'arch/[^/]+/boot/dts/.*\.(?:dtsi?|main_dts)$')
_KCONFIG_RE = re.compile(r'(?:^|/)Kconfig(?:\.[A-Za-z][^/]*)?$')
_MAKEFILE_RE = re.compile(r'(?:^|/)(?:Makefile|Kbuild)$')
_DEFCONFIG_RE = re.compile(r'(?:arch/[^/]+/configs/[^/]+_defconfig|[^/]+_defconfig)$')
_HAL_EXT_RE = re.compile(r'\.hal$')


@dataclass
class PartitionedHunk:
    hunk_id: str
    file_path: str
    component: str
    pipeline: str                      # "full" | "simple" | "skip"
    is_new_file: bool
    is_dts: bool
    is_kconfig: bool
    is_makefile: bool
    is_defconfig: bool
    is_hal_interface: bool
    kernel_subsystem: Optional[str]
    dts_board_prefix: Optional[str]
    defconfig_board: Optional[str]
    # Raw hunk lines for downstream extraction
    lines: list[str] = field(default_factory=list)
    bsp_subsystem: Optional[str] = None
    hunk_header: str = ""


def _classify_component_bsp(file_path: str) -> tuple[str, str]:
    """Return (component, pipeline) for a FULL_BSP file."""
    for prefix, component, pipeline in BSP_COMPONENTS:
        if file_path.startswith(prefix):
            return component, pipeline
    # Default: treat as kernel component with full pipeline
    return "kernel", "full"


def _classify_component_kernel(file_path: str, kernel_root: str) -> tuple[str, str]:
    """Return (component, pipeline) for a KERNEL_ONLY file."""
    return "kernel", "full"


def _rel_to_kernel(file_path: str, kernel_root: str) -> str:
    """Return path relative to kernel_root."""
    if kernel_root and file_path.startswith(kernel_root + "/"):
        return file_path[len(kernel_root) + 1:]
    elif kernel_root and file_path.startswith(kernel_root):
        return file_path[len(kernel_root):].lstrip("/")
    return file_path


def _discover_deep_roots(
    file_paths: list[str],
    kernel_root: str,
    threshold: int = _DEEP_ROOT_THRESHOLD,
    max_depth: int = _DEEP_ROOT_MAX_DEPTH,
) -> dict[str, int]:
    """Auto-discover subsystem prefixes that need more than 2 path components.

    For every depth-2 prefix (e.g. 'drivers/usb') that has more than *threshold*
    files in the diff, find the minimum depth at which the files split into at
    least two distinct sub-groups.  This is purely data-driven — no vendor-specific
    knowledge required.

    Returns a dict of {prefix: depth} suitable for use in _kernel_subsystem().
    """
    # Group files by their depth-2 prefix
    by_prefix: dict[str, list[str]] = defaultdict(list)
    for fp in file_paths:
        rel = _rel_to_kernel(fp, kernel_root)
        parts = rel.split("/")
        if len(parts) >= 2:
            by_prefix[f"{parts[0]}/{parts[1]}"].append(fp)

    deep_roots: dict[str, int] = {}
    for prefix, fps in by_prefix.items():
        if len(fps) <= threshold:
            continue
        # Walk deeper until the files split into at least two sub-groups that
        # each contain more than one file (prevents a single stray file from
        # triggering deep splitting for the entire prefix).
        for depth in range(3, max_depth + 1):
            sub_groups: dict[str, int] = defaultdict(int)
            for fp in fps:
                rel = _rel_to_kernel(fp, kernel_root)
                parts = list(Path(rel).parent.parts)
                if len(parts) < depth:
                    continue
                key = "/".join(parts[:depth])
                sub_groups[key] += 1
            meaningful = [k for k, cnt in sub_groups.items() if cnt > 1]
            if len(meaningful) > 1:
                deep_roots[prefix] = depth
                break

    return deep_roots


def _kernel_subsystem(
    file_path: str,
    kernel_root: str,
    deep_roots: Optional[dict[str, int]] = None,
) -> Optional[str]:
    """Top-N path components below kernel_root (e.g. 'drivers/usb').

    Defaults to top-2, but uses top-3 for include/linux/ paths, and uses
    deeper depths for any prefix present in *deep_roots* (auto-discovered or
    provided via CLI).
    """
    rel = _rel_to_kernel(file_path, kernel_root)
    parts = rel.split("/")
    roots = deep_roots or {}
    for prefix, depth in roots.items():
        prefix_parts = prefix.split("/")
        # `len(parts) > depth` means the file is nested deeper than the target
        # depth (i.e. it has at least depth+1 components including the filename).
        # Files that sit *at* the subsystem root (exactly `depth` components)
        # fall through to the default top-2 logic — they belong to the parent
        # subsystem, not a sub-group.
        if parts[:len(prefix_parts)] == prefix_parts and len(parts) > depth:
            return "/".join(parts[:depth])
    # include/linux/ → top-3 when the third component is a subdirectory (no extension).
    # Flat headers like include/linux/fscrypt.h stay at depth-2.
    if parts[0:2] == ["include", "linux"] and len(parts) >= 3 and "." not in parts[2]:
        return f"include/linux/{parts[2]}"
    # Default: top-2
    if len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
    if len(parts) == 1 and parts[0]:
        return parts[0]
    return None


def _dts_board_prefix(file_path: str) -> Optional[str]:
    """Extract board prefix from DTS filename stem.

    Upstream convention: hyphen-separated → first two parts (e.g. 'rockchip-rk3399').
    MStar/vendor convention: underscore-separated → first part (e.g. 'm7322').
    """
    stem = Path(file_path).stem
    hyphen_parts = stem.split("-")
    if len(hyphen_parts) >= 2:
        return f"{hyphen_parts[0]}-{hyphen_parts[1]}"
    # Vendor underscore convention (e.g. m7322_an, m7622_ramdisk)
    underscore_parts = stem.split("_")
    if len(underscore_parts) >= 2:
        return underscore_parts[0]
    return stem if stem else None


def _defconfig_board(file_path: str) -> Optional[str]:
    """Board name from defconfig filename (strip _defconfig)."""
    name = Path(file_path).name
    if name.endswith("_defconfig"):
        return name[: -len("_defconfig")]
    return name


def _bsp_subsystem(file_path: str) -> Optional[str]:
    """Extract a 2-component subsystem label for BSP (non-kernel) files.

    E.g. vendor/foo/drivers/bar.c → "vendor/foo/drivers"
         hardware/interfaces/power/aidl/default/foo.cpp → "hardware/interfaces/power"
    """
    parts = file_path.replace("\\", "/").split("/")
    if len(parts) >= 3:
        return "/".join(parts[:3])
    if len(parts) >= 2:
        return "/".join(parts[:2])
    return None


def classify_hunk(
    hunk,
    file_headers: dict,
    kernel_root: str,
    mode: RepoMode,
    deep_roots: Optional[dict[str, int]] = None,
) -> PartitionedHunk:
    """Classify a hunk into its structural partition."""
    fp: str = hunk.filepath
    lines: list[str] = list(hunk.lines)

    # Determine component + pipeline
    if mode == RepoMode.FULL_BSP:
        component, pipeline = _classify_component_bsp(fp)
    else:
        # KERNEL_ONLY or UNKNOWN — treat everything as kernel
        component, pipeline = _classify_component_kernel(fp, kernel_root or "")

    # is_new_file: check file header for "new file"
    is_new_file = False
    fh = file_headers.get(fp)
    if fh is not None:
        is_new_file = any("new file" in line for line in fh.lines)

    # File-type flags
    is_dts = bool(_DTS_RE.search(fp) or fp.endswith(".dtsi") or fp.endswith(".dts") or fp.endswith(".main_dts"))
    is_kconfig = bool(_KCONFIG_RE.search(fp))
    is_makefile = bool(_MAKEFILE_RE.search(fp))
    is_defconfig = bool(_DEFCONFIG_RE.search(fp))
    is_hal_interface = bool(
        _HAL_EXT_RE.search(fp)
        or "hardware/interfaces/" in fp
        or "/aidl/" in fp
    )

    # kernel_subsystem: for kernel-component files only
    ks: Optional[str] = None
    if component == "kernel" or mode == RepoMode.KERNEL_ONLY:
        ks = _kernel_subsystem(fp, kernel_root or "", deep_roots=deep_roots)

    # dts_board_prefix
    dts_bp: Optional[str] = _dts_board_prefix(fp) if is_dts else None

    # defconfig_board
    dc_board: Optional[str] = _defconfig_board(fp) if is_defconfig else None

    return PartitionedHunk(
        hunk_id=hunk.id,
        file_path=fp,
        component=component,
        pipeline=pipeline,
        is_new_file=is_new_file,
        is_dts=is_dts,
        is_kconfig=is_kconfig,
        is_makefile=is_makefile,
        is_defconfig=is_defconfig,
        is_hal_interface=is_hal_interface,
        kernel_subsystem=ks,
        dts_board_prefix=dts_bp,
        defconfig_board=dc_board,
        lines=lines,
        bsp_subsystem=_bsp_subsystem(fp) if component != "kernel" else None,
        hunk_header=getattr(hunk, 'header', ''),
    )


def extract_config_symbols_defined(hunk_body: str) -> list[str]:
    """Extract Kconfig symbol names defined in hunk (lines starting with '+config ...')."""
    return re.findall(r'^\+config\s+([\w]+)', hunk_body, re.MULTILINE)


def extract_config_symbols_referenced(hunk_body: str) -> list[str]:
    """Extract CONFIG_ symbol references from Makefile/Kbuild hunks."""
    return re.findall(r'\$\(CONFIG_([\w]+)\)', hunk_body)


@dataclass
class PartitionResult:
    hunks: list[PartitionedHunk]
    skipped: list[tuple[str, str]]          # (filepath, component)
    kconfig_symbols: dict[str, list[str]]   # hunk_id → defined symbols
    makefile_symbols: dict[str, list[str]]  # hunk_id → referenced CONFIG_ symbols


def partition(
    hunks,
    file_headers: dict,
    analysis: RepoAnalysis,
    deep_roots: Optional[dict[str, int]] = None,
) -> PartitionResult:
    """Partition all hunks into structural components.

    deep_roots: optional extra entries to merge with auto-discovered deep roots
                (e.g. from the --deep-subsystem-roots CLI flag).
    """
    partitioned: list[PartitionedHunk] = []
    skipped: list[tuple[str, str]] = []
    kconfig_symbols: dict[str, list[str]] = {}
    makefile_symbols: dict[str, list[str]] = {}

    kernel_root = analysis.kernel_root or ""
    mode = analysis.mode

    # Auto-discover which kernel subsystems are dense enough to warrant deeper
    # splitting, then merge with any explicit overrides from the caller.
    # Only kernel-component paths contribute — BSP paths (vendor/, hardware/, …)
    # are handled by component classification, not kernel_subsystem().
    kernel_file_paths = [
        hunk.filepath for hunk in hunks
        if not any(hunk.filepath.startswith(p) for p, _, _ in BSP_COMPONENTS)
    ]
    discovered = _discover_deep_roots(kernel_file_paths, kernel_root)
    effective_deep_roots = {**discovered, **(deep_roots or {})}

    for hunk in hunks:
        ph = classify_hunk(hunk, file_headers, kernel_root, mode, deep_roots=effective_deep_roots)

        # Extract kconfig / makefile symbols from hunk body
        body = "\n".join(ph.lines)
        if ph.is_kconfig:
            syms = extract_config_symbols_defined(body)
            if syms:
                kconfig_symbols[ph.hunk_id] = syms
        if ph.is_makefile:
            refs = extract_config_symbols_referenced(body)
            if refs:
                makefile_symbols[ph.hunk_id] = refs

        if ph.pipeline == "skip":
            skipped.append((ph.file_path, ph.component))
        else:
            partitioned.append(ph)

    return PartitionResult(
        hunks=partitioned,
        skipped=skipped,
        kconfig_symbols=kconfig_symbols,
        makefile_symbols=makefile_symbols,
    )
