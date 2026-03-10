#!/usr/bin/env python3
"""Stage 2: Structural partitioning — hard-separate diff into component streams."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

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

_DTS_RE = re.compile(r'arch/[^/]+/boot/dts/.*\.dtsi?$')
_KCONFIG_RE = re.compile(r'(?:^|/)Kconfig(?:\.[^/]*)?$')
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


def _kernel_subsystem(file_path: str, kernel_root: str) -> Optional[str]:
    """Top-2 path components below kernel_root (e.g. 'drivers/usb')."""
    rel = _rel_to_kernel(file_path, kernel_root)
    parts = rel.split("/")
    if len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
    if len(parts) == 1 and parts[0]:
        return parts[0]
    return None


def _dts_board_prefix(file_path: str) -> Optional[str]:
    """Extract board prefix from DTS filename stem (first 2 dash-parts)."""
    stem = Path(file_path).stem
    parts = stem.split("-")
    if len(parts) >= 2:
        return f"{parts[0]}-{parts[1]}"
    if parts:
        return parts[0]
    return None


def _defconfig_board(file_path: str) -> Optional[str]:
    """Board name from defconfig filename (strip _defconfig)."""
    name = Path(file_path).name
    if name.endswith("_defconfig"):
        return name[: -len("_defconfig")]
    return name


def classify_hunk(hunk, file_headers: dict, kernel_root: str, mode: RepoMode) -> PartitionedHunk:
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
    is_dts = bool(_DTS_RE.search(fp) or fp.endswith(".dtsi") or fp.endswith(".dts"))
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
        ks = _kernel_subsystem(fp, kernel_root or "")

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


def partition(hunks, file_headers: dict, analysis: RepoAnalysis) -> PartitionResult:
    """Partition all hunks into structural components."""
    partitioned: list[PartitionedHunk] = []
    skipped: list[tuple[str, str]] = []
    kconfig_symbols: dict[str, list[str]] = {}
    makefile_symbols: dict[str, list[str]] = {}

    kernel_root = analysis.kernel_root or ""
    mode = analysis.mode

    for hunk in hunks:
        ph = classify_hunk(hunk, file_headers, kernel_root, mode)

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
