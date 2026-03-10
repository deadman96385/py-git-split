#!/usr/bin/env python3
"""Stage 1: Repo analysis — mode detection, artifact suppression, HEAD release metadata."""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class RepoMode(Enum):
    FULL_BSP = "full_bsp"
    KERNEL_ONLY = "kernel_only"
    UNKNOWN = "unknown"


ARTIFACT_PATH_SEGMENTS = {"tmp-ws", "tmp", "out", ".repo", "goldfish", "build-artifacts"}

KERNEL_FINGERPRINTS = {
    "arch/":           10,
    "include/linux/":  10,
    "init/Kconfig":     9,
    "MAINTAINERS":      8,
    "Kconfig":          8,
    "scripts/Makefile": 6,
    "drivers/":         5,
    "mm/":              5,
    "fs/":              5,
    "net/":             4,
    "kernel/":          4,
}

DEPTH_PENALTY_PER_LEVEL = 0.15
CONFIDENCE_THRESHOLD = 20

# Match bare version numbers like 6.2.9.4 or 7.6.2.4.
# Uses only . and _ as separators (not -) and caps digit runs at 4 to avoid
# consuming an adjacent YYYYMMDD date stamp.
RELEASE_VERSION_RE = re.compile(r'(\d+\.\d+(?:[._]\d{1,4}){0,3})')
# Match 8-digit date stamps like 20221228 (YYYYMMDD).
RELEASE_DATE_RE = re.compile(r'\b(20\d{6})\b')


@dataclass
class ReleaseContext:
    version: Optional[str]
    date: Optional[str]
    raw_subject: str


def extract_release_context(repo_path: str) -> Optional[ReleaseContext]:
    """Run git log -1 --format=%s and parse version/date from the subject line."""
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%s"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
            cwd=repo_path,
        )
        subject = result.stdout.strip()
        if not subject:
            return None

        version_match = RELEASE_VERSION_RE.search(subject)
        date_match = RELEASE_DATE_RE.search(subject)

        version = version_match.group(1) if version_match else None
        date = date_match.group(1) if date_match else None

        if version is None and date is None:
            return None

        return ReleaseContext(version=version, date=date, raw_subject=subject)
    except Exception:
        return None


def _path_has_artifact_segment(path: str) -> bool:
    """Return True if any path component is in ARTIFACT_PATH_SEGMENTS."""
    parts = path.replace("\\", "/").split("/")
    return any(part in ARTIFACT_PATH_SEGMENTS for part in parts)


def _prefix_at_depth(path: str, depth: int) -> Optional[str]:
    """Return the path prefix at the given depth (number of components), or None if depth > parts."""
    parts = path.replace("\\", "/").split("/")
    if depth == 0:
        return ""
    if depth > len(parts):
        return None
    return "/".join(parts[:depth]) + "/"


def detect_repo_mode(
    file_paths: list[str],
    kernel_root_override: Optional[str] = None,
) -> tuple[RepoMode, Optional[str]]:
    """Detect whether this is a full BSP repo or kernel-only, and find the kernel root.

    Returns (mode, kernel_root).
    """
    if kernel_root_override is not None:
        return (RepoMode.FULL_BSP, kernel_root_override)

    if not file_paths:
        return (RepoMode.UNKNOWN, None)

    # Score candidate roots at depths 0..4
    candidate_scores: dict[str, float] = {}
    candidate_depths: dict[str, int] = {}

    for depth in range(0, 5):
        prefixes: dict[str, list[str]] = {}
        for fp in file_paths:
            prefix = _prefix_at_depth(fp, depth)
            if prefix is None:
                continue
            if prefix not in prefixes:
                prefixes[prefix] = []
            prefixes[prefix].append(fp)

        for prefix, paths in prefixes.items():
            # Skip candidates containing artifact segments
            if _path_has_artifact_segment(prefix.rstrip("/")):
                continue

            base_score = 0.0
            for fp in paths:
                # Compute relative path from prefix
                if depth == 0:
                    rel = fp
                else:
                    rel = fp[len(prefix):]

                for fingerprint, weight in KERNEL_FINGERPRINTS.items():
                    if rel.startswith(fingerprint) or rel == fingerprint.rstrip("/"):
                        base_score += weight

            if base_score == 0.0:
                continue

            penalty = DEPTH_PENALTY_PER_LEVEL * depth * base_score
            final_score = base_score - penalty

            if prefix not in candidate_scores or final_score > candidate_scores[prefix]:
                candidate_scores[prefix] = final_score
                candidate_depths[prefix] = depth

    if not candidate_scores:
        return (RepoMode.UNKNOWN, None)

    best_prefix = max(candidate_scores, key=lambda p: (candidate_scores[p], -candidate_depths[p]))
    best_score = candidate_scores[best_prefix]
    best_depth = candidate_depths[best_prefix]

    if best_score < CONFIDENCE_THRESHOLD:
        return (RepoMode.UNKNOWN, None)

    if best_depth == 0:
        return (RepoMode.KERNEL_ONLY, "")
    else:
        return (RepoMode.FULL_BSP, best_prefix.rstrip("/"))


@dataclass
class RepoAnalysis:
    mode: RepoMode
    kernel_root: Optional[str]
    release_context: Optional[ReleaseContext]
    file_paths: list[str]


def analyze_repo(
    file_paths: list[str],
    repo_path: str = ".",
    kernel_root_override: Optional[str] = None,
    no_release_context: bool = False,
) -> RepoAnalysis:
    """Run full repo analysis: mode detection + release context extraction."""
    mode, kernel_root = detect_repo_mode(file_paths, kernel_root_override=kernel_root_override)

    release_context: Optional[ReleaseContext] = None
    if not no_release_context:
        release_context = extract_release_context(repo_path)

    return RepoAnalysis(
        mode=mode,
        kernel_root=kernel_root,
        release_context=release_context,
        file_paths=list(file_paths),
    )
