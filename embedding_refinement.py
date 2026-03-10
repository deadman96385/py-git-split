#!/usr/bin/env python3
"""Stage 5: Embedding refinement — optional voyage-code-3 bundle-level similarity."""
from __future__ import annotations

import time
from typing import Optional

try:
    import voyageai  # type: ignore
    import numpy as np  # type: ignore
    _voyage_available = True
except ImportError:
    _voyage_available = False

from candidate_bundling import Bundle


_VOYAGE_MAX_ATTEMPTS = 6
_VOYAGE_INITIAL_WAIT = 60   # seconds; doubled each attempt, capped at 300
_VOYAGE_BATCH_SIZE = 50     # texts per request (Voyage max is 128; stay conservative)
_VOYAGE_INTER_BATCH_DELAY = 2.0  # seconds between batches

# Voyage's voyage-code-3 accepts up to 16k tokens per text; truncate long
# summaries to keep per-request token counts predictable.
_VOYAGE_SUMMARY_MAX_CHARS = 1500


def _truncate_summary(s: str) -> str:
    if len(s) <= _VOYAGE_SUMMARY_MAX_CHARS:
        return s
    return s[:_VOYAGE_SUMMARY_MAX_CHARS]


def _embed_batch_with_retry(client, batch: list[str]) -> tuple[Optional[list[list[float]]], float]:
    """Embed a single batch with exponential backoff on rate limit errors.

    Returns (embeddings, total_waited_seconds). Caller uses total_waited to
    decide whether to extend the next inter-batch delay.
    """
    wait = _VOYAGE_INITIAL_WAIT
    total_waited = 0.0
    for attempt in range(_VOYAGE_MAX_ATTEMPTS):
        try:
            result = client.embed(batch, model="voyage-code-3", input_type="document")
            return result.embeddings, total_waited
        except Exception as e:
            err_str = str(e).lower()
            is_rate_limit = "rate_limit" in err_str or "rate limit" in err_str or "429" in str(e)
            if not is_rate_limit or attempt == _VOYAGE_MAX_ATTEMPTS - 1:
                return None, total_waited
            retry_after = None
            try:
                if hasattr(e, "response") and hasattr(e.response, "headers"):
                    ra = e.response.headers.get("retry-after") or e.response.headers.get("x-ratelimit-reset-requests")
                    if ra:
                        retry_after = int(float(ra))
            except Exception:
                pass
            actual_wait = retry_after if retry_after is not None else wait
            print(f" voyage: rate limited, waiting {actual_wait}s (attempt {attempt + 1}/{_VOYAGE_MAX_ATTEMPTS})...", flush=True)
            time.sleep(actual_wait)
            total_waited += actual_wait
            wait = min(wait * 2, 300)
    return None, total_waited


def embed_bundles(summaries: list[str], api_key: str) -> Optional[list[list[float]]]:
    """Embed bundle summaries using voyage-code-3.

    Sends in batches of _VOYAGE_BATCH_SIZE with an inter-batch delay to stay
    within Voyage's token-per-minute rate limits. Returns None on failure.
    """
    if not _voyage_available:
        return None
    if not summaries:
        return []
    try:
        client = voyageai.Client(api_key=api_key)
        truncated = [_truncate_summary(s) for s in summaries]
        total_batches = (len(truncated) + _VOYAGE_BATCH_SIZE - 1) // _VOYAGE_BATCH_SIZE
        all_embeddings: list[list[float]] = []
        for i in range(total_batches):
            batch = truncated[i * _VOYAGE_BATCH_SIZE : (i + 1) * _VOYAGE_BATCH_SIZE]
            if total_batches > 1:
                print(f" voyage: embedding batch {i + 1}/{total_batches} ({len(batch)} texts)...", flush=True)
            embeddings, waited = _embed_batch_with_retry(client, batch)
            if embeddings is None:
                return None
            all_embeddings.extend(embeddings)
            if i < total_batches - 1:
                # If the last batch had to wait for a rate limit, the window just
                # reset — use the normal delay. If it sailed through instantly,
                # use the normal delay too. Either way _VOYAGE_INTER_BATCH_DELAY
                # provides breathing room; after a recovery we skip the extra
                # sleep since we already waited >> the normal delay.
                extra = max(0.0, _VOYAGE_INTER_BATCH_DELAY - waited)
                if extra > 0:
                    time.sleep(extra)
        return all_embeddings
    except Exception:
        return None


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two embedding vectors."""
    if not _voyage_available:
        # Pure-Python fallback
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    va = np.array(a, dtype=float)
    vb = np.array(b, dtype=float)
    norm_a = float(np.linalg.norm(va))
    norm_b = float(np.linalg.norm(vb))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))


def _merge_bundles(b1: Bundle, b2: Bundle) -> Bundle:
    """Merge two bundles into one, taking id and label from the larger one."""
    primary = b1 if len(b1.hunks) >= len(b2.hunks) else b2
    secondary = b2 if primary is b1 else b1
    merged = Bundle(
        bundle_id=primary.bundle_id,
        hunks=list(primary.hunks) + list(secondary.hunks),
        confidence="low",   # Flag for LLM confirmation
        preliminary_label=primary.preliminary_label,
        component=primary.component,
        release_context=primary.release_context or secondary.release_context,
        confidence_score=min(primary.confidence_score, secondary.confidence_score),
    )
    return merged


def refine_bundles(
    bundles: list[Bundle],
    voyage_api_key: Optional[str],
    merge_threshold: float = 0.75,
) -> list[Bundle]:
    """For low-confidence bundles in the same component, merge similar ones.

    Uses cosine similarity of voyage-code-3 embeddings. Pairs above merge_threshold
    are merged and flagged with confidence='low' for LLM confirmation.

    Returns the updated bundle list.
    """
    if not _voyage_available or not voyage_api_key:
        return bundles

    # Work only with low-confidence bundles
    low_conf = [b for b in bundles if b.confidence == "low"]
    high_conf = [b for b in bundles if b.confidence == "high"]

    if len(low_conf) < 2:
        return bundles

    # Generate summaries
    summaries = [b.summary() for b in low_conf]
    embeddings = embed_bundles(summaries, voyage_api_key)
    if embeddings is None or len(embeddings) != len(low_conf):
        return bundles

    # Group by component
    by_component: dict[str, list[int]] = {}
    for i, b in enumerate(low_conf):
        by_component.setdefault(b.component, []).append(i)

    merged_into: dict[int, int] = {}  # source_idx -> target_idx

    for component, indices in by_component.items():
        if len(indices) < 2:
            continue
        for i_pos in range(len(indices)):
            for j_pos in range(i_pos + 1, len(indices)):
                i = indices[i_pos]
                j = indices[j_pos]

                # Skip already-merged bundles
                if i in merged_into or j in merged_into:
                    continue

                sim = cosine_similarity(embeddings[i], embeddings[j])
                if sim >= merge_threshold:
                    merged_into[j] = i  # merge j into i

    # Apply merges
    merge_targets: dict[int, list[int]] = {}
    for src, tgt in merged_into.items():
        merge_targets.setdefault(tgt, []).append(src)

    result_low: list[Bundle] = []
    processed: set[int] = set()

    for i, bundle in enumerate(low_conf):
        if i in merged_into:
            processed.add(i)
            continue  # Will be merged into another
        if i in merge_targets:
            merged = bundle
            for src_idx in merge_targets[i]:
                merged = _merge_bundles(merged, low_conf[src_idx])
                processed.add(src_idx)
            result_low.append(merged)
        else:
            result_low.append(bundle)
        processed.add(i)

    return high_conf + result_low
