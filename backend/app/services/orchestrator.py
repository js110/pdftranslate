"""Per-page translation orchestration: chunking, memory, fallbacks."""
from __future__ import annotations

import logging
import re

from app.core.settings import get_settings
from app.services.heuristics import _looks_like_overlap_continuation
from app.services.jobtypes import PageProcessResult, _TextBlockJob
from app.services.promptprep import (
    _compact_text_for_prompt,
    _should_force_single_retry,
    _text_fingerprint,
)
from app.services.quality import (
    _fallback_translate_short_formula_phrase,
    _fallback_translate_table_narrative,
    _is_low_quality_translation,
    _normalize_translated_text,
)
from app.services.session_store import SessionStore
from app.services.textutil import ENGLISH_WORD_RE
from app.services.translator import (
    ProviderRuntime,
    TranslationError,
    translate_batch_with_fallback,
    translate_with_fallback,
)

logger = logging.getLogger(__name__)



def _translate_text_jobs(
    *,
    session_id: str | None,
    jobs: list[_TextBlockJob],
    result: PageProcessResult,
    page_no: int,
    primary_provider: ProviderRuntime,
    backup_provider: ProviderRuntime | None,
    style_profile: str,
    glossary: list[str] | None,
    max_retries: int,
) -> dict[int, str]:
    if not jobs:
        return {}

    settings = get_settings()
    chunk_size = max(1, int(settings.batch_segment_size))
    chunk_char_limit = max(1000, int(settings.batch_segment_char_limit))
    single_fallback = bool(settings.enable_single_block_fallback)
    glossary_first_chunk_only = bool(settings.glossary_first_chunk_only)
    drop_low_quality_cache_on_read = bool(settings.drop_low_quality_cache_on_read)
    translated_by_block: dict[int, str] = {}
    memory_updates: dict[str, str] = {}
    store = SessionStore() if session_id else None

    hash_to_jobs: dict[str, list[_TextBlockJob]] = {}
    prompt_by_block: dict[int, str] = {}
    ordered_unique_hashes: list[str] = []
    ordered_unique_jobs: list[_TextBlockJob] = []

    for job in jobs:
        prompt_text = _compact_text_for_prompt(job.text)
        if not prompt_text:
            continue
        prompt_by_block[job.block_index] = prompt_text
        fingerprint = _text_fingerprint(prompt_text)
        hash_to_jobs.setdefault(fingerprint, []).append(job)
        if fingerprint not in ordered_unique_hashes:
            ordered_unique_hashes.append(fingerprint)
            ordered_unique_jobs.append(job)

    if not ordered_unique_jobs:
        return translated_by_block

    def mark_untranslated(fingerprint: str, reason: str) -> None:
        for dup_job in hash_to_jobs.get(fingerprint, []):
            result.untranslated_items.append(
                {
                    "page_no": page_no,
                    "bbox": [dup_job.x0, dup_job.y0, dup_job.x1, dup_job.y1],
                    "source_excerpt": dup_job.text[:160],
                    "reason": reason,
                }
            )

    def assign_translation(fingerprint: str, normalized: str) -> None:
        memory_updates[fingerprint] = normalized
        for dup_job in hash_to_jobs.get(fingerprint, []):
            translated_by_block[dup_job.block_index] = _normalize_translated_text(dup_job.text, normalized)

    def apply_fallback_or_mark(job: _TextBlockJob, *, fingerprint: str, reason: str) -> bool:
        """Resolve a block via deterministic narrative/phrase fallbacks.

        Returns True and assigns the fallback translation when one applies;
        otherwise records the block as untranslated and returns False.
        """
        narrative_fallback = _fallback_translate_table_narrative(job.text)
        if narrative_fallback is not None:
            assign_translation(fingerprint, narrative_fallback)
            return True
        phrase_fallback = _fallback_translate_short_formula_phrase(job.text)
        if phrase_fallback is not None:
            assign_translation(fingerprint, phrase_fallback)
            return True
        mark_untranslated(fingerprint, reason)
        return False

    memory_hits: dict[str, str] = {}
    if store is not None:
        memory_hits = store.get_translation_memory_bulk(session_id, ordered_unique_hashes)

    unresolved_jobs: list[_TextBlockJob] = []
    stale_cache_fingerprints: set[str] = set()
    for fingerprint, job in zip(ordered_unique_hashes, ordered_unique_jobs, strict=False):
        cached = memory_hits.get(fingerprint)
        if not cached:
            unresolved_jobs.append(job)
            continue
        if _is_low_quality_translation(job.text, cached):
            unresolved_jobs.append(job)
            if drop_low_quality_cache_on_read:
                stale_cache_fingerprints.add(fingerprint)
            continue
        for dup_job in hash_to_jobs.get(fingerprint, []):
            translated_by_block[dup_job.block_index] = _normalize_translated_text(dup_job.text, cached)
    if store is not None and stale_cache_fingerprints:
        removed = store.delete_translation_memory_bulk(session_id, list(stale_cache_fingerprints))
        logger.info("page %s evicted low-quality cache entries: %s", page_no, removed)

    offset = 0

    for chunk_index, chunk in enumerate(
        _chunk_jobs_for_translation(
            unresolved_jobs,
            max_blocks=chunk_size,
            max_chars=chunk_char_limit,
            prompt_texts=prompt_by_block,
        ),
        start=1,
    ):
        start_no = offset + 1
        end_no = offset + len(chunk)
        offset = end_no

        texts = [prompt_by_block.get(job.block_index, _compact_text_for_prompt(job.text)) for job in chunk]
        chunk_glossary = glossary
        if glossary_first_chunk_only and chunk_index > 1:
            chunk_glossary = None

        try:
            translated_list, used_provider, switched_from = translate_batch_with_fallback(
                texts=texts,
                primary=primary_provider,
                backup=backup_provider,
                style_profile=style_profile,
                glossary=chunk_glossary,
                max_retries=max_retries,
            )
            if switched_from:
                result.fallback_events.append(
                    {
                        "page_no": page_no,
                        "from_provider": switched_from,
                        "to_provider": used_provider,
                        "reason": "primary provider failed on batch text translation",
                    }
                )

            for idx, job in enumerate(chunk):
                prompt_text = texts[idx]
                fingerprint = _text_fingerprint(prompt_text)
                normalized = _normalize_translated_text(job.text, translated_list[idx])
                normalized, strict_switch = _repair_low_quality_translation(
                    job=job,
                    translated=normalized,
                    primary_provider=primary_provider,
                    backup_provider=backup_provider,
                    style_profile=style_profile,
                    max_retries=max_retries,
                )
                if strict_switch is not None:
                    result.fallback_events.append(
                        {
                            "page_no": page_no,
                            "from_provider": strict_switch["from_provider"],
                            "to_provider": strict_switch["to_provider"],
                            "reason": "primary provider failed on strict prose retry",
                        }
                    )
                if _is_low_quality_translation(job.text, normalized):
                    apply_fallback_or_mark(job, fingerprint=fingerprint, reason="low_quality_keep_original")
                    continue
                assign_translation(fingerprint, normalized)
            continue

        except TranslationError as exc:
            result.warnings.append(
                f"page {page_no}: batch chunk [{start_no}-{end_no}] failed: {exc}"
            )
            try:
                split_map, split_switches = _translate_chunk_with_split_fallback(
                    chunk=chunk,
                    primary_provider=primary_provider,
                    backup_provider=backup_provider,
                    style_profile=style_profile,
                    max_retries=max_retries,
                )
                for event in split_switches:
                    result.fallback_events.append(
                        {
                            "page_no": page_no,
                            "from_provider": event["from_provider"],
                            "to_provider": event["to_provider"],
                            "reason": "primary provider failed on split batch text translation",
                        }
                    )
                for job in chunk:
                    normalized = split_map.get(job.block_index)
                    if normalized is None:
                        continue
                    normalized, strict_switch = _repair_low_quality_translation(
                        job=job,
                        translated=normalized,
                        primary_provider=primary_provider,
                        backup_provider=backup_provider,
                        style_profile=style_profile,
                        max_retries=max_retries,
                    )
                    if strict_switch is not None:
                        result.fallback_events.append(
                            {
                                "page_no": page_no,
                                "from_provider": strict_switch["from_provider"],
                                "to_provider": strict_switch["to_provider"],
                                "reason": "primary provider failed on strict prose retry",
                            }
                        )
                    prompt_text = prompt_by_block.get(job.block_index, _compact_text_for_prompt(job.text))
                    fingerprint = _text_fingerprint(prompt_text)
                    if _is_low_quality_translation(job.text, normalized):
                        apply_fallback_or_mark(job, fingerprint=fingerprint, reason="low_quality_keep_original")
                        continue
                    assign_translation(fingerprint, normalized)
                result.warnings.append(
                    f"page {page_no}: batch chunk [{start_no}-{end_no}] recovered by split retry"
                )
                continue
            except TranslationError:
                pass

        retry_candidates: set[int] = set()
        if not single_fallback:
            retry_candidates = {job.block_index for job in chunk if _should_force_single_retry(job.text)}
            if not retry_candidates:
                for job in chunk:
                    prompt_text = prompt_by_block.get(job.block_index, _compact_text_for_prompt(job.text))
                    fingerprint = _text_fingerprint(prompt_text)
                    apply_fallback_or_mark(job, fingerprint=fingerprint, reason="batch_translation_failed_keep_original")
                continue
            result.warnings.append(
                f"page {page_no}: batch chunk [{start_no}-{end_no}] forced single-block retry"
            )

        for job in chunk:
            prompt_text = prompt_by_block.get(job.block_index, _compact_text_for_prompt(job.text))
            fingerprint = _text_fingerprint(prompt_text)
            if not single_fallback and job.block_index not in retry_candidates:
                apply_fallback_or_mark(job, fingerprint=fingerprint, reason="batch_translation_failed_noncritical_keep_original")
                continue
            try:
                translated, used_provider, switched_from = translate_with_fallback(
                    text=prompt_text,
                    primary=primary_provider,
                    backup=backup_provider,
                    style_profile=style_profile,
                    glossary=chunk_glossary,
                    max_retries=max_retries,
                )
                if switched_from:
                    result.fallback_events.append(
                        {
                            "page_no": page_no,
                            "from_provider": switched_from,
                            "to_provider": used_provider,
                            "reason": "primary provider failed on single text block",
                        }
                    )
                normalized = _normalize_translated_text(job.text, translated)
                normalized, strict_switch = _repair_low_quality_translation(
                    job=job,
                    translated=normalized,
                    primary_provider=primary_provider,
                    backup_provider=backup_provider,
                    style_profile=style_profile,
                    max_retries=max_retries,
                )
                if strict_switch is not None:
                    result.fallback_events.append(
                        {
                            "page_no": page_no,
                            "from_provider": strict_switch["from_provider"],
                            "to_provider": strict_switch["to_provider"],
                            "reason": "primary provider failed on strict prose retry",
                        }
                    )
                if _is_low_quality_translation(job.text, normalized):
                    apply_fallback_or_mark(job, fingerprint=fingerprint, reason="low_quality_keep_original")
                    continue
                assign_translation(fingerprint, normalized)
            except TranslationError as exc:
                resolved = apply_fallback_or_mark(job, fingerprint=fingerprint, reason=f"translation_failed: {exc}")
                if not resolved:
                    result.warnings.append(f"page {page_no}: text block translation failed")

    if store is not None and memory_updates:
        store.set_translation_memory_bulk(session_id, memory_updates)

    return translated_by_block



def _chunk_jobs_for_translation(
    jobs: list[_TextBlockJob],
    *,
    max_blocks: int,
    max_chars: int,
    prompt_texts: dict[int, str] | None = None,
) -> list[list[_TextBlockJob]]:
    chunks: list[list[_TextBlockJob]] = []
    current: list[_TextBlockJob] = []
    current_chars = 0
    # Very long segments are sensitive to neighboring context in a large batch.
    # Translate them alone to reduce cross-segment contamination.
    long_segment_char_threshold = max(760, min(980, int(max_chars * 0.30)))

    for job in jobs:
        if prompt_texts is not None:
            text_chars = len(prompt_texts.get(job.block_index, ""))
        else:
            text_chars = len(_compact_text_for_prompt(job.text))
        if text_chars >= long_segment_char_threshold:
            if current:
                chunks.append(current)
                current = []
                current_chars = 0
            chunks.append([job])
            continue
        should_split = (
            bool(current)
            and (len(current) >= max_blocks or (current_chars + text_chars) > max_chars)
        )
        if should_split:
            chunks.append(current)
            current = []
            current_chars = 0

        current.append(job)
        current_chars += text_chars

    if current:
        chunks.append(current)
    return chunks



def _translate_chunk_with_split_fallback(
    *,
    chunk: list[_TextBlockJob],
    primary_provider: ProviderRuntime,
    backup_provider: ProviderRuntime | None,
    style_profile: str,
    max_retries: int,
    depth: int = 0,
    max_depth: int = 2,
) -> tuple[dict[int, str], list[dict[str, str]]]:
    if not chunk:
        return {}, []

    texts = [_compact_text_for_prompt(job.text) for job in chunk]
    try:
        translated_list, used_provider, switched_from = translate_batch_with_fallback(
            texts=texts,
            primary=primary_provider,
            backup=backup_provider,
            style_profile=style_profile,
            glossary=None,
            max_retries=max_retries,
        )
        mapped = {
            job.block_index: _normalize_translated_text(job.text, translated_list[idx])
            for idx, job in enumerate(chunk)
        }
        switches: list[dict[str, str]] = []
        if switched_from:
            switches.append({"from_provider": switched_from, "to_provider": used_provider})
        return mapped, switches
    except TranslationError:
        if len(chunk) <= 2 or depth >= max_depth:
            raise

    mid = len(chunk) // 2
    left_map, left_switches = _translate_chunk_with_split_fallback(
        chunk=chunk[:mid],
        primary_provider=primary_provider,
        backup_provider=backup_provider,
        style_profile=style_profile,
        max_retries=max_retries,
        depth=depth + 1,
        max_depth=max_depth,
    )
    right_map, right_switches = _translate_chunk_with_split_fallback(
        chunk=chunk[mid:],
        primary_provider=primary_provider,
        backup_provider=backup_provider,
        style_profile=style_profile,
        max_retries=max_retries,
        depth=depth + 1,
        max_depth=max_depth,
    )
    merged = {**left_map, **right_map}
    return merged, left_switches + right_switches



def _repair_low_quality_translation(
    *,
    job: _TextBlockJob,
    translated: str,
    primary_provider: ProviderRuntime,
    backup_provider: ProviderRuntime | None,
    style_profile: str,
    max_retries: int,
) -> tuple[str, dict[str, str] | None]:
    if not _is_low_quality_translation(job.text, translated):
        return translated, None
    settings = get_settings()
    should_retry = bool(settings.enable_strict_low_quality_retry) or _should_force_single_retry(job.text)
    if not should_retry:
        return translated, None

    prompt_text = _compact_text_for_prompt(job.text)
    if not prompt_text:
        return translated, None

    try:
        retried, used_provider, switched_from = translate_with_fallback(
            text=prompt_text,
            primary=primary_provider,
            backup=backup_provider,
            style_profile=style_profile,
            glossary=None,
            max_retries=max_retries,
            strict_mode=True,
        )
    except TranslationError:
        return translated, None

    repaired = _normalize_translated_text(job.text, retried)
    if _is_low_quality_translation(job.text, repaired):
        rescued = _rescue_low_quality_block_with_linewise_retry(
            job=job,
            primary_provider=primary_provider,
            backup_provider=backup_provider,
            style_profile=style_profile,
            max_retries=max_retries,
        )
        if rescued is not None:
            rescued_text, rescued_switch = rescued
            if rescued_switch is not None:
                return rescued_text, rescued_switch
            return rescued_text, None
        return translated, None

    if switched_from:
        return repaired, {"from_provider": switched_from, "to_provider": used_provider}
    return repaired, None



def _rescue_low_quality_block_with_linewise_retry(
    *,
    job: _TextBlockJob,
    primary_provider: ProviderRuntime,
    backup_provider: ProviderRuntime | None,
    style_profile: str,
    max_retries: int,
) -> tuple[str, dict[str, str] | None] | None:
    lines = [re.sub(r"\s+", " ", line).strip() for line in job.text.splitlines() if line.strip()]
    if len(lines) < 3:
        return None

    flat_source = " ".join(lines)
    source_words = len(ENGLISH_WORD_RE.findall(flat_source))
    if source_words < 14:
        return None
    sentence_punct = sum(flat_source.count(mark) for mark in (".", "?", "!", ";", ":"))
    if sentence_punct == 0 and source_words < 22:
        return None

    segments: list[str] = []
    current = lines[0]
    for line in lines[1:]:
        candidate = f"{current} {line}".strip()
        if _looks_like_overlap_continuation(prev_text=current, curr_text=line) and len(candidate) <= 320:
            current = candidate
            continue
        # Join short adjacent fragments to preserve sentence context.
        if len(current) < 72 and len(candidate) <= 220:
            current = candidate
            continue
        segments.append(current)
        current = line
    if current:
        segments.append(current)

    if len(segments) <= 1:
        return None

    translated_parts: list[str] = []
    switch_event: dict[str, str] | None = None
    for segment in segments:
        seg_words = len(ENGLISH_WORD_RE.findall(segment))
        if seg_words < 2:
            translated_parts.append(segment)
            continue
        try:
            seg_translated, used_provider, switched_from = translate_with_fallback(
                text=segment,
                primary=primary_provider,
                backup=backup_provider,
                style_profile=style_profile,
                glossary=None,
                max_retries=max_retries,
                strict_mode=True,
            )
        except TranslationError:
            return None

        seg_normalized = _normalize_translated_text(segment, seg_translated)
        if _is_low_quality_translation(segment, seg_normalized):
            return None
        translated_parts.append(seg_normalized)
        if switched_from and switch_event is None:
            switch_event = {"from_provider": switched_from, "to_provider": used_provider}

    rescued_text = "\n".join(part for part in translated_parts if part.strip()).strip()
    if not rescued_text:
        return None
    if _is_low_quality_translation(job.text, rescued_text):
        return None
    return rescued_text, switch_event


