"""PDF page pipeline entrypoints: rendering, region guards, page translation."""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

import fitz
from PIL import Image, ImageDraw

from app.core.settings import get_settings
from app.services.geom import (
    _dedupe_regions,
    _horizontal_overlap_ratio,
    _is_in_table_region,
    _max_overlap_ratio_to_regions,
    _rect_intersects,
    _rect_iou,
)
from app.services.heuristics import (
    ABBR_RE,
    ALGORITHM_HEADING_RE,
    REFERENCE_ENTRY_START_RE,
    REFERENCE_HEADING_RE,
    TABLE_NOTE_RE,
    _is_algorithm_block_text,
    _is_ieee_license_watermark_text,
    _is_reference_entry_text,
    _is_reference_page,
    _is_running_header_footer_text,
    _is_table_body_text,
    _is_table_caption_text,
    _is_table_header_text,
    _is_table_like_text,
    _log_block_skip,
    _looks_like_formula_narrative_fragment,
    _looks_like_narrative_prose,
    _looks_like_notation_style_line,
    _looks_like_overlap_continuation,
    _looks_like_paragraph_block,
)
from app.services.jobtypes import PageProcessResult, _TextBlockJob  # noqa: F401  (PageProcessResult re-exported)
from app.services.layout import (
    _get_grobid_reference_regions_cached,
    _layout_reference_start_y,
)
from app.services.orchestrator import _translate_text_jobs
from app.services.promptprep import _should_translate_text_block
from app.services.render import _render_text_job
from app.services.textutil import ENGLISH_WORD_RE, _extract_block_text_style
from app.services.translator import ProviderRuntime

logger = logging.getLogger(__name__)


MIN_RULE_LINE_WIDTH_RATIO = 0.24

MAX_RULE_SLOPE = 1.1

RULE_Y_MERGE_TOL = 2.0

RULE_BAND_MARGIN = 2.5

_reference_anchor_cache: dict[tuple[str, int, int], tuple[int, float] | None] = {}

def _collect_reference_heading_boxes(
    blocks: list[dict[str, Any]],
    zoom: float,
) -> list[tuple[float, float, float, float]]:
    boxes: list[tuple[float, float, float, float]] = []
    for block in blocks:
        if block.get("type", -1) != 0:
            continue
        text, _, _ = _extract_block_text_style(block)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            continue
        if not any(REFERENCE_HEADING_RE.match(re.sub(r"\s+", " ", line).strip()) for line in lines[:2]):
            continue
        x0, y0, x1, y1 = [float(v) * zoom for v in block.get("bbox", [0, 0, 0, 0])]
        boxes.append((x0, y0, x1, y1))
    return boxes


def extract_glossary_terms(source_pdf: Path, max_terms: int = 80) -> list[str]:
    terms: dict[str, int] = {}
    with fitz.open(source_pdf) as doc:
        page_limit = min(5, doc.page_count)
        for idx in range(page_limit):
            text = doc[idx].get_text("text")
            for term in ABBR_RE.findall(text):
                terms[term] = terms.get(term, 0) + 1

    ranked = sorted(terms.items(), key=lambda x: (-x[1], x[0]))
    return [term for term, _ in ranked[:max_terms]]


def render_original_pages(
    source_pdf: Path,
    original_dir: Path,
    pages: list[int] | None = None,
) -> int:
    """Render page images from the source PDF.

    ``pages`` limits rendering to the given 1-based page numbers (used for
    lazy rendering of priority pages). ``None`` renders every page. Writes are
    atomic (tmp file + replace) so concurrent readers never see partial PNGs.
    """
    settings = get_settings()
    original_dir.mkdir(parents=True, exist_ok=True)
    with fitz.open(source_pdf) as doc:
        if pages is None:
            page_numbers = range(1, doc.page_count + 1)
        else:
            page_numbers = sorted({p for p in pages if 1 <= p <= doc.page_count})
        zoom = settings.render_dpi / 72
        matrix = fitz.Matrix(zoom, zoom)
        for page_no in page_numbers:
            page = doc[page_no - 1]
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            out_path = original_dir / f"{page_no}.png"
            tmp_path = original_dir / f".{page_no}.png.tmp"
            # pix.save() dispatches on file extension (".tmp" is unknown),
            # so write explicit PNG bytes instead.
            tmp_path.write_bytes(pix.tobytes("png"))
            os.replace(tmp_path, out_path)
        return doc.page_count


def _reference_anchor_cache_key(source_pdf: Path, zoom: float) -> tuple[str, int, int]:
    stat = source_pdf.stat()
    return (source_pdf.resolve().as_posix(), stat.st_mtime_ns, int(zoom * 1000))


def _get_reference_anchor_cached(
    *,
    source_pdf: Path,
    doc: fitz.Document,
    zoom: float,
) -> tuple[int, float] | None:
    key = _reference_anchor_cache_key(source_pdf, zoom)
    if key in _reference_anchor_cache:
        return _reference_anchor_cache[key]

    anchor = _find_reference_chapter_anchor(doc, zoom)
    _reference_anchor_cache[key] = anchor
    if len(_reference_anchor_cache) > 256:
        _reference_anchor_cache.pop(next(iter(_reference_anchor_cache)))
    return anchor


def translate_page_to_image(
    session_id: str | None,
    source_pdf: Path,
    output_path: Path,
    page_no: int,
    primary_provider: ProviderRuntime,
    backup_provider: ProviderRuntime | None,
    style_profile: str,
    glossary: list[str] | None = None,
    max_retries: int = 2,
) -> PageProcessResult:
    settings = get_settings()
    result = PageProcessResult()
    zoom = settings.render_dpi / 72
    matrix = fitz.Matrix(zoom, zoom)

    with fitz.open(source_pdf) as doc:
        reference_anchor = _get_reference_anchor_cached(source_pdf=source_pdf, doc=doc, zoom=zoom)
        grobid_reference_map = _get_grobid_reference_regions_cached(source_pdf=source_pdf, zoom=zoom) or {}
        grobid_reference_regions = grobid_reference_map.get(page_no, [])
        page = doc[page_no - 1]
        page_dict = page.get_text("dict")
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        table_regions = _extract_table_regions(page, zoom)
        blocks = page_dict.get("blocks", [])
        non_translatable_regions = _collect_non_translatable_regions(
            page=page,
            blocks=blocks,
            zoom=zoom,
            table_regions=table_regions,
        )
        references_start_y = _find_references_start_y(blocks, zoom)
        reference_page_mode = _is_reference_page(blocks)

    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    draw = ImageDraw.Draw(image)
    if grobid_reference_regions:
        non_translatable_regions.extend(grobid_reference_regions)
    non_translatable_regions = _dedupe_regions(non_translatable_regions, iou_threshold=0.90)

    logger.info(
        "page %s region_summary: table=%s non_translatable=%s grobid_ref=%s",
        page_no,
        len(table_regions),
        len(non_translatable_regions),
        len(grobid_reference_regions),
    )
    page_height = float(pix.height)
    page_width = float(pix.width)
    reference_heading_boxes = _collect_reference_heading_boxes(blocks, zoom)
    grobid_ref_start_y = _layout_reference_start_y(grobid_reference_regions, page_height)
    if grobid_ref_start_y is not None:
        if references_start_y is None:
            references_start_y = grobid_ref_start_y
        else:
            references_start_y = min(references_start_y, grobid_ref_start_y)
        if grobid_ref_start_y <= page_height * 0.18 and len(grobid_reference_regions) >= 3:
            reference_page_mode = True

    # If references section is detected (via "References" chapter heading):
    # - On the References page itself: skip only content below the heading (references_start_y)
    # - On pages after References: skip entire page
    # - On pages before References: keep existing logic (if refs in lower portion, translate upper)
    # If references section is detected (via "References" chapter heading):
    # - On the References page itself: skip only content below the heading (references_start_y)
    # - On pages after References: skip entire page
    # - On pages before References: keep existing logic (if refs in lower portion, translate upper)
    if reference_anchor is not None:
        anchor_page_no, anchor_y = reference_anchor
        if page_no > anchor_page_no:
            # Pages after References section - skip entire page
            references_start_y = 0.0
            reference_page_mode = True
        elif page_no == anchor_page_no:
            # On the References page - force use anchor_y to skip only content below heading
            # This ensures content ABOVE the References heading is translated
            references_start_y = anchor_y
            reference_page_mode = True
            logger.debug(
                "Force set references_start_y=anchor_y=%.2f for page %s (anchor_page=%s)",
                anchor_y,
                page_no,
                anchor_page_no,
            )
        elif references_start_y is not None and references_start_y > page_height * 0.40:
            # For pages before references: if refs start in lower portion, translate upper content
            reference_page_mode = False
    logger.info(
        "page %s reference_guard: mode=%s start_y=%s anchor=%s headings=%s",
        page_no,
        reference_page_mode,
        f"{references_start_y:.1f}" if references_start_y is not None else "None",
        reference_anchor,
        len(reference_heading_boxes),
    )
    text_jobs: list[_TextBlockJob] = []
    for block_index, block in enumerate(blocks):
        if block.get("type", -1) != 0:
            continue
        job = _build_text_job(
            block_index=block_index,
            block=block,
            page_no=page_no,
            zoom=zoom,
            page_height=page_height,
            page_width=page_width,
            non_translatable_regions=non_translatable_regions,
            references_start_y=references_start_y,
            reference_page_mode=reference_page_mode,
            reference_heading_boxes=reference_heading_boxes,
        )
        if job is not None:
            text_jobs.append(job)
    merged_jobs = _merge_fragmented_text_jobs(text_jobs, page_width=page_width)
    if len(merged_jobs) != len(text_jobs):
        logger.info("page %s merged text jobs: %s -> %s", page_no, len(text_jobs), len(merged_jobs))
    text_jobs = merged_jobs

    translated_map = _translate_text_jobs(
        session_id=session_id,
        jobs=text_jobs,
        result=result,
        page_no=page_no,
        primary_provider=primary_provider,
        backup_provider=backup_provider,
        style_profile=style_profile,
        glossary=glossary,
        max_retries=max_retries,
    )
    job_by_index = {job.block_index: job for job in text_jobs}

    render_jobs: list[_TextBlockJob] = []
    for block_index in translated_map:
        job = job_by_index.get(block_index)
        if job is not None:
            render_jobs.append(job)

    render_jobs.sort(key=_text_job_render_sort_key)
    for job in render_jobs:
        translated = translated_map.get(job.block_index)
        if translated is None:
            continue
        _render_text_job(job=job, translated=translated, draw=draw, result=result, page_no=page_no)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return result


def _build_text_job(
    block_index: int,
    block: dict[str, Any],
    page_no: int,
    zoom: float,
    page_height: float,
    page_width: float,
    non_translatable_regions: list[tuple[float, float, float, float]],
    references_start_y: float | None = None,
    reference_page_mode: bool = False,
    reference_heading_boxes: list[tuple[float, float, float, float]] | None = None,
) -> _TextBlockJob | None:
    bbox = block.get("bbox", [0, 0, 0, 0])
    x0, y0, x1, y1 = [float(v) * zoom for v in bbox]
    text, base_size, color = _extract_block_text_style(block)
    if _is_running_header_footer_text(text=text, y0=y0, y1=y1, page_height=page_height):
        _log_block_skip(page_no=page_no, block_index=block_index, reason="running_header_footer", text=text)
        return None
    if _is_in_table_region((x0, y0, x1, y1), non_translatable_regions):
        # Guard against oversized / noisy table masks:
        # keep narrative prose and formula-explanation fragments translatable
        # unless they are clearly dominated by the non-translatable region.
        overlap_ratio = _max_overlap_ratio_to_regions((x0, y0, x1, y1), non_translatable_regions)
        narrative_like = (
            _looks_like_paragraph_block(text)
            or _looks_like_narrative_prose(text)
            or _looks_like_formula_narrative_fragment(text)
        )
        if overlap_ratio < 0.52:
            pass
        elif narrative_like and overlap_ratio < 0.82:
            pass
        else:
            _log_block_skip(page_no=page_no, block_index=block_index, reason="in_non_translatable_region", text=text)
            return None
    # Reference guard:
    # - whole-page references mode when start_y is None or <= 1
    # - otherwise only skip content below heading, and prefer same-column filtering
    #   to avoid suppressing left-column prose when "REFERENCES" starts in right column.
    if reference_page_mode and (references_start_y is None or references_start_y <= 1.0):
        _log_block_skip(page_no=page_no, block_index=block_index, reason="reference_page_mode", text=text)
        return None
    if references_start_y is not None and y0 >= references_start_y - 1.0:
        if _in_reference_heading_column(
            rect=(x0, y0, x1, y1),
            heading_boxes=reference_heading_boxes or [],
            page_width=page_width,
        ):
            _log_block_skip(page_no=page_no, block_index=block_index, reason="below_references_anchor", text=text)
            return None
    if _is_ieee_license_watermark_text(text):
        _log_block_skip(page_no=page_no, block_index=block_index, reason="ieee_license_watermark", text=text)
        return None
    if _is_algorithm_block_text(text):
        _log_block_skip(page_no=page_no, block_index=block_index, reason="algorithm_block_text", text=text)
        return None
    if _is_table_like_text(text):
        _log_block_skip(page_no=page_no, block_index=block_index, reason="table_like_text", text=text)
        return None

    if not _should_translate_text_block(text):
        _log_block_skip(page_no=page_no, block_index=block_index, reason="should_translate_false", text=text)
        return None
    if _is_reference_entry_text(text):
        _log_block_skip(page_no=page_no, block_index=block_index, reason="reference_entry", text=text)
        return None

    width = max(1.0, x1 - x0)
    height = max(1.0, y1 - y0)

    return _TextBlockJob(
        block_index=block_index,
        text=text,
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        width=width,
        height=height,
        base_font_size=max(base_size * zoom, 10.0),
        color=color,
    )


def _merge_fragmented_text_jobs(jobs: list[_TextBlockJob], *, page_width: float) -> list[_TextBlockJob]:
    if len(jobs) <= 1:
        return jobs

    def _column_key(job: _TextBlockJob) -> int:
        mid = (job.x0 + job.x1) * 0.5
        return 0 if mid < page_width * 0.5 else 1

    # Prefer PDF extraction order inside each column to avoid y-jitter inversions
    # where continuation fragments can be emitted with slightly earlier/later y0.
    ordered = sorted(jobs, key=lambda job: (_column_key(job), job.block_index, job.y0, job.x0))
    merged: list[_TextBlockJob] = []
    idx = 0
    while idx < len(ordered):
        group = [ordered[idx]]
        probe = idx + 1
        while probe < len(ordered):
            prev = group[-1]
            curr = ordered[probe]
            # Keep merge groups conservative to avoid cross-paragraph stitching.
            if len(group) >= 3:
                break
            if not _should_merge_text_jobs(prev=prev, curr=curr, page_width=page_width):
                break
            projected_y0 = min(group[0].y0, curr.y0)
            projected_y1 = max(group[-1].y1, curr.y1)
            projected_height = projected_y1 - projected_y0
            if projected_height > 260.0:
                break
            group.append(curr)
            probe += 1
        if len(group) == 1:
            merged.append(group[0])
        else:
            merged.append(_merge_text_job_group(group))
        idx = probe

    merged.sort(key=lambda job: job.block_index)
    return merged


def _should_merge_text_jobs(*, prev: _TextBlockJob, curr: _TextBlockJob, page_width: float) -> bool:
    prev_mid = (prev.x0 + prev.x1) * 0.5
    curr_mid = (curr.x0 + curr.x1) * 0.5
    prev_col = 0 if prev_mid < page_width * 0.5 else 1
    curr_col = 0 if curr_mid < page_width * 0.5 else 1
    if prev_col != curr_col:
        return False

    if max(prev.height, curr.height) > 260.0:
        return False

    prev_compact = re.sub(r"\s+", " ", prev.text).strip()
    curr_compact = re.sub(r"\s+", " ", curr.text).strip()
    if not prev_compact or not curr_compact:
        return False
    prev_notation_like = _looks_like_notation_style_line(prev_compact)
    curr_notation_like = _looks_like_notation_style_line(curr_compact)

    # Keep section titles and clear standalone headings separate.
    heading_re = r"^\s*(?:section\s+)?[ivxlcdm0-9.\-: ]+[A-Za-z][A-Za-z \-']+$"
    if len(prev_compact) <= 96 and re.match(heading_re, prev_compact, re.IGNORECASE):
        return False
    if len(curr_compact) <= 96 and re.match(heading_re, curr_compact, re.IGNORECASE):
        return False

    overlap = _horizontal_overlap_ratio((prev.x0, prev.x1), (curr.x0, curr.x1))
    if overlap < 0.68:
        return False

    gap = curr.y0 - prev.y1
    overlap_continuation = _looks_like_overlap_continuation(prev_text=prev_compact, curr_text=curr_compact)
    min_width = max(1.0, min(prev.width, curr.width))
    x0_delta = abs(prev.x0 - curr.x0)
    x1_delta = abs(prev.x1 - curr.x1)
    overlap_continuation_merge = gap < -2.5 and overlap >= 0.90 and overlap_continuation
    if overlap_continuation_merge:
        if x0_delta > max(34.0, min_width * 0.18):
            return False
        if x1_delta > max(190.0, min_width * 0.62):
            return False
    else:
        if x0_delta > max(24.0, min_width * 0.10):
            return False
        if x1_delta > max(30.0, min_width * 0.12):
            return False

    # Preserve list/notation visual rhythm by avoiding vertical merge for normal adjacency.
    # Allow only true overlap-continuation merge for these line-style fragments.
    if (prev_notation_like or curr_notation_like) and gap >= -1.0:
        if not overlap_continuation:
            return False
        if gap > 8.5:
            return False
    if gap < -2.5:
        if not (gap >= -42.0 and overlap >= 0.92 and overlap_continuation):
            return False
    max_gap = max(10.0, min(30.0, 0.38 * max(prev.height, curr.height) + 12.0))
    if gap > max_gap:
        return False

    prev_words = len(ENGLISH_WORD_RE.findall(prev_compact))
    curr_words = len(ENGLISH_WORD_RE.findall(curr_compact))
    # Large neighboring prose blocks are usually standalone paragraphs.
    if prev.height >= 44.0 and curr.height >= 44.0 and prev_words >= 18 and curr_words >= 18:
        return False
    if prev.height > 72.0 and curr.height > 72.0 and prev_words > 55 and curr_words > 55:
        return False

    return True


def _merge_text_job_group(group: list[_TextBlockJob]) -> _TextBlockJob:
    anchor = min(group, key=lambda job: job.block_index)
    x0 = min(job.x0 for job in group)
    y0 = min(job.y0 for job in group)
    x1 = max(job.x1 for job in group)
    y1 = max(job.y1 for job in group)

    ordered_group = sorted(group, key=lambda job: job.block_index)
    parts: list[str] = []
    for job in ordered_group:
        part = job.text.strip()
        if not part:
            continue
        if parts and parts[-1].endswith("-"):
            parts[-1] = parts[-1][:-1].rstrip() + part.lstrip()
        else:
            parts.append(part)

    merged_text = "\n".join(parts)
    width = max(1.0, x1 - x0)
    height = max(1.0, y1 - y0)
    base_font_size = max(job.base_font_size for job in group)

    return _TextBlockJob(
        block_index=anchor.block_index,
        text=merged_text,
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        width=width,
        height=height,
        base_font_size=base_font_size,
        color=anchor.color,
    )


def _text_job_render_sort_key(job: _TextBlockJob) -> tuple[float, float, float, int]:
    area = max(1.0, job.width * job.height)
    # Larger blocks first at similar position, then smaller overlays.
    return (job.y0, job.x0, -area, job.block_index)


def _in_reference_heading_column(
    *,
    rect: tuple[float, float, float, float],
    heading_boxes: list[tuple[float, float, float, float]],
    page_width: float,
) -> bool:
    if not heading_boxes:
        return True

    x0, _, x1, _ = rect
    block_mid = (x0 + x1) * 0.5
    for hx0, _, hx1, _ in heading_boxes:
        heading_width = max(1.0, hx1 - hx0)
        # Single-column headings should keep the previous behavior (global below-heading guard).
        if heading_width >= max(120.0, page_width * 0.58):
            return True

        overlap = _horizontal_overlap_ratio((x0, x1), (hx0, hx1))
        if overlap >= 0.28:
            return True

        heading_mid = (hx0 + hx1) * 0.5
        same_half = (block_mid < page_width * 0.5 and heading_mid < page_width * 0.5) or (
            block_mid >= page_width * 0.5 and heading_mid >= page_width * 0.5
        )
        if same_half and abs(block_mid - heading_mid) <= page_width * 0.34:
            return True
    return False


def _find_reference_chapter_anchor(doc: fitz.Document, zoom: float) -> tuple[int, float] | None:
    for idx in range(doc.page_count):
        page = doc[idx]
        blocks = page.get_text("dict").get("blocks", [])
        for block in blocks:
            if block.get("type", -1) != 0:
                continue
            text, _, _ = _extract_block_text_style(block)
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            if not lines:
                continue
            for candidate in lines[:2]:
                line = re.sub(r"\s+", " ", candidate).strip()
                if REFERENCE_HEADING_RE.match(line):
                    bbox = block.get("bbox", [0, 0, 0, 0])
                    _, y0, _, _ = [float(v) * zoom for v in bbox]
                    return idx + 1, y0
    return None


def _find_references_start_y(blocks: list[dict[str, Any]], zoom: float) -> float | None:
    text_blocks: list[tuple[float, float, bool, bool, str]] = []
    for block in blocks:
        if block.get("type", -1) != 0:
            continue
        bbox = block.get("bbox", [0, 0, 0, 0])
        _, y0, _, y1 = [float(v) * zoom for v in bbox]
        text, _, _ = _extract_block_text_style(block)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            continue
        for candidate in lines[:2]:
            line = re.sub(r"\s+", " ", candidate).strip()
            if REFERENCE_HEADING_RE.match(line):
                return y0

        has_entry_line = any(REFERENCE_ENTRY_START_RE.match(re.sub(r"\s+", " ", line).strip()) for line in lines)
        is_entry_block = _is_reference_entry_text(text)
        text_blocks.append((y0, y1, is_entry_block, has_entry_line, text))

    if not text_blocks:
        return None

    text_blocks.sort(key=lambda item: item[0])
    page_bottom = max(item[1] for item in text_blocks)

    # Check if page contains contributions section - if so, skip reference detection entirely
    # to avoid false positive detection of contribution list items as references
    has_contributions = any("contribution" in item[4].lower() for item in text_blocks)
    if has_contributions:
        # Further verify it's actually a contributions section (not just the word appearing in context)
        for y0, y1, is_entry_block, has_entry_line, block_text in text_blocks:
            if has_entry_line and "contribution" in block_text.lower():
                has_proposal_verbs = any(
                    verb in block_text.lower() for verb in ["propose", "design", "present", "develop", "introduce", "suggest", "we "]
                )
                if has_proposal_verbs:
                    # This is a contributions list, not references - skip reference detection
                    return None

    for idx, (y0, _, is_entry_block, has_entry_line, block_text) in enumerate(text_blocks):
        # Skip if this looks like a contributions list (not references).
        # Contributions typically contain proposal verbs and appear in early sections.
        if has_entry_line and "contribution" in block_text.lower():
            has_proposal_verbs = any(
                verb in block_text.lower() for verb in ["propose", "design", "present", "develop", "introduce", "suggest", "we "]
            )
            if has_proposal_verbs:
                continue

        if not (is_entry_block or has_entry_line):
            continue
        tail = text_blocks[idx:]
        entry_count = sum(1 for _, _, is_entry, has_entry, _ in tail if is_entry or has_entry)
        tail_count = len(tail)
        ratio = entry_count / max(tail_count, 1)
        lower_half = y0 >= page_bottom * 0.32
        if entry_count >= 6:
            return y0
        if lower_half and entry_count >= 3 and ratio >= 0.42:
            return y0
        if entry_count >= 4 and ratio >= 0.6:
            return y0
    return None


def _extract_table_regions(page: fitz.Page, zoom: float) -> list[tuple[float, float, float, float]]:
    regions: list[tuple[float, float, float, float]] = []
    table_boxes = _find_tables_with_fallback_strategies(page)
    if not table_boxes:
        table_boxes = _find_tables_with_caption_clips(page)
    if not table_boxes:
        return regions

    # Get text blocks to check for paragraph content below tables
    page_dict = page.get_text("dict")
    all_blocks = page_dict.get("blocks", [])
    page_width = float(page.rect.width)
    page_height = float(page.rect.height)
    page_area = max(1.0, page_width * page_height)
    page_text_blocks = _collect_page_text_blocks(page)

    for bbox in table_boxes:
        raw_x0, raw_y0, raw_x1, raw_y1 = [float(v) for v in bbox]
        raw_w = max(0.0, raw_x1 - raw_x0)
        raw_h = max(0.0, raw_y1 - raw_y0)
        area_ratio = (raw_w * raw_h) / page_area
        # Guard against broad false-positive table regions swallowing narrative prose
        # (e.g., first-page abstract blocks bounded by horizontal rules).
        if area_ratio >= 0.14 and _candidate_has_paragraph_overreach(page_text_blocks, (raw_x0, raw_y0, raw_x1, raw_y1)):
            continue

        x0, y0, x1, y1 = [float(v) * zoom for v in bbox]

        # Check if there's paragraph content below the table
        # and shrink the region to exclude it
        table_bottom = y1 / zoom

        # Look for paragraph-like text below the table (within a wider range)
        paragraph_y: float | None = None
        for block in all_blocks:
            if block.get("type", -1) != 0:
                continue
            block_bbox = block.get("bbox", [0, 0, 0, 0])
            bx0, by0, bx1, by1 = [float(v) for v in block_bbox]

            # Check if block is below the table (within 80 points)
            if by0 >= table_bottom and by0 < table_bottom + 80:
                # Check horizontal overlap with table (allow some margin)
                table_left = x0 / zoom
                table_right = x1 / zoom
                # If block overlaps significantly with table horizontally
                if bx0 < table_right and bx1 > table_left:
                    text, _, _ = _extract_block_text_style(block)
                    # Special handling for table notes (e.g., "Note: m, p are accuracy...")
                    # These should be translated, not excluded as table content
                    if TABLE_NOTE_RE.match(text.strip()):
                        # Table notes should not be part of the non-translatable region
                        # Shrink table region to exclude this note
                        if paragraph_y is None or by0 < paragraph_y:
                            paragraph_y = by0
                        continue
                    if _looks_like_paragraph_block(text):
                        # Found paragraph below table - mark it for exclusion
                        if paragraph_y is None or by0 < paragraph_y:
                            paragraph_y = by0

        # If we found paragraph text below, shrink the table region
        if paragraph_y is not None:
            y1 = min(y1, paragraph_y * zoom - 1.0)

        # Only expand slightly to ensure border text is not translated.
        # Avoid excessive expansion that would include surrounding text.
        regions.append((x0 - 1.0, y0 - 1.0, x1 + 1.0, y1 + 1.0))

    if regions:
        logger.info("detected table regions: %s", len(regions))
    return regions


def _find_tables_with_fallback_strategies(page: fitz.Page) -> list[tuple[float, float, float, float]]:
    # Prefer line-based detection first (stable for ruled tables), then fallback
    # to text-alignment strategies for borderless scientific tables.
    strategy_options: list[dict[str, Any]] = [
        {},
        {"vertical_strategy": "lines_strict", "horizontal_strategy": "lines_strict"},
        # NOTE: global text-text strategy often over-segments full column / page regions.
        # We keep text-based detection in clipped fallback near captions instead.
        {"vertical_strategy": "lines", "horizontal_strategy": "text", "min_words_horizontal": 1},
        {"vertical_strategy": "text", "horizontal_strategy": "lines", "min_words_vertical": 3},
    ]

    dedup_boxes: list[tuple[float, float, float, float]] = []
    merged_boxes: list[tuple[float, float, float, float]] = []

    page_area = max(1.0, float(page.rect.width * page.rect.height))
    page_width = max(1.0, float(page.rect.width))
    page_height = max(1.0, float(page.rect.height))
    page_text_blocks = _collect_page_text_blocks(page)

    for opts in strategy_options:
        try:
            finder = page.find_tables(**opts)
        except TypeError:
            # Older PyMuPDF builds may not support strategy kwargs.
            if opts:
                continue
            try:
                finder = page.find_tables()
            except Exception:  # noqa: BLE001
                continue
        except Exception:  # noqa: BLE001
            continue

        tables = getattr(finder, "tables", None) or []
        for table in tables:
            bbox = getattr(table, "bbox", None)
            if not bbox or len(bbox) != 4:
                continue
            box = tuple(float(v) for v in bbox)
            width = max(0.0, box[2] - box[0])
            height = max(0.0, box[3] - box[1])
            area_ratio = (width * height) / page_area
            width_ratio = width / page_width
            height_ratio = height / page_height
            row_count = int(getattr(table, "row_count", 0) or 0)
            col_count = int(getattr(table, "col_count", 0) or 0)

            is_text_strategy = opts.get("vertical_strategy") == "text" or opts.get("horizontal_strategy") == "text"
            # Text-based strategies may occasionally over-segment almost the whole page as a table.
            # Guard against these outliers and rely on caption-based table masking instead.
            if area_ratio >= 0.72:
                continue
            if is_text_strategy and area_ratio >= 0.38 and width_ratio >= 0.74 and height_ratio >= 0.45:
                continue
            if is_text_strategy and area_ratio >= 0.52 and width_ratio >= 0.88 and height_ratio >= 0.62:
                continue
            if is_text_strategy and row_count >= 55 and col_count <= 10 and area_ratio >= 0.45:
                continue
            if is_text_strategy and row_count >= 40 and area_ratio >= 0.30:
                continue
            if is_text_strategy and row_count * max(col_count, 1) >= 220 and area_ratio >= 0.20:
                continue
            # Text-alignment strategies may also produce narrow but extremely tall pseudo-tables
            # spanning half a page or more (common around table+paragraph mixed columns).
            if is_text_strategy and height_ratio >= 0.60 and width_ratio <= 0.40:
                continue
            if (
                is_text_strategy
                and height_ratio >= 0.48
                and width_ratio <= 0.32
                and area_ratio >= 0.08
                and row_count <= 18
                and col_count <= 8
            ):
                continue
            # Borrowed from scholarly layout parsing practice:
            # reject table candidates that swallow multiple body-text paragraphs.
            if is_text_strategy and _candidate_has_paragraph_overreach(page_text_blocks, box):
                continue

            if any(_rect_iou(box, kept) > 0.90 for kept in dedup_boxes):
                continue
            dedup_boxes.append(box)
            merged_boxes.append(box)

    return merged_boxes


def _find_tables_with_caption_clips(page: fitz.Page) -> list[tuple[float, float, float, float]]:
    page_dict = page.get_text("dict")
    blocks = page_dict.get("blocks", [])
    text_blocks: list[tuple[float, float, float, float, str]] = []
    for block in blocks:
        if block.get("type", -1) != 0:
            continue
        text, _, _ = _extract_block_text_style(block)
        bx0, by0, bx1, by1 = [float(v) for v in block.get("bbox", [0, 0, 0, 0])]
        stripped = text.strip()
        if not stripped:
            continue
        text_blocks.append((bx0, by0, bx1, by1, stripped))

    text_blocks.sort(key=lambda item: item[1])
    captions: list[tuple[float, float, float, float]] = [
        (bx0, by0, bx1, by1)
        for bx0, by0, bx1, by1, text in text_blocks
        if _is_table_caption_text(text)
    ]

    if not captions:
        return []

    page_width = float(page.rect.width)
    page_height = float(page.rect.height)
    dedup: list[tuple[float, float, float, float]] = []

    for cx0, _, cx1, cy1 in captions:
        paragraph_cut: float | None = None
        for bx0, by0, bx1, _, body_text in text_blocks:
            if by0 <= cy1 + 2.0:
                continue
            if by0 - cy1 > 300.0:
                break
            if _horizontal_overlap_ratio((cx0, cx1), (bx0, bx1)) < 0.30:
                continue
            if TABLE_NOTE_RE.match(body_text):
                paragraph_cut = by0
                break
            if _is_table_body_text(body_text) or _is_table_header_text(body_text):
                continue
            if _looks_like_paragraph_block(body_text):
                paragraph_cut = by0
                break

        clip_bottom = min(page_height, cy1 + min(260.0, page_height * 0.40))
        if paragraph_cut is not None:
            clip_bottom = min(clip_bottom, max(cy1 + 36.0, paragraph_cut - 2.0))
        clip = fitz.Rect(
            max(0.0, cx0 - page_width * 0.05),
            max(0.0, cy1 - 2.0),
            min(page_width, cx1 + page_width * 0.05),
            clip_bottom,
        )
        if clip.height < 40 or clip.width < 80:
            continue
        try:
            finder = page.find_tables(
                clip=clip,
                vertical_strategy="text",
                horizontal_strategy="text",
                min_words_vertical=2,
                min_words_horizontal=1,
            )
        except Exception:  # noqa: BLE001
            continue

        tables = getattr(finder, "tables", None) or []
        for table in tables:
            bbox = getattr(table, "bbox", None)
            if not bbox or len(bbox) != 4:
                continue
            x0, y0, x1, y1 = [float(v) for v in bbox]
            width = max(0.0, x1 - x0)
            height = max(0.0, y1 - y0)
            if width < 80 or height < 28:
                continue
            if y0 > clip.y1 + 6 or y1 < clip.y0 - 6:
                continue
            area_ratio = (width * height) / max(page_width * page_height, 1.0)
            # Clip-based fallback should capture local table body, not full-page zones.
            if area_ratio >= 0.52:
                continue
            box = (x0, y0, x1, y1)
            if any(_rect_iou(box, kept) > 0.90 for kept in dedup):
                continue
            dedup.append(box)

    return dedup


def _collect_non_translatable_regions(
    *,
    page: fitz.Page,
    blocks: list[dict[str, Any]],
    zoom: float,
    table_regions: list[tuple[float, float, float, float]],
) -> list[tuple[float, float, float, float]]:
    regions: list[tuple[float, float, float, float]] = []
    regions.extend(table_regions)
    regions.extend(_extract_caption_table_regions(blocks, zoom))
    regions.extend(_extract_algorithm_rule_regions(page=page, blocks=blocks, zoom=zoom))
    return regions


def _extract_algorithm_rule_regions(
    *,
    page: fitz.Page,
    blocks: list[dict[str, Any]],
    zoom: float,
) -> list[tuple[float, float, float, float]]:
    lines = _extract_horizontal_rule_lines(page)
    if len(lines) < 2:
        return []

    algorithm_headings = _collect_algorithm_heading_boxes(blocks, zoom)
    if not algorithm_headings:
        return []

    candidates = _build_rule_bands(lines, page_width=float(page.rect.width), zoom=zoom)
    regions: list[tuple[float, float, float, float]] = []
    for cx0, cy0, cx1, cy1 in candidates:
        if not _band_has_algorithm_heading((cx0, cy0, cx1, cy1), algorithm_headings):
            continue
        regions.append((cx0 - 2.0, cy0 - 2.0, cx1 + 2.0, cy1 + 2.0))
    if regions:
        logger.info("detected algorithm rule regions: %s", len(regions))
    return regions


def _extract_horizontal_rule_lines(page: fitz.Page) -> list[tuple[float, float, float]]:
    out: list[tuple[float, float, float]] = []
    page_width = float(page.rect.width)
    min_width = max(50.0, page_width * MIN_RULE_LINE_WIDTH_RATIO)

    try:
        drawings = page.get_drawings()
    except Exception:  # noqa: BLE001
        return out

    for drawing in drawings:
        items = drawing.get("items") or []
        for item in items:
            if not isinstance(item, tuple) or len(item) < 3:
                continue
            op = item[0]
            if op != "l":
                continue
            p0, p1 = item[1], item[2]
            x0, y0 = float(p0.x), float(p0.y)
            x1, y1 = float(p1.x), float(p1.y)
            if abs(y1 - y0) > MAX_RULE_SLOPE:
                continue
            left = min(x0, x1)
            right = max(x0, x1)
            width = right - left
            if width < min_width:
                continue
            y = (y0 + y1) / 2.0
            out.append((left, y, right))

    return out


def _build_rule_bands(
    lines: list[tuple[float, float, float]],
    *,
    page_width: float,
    zoom: float,
) -> list[tuple[float, float, float, float]]:
    if not lines:
        return []

    sorted_lines = sorted(lines, key=lambda x: x[1])
    merged: list[tuple[float, float, float]] = []
    for left, y, right in sorted_lines:
        if not merged:
            merged.append((left, y, right))
            continue

        m_left, m_y, m_right = merged[-1]
        if abs(y - m_y) <= RULE_Y_MERGE_TOL and _horizontal_overlap_ratio((left, right), (m_left, m_right)) >= 0.7:
            merged[-1] = (min(left, m_left), (y + m_y) / 2.0, max(right, m_right))
        else:
            merged.append((left, y, right))

    min_height = max(18.0, 38.0 * zoom / 2.22)
    max_height = max(220.0, 980.0 * zoom / 2.22)
    min_band_width = max(90.0, page_width * 0.20)

    bands: list[tuple[float, float, float, float]] = []
    for idx in range(len(merged) - 1):
        top_left, top_y, top_right = merged[idx]
        for j in range(idx + 1, len(merged)):
            bot_left, bot_y, bot_right = merged[j]
            height = bot_y - top_y
            if height < min_height:
                continue
            if height > max_height:
                break

            overlap_left = max(top_left, bot_left)
            overlap_right = min(top_right, bot_right)
            overlap_width = overlap_right - overlap_left
            if overlap_width < min_band_width:
                continue

            width_ratio = overlap_width / max(1.0, min(top_right - top_left, bot_right - bot_left))
            if width_ratio < 0.72:
                continue

            bands.append(
                (
                    overlap_left - RULE_BAND_MARGIN,
                    top_y - RULE_BAND_MARGIN,
                    overlap_right + RULE_BAND_MARGIN,
                    bot_y + RULE_BAND_MARGIN,
                )
            )
            break

    dedup: list[tuple[float, float, float, float]] = []
    for band in bands:
        if any(_rect_iou(band, kept) > 0.92 for kept in dedup):
            continue
        dedup.append(band)
    return dedup


def _collect_algorithm_heading_boxes(blocks: list[dict[str, Any]], zoom: float) -> list[tuple[float, float, float, float]]:
    boxes: list[tuple[float, float, float, float]] = []
    for block in blocks:
        if block.get("type", -1) != 0:
            continue
        text, _, _ = _extract_block_text_style(block)
        lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines() if line.strip()]
        if not lines:
            continue
        if not any(ALGORITHM_HEADING_RE.match(line) for line in lines[:2]):
            continue
        x0, y0, x1, y1 = [float(v) * zoom for v in block.get("bbox", [0, 0, 0, 0])]
        boxes.append((x0, y0, x1, y1))
    return boxes


def _band_has_algorithm_heading(
    band: tuple[float, float, float, float],
    headings: list[tuple[float, float, float, float]],
) -> bool:
    bx0, by0, bx1, by1 = band
    head_limit = by0 + (by1 - by0) * 0.34
    head_rect = (bx0, by0, bx1, head_limit)
    for heading in headings:
        if _rect_intersects(head_rect, heading):
            return True
    return False


def _extract_caption_table_regions(
    blocks: list[dict[str, Any]],
    zoom: float,
) -> list[tuple[float, float, float, float]]:
    text_blocks: list[tuple[float, float, float, float, str]] = []
    for block in blocks:
        if block.get("type", -1) != 0:
            continue
        bbox = block.get("bbox", [0, 0, 0, 0])
        x0, y0, x1, y1 = [float(v) * zoom for v in bbox]
        text, _, _ = _extract_block_text_style(block)
        stripped = text.strip()
        if not stripped:
            continue
        text_blocks.append((x0, y0, x1, y1, stripped))

    regions: list[tuple[float, float, float, float]] = []
    for idx, (cx0, cy0, cx1, cy1, caption_text) in enumerate(text_blocks):
        if not _is_table_caption_text(caption_text):
            continue

        # Guardrail: a very tall "caption" block is almost always narrative prose
        # (e.g., a full paragraph beginning with "Table I ..."), not a real table caption.
        caption_block_height = max(0.0, cy1 - cy0)
        if caption_block_height > max(96.0, 180.0 * zoom / 2.22):
            continue

        max_table_height = max(180.0, 460.0 * zoom / 2.22)
        min_x, max_x = cx0, cx1
        max_y = cy1
        last_y = cy1
        captured_body = False

        for bx0, by0, bx1, by1, body_text in text_blocks[idx + 1 :]:
            if by0 <= cy0:
                continue
            if by1 - cy0 > max_table_height:
                break
            if by0 - last_y > 72:
                break

            overlap = _horizontal_overlap_ratio((cx0, cx1), (bx0, bx1))
            if overlap < 0.35:
                continue

            # Check for table notes (e.g., "Note: m, p are accuracy...")
            # These should be translated, not excluded as table content
            if TABLE_NOTE_RE.match(body_text.strip()):
                # Stop expansion and don't include this as table content
                break

            # Check for paragraph text first - this stops table region expansion.
            # This handles table notes/captions like "Table III shows several p_b..."
            if _looks_like_paragraph_block(body_text):
                break

            if _is_table_body_text(body_text) or _is_table_header_text(body_text):
                captured_body = True
                min_x = min(min_x, bx0)
                max_x = max(max_x, bx1)
                max_y = max(max_y, by1)
                last_y = by1
                continue

        if captured_body:
            regions.append((min_x - 2.0, cy0 - 2.0, max_x + 2.0, max_y + 2.0))
        else:
            regions.append((cx0 - 2.0, cy0 - 2.0, cx1 + 2.0, cy1 + 2.0))

    return regions


def _collect_page_text_blocks(page: fitz.Page) -> list[tuple[float, float, float, float, str]]:
    out: list[tuple[float, float, float, float, str]] = []
    try:
        blocks = page.get_text("dict").get("blocks", [])
    except Exception:  # noqa: BLE001
        return out

    for block in blocks:
        if block.get("type", -1) != 0:
            continue
        text, _, _ = _extract_block_text_style(block)
        stripped = text.strip()
        if not stripped:
            continue
        x0, y0, x1, y1 = [float(v) for v in block.get("bbox", [0, 0, 0, 0])]
        out.append((x0, y0, x1, y1, stripped))
    return out


def _candidate_has_paragraph_overreach(
    text_blocks: list[tuple[float, float, float, float, str]],
    candidate_box: tuple[float, float, float, float],
) -> bool:
    cx0, cy0, cx1, cy1 = candidate_box
    c_area = max(1.0, (cx1 - cx0) * (cy1 - cy0))
    para_hits = 0

    for bx0, by0, bx1, by1, text in text_blocks:
        ix0, iy0 = max(cx0, bx0), max(cy0, by0)
        ix1, iy1 = min(cx1, bx1), min(cy1, by1)
        if ix1 <= ix0 or iy1 <= iy0:
            continue
        inter = (ix1 - ix0) * (iy1 - iy0)
        block_area = max(1.0, (bx1 - bx0) * (by1 - by0))
        overlap_to_block = inter / block_area
        overlap_to_candidate = inter / c_area
        if overlap_to_block < 0.40 and overlap_to_candidate < 0.08:
            continue

        if _is_table_caption_text(text) or _is_table_header_text(text) or _is_table_body_text(text):
            continue
        if not _looks_like_paragraph_block(text):
            continue

        words = len(ENGLISH_WORD_RE.findall(text))
        if words < 8:
            continue
        sentence_punct = sum(text.count(mark) for mark in (".", "?", "!", ";", ":"))
        # A table candidate that almost fully covers one long narrative paragraph
        # is typically a false positive around abstract/introduction zones.
        if overlap_to_block >= 0.78 and words >= 42 and sentence_punct >= 2:
            return True
        para_hits += 1
        if para_hits >= 2:
            return True

    return False


