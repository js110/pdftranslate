"""Text rendering onto page images: fonts, fitting, wrapping, drawing."""
from __future__ import annotations

import logging
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

from PIL import ImageDraw, ImageFont

from app.services.heuristics import _is_continuation_marker_line
from app.services.jobtypes import PageProcessResult, _TextBlockJob
from app.services.promptprep import _should_preserve_source_line_layout
from app.services.textutil import (
    CJK_CHAR_RE,
    ENGLISH_WORD_RE,
    _normalize_common_unicode_text,
    _trim_extra_numbered_items,
    _truncate_by_sentence,
)

logger = logging.getLogger(__name__)



def _render_text_job(
    *,
    job: _TextBlockJob,
    translated: str,
    draw: ImageDraw.ImageDraw,
    result: PageProcessResult,
    page_no: int,
) -> None:
    render_text = _prepare_text_for_render(source_text=job.text, translated_text=translated)
    if not render_text:
        return

    pad_x = min(8.0, max(2.0, job.width * 0.012))
    pad_y = min(6.0, max(1.5, job.height * 0.02))
    inner_width = max(1.0, job.width - pad_x * 2)
    inner_height = max(1.0, job.height - pad_y * 2)

    fit = _fit_text_to_box(
        render_text,
        width=inner_width,
        height=inner_height,
        base_font_size=job.base_font_size,
        min_scale=0.92,
        line_spacing=1.1,
        height_factor=1.0,
    )
    if fit is None:
        fit = _fit_text_to_box(
            render_text,
            width=inner_width,
            height=inner_height,
            base_font_size=job.base_font_size,
            min_scale=0.55,
            min_font_size=8.8,
            line_spacing=1.02,
            height_factor=1.0,
        )
    if fit is None:
        fit = _fit_text_to_box(
            render_text,
            width=inner_width,
            height=inner_height,
            base_font_size=job.base_font_size,
            min_scale=0.45,
            min_font_size=8.4,
            line_spacing=1.0,
            height_factor=1.0,
        )
    if fit is None:
        result.overflow_items.append(
            {
                "page_no": page_no,
                "bbox": [job.x0, job.y0, job.x1, job.y1],
                "reason": "text_overflow_after_wrap",
            }
        )
        result.untranslated_items.append(
            {
                "page_no": page_no,
                "bbox": [job.x0, job.y0, job.x1, job.y1],
                "source_excerpt": job.text[:160],
                "reason": "overflow_keep_original",
            }
        )
        return

    lines, font, line_height, total_height = fit
    rendered_font_size = float(getattr(font, "size", 0) or 0.0)
    min_readable_font = max(8.8, min(10.9, job.base_font_size * 0.55))
    if job.height <= 72.0 and job.width >= 240.0:
        min_readable_font = max(min_readable_font, 9.4)

    if rendered_font_size and rendered_font_size < min_readable_font:
        clipped_text = _clip_translation_for_tight_box(
            source_text=job.text,
            translated_text=render_text,
            box_height=job.height,
            box_width=job.width,
        )
        if clipped_text and clipped_text != render_text:
            retry_fit = _fit_text_to_box(
                clipped_text,
                width=inner_width,
                height=inner_height,
                base_font_size=max(8.8, job.base_font_size * 0.92),
                min_scale=0.52,
                min_font_size=8.8,
                line_spacing=1.0,
                height_factor=1.0,
            )
            if retry_fit is not None:
                lines, font, line_height, total_height = retry_fit
                rendered_font_size = float(getattr(font, "size", 0) or 0.0)

    if rendered_font_size and rendered_font_size < min_readable_font:
        result.overflow_items.append(
            {
                "page_no": page_no,
                "bbox": [job.x0, job.y0, job.x1, job.y1],
                "reason": "font_too_small_keep_original",
            }
        )
        result.untranslated_items.append(
            {
                "page_no": page_no,
                "bbox": [job.x0, job.y0, job.x1, job.y1],
                "source_excerpt": job.text[:160],
                "reason": "font_too_small_keep_original",
            }
        )
        return
    if len(lines) >= 3 and total_height < inner_height * 0.72:
        stretch = min(1.28, (inner_height * 0.9) / max(total_height, 1.0))
        if stretch > 1.03:
            line_height *= stretch
            total_height = line_height * len(lines)
    inner_top = job.y0 + pad_y
    limit_bottom = job.y1 - pad_y + 0.1
    current_y = _compute_text_start_y(inner_top=inner_top, inner_height=inner_height, total_height=total_height, line_count=len(lines))
    if not _can_render_all_lines(current_y=current_y, line_height=line_height, line_count=len(lines), limit_bottom=limit_bottom):
        tighter_fit = _fit_text_to_box(
            render_text,
            width=inner_width,
            height=inner_height,
            base_font_size=max(8.0, rendered_font_size - 0.9 if rendered_font_size else job.base_font_size * 0.9),
            min_scale=0.38,
            min_font_size=7.8,
            line_spacing=0.98,
            height_factor=0.98,
        )
        if tighter_fit is not None:
            lines, font, line_height, total_height = tighter_fit
            rendered_font_size = float(getattr(font, "size", 0) or rendered_font_size or 0.0)
            if rendered_font_size and rendered_font_size < max(7.8, min_readable_font - 0.6):
                tighter_fit = None
            else:
                current_y = _compute_text_start_y(
                    inner_top=inner_top,
                    inner_height=inner_height,
                    total_height=total_height,
                    line_count=len(lines),
                )
                if not _can_render_all_lines(
                    current_y=current_y,
                    line_height=line_height,
                    line_count=len(lines),
                    limit_bottom=limit_bottom,
                ):
                    tighter_fit = None
        if tighter_fit is None:
            # Last-resort rendering for very short-height definition lines.
            # Prefer a compact Chinese render over keeping a full English source line.
            src_compact = re.sub(r"\s+", " ", job.text).strip()
            src_en_words = len(ENGLISH_WORD_RE.findall(src_compact))
            src_punct = sum(src_compact.count(mark) for mark in (",", ";", ":", ".", "?", "!", "，", "；", "：", "。"))
            tgt_compact = re.sub(r"\s+", " ", render_text).strip()
            tgt_cjk = len(CJK_CHAR_RE.findall(tgt_compact))
            if (
                src_en_words >= 2
                and src_en_words <= 7
                and len(src_compact) <= 64
                and src_punct <= 1
                and job.height <= 34.0
                and job.width >= 120.0
                and len(tgt_compact) <= 24
                and tgt_cjk <= 12
            ):
                rescue_fit = _fit_text_to_box(
                    render_text,
                    width=inner_width,
                    height=inner_height,
                    base_font_size=max(7.8, min((rendered_font_size or job.base_font_size) * 0.88, job.base_font_size * 0.46)),
                    min_scale=0.26,
                    min_font_size=7.0,
                    line_spacing=0.95,
                    height_factor=0.98,
                )
                if rescue_fit is not None:
                    lines, font, line_height, total_height = rescue_fit
                    rendered_font_size = float(getattr(font, "size", 0) or rendered_font_size or 0.0)
                    if rendered_font_size >= 7.0:
                        current_y = _compute_text_start_y(
                            inner_top=inner_top,
                            inner_height=inner_height,
                            total_height=total_height,
                            line_count=len(lines),
                        )
                        if _can_render_all_lines(
                            current_y=current_y,
                            line_height=line_height,
                            line_count=len(lines),
                            limit_bottom=limit_bottom,
                        ):
                            tighter_fit = rescue_fit
        if tighter_fit is None:
            result.overflow_items.append(
                {
                    "page_no": page_no,
                    "bbox": [job.x0, job.y0, job.x1, job.y1],
                    "reason": "layout_clipped_keep_original",
                }
            )
            result.untranslated_items.append(
                {
                    "page_no": page_no,
                    "bbox": [job.x0, job.y0, job.x1, job.y1],
                    "source_excerpt": job.text[:160],
                    "reason": "layout_clipped_keep_original",
                }
            )
            return

    draw.rectangle([(job.x0, job.y0), (job.x1, job.y1)], fill=(255, 255, 255))
    current_x = job.x0 + pad_x
    for line in lines:
        draw.text((current_x, current_y), line, font=font, fill=job.color)
        current_y += line_height



def _prepare_text_for_render(*, source_text: str, translated_text: str) -> str:
    text = _normalize_common_unicode_text(translated_text.strip())
    if not text:
        return text

    if _should_preserve_source_line_layout(source_text):
        return _format_structured_render_text(text)

    # LLM output may contain hard wraps that over-fragment a single PDF block.
    # Collapse them before width-based wrapping to avoid tiny-font rendering.
    src_lines = [line.strip() for line in source_text.splitlines() if line.strip()]
    tgt_lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(tgt_lines) >= max(3, len(src_lines) + 2):
        text = " ".join(tgt_lines)
    else:
        text = re.sub(r"\s*\n+\s*", " ", text)

    text = re.sub(r"[ \t]{2,}", " ", text).strip()
    return text



def _format_structured_render_text(text: str) -> str:
    normalized = _normalize_common_unicode_text(text)
    normalized = re.sub(r"\s*\n+\s*", "\n", normalized).strip()
    raw_lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    if not raw_lines:
        return re.sub(r"\s+", " ", normalized).strip()

    lines: list[str] = []
    for line in raw_lines:
        item = re.sub(r"[ \t]{2,}", " ", line).strip()
        item = re.sub(r"^['`‘’]+\s*", "", item)
        if _is_continuation_marker_line(item):
            continue
        lines.append(item)

    if not lines:
        return ""

    return "\n".join(lines).strip()



def _clip_translation_for_tight_box(
    *,
    source_text: str,
    translated_text: str,
    box_height: float | None = None,
    box_width: float | None = None,
) -> str:
    source = re.sub(r"\s+", " ", source_text).strip()
    translated = re.sub(r"\s+", " ", translated_text).strip()
    if not source or not translated:
        return translated

    clipped = _trim_extra_numbered_items(source, translated)
    if clipped != translated:
        translated = clipped

    src_len = len(source)
    tgt_len = len(translated)
    src_en = len(ENGLISH_WORD_RE.findall(source))
    tgt_cjk = len(CJK_CHAR_RE.findall(translated))
    sentence_marks = sum(translated.count(mark) for mark in ("。", "！", "？", ".", ";", "；"))

    # Very short-height boxes are prone to unreadable tiny fonts.
    # Prefer a compact first-clause translation in these tight regions.
    if (
        box_height is not None
        and box_height <= 42.0
        and (box_width is None or box_width >= 150.0)
        and src_en >= 6
        and tgt_len >= 56
        and sentence_marks >= 1
    ):
        compact_cap = max(42, min(68, int(src_len * 0.62)))
        compact = _truncate_by_sentence(translated, max_len=compact_cap)
        if compact and len(compact) + 4 < len(translated):
            translated = compact

    if src_len <= 220 and src_en >= 5 and tgt_cjk >= 42 and tgt_len >= max(125, int(src_len * 1.45)) and sentence_marks >= 2:
        return _truncate_by_sentence(translated, max_len=max(108, int(src_len * 1.3)))
    return translated



def _compute_text_start_y(*, inner_top: float, inner_height: float, total_height: float, line_count: int) -> float:
    if total_height < inner_height:
        remaining = max(0.0, inner_height - total_height)
        if line_count <= 1:
            return inner_top + remaining * 0.18
        if line_count == 2:
            return inner_top + remaining * 0.12
        return inner_top + min(2.0, remaining * 0.04)
    return inner_top



def _can_render_all_lines(*, current_y: float, line_height: float, line_count: int, limit_bottom: float) -> bool:
    if line_count <= 0:
        return True
    required_bottom = current_y + line_height * line_count
    return required_bottom <= (limit_bottom + 0.01)



def _font_candidates() -> list[str]:
    return [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simsun.ttc",
    ]



@lru_cache(maxsize=1)
def _selected_font_path() -> str | None:
    for path in _font_candidates():
        if Path(path).exists():
            return path
    return None



@lru_cache(maxsize=192)
def _cached_font(size: int, font_path: str | None) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if font_path:
        return ImageFont.truetype(font_path, size)
    return ImageFont.load_default()



def _load_font(size: float) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    safe_size = max(1, int(round(size)))
    font_path = _selected_font_path()
    try:
        return _cached_font(safe_size, font_path)
    except Exception:  # noqa: BLE001
        return ImageFont.load_default()



def _wrap_text(text: str, font: ImageFont.FreeTypeFont | ImageFont.ImageFont, max_width: float) -> list[str]:
    lines: list[str] = []
    for raw_line in text.splitlines() or [text]:
        current = ""
        for ch in raw_line:
            candidate = current + ch
            if font.getlength(candidate) <= max_width or not current:
                current = candidate
            else:
                lines.append(current)
                current = ch
        if current:
            lines.append(current)
    return lines or [text]



def _normalize_math_glyphs_for_render(text: str) -> str:
    if not text:
        return text

    text = _normalize_common_unicode_text(text)

    out: list[str] = []
    for ch in text:
        cp = ord(ch)
        decomp = unicodedata.decomposition(ch)
        tagged = decomp.split(" ", 1)[0] if decomp else ""
        should_fold = (
            0x1D400 <= cp <= 0x1D7FF
            or 0x2070 <= cp <= 0x209F
            or tagged in {"<font>", "<super>", "<sub>"}
        )
        if not should_fold:
            out.append(ch)
            continue

        folded = unicodedata.normalize("NFKC", ch)
        out.append(folded if folded else ch)
    return "".join(out)



def _fit_text_to_box(
    text: str,
    width: float,
    height: float,
    base_font_size: float,
    min_scale: float = 0.88,
    min_font_size: float = 8.0,
    line_spacing: float = 1.2,
    height_factor: float = 1.08,
) -> tuple[list[str], ImageFont.FreeTypeFont | ImageFont.ImageFont, float, float] | None:
    text = _normalize_math_glyphs_for_render(text)
    min_size = max(min_font_size, base_font_size * min_scale)
    candidate = base_font_size
    while candidate >= min_size:
        font = _load_font(candidate)
        lines = _wrap_text(text, font, width)
        line_height = max(font.getbbox("Ag")[3] - font.getbbox("Ag")[1], candidate) * line_spacing
        total_height = line_height * len(lines)
        if total_height <= height * height_factor:
            return lines, font, line_height, total_height
        candidate -= 0.8
    return None


