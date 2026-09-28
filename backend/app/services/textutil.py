"""Low-level text normalization, extraction, and truncation helpers."""
from __future__ import annotations

import re
import unicodedata
from typing import Any


ENGLISH_WORD_RE = re.compile(r"\b[A-Za-z]{3,}\b")

CJK_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")

CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

NUMBERED_ITEM_MARKER_RE = re.compile(
    r"(?:^|[\n\r;；，,。：:\.]\s*)(?P<label>(?:[（(]?\d{1,2}[）)])|(?:\d{1,2}[\.、:：]))"
)

def _extract_block_text_style(block: dict[str, Any]) -> tuple[str, float, tuple[int, int, int]]:
    text_parts: list[str] = []
    font_sizes: list[float] = []
    colors: list[tuple[int, int, int]] = []

    for line in block.get("lines", []):
        for span in line.get("spans", []):
            text_parts.append(span.get("text", ""))
            if span.get("size"):
                font_sizes.append(float(span["size"]))
            color_int = int(span.get("color", 0))
            colors.append(_int_to_rgb(color_int))
        text_parts.append("\n")

    text = _clean_extracted_text("".join(text_parts).strip())
    base_size = max(font_sizes) if font_sizes else 11.0
    color = colors[0] if colors else (0, 0, 0)
    return text, base_size, color



def _clean_extracted_text(text: str) -> str:
    if not text:
        return text
    cleaned = CONTROL_CHAR_RE.sub("", text)
    cleaned = _normalize_common_unicode_text(cleaned)
    # Some PDF extractions contain replacement boxes that break readability.
    cleaned = cleaned.replace("□", "")
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()



def _normalize_common_unicode_text(text: str) -> str:
    if not text:
        return text

    out = unicodedata.normalize("NFKC", text)
    out = (
        out.replace("\u2217", "*")
        .replace("\u2212", "-")
        .replace("\u2032", "'")
        .replace("\u2033", "''")
        .replace("\u2034", "'''")
        .replace("\u2044", "/")
        .replace("\u2308", "[")
        .replace("\u2309", "]")
        .replace("\u230a", "[")
        .replace("\u230b", "]")
    )
    # Repair common mojibake artifacts seen in model outputs.
    out = re.sub(r"Â(?=[·±×÷])", "", out)
    out = out.replace("Â", "")
    out = re.sub(r"(?<=[,;:])(?=[A-Za-z])", " ", out)
    out = re.sub(r"([*=/<>≤≥±×÷])(?=[A-Za-z])", r"\1 ", out)
    # Replace unknown square placeholders inside formulas/variables.
    out = re.sub(r"(?<=[A-Za-z0-9])□(?=[A-Za-z0-9'*])", "*", out)
    out = out.replace("□", "")
    return out



def _int_to_rgb(color_int: int) -> tuple[int, int, int]:
    return (color_int >> 16 & 255, color_int >> 8 & 255, color_int & 255)



def _count_numbered_item_markers(text: str) -> int:
    if not text:
        return 0
    return len(list(NUMBERED_ITEM_MARKER_RE.finditer(text)))



def _numbered_item_positions(text: str) -> list[int]:
    positions: list[int] = []
    for match in NUMBERED_ITEM_MARKER_RE.finditer(text):
        positions.append(match.start("label"))
    return positions



def _truncate_by_sentence(text: str, *, max_len: int) -> str:
    clean = text.strip()
    if len(clean) <= max_len:
        return clean

    pieces = re.split(r"(?<=[。！？!?；;])\s*", clean)
    kept: list[str] = []
    total = 0
    for piece in pieces:
        item = piece.strip()
        if not item:
            continue
        if kept and (total + len(item)) > max_len:
            break
        kept.append(item)
        total += len(item)
        if total >= max_len:
            break

    if kept:
        return "".join(kept).strip()
    return clean[:max_len].rstrip(" ,;:.-，；：")



def _trim_extra_numbered_items(source_text: str, translated_text: str) -> str:
    translated = translated_text.strip()
    if not translated:
        return translated

    src_count = _count_numbered_item_markers(source_text)
    tgt_positions = _numbered_item_positions(translated)
    tgt_count = len(tgt_positions)
    if tgt_count <= 1:
        return translated

    # If source contains a single list marker but translation contains many items,
    # keep only the first item to prevent cross-segment expansion.
    if src_count <= 1 and tgt_count >= 2:
        cut = tgt_positions[1]
        trimmed = translated[:cut].rstrip(" ,;:.-，；：")
        if len(trimmed) >= 8:
            return trimmed
        return translated

    if src_count >= 2 and tgt_count > src_count + 1:
        cut = tgt_positions[src_count]
        trimmed = translated[:cut].rstrip(" ,;:.-，；：")
        if len(trimmed) >= 8:
            return trimmed

    return translated



def _trim_overlong_fragment_translation(source_text: str, translated_text: str) -> str:
    source = re.sub(r"\s+", " ", source_text).strip()
    translated = translated_text.strip()
    if not source or not translated:
        return translated

    src_len = len(source)
    if src_len > 240:
        return translated

    src_en = len(ENGLISH_WORD_RE.findall(source))
    tgt_len = len(translated)
    tgt_cjk = len(CJK_CHAR_RE.findall(translated))
    sentence_marks = sum(translated.count(mark) for mark in ("。", "！", "？", ".", ";", "；"))
    source_lower = source.lower()
    fragment_tail = source.endswith("-") or bool(
        re.search(r"(?:\b(for both the|for the|of the|and the|to the|in the|with the)\s*|[,;:])$", source_lower)
    )

    if src_en >= 5 and src_len <= 180 and tgt_cjk >= 80 and tgt_len >= int(src_len * 1.6) and sentence_marks >= 3:
        return _truncate_by_sentence(translated, max_len=max(120, int(src_len * 1.55)))

    if fragment_tail and src_len <= 200 and tgt_cjk >= 56 and tgt_len >= int(src_len * 1.45) and sentence_marks >= 2:
        return _truncate_by_sentence(translated, max_len=max(108, int(src_len * 1.35)))

    return translated



def _first_sentence(text: str) -> str:
    parts = re.split(r"[.!?;:]\s+", text, maxsplit=1)
    return parts[0].strip() if parts else text.strip()


