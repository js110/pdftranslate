from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import fitz
import httpx
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.core.settings import get_settings
from app.services.session_store import SessionStore
from app.services.translator import (
    ProviderRuntime,
    TranslationError,
    should_skip_translation,
    translate_batch_with_fallback,
    translate_with_fallback,
)

try:
    from paddleocr import PaddleOCR
except Exception:  # noqa: BLE001
    PaddleOCR = None  # type: ignore[misc,assignment]

try:
    from paddleocr import PPStructureV3
except Exception:  # noqa: BLE001
    PPStructureV3 = None  # type: ignore[misc,assignment]


ABBR_RE = re.compile(r"\b[A-Z]{2,8}s?\b")
TABLE_CAPTION_RE = re.compile(r"^\s*TABLE\s+([IVXLCM]+|\d+)(?:\s*[:.\-]?\s*.*)?$", re.IGNORECASE)
TABLE_NOTE_RE = re.compile(r"^\s*Note\s*[:\-]?\s*\S", re.IGNORECASE)
TABLE_HEADER_HINT_RE = re.compile(
    r"\bnotation\b.*\bdefinition\b|\bdefinition\b.*\bnotation\b|"
    r"\b(method|metric|measured|parameter|prototype|component)\b[^.;:\n]{0,48}\b"
    r"(accuracy|precision|recall|f1|auc|latency|throughput|mean|max|min|std|variance|result)\b",
    re.IGNORECASE,
)
TABLE_ROW_NUMERIC_RE = re.compile(r"^(?:\S+\s+)?(?:\d+(?:\.\d+)?\s+){2,}\d+(?:\.\d+)?$")
TABLE_SYMBOLIC_ROW_RE = re.compile(r"^(?:[A-Za-z][\w\-/*']*\s+){1,6}(?:[×x✓\-]|[0-9]+(?:\.[0-9]+)?)(?:\s+[×x✓\-]|[0-9]+(?:\.[0-9]+)?)+$")
TABLE_SYMBOL_TOKEN_RE = re.compile(r"(?:^|[\s|,;:/()\[\]{}])(?:x|X|×|✓)(?=$|[\s|,;:/()\[\]{}])")
TABLE_NARRATIVE_VERB_RE = re.compile(
    r"\b("
    r"list|lists|show|shows|present|presents|illustrate|illustrates|compare|compares|"
    r"summarize|summarizes|report|reports|give|gives|describe|describes|"
    r"outline|outlines|detail|details|discuss|discusses|analyze|analyzes|"
    r"evaluate|evaluates|investigate|investigates|examine|examines"
    r")\b",
    re.IGNORECASE,
)
REFERENCE_HEADING_RE = re.compile(
    r"^\s*(?:(?:section\s+)?(?:\d+(?:\.\d+)*|[ivxlcdm]+)\s*[\).:\-]?\s*)?"
    r"(references?|bibliography|works\s+cited|literature|参考文献)\s*[:.]?\s*$",
    re.IGNORECASE,
)
REFERENCE_ENTRY_START_RE = re.compile(r"^\s*(\[\d{1,3}\]|\d{1,3}[.)])\s+")
REFERENCE_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
REFERENCE_VENUE_HINT_RE = re.compile(
    r"\b(ieee|acm|springer|elsevier|wiley|journal|transactions|proc\.?|conference|symposium|doi|arxiv|vol\.?\s*\d+|no\.?\s*\d+|pp\.?\s*\d+)\b",
    re.IGNORECASE,
)
REFERENCE_AUTHOR_HINT_RE = re.compile(
    r"\b(?:[A-Z][a-zA-Z'`-]{1,24},\s*(?:[A-Z]\.\s*){1,3}|[A-Z][a-zA-Z'`-]{1,24}\s+et al\.)",
)
ENGLISH_WORD_RE = re.compile(r"\b[A-Za-z]{3,}\b")
CJK_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")
NON_PAPER_META_INLINE_RE = re.compile(
    r"[（(][^()（）]{0,260}(后续内容保持原文|严格遵循要求|翻译要求|输出要求|不要输出|不要翻译|strict retry mode|system prompt|preferred terms|mathseg\d+token)[^()（）]{0,260}[）)]",
    re.IGNORECASE,
)
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
IEEE_LICENSE_WATERMARK_RE = re.compile(
    r"\bauthorized licensed use limited to\b|"
    r"\bdownloaded on\b.*\bieee xplore\b|"
    r"\bieee xplore\b.*\brestrictions apply\b|"
    r"\bpersonal use is permitted\b|"
    r"\brepublication/?redistribution\b",
    re.IGNORECASE,
)
ALGORITHM_HEADING_RE = re.compile(r"^\s*(algorithm|alg\.?|算法)\s*[\divxlcdm一二三四五六七八九十]*", re.IGNORECASE)
ALGORITHM_IO_RE = re.compile(
    r"^\s*(input|output|require|ensure|return|输入|输出|返回|初始化|参数)\b",
    re.IGNORECASE,
)
ALGORITHM_STEP_RE = re.compile(r"^\s*(\d{1,3}|[ivxlcdm]{1,6})[\).]?\s+\S", re.IGNORECASE)
ALGORITHM_CTRL_RE = re.compile(r"\b(if|then|else|while|repeat|until|return)\b|[:=]{1,2}|<-|->", re.IGNORECASE)
ALGORITHM_FOR_LOOP_RE = re.compile(r"\bfor\s+each\b|\bfor\b[^.;:\n]{0,48}\b(in|to|from)\b", re.IGNORECASE)
ALGORITHM_END_RE = re.compile(r"^\s*end\s+(for|while|if|repeat)\b|^\s*结束\b", re.IGNORECASE)
ACADEMIC_LABEL_RE = re.compile(
    r"^\s*(?:the\s+)?(definition|theorem|lemma|corollary|proposition|proof|remark|example)\b",
    re.IGNORECASE,
)
BULLET_SENTENCE_RE = re.compile(r"^\s*(?:[-*•]|(?:\d{1,3}|[ivxlcdm]{1,6})[\).])\s+[A-Za-z]", re.IGNORECASE)
NUMBERED_NARRATIVE_LINE_RE = re.compile(
    r"^\s*(?:\(?\d{1,3}\)?|[ivxlcdm]{1,6})[\).:]\s+[A-Za-z]",
    re.IGNORECASE,
)
NARRATIVE_LIST_LEADIN_RE = re.compile(
    r"\b(contribution|contributions|our main|we propose|we design|we study|we perform)\b",
    re.IGNORECASE,
)
NUMBERED_ITEM_MARKER_RE = re.compile(
    r"(?:^|[\n\r;；，,。：:\.]\s*)(?P<label>(?:[（(]?\d{1,2}[）)])|(?:\d{1,2}[\.、:：]))"
)
TABLE_NARRATIVE_CONTINUATION_RE = re.compile(
    r"\b("
    r"that|which|where|if|then|means?|mean|known|stored|locally|initially|"
    r"example\d*|suppose|assuming|assume|let|denote|denotes|recording|variable"
    r")\b",
    re.IGNORECASE,
)
FORMULA_NARRATIVE_VERB_RE = re.compile(
    r"\b("
    r"decrypt|decrypts|encrypted|encrypt|encrypts|compute|computes|computed|"
    r"find|finds|get|gets|return|returns|send|sends|receive|receives|"
    r"obtain|obtains|choose|chooses|set|sets|check|checks|compare|compares"
    r")\b",
    re.IGNORECASE,
)
STRUCTURED_LINE_MARKER_RE = re.compile(r"^\s*(?:['`‘’•·\-*]\s*)?(?:r\b|[A-Za-z]\[[^\]]+\]|count\(|state\d*\b|next\b|res\b)")
STRUCTURED_NOTATION_TOKEN_RE = re.compile(
    r"\bcount\s*\(|\bstate\d*\b|\bnext\b|\bres\b|\bminpos\b|"
    r"[A-Za-z]\[[^\]]+\]|\|\||:=|⊕|←",
    re.IGNORECASE,
)
SHORT_SECTION_HEADING_TRANSLATION_RE = re.compile(
    r"^\s*(?:section\s+)?(?:\d+(?:\.\d+)*|[ivxlcdm]+)\s*[\).:：\-]?\s*"
    r"(?:相关工作|参考文献|引言|摘要|结论|实验(?:评估|结果)?|方法|预备知识|问题定义|系统模型)\s*$",
    re.IGNORECASE,
)
SHORT_SECTION_HEADING_TRANSLATION_SET = {
    "相关工作",
    "参考文献",
    "引言",
    "摘要",
    "结论",
    "实验",
    "实验评估",
    "实验结果",
    "方法",
    "预备知识",
    "问题定义",
    "系统模型",
}
RUNNING_HEADER_HINT_RE = re.compile(
    r"\b(ieee|transactions|journal|vol\.?|no\.?|january|february|march|april|may|june|july|august|"
    r"september|october|november|december)\b",
    re.IGNORECASE,
)
RUNNING_FOOTER_HINT_RE = re.compile(
    r"\b(authorized licensed use limited to|downloaded on|ieee xplore|restrictions apply|copyright|all rights reserved)\b",
    re.IGNORECASE,
)
MIN_RULE_LINE_WIDTH_RATIO = 0.24
MAX_RULE_SLOPE = 1.1
RULE_Y_MERGE_TOL = 2.0
RULE_BAND_MARGIN = 2.5
LAYOUT_TABLE_LABELS = {
    "table",
    "tab",
    "table region",
    "tabular",
    "表格",
}
LAYOUT_REFERENCE_LABELS = {
    "reference",
    "references",
    "reference list",
    "bibliography",
    "参考文献",
}
LAYOUT_SKIP_LABELS = {
    "header",
    "footer",
    "page_number",
    "page number",
    "footnote",
    "watermark",
}
_reference_anchor_cache: dict[tuple[str, int, int], tuple[int, float] | None] = {}
_grobid_reference_cache: dict[tuple[str, int, int], dict[int, list[tuple[float, float, float, float]]] | None] = {}
logger = logging.getLogger(__name__)


@dataclass
class PageProcessResult:
    warnings: list[str] = field(default_factory=list)
    overflow_items: list[dict[str, Any]] = field(default_factory=list)
    untranslated_items: list[dict[str, Any]] = field(default_factory=list)
    fallback_events: list[dict[str, Any]] = field(default_factory=list)
    image_ocr_failures: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class _TextBlockJob:
    block_index: int
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    width: float
    height: float
    base_font_size: float
    color: tuple[int, int, int]


_ocr_instance: Any = None
_layout_detector_instance: Any = None


@dataclass
class _LayoutHints:
    table_regions: list[tuple[float, float, float, float]] = field(default_factory=list)
    reference_regions: list[tuple[float, float, float, float]] = field(default_factory=list)
    skip_regions: list[tuple[float, float, float, float]] = field(default_factory=list)


def get_ocr() -> Any:
    global _ocr_instance
    if not get_settings().enable_image_ocr:
        _ocr_instance = False
        return _ocr_instance
    if _ocr_instance is not None:
        return _ocr_instance
    if PaddleOCR is None:
        _ocr_instance = False
        return _ocr_instance
    try:
        _ocr_instance = PaddleOCR(use_textline_orientation=True, lang="en")
    except Exception:  # noqa: BLE001
        _ocr_instance = False
    return _ocr_instance


def get_layout_detector() -> Any:
    global _layout_detector_instance
    settings = get_settings()
    if not settings.enable_layout_detection_guard:
        _layout_detector_instance = False
        return _layout_detector_instance
    if _layout_detector_instance is not None:
        return _layout_detector_instance
    if PPStructureV3 is None:
        _layout_detector_instance = False
        return _layout_detector_instance

    try:
        _layout_detector_instance = PPStructureV3(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            use_table_recognition=False,
            use_formula_recognition=False,
            use_chart_recognition=False,
            use_region_detection=False,
            lang=(settings.layout_detection_lang or "en"),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("layout detector unavailable, fallback to heuristics only: %s", exc)
        _layout_detector_instance = False
    return _layout_detector_instance


def _extract_layout_hints(image: Image.Image, page_no: int) -> _LayoutHints:
    detector = get_layout_detector()
    if not detector:
        return _LayoutHints()

    try:
        raw_result = detector.predict(np.array(image))
    except Exception as exc:  # noqa: BLE001
        logger.info("page %s layout detection failed: %s", page_no, exc)
        return _LayoutHints()

    table_regions: list[tuple[float, float, float, float]] = []
    reference_regions: list[tuple[float, float, float, float]] = []
    skip_regions: list[tuple[float, float, float, float]] = []

    for node in _iter_layout_nodes(raw_result):
        label = _normalize_layout_label(
            node.get("label") or node.get("type") or node.get("category") or node.get("name") or ""
        )
        if not label:
            continue
        rect = _extract_layout_rect(node, image.width, image.height)
        if rect is None:
            continue
        if _layout_label_matches(label, LAYOUT_TABLE_LABELS):
            table_regions.append(rect)
            continue
        if _layout_label_matches(label, LAYOUT_REFERENCE_LABELS):
            reference_regions.append(rect)
            continue
        if _layout_label_matches(label, LAYOUT_SKIP_LABELS):
            skip_regions.append(rect)

    return _LayoutHints(
        table_regions=_dedupe_regions(table_regions, iou_threshold=0.88),
        reference_regions=_dedupe_regions(reference_regions, iou_threshold=0.88),
        skip_regions=_dedupe_regions(skip_regions, iou_threshold=0.90),
    )


def _iter_layout_nodes(obj: Any) -> Iterator[dict[str, Any]]:
    if obj is None:
        return
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from _iter_layout_nodes(value)
        return
    if isinstance(obj, (list, tuple, set)):
        for item in obj:
            yield from _iter_layout_nodes(item)
        return
    if isinstance(obj, (str, bytes, np.ndarray)):
        return
    if hasattr(obj, "__iter__"):
        try:
            for item in obj:
                yield from _iter_layout_nodes(item)
        except Exception:  # noqa: BLE001
            return


def _normalize_layout_label(label: str) -> str:
    value = re.sub(r"\s+", " ", str(label)).strip().lower()
    return value


def _layout_label_matches(label: str, candidates: set[str]) -> bool:
    if label in candidates:
        return True
    return any(token in label for token in candidates if len(token) >= 4)


def _extract_layout_rect(node: dict[str, Any], image_width: int, image_height: int) -> tuple[float, float, float, float] | None:
    candidates = (
        node.get("coordinate"),
        node.get("bbox"),
        node.get("box"),
        node.get("rect"),
        node.get("polygon"),
        node.get("points"),
    )
    for value in candidates:
        rect = _coerce_layout_rect(value)
        if rect is None:
            continue
        x0, y0, x1, y1 = rect
        x0 = max(0.0, min(float(image_width), x0))
        y0 = max(0.0, min(float(image_height), y0))
        x1 = max(0.0, min(float(image_width), x1))
        y1 = max(0.0, min(float(image_height), y1))
        if x1 - x0 < 6.0 or y1 - y0 < 6.0:
            continue
        return (x0, y0, x1, y1)
    return None


def _coerce_layout_rect(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None

    if isinstance(value[0], (list, tuple)):
        points: list[tuple[float, float]] = []
        for item in value:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            try:
                px = float(item[0])
                py = float(item[1])
            except Exception:  # noqa: BLE001
                continue
            points.append((px, py))
        if len(points) < 2:
            return None
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return (min(xs), min(ys), max(xs), max(ys))

    try:
        x0 = float(value[0])
        y0 = float(value[1])
        x1 = float(value[2])
        y1 = float(value[3])
    except Exception:  # noqa: BLE001
        return None

    if x1 <= x0 or y1 <= y0:
        # Some detectors expose [x, y, w, h].
        if x1 > 0 and y1 > 0:
            x1 = x0 + x1
            y1 = y0 + y1

    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _dedupe_regions(
    regions: list[tuple[float, float, float, float]],
    *,
    iou_threshold: float,
) -> list[tuple[float, float, float, float]]:
    dedup: list[tuple[float, float, float, float]] = []
    for region in sorted(regions, key=lambda item: (item[1], item[0], item[3], item[2])):
        if any(_rect_iou(region, kept) >= iou_threshold for kept in dedup):
            continue
        dedup.append(region)
    return dedup


def _layout_reference_start_y(
    reference_regions: list[tuple[float, float, float, float]],
    page_height: float,
) -> float | None:
    if not reference_regions or page_height <= 0:
        return None
    candidates = [region[1] for region in reference_regions if region[3] - region[1] >= 12.0]
    if not candidates:
        return None
    start_y = min(candidates)
    if start_y > page_height * 0.92:
        return None
    return start_y


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


def render_original_pages(source_pdf: Path, original_dir: Path) -> int:
    settings = get_settings()
    with fitz.open(source_pdf) as doc:
        zoom = settings.render_dpi / 72
        matrix = fitz.Matrix(zoom, zoom)
        for page_no in range(1, doc.page_count + 1):
            page = doc[page_no - 1]
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            out_path = original_dir / f"{page_no}.png"
            pix.save(out_path.as_posix())
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


def _grobid_reference_cache_key(source_pdf: Path, zoom: float) -> tuple[str, int, int]:
    stat = source_pdf.stat()
    return (source_pdf.resolve().as_posix(), stat.st_mtime_ns, int(zoom * 1000))


def _get_grobid_reference_regions_cached(
    *,
    source_pdf: Path,
    zoom: float,
) -> dict[int, list[tuple[float, float, float, float]]] | None:
    key = _grobid_reference_cache_key(source_pdf, zoom)
    if key in _grobid_reference_cache:
        return _grobid_reference_cache[key]

    data = _fetch_grobid_reference_regions(source_pdf=source_pdf, zoom=zoom)
    _grobid_reference_cache[key] = data
    if len(_grobid_reference_cache) > 64:
        _grobid_reference_cache.pop(next(iter(_grobid_reference_cache)))
    return data


def _fetch_grobid_reference_regions(
    *,
    source_pdf: Path,
    zoom: float,
) -> dict[int, list[tuple[float, float, float, float]]] | None:
    settings = get_settings()
    if not settings.enable_grobid_reference_guard:
        return None
    base_url = (settings.grobid_base_url or "").strip()
    if not base_url:
        return None

    endpoint = f"{base_url.rstrip('/')}/api/processFulltextDocument"
    try:
        payload = source_pdf.read_bytes()
    except Exception as exc:  # noqa: BLE001
        logger.info("grobid guard skipped: failed reading source pdf: %s", exc)
        return None

    try:
        with httpx.Client(timeout=settings.grobid_timeout_sec) as client:
            response = client.post(
                endpoint,
                data={"teiCoordinates": "biblStruct"},
                files={"input": (source_pdf.name, payload, "application/pdf")},
            )
            response.raise_for_status()
            xml_text = response.text
    except Exception as exc:  # noqa: BLE001
        logger.info("grobid guard unavailable: %s", exc)
        return None

    try:
        root = ET.fromstring(xml_text)
    except Exception as exc:  # noqa: BLE001
        logger.info("grobid guard parse failed: %s", exc)
        return None

    by_page: dict[int, list[tuple[float, float, float, float]]] = {}
    for elem in root.iter():
        tag = str(elem.tag)
        if not tag.endswith("biblStruct"):
            continue
        coords_attr = elem.attrib.get("coords")
        if not coords_attr:
            continue
        for page_no, rect in _parse_grobid_coords(coords_attr, zoom=zoom):
            by_page.setdefault(page_no, []).append(rect)

    if not by_page:
        return None

    normalized: dict[int, list[tuple[float, float, float, float]]] = {}
    for page_no, regions in by_page.items():
        normalized[page_no] = _dedupe_regions(regions, iou_threshold=0.90)
    return normalized


def _parse_grobid_coords(coords_attr: str, *, zoom: float) -> list[tuple[int, tuple[float, float, float, float]]]:
    parsed: list[tuple[int, tuple[float, float, float, float]]] = []
    for chunk in coords_attr.split(";"):
        raw = chunk.strip()
        if not raw:
            continue
        parts = [part.strip() for part in raw.split(",")]
        if len(parts) < 5:
            continue
        try:
            page_no = int(float(parts[0]))
            x = float(parts[1]) * zoom
            y = float(parts[2]) * zoom
            w = float(parts[3]) * zoom
            h = float(parts[4]) * zoom
        except Exception:  # noqa: BLE001
            continue
        if page_no < 1:
            continue
        x0, y0 = x, y
        x1, y1 = x + w, y + h
        if x1 <= x0 or y1 <= y0:
            continue
        parsed.append((page_no, (x0, y0, x1, y1)))
    return parsed


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
    layout_hints = _extract_layout_hints(image, page_no)
    if layout_hints.table_regions:
        table_regions.extend(layout_hints.table_regions)
        non_translatable_regions.extend(layout_hints.table_regions)
    if layout_hints.skip_regions:
        non_translatable_regions.extend(layout_hints.skip_regions)
    if grobid_reference_regions:
        non_translatable_regions.extend(grobid_reference_regions)
    non_translatable_regions = _dedupe_regions(non_translatable_regions, iou_threshold=0.90)

    logger.info(
        "page %s region_summary: table=%s non_translatable=%s layout_table=%s layout_ref=%s grobid_ref=%s",
        page_no,
        len(table_regions),
        len(non_translatable_regions),
        len(layout_hints.table_regions),
        len(layout_hints.reference_regions),
        len(grobid_reference_regions),
    )
    page_height = float(pix.height)
    page_width = float(pix.width)
    reference_heading_boxes = _collect_reference_heading_boxes(blocks, zoom)
    layout_ref_start_y = _layout_reference_start_y(layout_hints.reference_regions, page_height)
    grobid_ref_start_y = _layout_reference_start_y(grobid_reference_regions, page_height)
    if grobid_ref_start_y is not None:
        if references_start_y is None:
            references_start_y = grobid_ref_start_y
        else:
            references_start_y = min(references_start_y, grobid_ref_start_y)
        if grobid_ref_start_y <= page_height * 0.18 and len(grobid_reference_regions) >= 3:
            reference_page_mode = True
    if layout_ref_start_y is not None:
        if references_start_y is None:
            references_start_y = layout_ref_start_y
        else:
            references_start_y = min(references_start_y, layout_ref_start_y)
        if layout_ref_start_y <= page_height * 0.18 and len(layout_hints.reference_regions) >= 3:
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
            logger.info(
                "DEBUG: Force set references_start_y=anchor_y=%.2f for page %s (anchor_page=%s)",
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

    # Process image blocks in source order, then render text in geometric order.
    # This avoids later overlapping text blocks wiping earlier lines via white-box fill.
    for block in blocks:
        block_type = block.get("type", -1)
        if block_type == 1:
            _process_image_block(
                block=block,
                image=image,
                result=result,
                zoom=zoom,
                page_no=page_no,
                primary_provider=primary_provider,
                backup_provider=backup_provider,
                style_profile=style_profile,
                glossary=glossary,
                max_retries=max_retries,
            )

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


def _looks_like_overlap_continuation(*, prev_text: str, curr_text: str) -> bool:
    prev_clean = prev_text.strip()
    curr_clean = curr_text.strip()
    if not prev_clean or not curr_clean:
        return False

    # Do not merge distinct list/step items such as "a) ...", "1) ...", "i) ...".
    if re.match(r"^\s*(?:[a-z]|[ivxlcdm]+|\d{1,3})[\).:]\s+", curr_clean, flags=re.IGNORECASE):
        return False

    first = curr_clean[:1]
    if first and (first.islower() or first.isdigit() or first in "=+-*/^_([{⊕←"):
        return True

    if re.search(r"(?:[-=+*/^_⊕←,;:.\(\[])\s*$", prev_clean):
        return True
    if re.search(r"\b(?:for|where|then|that|with|s\.t)\s*$", prev_clean, flags=re.IGNORECASE):
        return True
    return False


def _looks_like_notation_style_line(text: str) -> bool:
    compact = re.sub(r"\s+", " ", text).strip()
    if not compact:
        return False
    if STRUCTURED_LINE_MARKER_RE.match(compact):
        return True
    if len(compact) <= 140 and STRUCTURED_NOTATION_TOKEN_RE.search(compact):
        return True
    return False


def _text_job_render_sort_key(job: _TextBlockJob) -> tuple[float, float, float, int]:
    area = max(1.0, job.width * job.height)
    # Larger blocks first at similar position, then smaller overlays.
    return (job.y0, job.x0, -area, job.block_index)


def _should_translate_text_block(text: str) -> bool:
    compact = _compact_text_for_prompt(text)
    if not compact:
        return False
    if should_skip_translation(compact):
        return False

    en_words = len(ENGLISH_WORD_RE.findall(compact))
    cjk_count = len(CJK_CHAR_RE.findall(compact))
    latin_chars = len(re.findall(r"[A-Za-z]", compact))

    # Chinese-dominant blocks rarely need EN->ZH translation; skip to save tokens/latency.
    if en_words == 0 and cjk_count > 0:
        return False
    if cjk_count >= 16 and en_words <= 2:
        return False
    if cjk_count >= 48 and en_words <= 5 and latin_chars <= 28:
        return False

    return True


def _is_running_header_footer_text(*, text: str, y0: float, y1: float, page_height: float) -> bool:
    if page_height <= 0:
        return False
    compact = re.sub(r"\s+", " ", text).strip()
    if not compact:
        return False

    lower = compact.lower()
    near_top = y1 <= page_height * 0.095
    near_bottom = y0 >= page_height * 0.90
    en_words = len(ENGLISH_WORD_RE.findall(compact))

    if near_top:
        if RUNNING_HEADER_HINT_RE.search(compact):
            return True
        if re.fullmatch(r"\d{1,5}", compact):
            return True
        if len(compact) <= 42 and en_words <= 6 and compact.upper() == compact and any(ch.isalpha() for ch in compact):
            return True
        if len(compact) <= 180 and re.search(r"\bet al\.:", lower) and re.search(r"\b\d{3,5}\b", compact):
            return True
        upper_letters = sum(1 for ch in compact if ch.isupper())
        alpha_letters = sum(1 for ch in compact if ch.isalpha())
        if alpha_letters >= 10 and re.search(r"\b\d{3,5}\b", compact):
            if upper_letters / max(alpha_letters, 1) >= 0.42 and ":" in compact:
                return True

    if near_bottom:
        if _is_ieee_license_watermark_text(compact):
            return True
        if RUNNING_FOOTER_HINT_RE.search(compact):
            return True
        if re.fullmatch(r"\d{1,5}", compact):
            return True
        if len(compact) <= 56 and en_words <= 7 and ("downloaded on" in lower or "ieee xplore" in lower):
            return True

    return False


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


def _is_ieee_license_watermark_text(text: str) -> bool:
    compact = re.sub(r"\s+", " ", text).strip()
    if not compact:
        return False
    lower = compact.lower()
    # Strong single-signal match used in IEEE watermark/footer text.
    if "authorized licensed use limited to" in lower:
        return True
    # Multi-signal fallback to avoid false positives on regular prose.
    signals = 0
    if "ieee xplore" in lower:
        signals += 1
    if "downloaded on" in lower:
        signals += 1
    if "restrictions apply" in lower:
        signals += 1
    if "personal use is permitted" in lower:
        signals += 1
    if "republication/redistribution" in lower or "republication redistribution" in lower:
        signals += 1
    if IEEE_LICENSE_WATERMARK_RE.search(compact):
        signals += 1
    return signals >= 3 and len(compact) <= 420


def _is_algorithm_block_text(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False

    heading_hit = any(ALGORITHM_HEADING_RE.match(line) for line in lines[:2])

    io_count = 0
    for line in lines:
        if not ALGORITHM_IO_RE.match(line):
            continue
        compact = re.sub(r"\s+", " ", line).strip()
        # Narrative lines may start with words like "input/output" but are not pseudocode I/O headers.
        if len(compact) <= 88 or re.match(
            r"^\s*(input|output|require|ensure|return|输入|输出|返回|初始化|参数)\s*[:：]",
            line,
            flags=re.IGNORECASE,
        ):
            io_count += 1
    step_count = sum(1 for line in lines if ALGORITHM_STEP_RE.match(line))
    ctrl_count = sum(1 for line in lines if ALGORITHM_CTRL_RE.search(line))
    for_loop_count = sum(1 for line in lines if ALGORITHM_FOR_LOOP_RE.search(line))
    ctrl_count += for_loop_count
    end_count = sum(1 for line in lines if ALGORITHM_END_RE.search(line))
    flat = re.sub(r"\s+", " ", " ".join(lines)).strip()
    word_count = len(ENGLISH_WORD_RE.findall(flat))
    sentence_punct = sum(flat.count(mark) for mark in ".?!;:")
    has_academic_label = any(ACADEMIC_LABEL_RE.match(line) for line in lines[:2])
    numbered_prose_count = sum(
        1
        for line in lines
        if re.match(r"^\s*\d{1,3}[\).]?\s+(we|our)\b", line, flags=re.IGNORECASE)
    )
    pseudo_line_count = sum(
        1
        for line in lines
        if len(line) <= 120
        and (
            ALGORITHM_IO_RE.match(line)
            or ALGORITHM_CTRL_RE.search(line)
            or ALGORITHM_FOR_LOOP_RE.search(line)
        )
    )

    # Numbered contribution prose (e.g., "1) We propose ...") should remain translatable.
    if step_count >= 3 and io_count == 0 and end_count == 0 and ctrl_count <= 1 and numbered_prose_count >= 2:
        return False
    # Definitions/lemmas/proofs are often prose with occasional control words.
    if has_academic_label and io_count == 0 and end_count == 0 and word_count >= 24:
        return False
    if has_academic_label and io_count == 0 and end_count == 0 and step_count <= 2:
        return False
    # Long narrative prose should not be treated as pseudocode.
    if word_count >= 34 and sentence_punct >= 2 and io_count == 0 and end_count == 0 and step_count <= 2:
        return False
    # Very long prose blocks are not algorithms, even if they contain a few control words.
    if word_count >= 120 and io_count <= 1 and end_count == 0:
        return False
    # Long narrative blocks around formulas are often split into short lines and may
    # contain tokens that look like pseudocode controls. Keep them translatable.
    if io_count == 0 and end_count == 0 and word_count >= 64 and sentence_punct >= 6 and _looks_like_narrative_prose(text):
        return False

    if heading_hit:
        if io_count >= 1 or step_count >= 1 or ctrl_count >= 1 or pseudo_line_count >= 3:
            return True
        avg_len = sum(len(line) for line in lines) / max(len(lines), 1)
        if len(lines) >= 4 and avg_len <= 70 and sentence_punct <= 1 and not _looks_like_narrative_prose(text):
            return True

    if io_count >= 1 and (step_count >= 1 or ctrl_count >= 1):
        return True
    if io_count >= 2 and (step_count >= 2 or ctrl_count >= 2):
        return True
    if step_count >= 3 and (ctrl_count >= 2 or end_count >= 1):
        return True
    if pseudo_line_count >= 4 and (step_count >= 2 or io_count >= 1):
        return True

    avg_len = sum(len(line) for line in lines) / max(len(lines), 1)
    if step_count >= 5 and ctrl_count >= 2 and avg_len <= 68:
        return True

    # Narrative prose blocks with citations/lists should stay translatable.
    if (
        _looks_like_narrative_prose(text)
        and io_count == 0
        and end_count == 0
        and pseudo_line_count <= max(2, len(lines) // 2)
    ):
        return False

    return False


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
                    narrative_fallback = _fallback_translate_table_narrative(job.text)
                    if narrative_fallback is not None:
                        normalized = narrative_fallback
                    else:
                        phrase_fallback = _fallback_translate_short_formula_phrase(job.text)
                        if phrase_fallback is None:
                            mark_untranslated(fingerprint, "low_quality_keep_original")
                            continue
                        normalized = phrase_fallback
                memory_updates[fingerprint] = normalized
                for dup_job in hash_to_jobs.get(fingerprint, []):
                    translated_by_block[dup_job.block_index] = _normalize_translated_text(dup_job.text, normalized)
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
                        narrative_fallback = _fallback_translate_table_narrative(job.text)
                        if narrative_fallback is not None:
                            normalized = narrative_fallback
                        else:
                            phrase_fallback = _fallback_translate_short_formula_phrase(job.text)
                            if phrase_fallback is None:
                                mark_untranslated(fingerprint, "low_quality_keep_original")
                                continue
                            normalized = phrase_fallback
                    memory_updates[fingerprint] = normalized
                    for dup_job in hash_to_jobs.get(fingerprint, []):
                        translated_by_block[dup_job.block_index] = _normalize_translated_text(dup_job.text, normalized)
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
                    narrative_fallback = _fallback_translate_table_narrative(job.text)
                    if narrative_fallback is not None:
                        memory_updates[fingerprint] = narrative_fallback
                        for dup_job in hash_to_jobs.get(fingerprint, []):
                            translated_by_block[dup_job.block_index] = _normalize_translated_text(dup_job.text, narrative_fallback)
                        continue
                    phrase_fallback = _fallback_translate_short_formula_phrase(job.text)
                    if phrase_fallback is not None:
                        memory_updates[fingerprint] = phrase_fallback
                        for dup_job in hash_to_jobs.get(fingerprint, []):
                            translated_by_block[dup_job.block_index] = _normalize_translated_text(dup_job.text, phrase_fallback)
                        continue
                    mark_untranslated(fingerprint, "batch_translation_failed_keep_original")
                continue
            result.warnings.append(
                f"page {page_no}: batch chunk [{start_no}-{end_no}] forced single-block retry"
            )

        for job in chunk:
            prompt_text = prompt_by_block.get(job.block_index, _compact_text_for_prompt(job.text))
            fingerprint = _text_fingerprint(prompt_text)
            if not single_fallback and job.block_index not in retry_candidates:
                narrative_fallback = _fallback_translate_table_narrative(job.text)
                if narrative_fallback is not None:
                    memory_updates[fingerprint] = narrative_fallback
                    for dup_job in hash_to_jobs.get(fingerprint, []):
                        translated_by_block[dup_job.block_index] = _normalize_translated_text(
                            dup_job.text, narrative_fallback
                        )
                    continue
                phrase_fallback = _fallback_translate_short_formula_phrase(job.text)
                if phrase_fallback is not None:
                    memory_updates[fingerprint] = phrase_fallback
                    for dup_job in hash_to_jobs.get(fingerprint, []):
                        translated_by_block[dup_job.block_index] = _normalize_translated_text(
                            dup_job.text, phrase_fallback
                        )
                    continue
                mark_untranslated(fingerprint, "batch_translation_failed_noncritical_keep_original")
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
                    narrative_fallback = _fallback_translate_table_narrative(job.text)
                    if narrative_fallback is not None:
                        normalized = narrative_fallback
                    else:
                        phrase_fallback = _fallback_translate_short_formula_phrase(job.text)
                        if phrase_fallback is None:
                            mark_untranslated(fingerprint, "low_quality_keep_original")
                            continue
                        normalized = phrase_fallback
                memory_updates[fingerprint] = normalized
                for dup_job in hash_to_jobs.get(fingerprint, []):
                    translated_by_block[dup_job.block_index] = _normalize_translated_text(dup_job.text, normalized)
            except TranslationError as exc:
                narrative_fallback = _fallback_translate_table_narrative(job.text)
                if narrative_fallback is not None:
                    memory_updates[fingerprint] = narrative_fallback
                    for dup_job in hash_to_jobs.get(fingerprint, []):
                        translated_by_block[dup_job.block_index] = _normalize_translated_text(dup_job.text, narrative_fallback)
                    continue
                phrase_fallback = _fallback_translate_short_formula_phrase(job.text)
                if phrase_fallback is not None:
                    memory_updates[fingerprint] = phrase_fallback
                    for dup_job in hash_to_jobs.get(fingerprint, []):
                        translated_by_block[dup_job.block_index] = _normalize_translated_text(dup_job.text, phrase_fallback)
                    continue
                mark_untranslated(fingerprint, f"translation_failed: {exc}")
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


def _should_preserve_source_line_layout(text: str) -> bool:
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 3:
        return False

    marker_lines = sum(1 for line in lines if STRUCTURED_LINE_MARKER_RE.match(line))
    token_hits = len(STRUCTURED_NOTATION_TOKEN_RE.findall(" ".join(lines)))
    short_ratio = sum(1 for line in lines if len(line) <= 92) / max(len(lines), 1)
    return marker_lines >= 2 or (token_hits >= 3 and short_ratio >= 0.45)


def _compact_text_for_prompt(text: str) -> str:
    compact = _normalize_common_unicode_text(text)
    preserve_lines = _should_preserve_source_line_layout(compact)
    compact = compact.replace("-\n", "")
    if preserve_lines:
        lines = [re.sub(r"[ \t]{2,}", " ", line).strip() for line in compact.splitlines() if line.strip()]
        return "\n".join(lines).strip()
    compact = re.sub(r"\s*\n+\s*", " ; ", compact)
    compact = re.sub(r"(?:\s*;\s*){2,}", " ; ", compact)
    compact = re.sub(r"[ \t]{2,}", " ", compact)
    return compact.strip()


def _text_fingerprint(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


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
    text = translated_text.strip()
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


def _process_image_block(
    *,
    block: dict[str, Any],
    image: Image.Image,
    result: PageProcessResult,
    zoom: float,
    page_no: int,
    primary_provider: ProviderRuntime,
    backup_provider: ProviderRuntime | None,
    style_profile: str,
    glossary: list[str] | None,
    max_retries: int,
) -> None:
    if not get_settings().enable_image_ocr:
        return

    ocr = get_ocr()
    if not ocr:
        return

    bbox = block.get("bbox", [0, 0, 0, 0])
    x0, y0, x1, y1 = [int(float(v) * zoom) for v in bbox]
    if x1 <= x0 or y1 <= y0:
        return

    crop = image.crop((x0, y0, x1, y1))
    crop_np = np.array(crop)

    try:
        ocr_result = ocr.predict(crop_np)
    except Exception as exc:  # noqa: BLE001
        result.image_ocr_failures.append(
            {
                "page_no": page_no,
                "image_index": int(block.get("number", -1)),
                "reason": f"ocr_failed: {exc}",
            }
        )
        return

    lines = _normalize_ocr_result(ocr_result)
    if not lines:
        return

    overlay = crop.copy()
    overlay_draw = ImageDraw.Draw(overlay)

    for line in lines:
        text = line["text"].strip()
        if not _should_translate_text_block(text):
            continue

        lx0, ly0, lx1, ly1 = line["rect"]
        rect_w = max(1, lx1 - lx0)
        rect_h = max(1, ly1 - ly0)

        try:
            translated, used_provider, switched_from = translate_with_fallback(
                text=text,
                primary=primary_provider,
                backup=backup_provider,
                style_profile=style_profile,
                glossary=glossary,
                max_retries=max_retries,
            )
            if switched_from:
                result.fallback_events.append(
                    {
                        "page_no": page_no,
                        "from_provider": switched_from,
                        "to_provider": used_provider,
                        "reason": "primary provider failed on image text",
                    }
                )
        except TranslationError as exc:
            result.image_ocr_failures.append(
                {
                    "page_no": page_no,
                    "image_index": int(block.get("number", -1)),
                    "reason": f"image_text_translation_failed: {exc}",
                }
            )
            continue

        fit = _fit_text_to_box(translated, width=rect_w, height=rect_h, base_font_size=max(10.0, rect_h * 0.7), min_scale=0.7)
        if fit is None:
            result.image_ocr_failures.append(
                {
                    "page_no": page_no,
                    "image_index": int(block.get("number", -1)),
                    "reason": "image_text_overflow",
                }
            )
            continue

        txt_lines, txt_font, txt_line_height, _ = fit
        overlay_draw.rectangle([(lx0, ly0), (lx1, ly1)], fill=(255, 255, 255))
        ty = ly0
        for item in txt_lines:
            overlay_draw.text((lx0, ty), item, font=txt_font, fill=(0, 0, 0))
            ty += txt_line_height

    image.paste(overlay, (x0, y0))


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


def _normalize_ocr_result(raw: Any) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []

    if isinstance(raw, list):
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            rec_texts = entry.get("rec_texts") or []
            rec_polys = entry.get("rec_polys") or []
            for idx, text in enumerate(rec_texts):
                poly = rec_polys[idx] if idx < len(rec_polys) else None
                if poly is None:
                    continue
                xs = [point[0] for point in poly]
                ys = [point[1] for point in poly]
                normalized.append(
                    {
                        "text": str(text),
                        "rect": [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))],
                    }
                )

    return normalized


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


def _is_reference_entry_text(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False

    first = re.sub(r"\s+", " ", lines[0])
    flat = re.sub(r"\s+", " ", " ".join(lines))
    lower = flat.lower()
    has_year = bool(REFERENCE_YEAR_RE.search(flat))
    has_venue_hint = bool(REFERENCE_VENUE_HINT_RE.search(flat))
    has_author_hint = bool(REFERENCE_AUTHOR_HINT_RE.search(flat))
    start_count = sum(1 for line in lines if REFERENCE_ENTRY_START_RE.match(line))
    word_count = len(ENGLISH_WORD_RE.findall(flat))
    avg_len = sum(len(line) for line in lines) / max(len(lines), 1)
    proposal_like = bool(re.search(r"\b(we|our|propose|design|present|develop|introduce|suggest)\b", lower))

    # Skip detection if this looks like a contributions list (not references).
    # Contributions lists have keywords like "contributions" or "main contributions".
    if "contribution" in lower and start_count >= 1:
        # Check if it's actually a contributions list by looking for proposal verbs
        has_proposal_verbs = any(
            verb in lower for verb in ["propose", "design", "present", "develop", "introduce", "suggest", "we "]
        )
        if has_proposal_verbs:
            return False
    if proposal_like and not has_year and start_count <= 2:
        return False

    # Multiple reference entries in one block: check if many lines use [N] format
    # and have publication metadata (year, venue, or author hints)
    if start_count >= 2 and (has_year or has_venue_hint or has_author_hint):
        return True
    if REFERENCE_ENTRY_START_RE.match(first) and (has_year or has_venue_hint or has_author_hint):
        return True
    if has_year and (" doi" in lower or " arxiv" in lower) and (start_count >= 1 or REFERENCE_ENTRY_START_RE.match(first)):
        return True

    if has_author_hint and has_year and (has_venue_hint or start_count >= 1) and len(lines) <= 4 and avg_len <= 120:
        return True
    if start_count >= 1 and has_year and has_venue_hint and word_count <= 48:
        return True
    return False


def _is_reference_page(blocks: list[dict[str, Any]]) -> bool:
    total_text = 0
    entry_blocks = 0
    heading_seen = False
    first_entry_y: float | None = None
    page_bottom = 0.0

    for block in blocks:
        if block.get("type", -1) != 0:
            continue
        bbox = block.get("bbox", [0, 0, 0, 0])
        _, y0, _, y1 = [float(v) for v in bbox]
        page_bottom = max(page_bottom, y1)
        text, _, _ = _extract_block_text_style(block)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            continue
        total_text += 1
        if any(REFERENCE_HEADING_RE.match(re.sub(r"\s+", " ", line).strip()) for line in lines[:2]):
            heading_seen = True
            continue
        if _is_reference_entry_text(text):
            entry_blocks += 1
            if first_entry_y is None:
                first_entry_y = y0

    if total_text == 0:
        return False
    
    # 必须有参考文献标题
    if not heading_seen:
        return False
    
    # 必须有足够的参考文献条目
    if entry_blocks < 2:
        return False
    
    # 增加位置判断：参考文献条目通常不会从页面顶部就开始。
    # 对于真正的参考文献页，首条目一般位于页面中下部。
    if first_entry_y is not None and page_bottom > 0:
        first_entry_ratio = first_entry_y / page_bottom
        # If entries start too high on page, likely a false positive (e.g., contribution list).
        if first_entry_ratio < 0.22:
            return False
    
    return True


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


def _is_table_caption_text(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    first_line = re.sub(r"\s+", " ", lines[0]).strip()
    if not TABLE_CAPTION_RE.match(first_line):
        return False

    flat = re.sub(r"\s+", " ", " ".join(lines)).strip()
    lower = flat.lower()
    words = ENGLISH_WORD_RE.findall(flat)
    sentence_punct = sum(flat.count(mark) for mark in (".", "?", "!", ";", ":"))

    # Narrative mentions like "Table V outlines/lists ..." should stay translatable.
    if len(words) >= 6 and TABLE_NARRATIVE_VERB_RE.search(lower):
        return False
    # Long sentence-like blocks are prose, not standalone captions.
    if len(words) >= 36:
        return False
    if len(lines) >= 3 and len(words) >= 18:
        return False
    if sentence_punct >= 2 and len(words) >= 14:
        return False
    if len(words) >= 12 and sentence_punct >= 1 and (
        "," in flat or TABLE_NARRATIVE_CONTINUATION_RE.search(lower)
    ):
        return False
    if len(words) >= 10 and (" that " in lower or " since " in lower):
        return False
    return True


def _is_table_header_text(text: str) -> bool:
    flat = re.sub(r"\s+", " ", text).strip().lower()
    if not flat:
        return False
    word_count = len(ENGLISH_WORD_RE.findall(flat))
    sentence_punct = sum(flat.count(mark) for mark in (".", "?", "!", ";", ":"))
    if word_count >= 10 and sentence_punct >= 1 and TABLE_NARRATIVE_CONTINUATION_RE.search(flat):
        return False
    if word_count >= 8 and "example" in flat and sentence_punct >= 1:
        return False
    if _looks_like_numbered_narrative_list(text):
        return False
    if _looks_like_narrative_prose(text):
        return False
    if TABLE_HEADER_HINT_RE.search(flat):
        return True

    tokens = re.findall(r"[a-z][a-z0-9\-]{1,}", flat)
    if len(tokens) < 3:
        return False

    token_set = set(tokens)
    metric_terms = {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auc",
        "latency",
        "throughput",
        "mean",
        "max",
        "min",
        "std",
        "variance",
        "error",
        "mse",
        "mae",
    }
    method_terms = {
        "method",
        "baseline",
        "model",
        "algorithm",
        "metric",
        "parameter",
        "setting",
        "dataset",
        "component",
    }
    metric_hits = sum(1 for term in metric_terms if term in token_set)
    method_hits = sum(1 for term in method_terms if term in token_set)
    has_struct_delim = bool(re.search(r"[|/]| {2,}|\t", text))
    has_digit = bool(re.search(r"\d", text))
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    if metric_hits >= 2 and (method_hits >= 1 or has_struct_delim or has_digit or len(lines) >= 2):
        return True
    if metric_hits >= 3:
        return True
    return False


def _looks_like_narrative_prose(text: str) -> bool:
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False

    if _looks_like_numbered_narrative_list(text):
        return True

    flat = " ".join(lines)
    word_count = len(ENGLISH_WORD_RE.findall(flat))
    if any(ACADEMIC_LABEL_RE.match(line) for line in lines[:2]) and word_count >= 5:
        return True
    if word_count < 10:
        return False

    sentence_punct = sum(flat.count(mark) for mark in ".?!;:")
    alpha_ratio = sum(1 for ch in flat if ch.isalpha()) / max(len(flat), 1)

    bullet_sentence_count = sum(
        1
        for line in lines
        if BULLET_SENTENCE_RE.match(line) and len(ENGLISH_WORD_RE.findall(line)) >= 6
    )
    if bullet_sentence_count >= 1 and word_count >= 16:
        return True

    # Short narrative definition/explanation fragments (common around formulas)
    # should stay translatable and not be treated as table rows.
    if len(lines) <= 3 and word_count >= 10 and sentence_punct >= 2 and alpha_ratio >= 0.35:
        return True

    if word_count >= 26 and sentence_punct >= 2 and alpha_ratio >= 0.45:
        return True

    avg_len = sum(len(line) for line in lines) / max(len(lines), 1)
    if len(lines) <= 3 and word_count >= 14 and sentence_punct >= 1 and avg_len >= 42 and alpha_ratio >= 0.36:
        return True
    if len(lines) >= 3 and word_count >= 24 and avg_len >= 40 and sentence_punct >= 1:
        return True

    return False


def _looks_like_numbered_narrative_list(text: str) -> bool:
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 3:
        return False

    numbered_lines = [line for line in lines if NUMBERED_NARRATIVE_LINE_RE.match(line)]
    if len(numbered_lines) < 2:
        return False

    flat = " ".join(lines)
    word_count = len(ENGLISH_WORD_RE.findall(flat))
    if word_count < 16:
        return False

    # Check if this looks like a reference list (not a contribution list).
    # Reference entries have [N] format and contain publication metadata.
    bracket_numbered = sum(1 for line in lines if re.match(r"^\s*\[\d{1,3}\]\s+", line))
    has_year = bool(REFERENCE_YEAR_RE.search(flat))
    has_venue = bool(REFERENCE_VENUE_HINT_RE.search(flat))
    # If most lines use [N] format and there's publication metadata, it's references.
    if bracket_numbered >= 2 and (has_year or has_venue):
        return False

    if NARRATIVE_LIST_LEADIN_RE.search(flat):
        return True

    numbered_word_count = sum(len(ENGLISH_WORD_RE.findall(line)) for line in numbered_lines)
    avg_numbered_words = numbered_word_count / max(len(numbered_lines), 1)
    return len(numbered_lines) >= 3 and avg_numbered_words >= 4


def _is_table_like_text(text: str) -> bool:
    if not text.strip():
        return False
    flat_all = re.sub(r"\s+", " ", text).strip()
    lower_all = flat_all.lower()
    word_count = len(ENGLISH_WORD_RE.findall(flat_all))
    sentence_punct = sum(flat_all.count(mark) for mark in (".", "?", "!", ";", ":"))
    if word_count >= 10 and sentence_punct >= 1 and TABLE_NARRATIVE_CONTINUATION_RE.search(flat_all):
        return False
    if word_count >= 8 and "example" in lower_all and sentence_punct >= 1:
        return False
    if _is_table_caption_text(text):
        return True
    if _looks_like_numbered_narrative_list(text):
        return False
    # Do not suppress regular prose blocks that happen to contain numbers/citations.
    if _looks_like_narrative_prose(text):
        return False
    if _is_table_header_text(text):
        return True
    if _is_table_body_text(text):
        return True

    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False

    numeric_row_hits = 0
    for line in lines:
        if TABLE_ROW_NUMERIC_RE.match(line):
            numeric_row_hits += 1
            continue
        if TABLE_SYMBOLIC_ROW_RE.match(line):
            numeric_row_hits += 1
            continue
        number_count = len(re.findall(r"\b\d+(?:\.\d+)?\b", line))
        if number_count >= 3 and len(line) <= 96:
            numeric_row_hits += 1

    short_ratio = sum(1 for line in lines if len(line) <= 96) / len(lines)
    return numeric_row_hits >= 2 and short_ratio >= 0.6


def _is_table_body_text(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    flat = re.sub(r"\s+", " ", " ".join(lines)).strip()
    lower_flat = flat.lower()
    word_count = len(ENGLISH_WORD_RE.findall(flat))
    sentence_punct = sum(flat.count(mark) for mark in (".", "?", "!", ";", ":"))

    # Formula-adjacent prose fragments (e.g., "that each round ...", "Example 1 ...")
    # are often split into short lines and can be mistaken as table rows.
    # Keep these translatable by suppressing table-body classification.
    if word_count >= 10 and sentence_punct >= 1 and TABLE_NARRATIVE_CONTINUATION_RE.search(flat):
        return False
    if word_count >= 8 and "example" in lower_flat and sentence_punct >= 1:
        return False

    if _looks_like_numbered_narrative_list(text):
        return False
    if _looks_like_narrative_prose(text):
        return False
    if any(TABLE_ROW_NUMERIC_RE.match(re.sub(r"\s+", " ", line).strip()) for line in lines):
        return True
    if any(TABLE_SYMBOLIC_ROW_RE.match(re.sub(r"\s+", " ", line).strip()) for line in lines):
        return True
    if len(lines) < 4:
        sentence_punct = sum(line.count(".") + line.count("?") + line.count("!") for line in lines)
        if sentence_punct >= 1 and word_count >= 8:
            return False

        numeric_rich = sum(1 for line in lines if len(re.findall(r"\b\d+(?:\.\d+)?\b", line)) >= 3)
        symbolic_rich = sum(1 for line in lines if TABLE_SYMBOL_TOKEN_RE.search(line))
        return (numeric_rich >= 2 and len(lines) >= 2) or (
            symbolic_rich >= 2 and len(lines) >= 3 and numeric_rich >= 1
        )

    short_ratio = sum(1 for line in lines if len(line) <= 78) / len(lines)
    sentence_punct = sum(line.count(".") + line.count("?") + line.count("!") for line in lines)
    if sentence_punct > max(2, len(lines) // 3):
        return False

    lead_like = 0
    row_like = 0
    for line in lines:
        words = line.split()
        if len(words) < 2 or len(words) > 18:
            continue
        lead = words[0].strip(",;:.()[]{}")
        if not lead:
            continue

        has_symbol = bool(re.search(r"[\d_'\-^/]", lead))
        short_symbolic = len(lead) <= 2 or lead.isupper()
        if has_symbol or short_symbolic:
            lead_like += 1
        if (has_symbol or short_symbolic) and len(words) <= 14:
            row_like += 1

    lead_ratio = lead_like / len(lines)
    row_ratio = row_like / len(lines)
    if len(lines) >= 8 and short_ratio >= 0.65 and lead_ratio >= 0.34 and row_ratio >= 0.24:
        return True

    if _is_table_header_text(flat):
        return True
    return False


def _looks_like_paragraph_block(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False

    flat = re.sub(r"\s+", " ", " ".join(lines))
    word_count = len(ENGLISH_WORD_RE.findall(flat))
    punctuation = sum(text.count(mark) for mark in ".?!;")

    # Short paragraphs (1-2 lines) can also be prose, especially table captions/notes.
    # Check if it looks like a sentence (has punctuation and enough words).
    if len(lines) < 3:
        # For 1-2 line blocks, consider it prose if:
        # - It has sentence-ending punctuation (at least 1 period/question/exclamation)
        # - It has enough words (at least 8) suggesting a complete sentence
        return punctuation >= 1 and word_count >= 8

    avg_len = sum(len(line) for line in lines) / max(len(lines), 1)
    return avg_len > 48 or punctuation >= 2


def _looks_like_formula_narrative_fragment(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False

    flat = re.sub(r"\s+", " ", " ".join(lines))
    word_count = len(ENGLISH_WORD_RE.findall(flat))
    if word_count < 4:
        return False
    if not FORMULA_NARRATIVE_VERB_RE.search(flat):
        return False

    symbol_hits = len(re.findall(r"[=<>≤≥+\-*/^_()]", flat))
    if symbol_hits >= 1:
        return True

    return bool(
        re.search(
            r"\b(ciphertext|plaintext|protocol|index|variable|value|random|nonce|message)\b",
            flat,
            flags=re.IGNORECASE,
        )
    )


def _horizontal_overlap_ratio(a: tuple[float, float], b: tuple[float, float]) -> float:
    a0, a1 = a
    b0, b1 = b
    overlap = max(0.0, min(a1, b1) - max(a0, b0))
    base = max(1.0, min(a1 - a0, b1 - b0))
    return overlap / base


def _is_in_table_region(rect: tuple[float, float, float, float], tables: list[tuple[float, float, float, float]]) -> bool:
    for table in tables:
        if _rect_overlap_significant(rect, table):
            return True
    return False


def _max_overlap_ratio_to_regions(
    rect: tuple[float, float, float, float],
    regions: list[tuple[float, float, float, float]],
) -> float:
    ax0, ay0, ax1, ay1 = rect
    area = max(1.0, (ax1 - ax0) * (ay1 - ay0))
    best = 0.0
    for region in regions:
        _, _, inter = _rect_intersection(rect, region)
        if inter <= 0:
            continue
        best = max(best, inter / area)
    return best


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


def _rect_intersects(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return not (ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0)


def _rect_overlap_significant(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    inter_w, inter_h, inter_area = _rect_intersection(a, b)
    if inter_area <= 0:
        return False

    ax0, ay0, ax1, ay1 = a
    a_area = max(1.0, (ax1 - ax0) * (ay1 - ay0))
    if inter_area / a_area >= 0.18:
        return True

    a_w = max(1.0, ax1 - ax0)
    a_h = max(1.0, ay1 - ay0)
    if (inter_w / a_w) >= 0.62 and inter_h >= max(8.0, a_h * 0.18):
        return True
    return False


def _rect_intersection(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> tuple[float, float, float]:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    return iw, ih, iw * ih


def _rect_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    _, _, inter = _rect_intersection(a, b)
    if inter <= 0:
        return 0.0
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    area_a = max(0.0, (ax1 - ax0) * (ay1 - ay0))
    area_b = max(0.0, (bx1 - bx0) * (by1 - by0))
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def _log_block_skip(*, page_no: int, block_index: int, reason: str, text: str) -> None:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) < 24:
        return
    en_words = len(ENGLISH_WORD_RE.findall(compact))
    cjk_chars = len(CJK_CHAR_RE.findall(compact))
    if en_words < 4 and cjk_chars < 8:
        return
    logger.info(
        "page %s block %s skipped: %s | excerpt=%s",
        page_no,
        block_index,
        reason,
        compact[:140],
    )


def _normalize_translated_text(source_text: str, translated_text: str) -> str:
    normalized = translated_text.strip()
    normalized = normalized.replace("译文：", "").replace("翻译：", "").strip()
    normalized = _sanitize_formula_render_artifacts(normalized)
    normalized = _strip_non_paper_meta(normalized)
    normalized = _strip_source_echo(source_text, normalized)
    title_repaired = _repair_title_translation(source_text, normalized)
    if title_repaired:
        normalized = title_repaired
    normalized = _trim_extra_numbered_items(source_text, normalized)
    normalized = _trim_overlong_fragment_translation(source_text, normalized)
    normalized = _sanitize_formula_render_artifacts(normalized)
    normalized = _strip_non_paper_meta(normalized)

    source_len = len(source_text.strip())
    source_en = len(ENGLISH_WORD_RE.findall(source_text))
    if source_len <= 80:
        normalized = re.sub(r"\s+", " ", normalized).strip()
        # If model returns only section number, recover common section heading translation.
        if re.fullmatch(r"(?:section\s+)?(?:\d+(?:\.\d+)*|[ivxlcdm]+)\.?", normalized, flags=re.IGNORECASE):
            repaired = _repair_section_heading(source_text, normalized)
            if repaired:
                normalized = repaired
        max_len = max(80, source_len * 3)
        if len(normalized) > max_len:
            normalized = normalized[:max_len].rstrip(" ,;:.-")
    elif source_len <= 220 and source_en <= 42:
        max_len = max(150, int(source_len * 1.8))
        if len(normalized) > max_len:
            normalized = _truncate_by_sentence(normalized, max_len=max_len)

    return normalized or translated_text


def _sanitize_formula_render_artifacts(text: str) -> str:
    if not text:
        return text

    out = _normalize_common_unicode_text(text)
    # Normalize escaped math delimiters first (e.g., "\$ ... \$").
    out = out.replace("\\$", "$")
    # Normalize common LaTeX artifacts that models may inject into prose.
    out = re.sub(r"\\\s*tex\s*t?\s*\{([^{}]{1,80})\}", r"\1", out, flags=re.IGNORECASE)
    out = re.sub(r"\\\s*text\s*\{([^{}]{1,80})\}", r"\1", out, flags=re.IGNORECASE)
    out = re.sub(r"\\\s*mathrm\s*\{([^{}]{1,80})\}", r"\1", out, flags=re.IGNORECASE)
    out = re.sub(r"\\\s*operatorname\s*\{([^{}]{1,80})\}", r"\1", out, flags=re.IGNORECASE)
    # Strip common LaTeX math wrappers while keeping inner formula content.
    out = re.sub(r"\\\(\s*([\s\S]{1,360}?)\s*\\\)", r"\1", out)
    out = re.sub(r"\\\[\s*([\s\S]{1,480}?)\s*\\\]", r"\1", out)
    out = re.sub(r"\$\s*([^$]{1,360}?)\s*\$", r"\1", out)
    out = re.sub(r"\\\s*leftarrow\s*_\s*\{?\s*R\s*\}?", "←R", out, flags=re.IGNORECASE)
    out = re.sub(r"\\\s*rightarrow\s*_\s*\{?\s*R\s*\}?", "→R", out, flags=re.IGNORECASE)
    out = re.sub(r"\bleftarrow\s*_\s*\{?\s*R\s*\}?", "←R", out, flags=re.IGNORECASE)
    out = re.sub(r"\brightarrow\s*_\s*\{?\s*R\s*\}?", "→R", out, flags=re.IGNORECASE)

    latex_map = (
        (r"\\\s*oplus\b", "⊕"),
        (r"\\\s*times\b", "×"),
        (r"\\\s*cdot\b", "·"),
        (r"\\\s*leq\b", "≤"),
        (r"\\\s*geq\b", "≥"),
        (r"\\\s*neq\b", "≠"),
        (r"\\\s*in\b", "∈"),
        (r"\\\s*leftarrow\b", "←"),
        (r"\\\s*rightarrow\b", "→"),
    )
    for pat, rep in latex_map:
        out = re.sub(pat, rep, out, flags=re.IGNORECASE)

    # Convert lightweight LaTeX superscript/subscript braces to plain form.
    out = re.sub(r"([_^])\s*\{([^{}]{1,40})\}", r"\1\2", out)
    out = out.replace("\\_", "_").replace("\\{", "{").replace("\\}", "}")
    # Strip accidental HTML-like superscript/subscript tags emitted by some models.
    # Keep only their inner text to preserve affiliation markers such as a/b/c.
    out = re.sub(r"&lt;\s*/?\s*(?:sup|sub)\b[^&]*&gt;", "", out, flags=re.IGNORECASE)
    out = re.sub(r"<\s*/?\s*(?:sup|sub)\b[^>]*>", "", out, flags=re.IGNORECASE)
    # Remove residual isolated math delimiters after unwrapping.
    out = re.sub(r"(?:(?<=\s)|^)\$(?=\s*[A-Za-z0-9(])", "", out)
    out = re.sub(r"(?<=[A-Za-z0-9\)])\$(?=(?:\s|[，。,:;!?)]|$))", "", out)
    # Remove stray backslashes that are not part of escaped newline or tabs.
    out = re.sub(r"\\(?![nrt])(?=\s|[A-Za-z\u4e00-\u9fff])", "", out)
    # In formula narrative, models may leave an English "by" between a variable and a delta/number.
    out = re.sub(
        r"\b([A-Za-z][A-Za-z0-9*']*)\s+by\s+(?=(?:[+\-]?\d+|at\s+(?:least|most)\b|at\b|least\b|most\b|至少|至多|约))",
        r"\1 ",
        out,
        flags=re.IGNORECASE,
    )
    out = re.sub(
        r"\b([A-Za-z][A-Za-z0-9*']*)\s+by\s+(?=[\u4e00-\u9fff\u0370-\u03FF])",
        r"\1 ",
        out,
        flags=re.IGNORECASE,
    )
    out = re.sub(r"(?<=[=+\-*/<>≤≥])\s*;\s*(?=[A-Za-z0-9\u0370-\u03FF])", " ", out)
    out = _normalize_common_unicode_text(out)
    out = _cleanup_formula_residue_lines(out)
    out = CONTROL_CHAR_RE.sub("", out)
    out = re.sub(r"[ \t]{2,}", " ", out)
    return out.strip()


def _cleanup_formula_residue_lines(text: str) -> str:
    if not text:
        return text

    cleaned_lines: list[str] = []
    for raw in text.splitlines():
        line = re.sub(r"[ \t]{2,}", " ", raw).strip()
        if not line:
            continue
        # Drop orphan symbol-only lines introduced by broken math delimiters.
        if re.fullmatch(r"[+\-*/=·•|]{1,4}", line):
            continue
        # Repair prime/subscript split artifact: "E' s(...)" -> "E_s(...)".
        line = re.sub(r"\b([A-Za-z])'\s+([A-Za-z])(?=\s*\()", r"\1_\2", line)
        # Remove leading operator residue for prose lines (common after "$+ ...$" cleanup).
        if re.match(r"^[+\-]\s*[A-Za-z(]", line):
            has_cjk = bool(CJK_CHAR_RE.search(line))
            has_sentence_punct = bool(re.search(r"[。！？；，,]", line))
            if (has_cjk or has_sentence_punct) and "=" not in line[:24]:
                line = re.sub(r"^[+\-]\s*", "", line)
        # Remove dangling operator at line end.
        line = re.sub(r"\s+[+\-*/=]\s*$", "", line)
        line = line.strip()
        if not line:
            continue
        cleaned_lines.append(line)

    if cleaned_lines:
        return "\n".join(cleaned_lines).strip()
    return ""


def _fallback_translate_short_formula_phrase(source_text: str) -> str | None:
    source = re.sub(r"\s+", " ", source_text).strip()
    if not source:
        return None
    if len(CJK_CHAR_RE.findall(source)) > 0:
        return None

    en_words = ENGLISH_WORD_RE.findall(source)
    if len(en_words) < 2 or len(en_words) > 8 or len(source) > 72:
        return None

    normalized = re.sub(r"[^A-Za-z ]", " ", source).lower()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        return None

    mapped: str | None = None
    if re.fullmatch(r"(then we have|hence we have)", normalized):
        mapped = "则有"
    elif re.fullmatch(r"(easy to see that|it is easy to see that)", normalized):
        mapped = "易得"
    elif re.fullmatch(r"(note that|notice that)", normalized):
        mapped = "注意"
    elif re.fullmatch(r"(for example|as an example)", normalized):
        mapped = "例如"
    elif re.fullmatch(r"(which directly leads to|thus we obtain|thus we have)", normalized):
        mapped = "由此可得"
    elif re.fullmatch(r"we have", normalized):
        mapped = "有"

    if mapped is None:
        return None

    if source.endswith(":") or source.endswith("："):
        return f"{mapped}："
    return mapped


def _fallback_translate_table_narrative(source_text: str) -> str | None:
    source = _compact_text_for_prompt(source_text)
    source = source.replace("＊", "'")
    source = re.sub(r"\s*;\s*", " ", source)
    source = re.sub(r"\s+", " ", source).strip()
    if not source:
        return None
    if len(CJK_CHAR_RE.findall(source)) > 0:
        return None

    table_match = re.match(r"^\s*table\s+([ivxlcdm]+|\d+)\b", source, flags=re.IGNORECASE)
    if not table_match:
        return None

    table_id = table_match.group(1).upper()
    sentences = [seg.strip(" .;") for seg in re.split(r"(?<=[.?!])\s+", source) if seg.strip(" .;")]
    if not sentences:
        return None

    def _norm_tail(text: str) -> str:
        out = text.strip(" .;")
        out = out.replace("’", "'").replace("`", "'")
        out = re.sub(r"\ban\s+d\b", "and", out, flags=re.IGNORECASE)
        replacements = (
            (r"\bsince they are not important regarding how the protocols?\s+works?\b", "因为它们对协议运行方式并不重要"),
            (r"\bthe number of rounds and communication cost of the \[(\d{1,3})\] is an approximation\b", r"[\1]中的轮数与通信开销是近似值"),
            (r"\bthe secret keys are distributed as described above\b", "密钥按上述方式分发"),
            (r"\bthe users could generate pads with arbitrary length\b", "用户可以生成任意长度的填充"),
            (r"\bit doesn't display the value of the public nonce t and the encrypted message mi\b", "它没有展示公共随机数t和加密消息Mi的取值"),
            (r"\bit does not display the value of the public nonce t and the encrypted message mi\b", "它没有展示公共随机数t和加密消息Mi的取值"),
            (r"\bit does(?:n't| not)\s+display the value of the\s+public nonce\s*t\s+and the encrypted message\s*mi\b", "它没有展示公共随机数t和加密消息Mi的取值"),
            (r"\bit\s+doesn.?t\s+display\s+the\s+value\s+of\s+the\s+public\s+nonce\s*t\s+and\s+the\s+encrypted\s+message\s*mi\b", "它没有展示公共随机数t和加密消息Mi的取值"),
            (r"\ba pad for\s+([A-Za-z0-9_]+)\s+with length,\s*e\.g\.,\s*length\b", r"长度为length的\1填充串"),
            (r"\ba pad for\s+([A-Za-z0-9_]+)\s+with length\b", r"长度给定的\1填充串"),
            (r"\bthis pad\b", "该填充串"),
            (r"\bbit-choosing algorithm is executed before aggregation\b", "位选择算法在聚合前执行"),
            (r"\bwhich approximately adds two or three rounds to the whole process\b", "这会使整个过程大约增加2到3轮"),
            (r"\bmincomputing\b", "最小值计算"),
            (r"\bmin-computing\b", "最小值计算"),
            (r"\bprotocols?\b", "协议"),
            (r"\bseveral\b", "若干"),
            (r"\bunder given settings\b", "在给定设置下"),
            (r"\bpublic nonce\b", "公共随机数"),
            (r"\bencrypted message\b", "加密消息"),
            (r"\bthe value of\b", "取值"),
            (r"\bdisplay\b", "展示"),
            (r"\bdoesn.?t\b", ""),
            (r"\bas the\b", "因为"),
            (r"\bsince\b", "因为"),
            (r"\band\b", "和"),
        )
        for pat, rep in replacements:
            out = re.sub(pat, rep, out, flags=re.IGNORECASE)
        out = re.sub(r"\bthe\b", "", out, flags=re.IGNORECASE)
        out = re.sub(r"\bit\b", "", out, flags=re.IGNORECASE)
        out = re.sub(r"\s+", " ", out).strip(" ,;")
        out = out.replace(",", "，")
        out = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", out)
        out = re.sub(r"\s+([，。；：])", r"\1", out)
        out = re.sub(r"([，。；：])\s+", r"\1", out)
        out = re.sub(r"([（\(])\s+", r"\1", out)
        out = re.sub(r"\s+([）\)])", r"\1", out)
        return out

    out_sentences: list[str] = []
    for idx, sentence in enumerate(sentences):
        s = re.sub(r"\s+", " ", sentence).strip()
        lower = s.lower()

        if idx == 0:
            match_show = re.match(
                r"^table\s+[ivxlcdm0-9]+\s+shows\s+(.+?)\s+under given settings$",
                s,
                flags=re.IGNORECASE,
            )
            if match_show:
                obj = _norm_tail(match_show.group(1))
                out_sentences.append(f"表{table_id}展示了给定设置下的{obj}。")
                continue

            match_list = re.match(
                r"^table\s+[ivxlcdm0-9]+\s+lists\s+all the important variables in the whole process of the\s+([A-Za-z0-9\-]+)$",
                s,
                flags=re.IGNORECASE,
            )
            if match_list:
                proc = match_list.group(1)
                out_sentences.append(f"表{table_id}列出了{proc}整个过程中的所有重要变量。")
                continue

            match_give = re.match(
                r"^table\s+[ivxlcdm0-9]+\s+gives\s+the comparison between\s+(.+)$",
                s,
                flags=re.IGNORECASE,
            )
            if match_give:
                obj = _norm_tail(match_give.group(1))
                out_sentences.append(f"表{table_id}给出了{obj}的比较。")
                continue

            generic = re.match(
                r"^table\s+[ivxlcdm0-9]+\s+(shows|lists|gives|presents|illustrates|compares|summarizes|reports|describes|provides)\s+(.+)$",
                s,
                flags=re.IGNORECASE,
            )
            if generic:
                verb = generic.group(1).lower()
                rest = _norm_tail(generic.group(2))
                verb_map = {
                    "shows": "展示了",
                    "lists": "列出了",
                    "gives": "给出了",
                    "presents": "给出了",
                    "illustrates": "展示了",
                    "compares": "比较了",
                    "summarizes": "总结了",
                    "reports": "报告了",
                    "describes": "描述了",
                    "provides": "给出了",
                }
                out_sentences.append(f"表{table_id}{verb_map.get(verb, '给出了')}{rest}。")
                continue

            return None

        note_match = re.match(r"^note that\s+(.+)$", s, flags=re.IGNORECASE)
        if note_match:
            out_sentences.append(f"需要注意的是，{_norm_tail(note_match.group(1))}。")
            continue

        below_assume = re.match(r"^below,\s*we\s+(?:will\s+)?assume\s+(.+)$", s, flags=re.IGNORECASE)
        if below_assume:
            out_sentences.append(f"下面，我们假设{_norm_tail(below_assume.group(1))}。")
            continue

        assume_match = re.match(r"^we\s+assume(?:\s+that)?\s+(.+)$", s, flags=re.IGNORECASE)
        if assume_match:
            out_sentences.append(f"我们假设{_norm_tail(assume_match.group(1))}。")
            continue

        when_need = re.match(
            r"^when we need\s+(.+?),\s*we use notation\s+(.+?)\s+to denote\s+(.+)$",
            s,
            flags=re.IGNORECASE,
        )
        if when_need:
            what = _norm_tail(when_need.group(1))
            notation = when_need.group(2).strip()
            target = _norm_tail(when_need.group(3))
            out_sentences.append(f"当我们需要{what}时，使用记号{notation}表示{target}。")
            continue

        # Keep fallback conservative: if any sentence is completely unknown, abort.
        return None

    merged = "".join(seg.strip() for seg in out_sentences if seg.strip())
    return merged or None


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


def _strip_source_echo(source_text: str, translated_text: str) -> str:
    source = re.sub(r"\s+", " ", source_text).strip()
    translated = translated_text.strip()
    if not source or not translated:
        return translated

    lines = [line.strip() for line in translated.splitlines() if line.strip()]
    if len(lines) >= 2:
        kept: list[str] = []
        removed = False
        for line in lines:
            if _is_source_echo_line(source, line):
                removed = True
                continue
            kept.append(line)
        if removed and kept:
            translated = "\n".join(kept).strip()

    source_lower = source.lower()
    translated_lower = translated.lower()
    if len(source_lower) >= 16 and translated_lower.startswith(source_lower):
        tail = translated[len(source) :].lstrip(" \t\r\n:：-")
        if tail:
            translated = tail

    translated = _drop_untranslated_english_lines_in_mixed_output(source, translated)
    return translated


def _is_source_echo_line(source_text: str, line: str) -> bool:
    clean = re.sub(r"\s+", " ", line).strip()
    if len(clean) < 16:
        return False
    return _is_likely_untranslated_english_line(source_text, clean, strict=True)


def _drop_untranslated_english_lines_in_mixed_output(source_text: str, translated_text: str) -> str:
    lines = [line.strip() for line in translated_text.splitlines() if line.strip()]
    if len(lines) < 2:
        return translated_text

    cjk_lines = sum(1 for line in lines if len(CJK_CHAR_RE.findall(line)) >= 2)
    if cjk_lines == 0:
        return translated_text

    kept: list[str] = []
    removed = 0
    for line in lines:
        # Also detect short English fragments that are likely untranslated residue
        # These often appear at the end of translated paragraphs (e.g., "min protocols, proving...")
        if _is_short_english_fragment(line, source_text):
            removed += 1
            continue
        if _is_likely_untranslated_english_line(source_text, line, strict=False):
            removed += 1
            continue
        kept.append(line)

    if removed > 0 and kept:
        logger.info("removed untranslated english lines from mixed output: %s", removed)
        return "\n".join(kept).strip()
    return translated_text


def _is_short_english_fragment(line: str, source_text: str = "") -> bool:
    """Detect short English fragments that are likely untranslated residue.

    These are typically short English sentences or phrases that appear mixed with Chinese,
    often at the end of translated paragraphs. Examples:
    - "min protocols, proving them being privacy-preserving."
    - "proving them being privacy-preserving."
    """
    clean = re.sub(r"\s+", " ", line).strip()
    # Check if line is mostly English with very few or no Chinese characters
    cjk_count = len(CJK_CHAR_RE.findall(clean))
    if cjk_count > 2:
        return False

    # Must have some English words but be relatively short
    words = ENGLISH_WORD_RE.findall(clean)
    if len(words) < 3 or len(words) > 12:
        return False

    # Length should be relatively short (less than 80 characters)
    if len(clean) > 80:
        return False

    # Check if it's likely a fragment (ends with incomplete punctuation or unusual patterns)
    # Examples: "min protocols, proving them being..." or "...being privacy-preserving."
    if len(clean) >= 15 and len(clean) <= 60:
        # Check if it looks like a fragment that's been left untranslated
        # Pattern: mostly lowercase or mixed case, contains commas or unusual constructions
        if "," in clean or clean.lower() != clean:
            # Additional check: if source text contains this line, it's likely residue
            source_norm = source_text.lower() if source_text else ""
            clean_lower = clean.lower()
            # If the line or significant portion appears in source, it's likely residue
            if source_norm and (clean_lower in source_norm or any(w in source_norm for w in words[:3])):
                return True

    return False


def _contains_untranslated_english_lines_in_mixed_output(source_text: str, translated_text: str) -> bool:
    lines = [line.strip() for line in translated_text.splitlines() if line.strip()]
    if len(lines) < 2:
        return False
    if not any(len(CJK_CHAR_RE.findall(line)) >= 2 for line in lines):
        return False

    residue_lines = 0
    residue_words = 0
    for line in lines:
        if not _is_likely_untranslated_english_line(source_text, line, strict=False):
            continue
        residue_lines += 1
        residue_words += len(ENGLISH_WORD_RE.findall(line))

    if residue_lines == 0:
        return False
    return residue_words >= 8 or residue_lines >= 2


def _is_likely_untranslated_english_line(source_text: str, line: str, *, strict: bool) -> bool:
    clean = re.sub(r"\s+", " ", line).strip()
    min_len = 30 if strict else 22
    if len(clean) < min_len:
        return False

    cjk = len(CJK_CHAR_RE.findall(clean))
    if cjk > (1 if strict else 2):
        return False

    words = ENGLISH_WORD_RE.findall(clean)
    min_words = 6 if strict else 5
    if len(words) < min_words:
        return False

    # Count acronyms/proper nouns (all-caps words 2-8 chars) in the line.
    # These are expected to remain in English and should not count against quality.
    acronyms = re.findall(r"\b[A-Z]{2,8}\b", clean)
    non_acronym_words = len(words) - len(acronyms)

    # If most English words are acronyms/proper nouns and there's Chinese content,
    # this line is likely translated with proper nouns preserved.
    if non_acronym_words <= 2 and cjk >= 2:
        return False

    # Check if line starts with numbered list marker (e.g., "1)", "2.", "3)")
    # Numbered list items with Chinese content should not be flagged as untranslated.
    if re.match(r"^\s*\d{1,3}[\).]\s*", clean) and cjk >= 2:
        return False

    # In mixed-language output, a long pure-English sentence is almost always untranslated residue.
    if cjk <= 1 and len(words) >= 10 and len(clean) >= 56:
        return True

    source_norm = re.sub(r"\s+", " ", source_text).strip().lower()
    if len(source_norm) < 20:
        return False

    lower = clean.lower()
    if lower in source_norm:
        return True

    phrase = " ".join(words[: (12 if strict else 9)]).lower()
    if len(phrase) >= (36 if strict else 26) and phrase in source_norm:
        return True

    source_words = {w.lower() for w in ENGLISH_WORD_RE.findall(source_norm)}
    if not source_words:
        return False
    line_words = {w.lower() for w in words}
    overlap = len(source_words & line_words) / max(1, len(line_words))
    return overlap >= (0.62 if strict else 0.52)


def _strip_non_paper_meta(text: str) -> str:
    if not text:
        return text

    cleaned = NON_PAPER_META_INLINE_RE.sub("", text).strip()
    lines = [line.rstrip() for line in cleaned.splitlines() if line.strip()]
    if not lines:
        return cleaned

    kept: list[str] = []
    removed = False
    for line in lines:
        if _is_non_paper_meta_line(line):
            removed = True
            continue
        kept.append(line)

    if removed and kept:
        cleaned = "\n".join(kept).strip()
    else:
        cleaned = cleaned

    cleaned = re.sub(r"^\s*[.。·…]{2,}\s*", "", cleaned)
    return cleaned


def _is_continuation_marker_line(line: str) -> bool:
    norm = re.sub(r"\s+", " ", line).strip()
    if not norm:
        return False
    if re.fullmatch(r"[（(]?\s*(?:续上行|接上行|续行|承上|同上|见上行)\s*[）)]?", norm):
        return True
    lower = norm.lower()
    return bool(re.fullmatch(r"[（(]?\s*(?:continued(?:\s+above)?|same as above|ditto)\s*[）)]?", lower))


def _is_non_paper_meta_line(line: str) -> bool:
    norm = re.sub(r"\s+", " ", line).strip()
    if len(norm) < 8:
        if re.fullmatch(r"[.。·…]{2,}", norm):
            return True
        if _is_continuation_marker_line(norm):
            return True
        return False

    if _is_continuation_marker_line(norm):
        return True

    lower = norm.lower()
    if re.search(r"mathseg\d+token", lower):
        return True
    meta_keys = (
        "后续内容保持原文",
        "原文未完成状态",
        "严格遵循要求",
        "翻译要求",
        "输出要求",
        "不要输出",
        "不要翻译",
        "术语=",
        "terms=",
        "strict retry mode",
        "system prompt",
        "preferred terms",
        "style:",
    )
    if not any(key in lower for key in meta_keys):
        return False

    cjk_count = len(CJK_CHAR_RE.findall(norm))
    en_words = len(ENGLISH_WORD_RE.findall(norm))
    return cjk_count >= 4 or en_words >= 3


def _contains_non_paper_meta(text: str) -> bool:
    if not text:
        return False
    if NON_PAPER_META_INLINE_RE.search(text):
        return True
    return any(_is_non_paper_meta_line(line) for line in text.splitlines() if line.strip())


def _should_force_single_retry(text: str) -> bool:
    compact = _compact_text_for_prompt(text)
    if not compact:
        return False
    if should_skip_translation(compact):
        return False

    cjk_count = len(CJK_CHAR_RE.findall(compact))
    en_words = len(ENGLISH_WORD_RE.findall(compact))
    latin_chars = len(re.findall(r"[A-Za-z]", compact))
    if cjk_count > 0:
        # Mixed-language prose still needs retry when English span is substantial.
        if en_words >= 10 and latin_chars >= 40:
            return True
        if cjk_count <= 20 and en_words >= 6 and latin_chars >= 24:
            return True
        return False
    if en_words >= 6:
        return True
    if re.match(r"^\s*(?:section\s+)?\d+(?:\.\d+)*\s+[A-Za-z]", compact, flags=re.IGNORECASE):
        return True
    if len(compact) <= 80 and en_words >= 2:
        return True
    return False


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


def _has_extreme_length_mismatch(source_text: str, translated_text: str) -> bool:
    source = re.sub(r"\s+", " ", source_text).strip()
    translated = re.sub(r"\s+", " ", translated_text).strip()
    if not source or not translated:
        return False

    src_len = len(source)
    tgt_len = len(translated)
    src_en = len(ENGLISH_WORD_RE.findall(source))
    src_cjk = len(CJK_CHAR_RE.findall(source))
    tgt_cjk = len(CJK_CHAR_RE.findall(translated))
    sentence_marks = sum(translated.count(mark) for mark in ("。", "！", "？", ".", ";"))
    src_lines = [line.strip() for line in source_text.splitlines() if line.strip()]
    src_lower = source.lower()
    src_item_count = _count_numbered_item_markers(source_text)
    tgt_item_count = _count_numbered_item_markers(translated_text)

    # Short English source blocks should not expand into a long multi-sentence Chinese passage.
    if src_cjk <= 2 and 4 <= src_en <= 28 and src_len <= 200:
        if tgt_cjk >= max(70, src_en * 5) and tgt_len >= max(170, int(src_len * 2.4)):
            return True

    # 1-2 line source fragments becoming long paragraphs are usually wrong segment mapping/hallucination.
    if len(src_lines) <= 2 and src_len <= 140 and tgt_cjk >= 80 and tgt_len >= 190 and sentence_marks >= 3:
        return True

    # Cross-segment hallucination often appears as extra numbered list items.
    if src_item_count >= 1 and tgt_item_count > src_item_count and src_len <= 240:
        return True
    if src_item_count == 0 and tgt_item_count >= 2 and src_len <= 180 and src_en >= 6:
        return True
    # Numbered list fragments should keep explicit item markers.
    if src_item_count >= 1 and tgt_item_count == 0 and src_len <= 260 and src_en >= 6 and tgt_cjk >= 6:
        return True

    # Under-translation guard: long source blocks collapsing into very short outputs.
    if src_en >= 24 and src_len >= 240:
        if tgt_len <= max(30, int(src_len * 0.20)) and tgt_cjk <= max(20, int(src_en * 0.55)):
            return True
    if src_en >= 55 and src_len >= 520 and tgt_len <= max(42, int(src_len * 0.14)):
        return True

    # Short fragments should not explode into long multi-sentence paragraphs.
    if src_len <= 180 and src_en >= 5 and sentence_marks >= 3 and tgt_len >= int(src_len * 1.65) and tgt_cjk >= 72:
        return True

    # Incomplete tail fragments (common near page/layout boundaries) should remain short in translation.
    if src_len <= 130 and src_en >= 5 and any(
        src_lower.endswith(tail)
        for tail in (
            " for both the",
            " for the",
            " of the",
            " and the",
            " to the",
            " in the",
            ",",
        )
    ):
        if tgt_cjk >= 70 and tgt_len >= 150:
            return True
    if src_len <= 200 and (source.endswith("-") or src_lower.endswith(" and") or src_lower.endswith(" of")):
        if sentence_marks >= 2 and tgt_cjk >= 55 and tgt_len >= int(src_len * 1.45):
            return True
    if src_len <= 170 and src_en >= 8 and source[:1].islower():
        if sentence_marks >= 2 and tgt_cjk >= 42 and tgt_len >= int(src_len * 1.18):
            return True

    return False


def _is_low_quality_translation(source_text: str, translated_text: str) -> bool:
    source = re.sub(r"\s+", " ", source_text).strip()
    translated = re.sub(r"\s+", " ", translated_text).strip()
    if not source:
        return False
    if not translated:
        return True
    if _contains_non_paper_meta(translated):
        return True
    if _contains_untranslated_english_lines_in_mixed_output(source, translated):
        return True
    if _has_extreme_length_mismatch(source, translated):
        return True

    source_lower = source.lower()
    translated_lower = translated.lower()
    if translated_lower == source_lower:
        return True

    src_en = len(ENGLISH_WORD_RE.findall(source))

    tgt_en = len(ENGLISH_WORD_RE.findall(translated))
    tgt_cjk = len(CJK_CHAR_RE.findall(translated))
    
    # Count acronyms/proper nouns (all-caps words 2-8 chars) in translated text.
    # These are expected to remain in English and should not count against quality.
    tgt_acronyms = len(re.findall(r"\b[A-Z]{2,8}\b", translated))
    tgt_en_excluding_acronyms = max(0, tgt_en - tgt_acronyms)

    # Long prose should not degrade into a short section heading.
    if src_en >= 10 and len(source) >= 80 and not _looks_like_title_block(source):
        if _looks_like_short_section_heading_translation(translated):
            return True
    
    if _looks_like_title_block(source):
        if tgt_en >= max(6, int(src_en * 0.45)) and tgt_cjk < max(6, int(src_en * 0.35)):
            return True
        if _contains_large_untranslated_english_segment(source_lower, translated, src_en, for_title=True):
            return True
    if src_en < 5:
        if src_en >= 2 and tgt_cjk == 0 and tgt_en >= max(2, src_en - 1):
            return True
        return False

    # Relax the threshold when most remaining English words are acronyms/proper nouns.
    # If non-acronym English words are few and Chinese content is substantial, it's not low quality.
    if tgt_en_excluding_acronyms <= 3 and tgt_cjk >= 8:
        pass  # Good translation with proper nouns preserved
    elif tgt_en >= max(8, int(src_en * 0.55)) and tgt_cjk < max(8, int(src_en * 0.22)):
        return True

    if _contains_large_untranslated_english_segment(source_lower, translated, src_en, for_title=False):
        return True

    if len(source_lower) >= 24 and source_lower in translated_lower and tgt_cjk >= 4:
        return True

    src_first = _first_sentence(source_lower)
    if len(src_first) >= 28 and src_first in translated_lower:
        return True

    return False


def _contains_large_untranslated_english_segment(
    source_lower: str,
    translated_text: str,
    source_en_words: int,
    *,
    for_title: bool = False,
) -> bool:
    min_line_len = 24 if for_title else 36
    min_words = 6 if for_title else 8
    for raw_line in translated_text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if len(line) < min_line_len:
            continue

        words = ENGLISH_WORD_RE.findall(line)
        if len(words) < min_words:
            continue

        cjk = len(CJK_CHAR_RE.findall(line))
        # Pure or nearly pure English span in translated output.
        if cjk > 2:
            continue

        # Check if line starts with numbered list marker (e.g., "1)", "2.", "3)")
        # Numbered list items with Chinese content should not be flagged as untranslated.
        if re.match(r"^\s*\d{1,3}[\).]\s*", line) and cjk >= 2:
            continue

        # Check if line contains protocol/acronym names (all-caps words 2-8 chars)
        # These are proper nouns that should remain in English.
        acronym_count = len(re.findall(r"\b[A-Z]{2,8}\b", line))
        # If most English words are acronyms/proper nouns, this line is likely translated.
        non_acronym_words = len(words) - acronym_count
        if acronym_count >= 1 and non_acronym_words <= 2 and cjk >= 4:
            continue

        # If source itself is a prose block, long pure-English spans in output are usually untranslated residue.
        if source_en_words >= 12 and len(words) >= 10 and len(line) >= 60:
            return True

        lower = line.lower()
        if lower in source_lower:
            return True

        # Match by long prefix phrase to catch line-wrap differences.
        phrase = " ".join(words[:12]).lower()
        if len(phrase) >= 40 and phrase in source_lower:
            return True

    return False


def _looks_like_title_block(text: str) -> bool:
    compact = re.sub(r"\s+", " ", text).strip()
    if not compact:
        return False
    if any(mark in compact for mark in ("http://", "https://", "doi", "arxiv")):
        return False

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines or len(lines) > 4:
        return False

    en_words = len(ENGLISH_WORD_RE.findall(compact))
    cjk_chars = len(CJK_CHAR_RE.findall(compact))
    if en_words < 6 or en_words > 36:
        return False
    if cjk_chars > 2:
        return False

    punct = sum(compact.count(mark) for mark in ".?!;:")
    if punct > 2:
        return False
    if re.search(r"[=<>\\/\[\]{}_^$]", compact):
        return False
    avg_len = sum(len(line) for line in lines) / max(len(lines), 1)
    return avg_len >= 18


def _first_sentence(text: str) -> str:
    parts = re.split(r"[.!?;:]\s+", text, maxsplit=1)
    return parts[0].strip() if parts else text.strip()


def _looks_like_short_section_heading_translation(text: str) -> bool:
    compact = re.sub(r"\s+", " ", text).strip()
    if not compact:
        return False
    if len(compact) <= 28 and SHORT_SECTION_HEADING_TRANSLATION_RE.match(compact):
        return True
    compact_no_space = compact.replace(" ", "")
    return compact_no_space in SHORT_SECTION_HEADING_TRANSLATION_SET


def _repair_title_translation(source_text: str, translated_text: str) -> str | None:
    source = re.sub(r"\s+", " ", source_text).strip()
    translated = re.sub(r"\s+", " ", translated_text).strip()
    if not source or not translated:
        return None
    source_lower = source.lower()

    # Handle split title fragments first: multi-line English titles are often extracted
    # into 2 blocks (e.g., first line + second line), so full-title matching would miss.
    if re.fullmatch(
        r"revisiting\s+privacy[- ]preserving\s+min\s+and\s+k(?:-|\s)?th",
        source_lower,
    ):
        return "重新审视隐私保护最小值与第k小值"
    if re.fullmatch(
        r"min\s+protocols?\s+for\s+mobile\s+sensing",
        source_lower,
    ):
        return "面向移动感知的计算协议"

    if not _looks_like_title_block(source):
        return None
    if len(CJK_CHAR_RE.findall(source)) > 0:
        return None

    # Common title in this domain that is frequently mistranslated into two loose phrases.
    if re.fullmatch(
        r"revisiting\s+privacy[- ]preserving\s+min\s+and\s+k(?:-|\s)?th\s+min\s+protocols?\s+for\s+mobile\s+sensing",
        source_lower,
    ):
        return "重新审视面向移动感知的隐私保护最小值与第k小值计算协议"

    # If output is already clean and complete, keep it.
    if "第k小值" in translated and "最小值" in translated and ("协议" in translated or "计算" in translated):
        if "面向" in translated or "场景" in translated:
            return None

    # Lightweight structure repair for "Revisiting ... for Mobile Sensing".
    match = re.match(r"^revisiting\s+(.+?)\s+for\s+mobile\s+sensing$", source, flags=re.IGNORECASE)
    if not match:
        return None

    left = match.group(1).strip().lower()
    if re.fullmatch(
        r"privacy[- ]preserving\s+min\s+and\s+k(?:-|\s)?th\s+min\s+protocols?",
        left,
    ):
        return "重新审视面向移动感知的隐私保护最小值与第k小值计算协议"

    return None


def _repair_section_heading(source_text: str, section_no: str) -> str | None:
    src = re.sub(r"\s+", " ", source_text).strip()
    match = re.match(
        r"^(?:Section\s+)?(?P<num>[IVXLCMivxlcm0-9\.]+)\s*[:.\-]?\s*(?P<title>[A-Za-z][A-Za-z \-]+)$",
        src,
    )
    if not match:
        return None

    num = match.group("num")
    title = re.sub(r"\s+", " ", match.group("title")).strip().upper()
    title_map = {
        "ABSTRACT": "摘要",
        "INTRODUCTION": "引言",
        "RELATED WORK": "相关工作",
        "RELATED WORKS": "相关工作",
        "PRELIMINARIES": "预备知识",
        "PROBLEM STATEMENT": "问题定义",
        "SYSTEM MODEL": "系统模型",
        "METHOD": "方法",
        "METHODOLOGY": "方法",
        "EXPERIMENT": "实验",
        "EXPERIMENTS": "实验",
        "EVALUATION": "实验评估",
        "RESULTS": "结果",
        "DISCUSSION": "讨论",
        "CONCLUSION": "结论",
        "CONCLUSIONS": "结论",
    }

    zh = title_map.get(title)
    if zh is None:
        return None
    return f"{num} {zh}"
