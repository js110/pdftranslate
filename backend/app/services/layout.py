"""Layout detection (PP-StructureV3) and GROBID reference-region guard."""
from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from PIL import Image

from app.core.settings import get_settings
from app.services.geom import _dedupe_regions

try:
    from paddleocr import PPStructureV3
except Exception:  # noqa: BLE001
    PPStructureV3 = None  # type: ignore[misc,assignment]

logger = logging.getLogger(__name__)


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

_grobid_reference_cache: dict[tuple[str, int, int], dict[int, list[tuple[float, float, float, float]]] | None] = {}

_layout_detector_instance: Any = None

@dataclass
class _LayoutHints:
    table_regions: list[tuple[float, float, float, float]] = field(default_factory=list)
    reference_regions: list[tuple[float, float, float, float]] = field(default_factory=list)
    skip_regions: list[tuple[float, float, float, float]] = field(default_factory=list)



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


