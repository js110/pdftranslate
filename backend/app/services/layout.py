"""GROBID reference-region guard."""
from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx

from app.core.settings import get_settings
from app.services.geom import _dedupe_regions

logger = logging.getLogger(__name__)

_grobid_reference_cache: dict[tuple[str, int, int], dict[int, list[tuple[float, float, float, float]]] | None] = {}


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
