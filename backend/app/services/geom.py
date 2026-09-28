"""Rectangle / region geometry helpers (pure functions)."""
from __future__ import annotations


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


