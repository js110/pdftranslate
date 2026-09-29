"""Shared dataclasses for the PDF translation pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PageProcessResult:
    warnings: list[str] = field(default_factory=list)
    overflow_items: list[dict[str, Any]] = field(default_factory=list)
    untranslated_items: list[dict[str, Any]] = field(default_factory=list)
    fallback_events: list[dict[str, Any]] = field(default_factory=list)



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


