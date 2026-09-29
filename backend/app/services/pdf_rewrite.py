"""In-place PDF translation: replace source glyphs with Chinese at the same
bboxes so the original layout (rules, images, columns, margins) is preserved.

Ported from the validated spike (spike.py), hardened for whole-document use:
- paragraphs are extracted per page and translated via the shared pipeline
- a paragraph whose translation fails/skips keeps its original text (no redact)
- one subsetted CJK font per document, embedded once per page
"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import fitz
from fontTools import subset as fts
from fontTools.ttLib import TTFont

logger = logging.getLogger(__name__)

_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9@._:/+\-]*[A-Za-z0-9]|[A-Za-z0-9]")
_NO_START = "，。、；：！？）】》」』%”’…·"
_CJK_RE = re.compile(r"[㐀-䶿一-鿿豈-﫿]")


def has_cjk(text: str) -> bool:
    """True when the text already contains CJK characters (nothing to translate)."""
    return bool(_CJK_RE.search(text))


_MATH_CHARS = set("ΓΔΘΛΞΠΣΦΨΩαβγδεζηθικλμνξπρστυφχψωϵ")


def _math_ratio(text: str) -> float:
    core = [c for c in text if not c.isspace()]
    if not core:
        return 0.0
    mathy = sum(1 for c in core if c in _MATH_CHARS or c in "−≤≥≈≠∈∉∪∩∂∇∀∃∑∏∫±×÷→←′∗")
    return mathy / len(core)


def is_translatable(p: RewriteParagraph) -> bool:
    """False for rotated/stacked/formula-fragment blocks that must stay as-is.

    Display equations are extracted as tiny side-by-side fragments; rewriting
    them wraps at ~1 char/line and cascades over the rest of the page."""
    if not p.horizontal or p.tight_pitch or p.blocked:
        return False
    if p.rect.width < 60:
        return False
    ratio = _math_ratio(p.text)
    if p.size < 8 and ratio > 0.10:
        return False
    if ratio > 0.35:
        return False
    return True

# Face preference inside font collections (.ttc): serif first (academic look).
_REGULAR_FACE_PREFER = ("songtisc", "notoserifcjksc", "notosanscjksc", "microsoftyahei", "simsun")
_BOLD_FACE_PREFER = ("songtiscbold", "notoserifcjksc", "notosanscjkscbold", "microsoftyaheibold")


@dataclass
class SrcLine:
    text: str
    bbox: fitz.Rect
    x0: float
    base: float
    size: float
    dir: tuple[float, float] = (1.0, 0.0)


@dataclass
class RewriteParagraph:
    rect: fitz.Rect
    text: str
    size: float
    bold: bool
    color: int
    align: str
    justify: bool
    bases: list[float]
    pitch: float
    src_lines: list[SrcLine]
    horizontal: bool = True
    tight_pitch: bool = False
    blocked: bool = False


# --------------------------------------------------------------------- extraction
def _extract_blocks(page: fitz.Page) -> list[dict]:
    raw = page.get_text("dict")
    blocks = []
    for b in raw["blocks"]:
        if b["type"] != 0 or not b["lines"]:
            continue
        lines = []
        for ln in b["lines"]:
            text = "".join(s["text"] for s in ln["spans"])
            origin = ln["spans"][0]["origin"]
            lines.append(
                SrcLine(
                    text=text,
                    bbox=fitz.Rect(ln["bbox"]),
                    x0=origin[0],
                    base=origin[1],
                    size=max(s["size"] for s in ln["spans"]),
                    dir=tuple(ln.get("dir", (1.0, 0.0))),
                )
            )
        sp = b["lines"][0]["spans"][0]
        # majority span size by char count — one drop-cap/heading span must not
        # inflate the whole paragraph's size (breaks baseline pitch => overlap)
        by_size: dict[float, int] = {}
        for ln in b["lines"]:
            for s in ln["spans"]:
                if s["text"]:
                    by_size[float(s["size"])] = by_size.get(float(s["size"]), 0) + len(s["text"])
        maj_size = max(by_size, key=lambda k: (by_size[k], -k)) if by_size else float(sp["size"])
        blocks.append(
            {
                "bbox": fitz.Rect(b["bbox"]),
                "lines": lines,
                "font": sp["font"],
                "size": maj_size,
                "chars": sum(by_size.values()),
                "flags": int(sp["flags"]),
                "color": int(sp["color"]),
            }
        )
    blocks.sort(key=lambda x: (round(x["bbox"].y0, 1), x["bbox"].x0))
    return blocks


def _font_key(name: str) -> str:
    return re.sub(r"[-+]?(Bold|Black|Italic|Oblique|Light|Medium)", "", name, flags=re.I)


def _build_para(
    rect: fitz.Rect,
    raw_lines: list[SrcLine],
    *,
    by_size: dict[float, int],
    bold: bool,
    color: int,
) -> RewriteParagraph | None:
    # PDF content-stream order is not visual order — sort by baseline first,
    # otherwise wrapped/algorithm lines overlap with negative pitch.
    raw_lines.sort(key=lambda ln: (round(ln.base, 1), ln.x0))
    logical: list[SrcLine] = []
    for ln in raw_lines:
        if logical:
            prev = logical[-1]
            # same visual line: drop-cap fragments sit ~0.4*size apart
            if abs(prev.base - ln.base) < max(1.0, 0.4 * min(prev.size, ln.size)):
                prev.text = (prev.text.rstrip() + " " + ln.text.lstrip()).strip()
                prev.x0 = min(prev.x0, ln.x0)
                prev.bbox |= ln.bbox
                prev.size = max(prev.size, ln.size)
                prev.base = ln.base  # keep the main baseline (drop-cap case)
                continue
        logical.append(ln)
    parts: list[str] = []
    for ln in logical:
        if not parts:
            parts.append(ln.text)
        elif parts[-1].endswith("-") and ln.text[:1].islower():
            parts[-1] = parts[-1][:-1] + ln.text
        else:
            parts.append(ln.text)
    text = re.sub(r"\s+", " ", " ".join(parts)).strip()
    if not text:
        return None
    bases = [ln.base for ln in logical]
    para_size = max(by_size, key=lambda k: (by_size[k], -k)) if by_size else max(ln.size for ln in logical)
    # baselines closer than 0.7*size = stacked formula fragments (numerator/
    # denominator lines) mixed into the block — rewriting would collide them
    tight = any(bases[k + 1] - bases[k] < 0.7 * para_size for k in range(len(bases) - 1))
    return RewriteParagraph(
        rect=rect,
        text=text,
        size=para_size,
        bold=bold,
        color=color,
        align="left",  # set below
        justify=_detect_justify(logical),
        bases=bases,
        pitch=(bases[-1] - bases[0]) / (len(bases) - 1) if len(bases) > 1 else para_size * 1.2,
        src_lines=logical,
        horizontal=all(abs(ln.dir[0]) > 0.99 for ln in logical),
        tight_pitch=tight,
    )


def _resolve_overlaps(paras: list[RewriteParagraph]) -> list[RewriteParagraph]:
    """Group paragraphs whose rects intersect (font-split blocks, inline math).

    - all members are full-width prose → merge into one paragraph (the split
      theorem/proof statement must be rewritten as a single flowing text)
    - otherwise block every member → original stays untouched, so a redact
      rect can never erase an overlapping equation/fragment."""
    n = len(paras)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            inter = paras[i].rect & paras[j].rect
            if inter.is_valid and inter.width >= 6 and inter.height >= 4:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    result: list[RewriteParagraph | None] = [None] * n
    for members in groups.values():
        if len(members) == 1:
            result[members[0]] = paras[members[0]]
            continue
        ps = [paras[k] for k in members]
        mergeable = all(
            p.horizontal
            and not p.tight_pitch
            and p.rect.width >= 100
            and p.size >= 8
            and _math_ratio(p.text) <= 0.30
            for p in ps
        )
        if mergeable:
            rect = fitz.Rect(ps[0].rect)
            lines: list[SrcLine] = []
            by_size: dict[float, int] = {}
            for p in ps:
                rect |= p.rect
                lines.extend(p.src_lines)
                by_size[p.size] = by_size.get(p.size, 0) + len(p.text)
            merged = _build_para(rect, lines, by_size=by_size, bold=ps[0].bold, color=ps[0].color)
            result[members[0]] = merged
            for k in members[1:]:
                result[k] = None
        else:
            for k, p in zip(members, ps, strict=True):
                p.blocked = True
                result[k] = p
    return [q for q in result if q is not None]


def extract_page_paragraphs(page: fitz.Page) -> list[RewriteParagraph]:
    blocks = _extract_blocks(page)

    paras: list[dict] = []
    for blk in blocks:
        if paras:
            prev = paras[-1]["blocks"][-1]
            gap = blk["bbox"].y0 - prev["bbox"].y1
            merge = (
                _font_key(prev["font"]) == _font_key(blk["font"])
                and abs(prev["size"] - blk["size"]) < 0.3
                and abs(prev["bbox"].x0 - blk["bbox"].x0) < 2.5
                and -0.4 * prev["size"] < gap < 0.95 * prev["size"]
                and (prev["flags"] & 16) == (blk["flags"] & 16)
            )
            if merge:
                paras[-1]["blocks"].append(blk)
                continue
        paras.append({"blocks": [blk]})

    out: list[RewriteParagraph] = []
    for p in paras:
        rect = fitz.Rect(p["blocks"][0]["bbox"])
        raw_lines: list[SrcLine] = []
        by_size: dict[float, int] = {}
        for blk in p["blocks"]:
            rect |= blk["bbox"]
            raw_lines.extend(blk["lines"])
            by_size[blk["size"]] = by_size.get(blk["size"], 0) + int(blk["chars"])
        para = _build_para(
            rect,
            raw_lines,
            by_size=by_size,
            bold=bool(p["blocks"][0]["flags"] & 16),
            color=p["blocks"][0]["color"],
        )
        if para is not None:
            out.append(para)

    out = _resolve_overlaps(out)
    # A rect that straddles an image (bio blocks: lines beside the photo plus
    # lines below it) would be re-wrapped straight across the picture when
    # redrawn at full width — keep those paragraphs in the original language.
    image_rects = [fitz.Rect(im["bbox"]) for im in page.get_image_info()]
    if image_rects:
        for q in out:
            for ib in image_rects:
                inter = fitz.Rect(q.rect)
                inter.intersect(ib)
                if inter.is_valid and inter.width >= 4 and inter.height >= 4:
                    q.blocked = True
                    break
    if out:
        widest = max(out, key=lambda q: q.rect.width)
        col_center = (widest.rect.x0 + widest.rect.x1) / 2
        for q in out:
            q.align = _detect_align(q, col_center)
    return out


def _detect_align(p: RewriteParagraph, col_center: float) -> str:
    lines = p.src_lines
    if len(lines) == 1:
        ln = lines[0]
        line_center = (ln.bbox.x0 + ln.bbox.x1) / 2
        if abs(line_center - col_center) < 6:
            return "center"
        return "left"
    xs0 = [ln.bbox.x0 for ln in lines]
    if max(xs0) - min(xs0) < 2:
        return "left"
    cs = [(ln.bbox.x0 + ln.bbox.x1) / 2 for ln in lines]
    if max(cs) - min(cs) < 3:
        return "center"
    return "left"


def _detect_justify(lines: list[SrcLine]) -> bool:
    if len(lines) < 3:
        return False
    x0s = [ln.bbox.x0 for ln in lines]
    x1s = [ln.bbox.x1 for ln in lines]
    return (max(x0s) - min(x0s) < 2) and (max(x1s[:-1]) - min(x1s[:-1]) < 3)


# ------------------------------------------------------------------------- fonts
class RewriteFonts:
    """Resolve + subset CJK fonts for embedding."""

    def __init__(self, cache_dir: Path, *, regular_override: str | None = None, bold_override: str | None = None) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.regular_override = regular_override
        self.bold_override = bold_override
        self._regular: Path | None = None
        self._bold: Path | None | bool = False  # False = not resolved yet, None = unavailable

    @staticmethod
    def _is_collection(path: Path) -> bool:
        try:
            with open(path, "rb") as fh:
                return fh.read(4) == b"ttcf"
        except OSError:
            return False

    @staticmethod
    def _default_candidates(bold: bool) -> list[str]:
        prefer = _BOLD_FACE_PREFER if bold else _REGULAR_FACE_PREFER
        names = "-Bold" if bold else "-Regular"
        return [
            "/System/Library/Fonts/Supplemental/Songti.ttc",
            f"/usr/share/fonts/opentype/noto/NotoSerifCJK{names}.ttc",
            f"/usr/share/fonts/opentype/noto/NotoSansCJK{names}.ttc",
            f"/usr/share/fonts/truetype/noto/NotoSansCJK{names}.ttc",
            "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/simsunb.ttc" if bold else "C:/Windows/Fonts/simsun.ttc",
        ], prefer

    @classmethod
    def _select_face_index(cls, path: Path, prefer: tuple[str, ...]) -> int:
        if not cls._is_collection(path):
            return 0
        for i in range(16):
            try:
                font = TTFont(path, fontNumber=i, lazy=True)
            except Exception:  # noqa: BLE001
                break
            try:
                name = font["name"].getDebugName(4) or ""
            finally:
                font.close()
            norm = re.sub(r"[\s\-]+", "", name).lower()
            if any(pref in norm for pref in prefer):
                return i
        return 0

    @classmethod
    def _face_file(cls, path: Path, prefer: tuple[str, ...], tag: str, cache_dir: Path) -> Path | None:
        if not path.exists():
            return None
        if not cls._is_collection(path):
            return path
        idx = cls._select_face_index(path, prefer)
        stem = f"{path.stem}-face{idx}-{tag}"
        for ext in (".otf", ".ttf"):
            cached = cache_dir / f"{stem}{ext}"
            if cached.exists():
                return cached
        font = TTFont(path, fontNumber=idx)
        ext = ".otf" if "CFF " in font else ".ttf"
        out = cache_dir / f"{stem}{ext}"
        font.save(out)
        font.close()
        logger.info("extracted font face %s -> %s", path, out)
        return out

    def resolve(self) -> tuple[Path, Path | None]:
        if self._regular is None:
            regular = self._resolve_one(bold=False)
            if regular is None:
                raise RuntimeError(
                    "no CJK font found for rewrite; set REWRITE_FONT_REGULAR (ttf/otf/ttc)"
                )
            self._regular = regular
        if self._bold is False:
            self._bold = self._resolve_one(bold=True)
        return self._regular, (self._bold if isinstance(self._bold, Path) else None)

    def _resolve_one(self, *, bold: bool) -> Path | None:
        override = self.bold_override if bold else self.regular_override
        if override:
            candidates, prefer = [override], (_BOLD_FACE_PREFER if bold else _REGULAR_FACE_PREFER)
        else:
            candidates, prefer = self._default_candidates(bold)
        for cand in candidates:
            face = self._face_file(Path(cand), prefer, "bold" if bold else "reg", self.cache_dir)
            if face is not None:
                return face
        # No dedicated bold face: fall back to the regular face's bold sibling,
        # otherwise plain regular (bold paragraphs render regular).
        if bold:
            return None
        return None

    def has_bold(self) -> bool:
        _, bold = self.resolve()
        return bold is not None

    def metric_font(self, *, bold: bool) -> fitz.Font:
        regular, bold_face = self.resolve()
        path = bold_face if (bold and bold_face is not None) else regular
        return fitz.Font(fontfile=path.as_posix())

    def subset(self, chars: set[str], *, bold: bool) -> Path:
        regular, bold_face = self.resolve()
        face = bold_face if (bold and bold_face is not None) else regular
        payload = "".join(sorted(chars))
        digest = hashlib.sha1(f"g2-{payload}".encode()).hexdigest()[:16]
        key = f"{face.stem}-{digest}"
        out = self.cache_dir / f"sub-{key}{face.suffix}"
        if out.exists():
            return out
        opts = fts.Options()
        opts.layout_features = ["*"]
        opts.notdef_outline = True
        opts.recalc_bounds = True
        # GID remapping breaks MuPDF rendering of subset CFF/CID fonts
        # (text layer fine, glyphs wrong) — keep glyph ids stable.
        opts.retain_gids = True
        font = fts.load_font(str(face), opts)
        subsetter = fts.Subsetter(options=opts)
        subsetter.populate(text=payload)
        subsetter.subset(font)
        fts.save_font(font, str(out), opts)
        font.close()
        logger.info("subset font %s (%d chars) -> %s", face.name, len(chars), out)
        return out


# ------------------------------------------------------------------------ layout
def wrap(text: str, font: fitz.Font, size: float, width: float) -> list[str]:
    # keep latin/number runs whole ("EdDSA/MiMC") — only CJK breaks char-wise
    tokens: list[str] = []
    pos = 0
    for m in _WORD.finditer(text):
        if m.start() > pos:
            tokens.append(text[pos : m.start()])
        tokens.append(m.group())
        pos = m.end()
    if pos < len(text):
        tokens.append(text[pos:])

    def fits(s: str) -> bool:
        return font.text_length(s, fontsize=size) <= width

    lines: list[str] = []
    cur = ""
    for tok in tokens:
        if fits(cur + tok):
            cur += tok
            continue
        if _WORD.fullmatch(tok) and fits(tok):
            lines.append(cur)
            cur = tok
            continue
        for ch in tok:
            if fits(cur + ch) or not cur:
                cur += ch
            elif ch in _NO_START and len(cur) > 1:  # no punctuation at line start
                lines.append(cur[:-1])
                cur = cur[-1] + ch
            else:
                lines.append(cur)
                cur = ch
    if cur:
        lines.append(cur)
    return lines


def _rule_above(page: fitz.Page, rect: fitz.Rect) -> float | None:
    """Horizontal rule sitting just above this block (footnote/section rule)."""
    for dr in page.get_drawings():
        r = dr["rect"]
        if r.height > 1.5 or r.width < 15:
            continue
        if not (rect.y0 - 5 <= r.y0 <= rect.y0 + 2):
            continue
        if min(r.x1, rect.x1) - max(r.x0, rect.x0) < min(30, rect.width * 0.5):
            continue
        return r.y0
    return None


def _layout_para(
    p: RewriteParagraph,
    zh: str,
    size: float,
    font: fitz.Font,
    rule_y: float | None,
) -> tuple[list[tuple[float, float, str, float, float]], float]:
    rect = p.rect
    lines = wrap(zh, font, size, rect.width - 1)
    orig = p.bases
    # CJK glyphs fill the full em: source baselines tighter than ~1.06x size
    # (algorithm boxes, mixed stacks) would overlap when redrawn in Chinese.
    min_gap = 1.06 * size
    pitch = max(p.pitch * (size / p.size), min_gap)
    if len(lines) <= len(orig):
        shift = 0.0
        if p.align == "center" and len(orig) > 1:
            shift = (len(orig) - len(lines)) * pitch / 2
        bases = [b + shift for b in orig[: len(lines)]]
    else:
        bases = list(orig) + [orig[-1] + pitch * k for k in range(1, len(lines) - len(orig) + 1)]
    if rule_y is not None and bases:
        need = rule_y + 2.5 + 0.78 * size
        if bases[0] < need:
            delta = need - bases[0]
            bases = [b + delta for b in bases]
    for i in range(1, len(bases)):
        if bases[i] - bases[i - 1] < min_gap:
            bases[i] = bases[i - 1] + min_gap
    out: list[tuple[float, float, str, float, float]] = []
    for i, line in enumerate(lines):
        w = font.text_length(line, fontsize=size)
        gap = 0.0
        if p.align == "center":
            x = rect.x0 + (rect.width - w) / 2
        elif p.align == "right":
            x = rect.x1 - w
        else:
            x = rect.x0
            if p.justify and i < len(lines) - 1 and line and rect.width - w > 0.5:
                gap = (rect.width - w) / (len(line) - 1) if len(line) > 1 else 0.0
        out.append((bases[i], x, line, gap, w))
    bottom = out[-1][0] + size * 0.28 if out else 0.0
    return out, bottom


# ------------------------------------------------------------------------- write
def _sanitize(text: str, font: fitz.Font) -> str:
    out = []
    for ch in text:
        if ch == " " or font.has_glyph(ord(ch)):
            out.append(ch)
        elif font.has_glyph(ord("*")):
            out.append("*")
        else:
            out.append(" ")
    return "".join(out)


def write_page(
    page: fitz.Page,
    paras: list[RewriteParagraph],
    translations: list[str | None],
    *,
    reg_subset: Path,
    bold_subset: Path | None,
    fonts: RewriteFonts,
    reg_metric: fitz.Font,
    bold_metric: fitz.Font | None,
) -> int:
    """Redact translated paragraphs and write Chinese back. Returns written count."""
    pending = [(p, zh) for p, zh in zip(paras, translations, strict=False) if zh]
    if not pending:
        return 0

    # layout dry-run BEFORE redacting: a paragraph that still overflows at the
    # minimum size would cascade over its neighbours — keep the original instead
    prepared: list[tuple[RewriteParagraph, float, str, list, tuple, fitz.Font]] = []
    for p, zh in pending:
        metric = bold_metric if (p.bold and bold_metric is not None) else reg_metric
        text = _sanitize(zh, metric)
        rule_y = _rule_above(page, p.rect)
        size = p.size
        min_size = p.size * 0.62
        allowed = p.rect.y1 + p.size * 0.45
        while True:
            laid, bottom = _layout_para(p, text, size, metric, rule_y)
            if bottom <= allowed or size <= min_size + 1e-6:
                break
            size -= 0.3
        if bottom > allowed:
            logger.info("rewrite: keep original (no room) p=(%.0f,%.0f) '%s'", p.rect.x0, p.rect.y0, p.text[:40])
            continue
        color = ((p.color >> 16 & 255) / 255, (p.color >> 8 & 255) / 255, (p.color & 255) / 255)
        prepared.append((p, size, text, laid, color, metric))

    if not prepared:
        return 0

    for p, *_ in prepared:
        page.add_redact_annot(p.rect)
    page.apply_redactions(images=0, graphics=0, text=0)

    page.insert_font(fontname="rw1", fontfile=reg_subset.as_posix())
    bold_face = bold_subset if bold_subset is not None else reg_subset
    if bold_subset is not None:
        page.insert_font(fontname="rw2", fontfile=bold_face.as_posix())

    written = 0
    for p, size, _text, laid, color, metric in prepared:
        fontname = "rw2" if (p.bold and bold_subset is not None) else "rw1"
        for base, x, line, gap, _w in laid:
            if gap > 0:
                cx = x
                for ch in line:
                    page.insert_text((cx, base), ch, fontname=fontname, fontsize=size, color=color)
                    cx += metric.text_length(ch, fontsize=size) + gap
            else:
                page.insert_text((x, base), line, fontname=fontname, fontsize=size, color=color)
        written += 1
    return written


# ------------------------------------------------------------------ orchestration
def rewrite_document(
    source_pdf: Path,
    out_pdf: Path,
    *,
    translate: Callable[[int, list[RewriteParagraph]], list[str | None]],
    fonts: RewriteFonts,
    on_progress: Callable[[str, int, int], None] | None = None,
) -> dict[str, int]:
    """Translate + rewrite the whole document. `translate(page_no, paras)` must
    return one entry per paragraph; None means "keep the original text"."""
    doc = fitz.open(source_pdf)
    stats = {"pages": doc.page_count, "paragraphs": 0, "written": 0, "kept": 0}
    try:
        total = doc.page_count
        plans: list[list[RewriteParagraph]] = []
        for i in range(total):
            plans.append(extract_page_paragraphs(doc[i]))
            if on_progress:
                on_progress("extract", i + 1, total)

        translated_pages: list[list[str | None]] = []
        for i in range(total):
            paras = plans[i]
            if paras:
                results = translate(i + 1, paras)
                if len(results) != len(paras):
                    raise RuntimeError(f"translate returned {len(results)} for {len(paras)} paragraphs on page {i + 1}")
                translated_pages.append(results)
            else:
                translated_pages.append([])
            if on_progress:
                on_progress("translate", i + 1, total)

        used_reg: set[str] = set()
        used_bold: set[str] = set()
        has_bold = fonts.has_bold()
        for paras, results in zip(plans, translated_pages, strict=False):
            for p, zh in zip(paras, results, strict=False):
                if not zh:
                    stats["kept"] += 1
                    continue
                stats["paragraphs"] += 1
                target = used_bold if (p.bold and has_bold) else used_reg
                target.update(zh)

        reg_metric = fonts.metric_font(bold=False)
        bold_metric = fonts.metric_font(bold=True) if has_bold else None
        reg_subset = fonts.subset(used_reg or {"转"}, bold=False)
        bold_subset = fonts.subset(used_bold, bold=True) if (has_bold and used_bold) else None

        for i in range(total):
            stats["written"] += write_page(
                doc[i],
                plans[i],
                translated_pages[i],
                reg_subset=reg_subset,
                bold_subset=bold_subset,
                fonts=fonts,
                reg_metric=reg_metric,
                bold_metric=bold_metric,
            )
            if on_progress:
                on_progress("write", i + 1, total)

        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        doc.save(out_pdf.as_posix(), garbage=0, deflate=True)
    finally:
        doc.close()
    return stats
