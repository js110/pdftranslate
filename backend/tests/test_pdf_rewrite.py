from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from app.services.pdf_rewrite import (
    RewriteFonts,
    RewriteParagraph,
    SrcLine,
    _detect_justify,
    extract_page_paragraphs,
    has_cjk,
    is_translatable,
    rewrite_document,
    wrap,
)


def _make_source(path: Path) -> None:
    doc = fitz.open()
    page = doc.new_page(width=400, height=500)
    page.insert_text((50, 60), "Rewrite Me Title", fontsize=16, fontname="helv")
    for i in range(4):
        page.insert_text(
            (50, 100 + i * 14),
            f"This is line {i} of the abstract body text for testing" + "." * i,
            fontsize=10,
            fontname="helv",
        )
    page.draw_line((50, 206), (200, 206))
    page.insert_text((50, 215), "*Corresponding author: Jane Doe.", fontsize=9, fontname="helv")
    doc.save(path.as_posix())
    doc.close()


def _fonts(tmp_path: Path) -> RewriteFonts:
    fonts = RewriteFonts(tmp_path / "fonts")
    try:
        fonts.resolve()
    except RuntimeError:
        pytest.skip("no CJK font available on this host")
    return fonts


def test_wrap_keeps_latin_runs_whole():
    font = fitz.Font(fontname="china-s")
    lines = wrap("验证 EdDSA/MiMC 签名的证据并绑定 Groth16/BN254", font, 10, 60)
    assert len(lines) > 1
    assert all("EdDSA" in line or "MiMC" not in line for line in lines)
    assert any("EdDSA/MiMC" in line for line in lines)


def test_wrap_avoids_line_start_punctuation():
    font = fitz.Font(fontname="china-s")
    text = "甲" * 19 + "，" + "乙" * 10
    lines = wrap(text, font, 10, 62)  # ~6 chars per line at size 10
    for line in lines[1:]:
        assert line[0] not in "，。、；：！？"


def test_detect_justify_requires_flush_edges():
    def line(x0: float, x1: float) -> SrcLine:
        return SrcLine(text="x", bbox=fitz.Rect(x0, 0, x1, 10), x0=x0, base=10.0, size=10.0)

    flush = [line(50, 300), line(50, 301), line(50, 299), line(50, 180)]
    assert _detect_justify(flush) is True
    ragged = [line(50, 300), line(52, 280), line(50, 310), line(50, 180)]
    assert _detect_justify(ragged) is False
    assert _detect_justify(flush[:2]) is False


def test_has_cjk():
    assert has_cjk("中文测试")
    assert not has_cjk("English 123")


def _para(
    text: str = "Plain English prose sentence for the body.",
    *,
    rect: fitz.Rect | None = None,
    size: float = 10.0,
    horizontal: bool = True,
    tight_pitch: bool = False,
) -> RewriteParagraph:
    r = rect or fitz.Rect(50, 100, 300, 140)
    return RewriteParagraph(
        rect=r,
        text=text,
        size=size,
        bold=False,
        color=0,
        align="left",
        justify=False,
        bases=[110.0, 122.0],
        pitch=12.0,
        src_lines=[SrcLine(text=text, bbox=r, x0=r.x0, base=110.0, size=size)],
        horizontal=horizontal,
        tight_pitch=tight_pitch,
    )


def test_is_translatable_rejects_fragments_and_formulas():
    assert is_translatable(_para()) is True
    assert is_translatable(_para(horizontal=False)) is False
    assert is_translatable(_para(tight_pitch=True)) is False
    assert is_translatable(_para(rect=fitz.Rect(50, 100, 58, 140))) is False
    assert is_translatable(_para("l Y", size=6.0, rect=fitz.Rect(50, 100, 61, 140))) is False
    assert is_translatable(_para("ϵbj exp(− 2(bmax−bmin) ) Pr(bi)", size=5.0)) is False
    assert is_translatable(_para("The mechanism satisfies individual rationality.")) is True


def test_extract_page_paragraphs(tmp_path: Path):
    src = tmp_path / "src.pdf"
    _make_source(src)
    doc = fitz.open(src)
    paras = extract_page_paragraphs(doc[0])
    doc.close()
    texts = [p.text for p in paras]
    assert any("Rewrite Me Title" in t for t in texts)
    assert any("abstract body" in t for t in texts)
    body = next(p for p in paras if "abstract body" in p.text)
    assert len(body.src_lines) == 4
    assert body.justify is False


def test_rewrite_document_roundtrip(tmp_path: Path):
    src = tmp_path / "src.pdf"
    out = tmp_path / "out.pdf"
    _make_source(src)
    fonts = _fonts(tmp_path)
    seen_pages: list[int] = []

    def translate(page_no: int, paras: list[RewriteParagraph]) -> list[str | None]:
        seen_pages.append(page_no)
        return [f"第{i}段中文译文内容用于验证。" for i in range(len(paras))]

    stats = rewrite_document(src, out, translate=translate, fonts=fonts)
    assert out.exists()
    assert stats["pages"] == 1
    assert stats["written"] > 0
    assert seen_pages == [1]

    doc = fitz.open(out)
    page = doc[0]
    text = page.get_text()
    drawings = page.get_drawings()
    doc.close()
    assert "中文译文" in text
    assert "abstract body" not in text
    # vector art survives redaction
    assert any(dr["rect"].height < 2 for dr in drawings)


def test_rewrite_keeps_untranslated_paragraphs(tmp_path: Path):
    src = tmp_path / "src.pdf"
    out = tmp_path / "out.pdf"
    _make_source(src)
    fonts = _fonts(tmp_path)

    def translate(page_no: int, paras: list[RewriteParagraph]) -> list[str | None]:
        return [None if "abstract" in p.text else f"中文{i}段译文。" for i, p in enumerate(paras)]

    stats = rewrite_document(src, out, translate=translate, fonts=fonts)
    assert stats["kept"] >= 1

    doc = fitz.open(out)
    text = doc[0].get_text()
    doc.close()
    assert "abstract body" in text  # untouched paragraph kept its English


def test_rewrite_keeps_original_when_translation_cannot_fit(tmp_path: Path):
    src = tmp_path / "src.pdf"
    out = tmp_path / "out.pdf"
    _make_source(src)
    fonts = _fonts(tmp_path)

    def translate(page_no: int, paras: list[RewriteParagraph]) -> list[str | None]:
        return ["中" * 400 if "abstract" in p.text else "标题已译。" for p in paras]

    stats = rewrite_document(src, out, translate=translate, fonts=fonts)
    assert stats["written"] < stats["paragraphs"]  # overflow paragraph dropped

    doc = fitz.open(out)
    text = doc[0].get_text()
    doc.close()
    assert "abstract body" in text  # too-long translation never redacted the source
