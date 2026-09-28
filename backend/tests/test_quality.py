"""Tests for translation quality evaluation, repair, and truncation."""
from __future__ import annotations

from app.services.quality import (
    _fallback_translate_short_formula_phrase,
    _fallback_translate_table_narrative,
    _is_low_quality_translation,
    _looks_like_title_block,
    _normalize_translated_text,
    _repair_section_heading,
    _truncate_by_sentence,
    _trim_extra_numbered_items,
)
from app.services.textutil import _trim_overlong_fragment_translation


class TestNormalizeTranslatedText:
    def test_plain_translation(self):
        assert _normalize_translated_text("Table I shows the results.", "表I展示了结果。") == "表I展示了结果。"

    def test_strips_yiwen_prefix(self):
        assert _normalize_translated_text("Hello.", "译文：你好。") == "你好。"

    def test_echo_kept_for_later_guards(self):
        out = _normalize_translated_text("Some English sentence here.", "Some English sentence here.")
        assert out  # normalized output is not empty


class TestLowQuality:
    def test_echo_is_low_quality(self):
        assert _is_low_quality_translation(
            "This is a longer English sentence about cryptographic protocols.",
            "This is a longer English sentence about cryptographic protocols.",
        )

    def test_good_translation_not_flagged(self):
        assert not _is_low_quality_translation(
            "We propose a new protocol for privacy-preserving aggregation in mobile sensing.",
            "我们提出了一种用于移动感知中隐私保护聚合的新协议。",
        )

    def test_meta_leak_is_low_quality(self):
        assert _is_low_quality_translation(
            "This is the source sentence about protocols.",
            "这是输出。strict retry mode: 请严格遵循要求。",
        )

    def test_mathseg_token_leak_is_low_quality(self):
        assert _is_low_quality_translation("Source text here.", "结果包含 MATHSEG3TOKEN 占位符。")

    def test_empty_translation_is_low_quality(self):
        assert _is_low_quality_translation("Some source text.", "")


class TestTrimming:
    def test_truncate_by_sentence(self):
        text = "第一句话。第二句话。第三句话。"
        trimmed = _truncate_by_sentence(text, max_len=6)
        assert trimmed == "第一句话。"

    def test_trim_extra_numbered_items(self):
        src = "We do one thing."
        translated = "1) 第一项内容比较长一些。2) 第二项。3) 第三项。"
        trimmed = _trim_extra_numbered_items(src, translated)
        assert "第二项" not in trimmed

    def test_trim_keeps_matching_items(self):
        src = "1) one thing 2) another"
        translated = "1) 第一项。2) 第二项。"
        assert _trim_extra_numbered_items(src, translated) == translated

    def test_overlong_fragment_trimmed(self):
        src = "of the"
        translated = "很长的译文。" * 30
        out = _trim_overlong_fragment_translation(src, translated)
        assert len(out) < len(translated)


class TestDeterministicFallbacks:
    def test_table_narrative(self):
        out = _fallback_translate_table_narrative("Table I shows several protocols under given settings.")
        assert out is not None
        assert out.startswith("表I")

    def test_table_narrative_rejects_non_table(self):
        assert _fallback_translate_table_narrative("This is a normal sentence about protocols.") is None

    def test_short_phrase_note_that(self):
        assert _fallback_translate_short_formula_phrase("Note that") == "注意"

    def test_short_phrase_with_colon(self):
        assert _fallback_translate_short_formula_phrase("For example:") == "例如："

    def test_short_phrase_rejects_long_text(self):
        assert _fallback_translate_short_formula_phrase("This is a much longer sentence than eight words maybe more") is None


class TestTitleAndHeadings:
    def test_title_block_detection(self):
        assert _looks_like_title_block("Privacy-Preserving Aggregation for Mobile Sensing")

    def test_not_title_block_with_url(self):
        assert not _looks_like_title_block("See https://example.com for details")

    def test_repair_section_heading(self):
        assert _repair_section_heading("I. Introduction", "I") == "I. 引言"
        assert _repair_section_heading("2 Related Work", "2") == "2 相关工作"
        assert _repair_section_heading("Something else entirely", "1") is None
