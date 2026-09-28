"""Tests for translator response parsing, math protection, and prompts."""
from __future__ import annotations

import pytest

from app.services.translator import (
    ProviderRuntime,
    _is_untranslated_echo,
    _parse_batch_output,
    _parse_marked_output,
    _protect_math_content,
    _restore_math_content,
    build_batch_prompt,
    build_prompt,
    should_skip_translation,
)
from app.services.translator import TranslationError


class TestParseMarkedOutput:
    def test_valid(self):
        assert _parse_marked_output("[1] 你好\n[2] 世界", 2) == ["你好", "世界"]

    def test_out_of_range_index_fails(self):
        assert _parse_marked_output("[1] 你好\n[3] 世界", 2) is None

    def test_missing_index_fails(self):
        assert _parse_marked_output("[1] 你好", 2) is None

    def test_duplicate_index_fails(self):
        assert _parse_marked_output("[1] 你好\n[1] 世界", 2) is None

    def test_multiline_value(self):
        raw = "[1] 第一行\n第二行\n[2] 世界"
        assert _parse_marked_output(raw, 2) == ["第一行\n第二行", "世界"]


class TestParseBatchOutput:
    def test_json_list(self):
        assert _parse_batch_output('["你", "世"]', 2, ["a", "b"]) == ["你", "世"]

    def test_json_dict_translations(self):
        assert _parse_batch_output('{"translations": ["你", "世"]}', 2, ["a", "b"]) == ["你", "世"]

    def test_marked_format(self):
        assert _parse_batch_output("[1] 你\n[2] 世", 2, ["a", "b"]) == ["你", "世"]

    def test_plain_lines(self):
        assert _parse_batch_output("你\n世", 2, ["a", "b"]) == ["你", "世"]

    def test_single_segment(self):
        assert _parse_batch_output("只是一个翻译", 1, ["a"]) == ["只是一个翻译"]

    def test_size_mismatch_raises(self):
        with pytest.raises(TranslationError):
            _parse_batch_output('["你"]', 2, ["a", "b"])

    def test_empty_line_falls_back_to_source(self):
        assert _parse_batch_output('["", "世"]', 2, ["a", "b"]) == ["a", "世"]


class TestMathProtection:
    def test_roundtrip(self):
        text = "The value E = m c^2 and \\frac{a}{b} are preserved."
        protected, replacements = _protect_math_content(text)
        assert "MATHSEG0TOKEN" in protected
        restored = _restore_math_content(protected, replacements)
        assert restored == text

    def test_no_math(self):
        text = "Plain sentence without math."
        protected, replacements = _protect_math_content(text)
        assert protected == text
        assert replacements == []


class TestPrompts:
    def test_prompt_contains_text_and_rules(self):
        messages = build_prompt("Hello world.", style_profile="")
        assert messages[0]["role"] == "system"
        assert "MATHSEG" in messages[0]["content"]
        assert "Hello world." in messages[1]["content"]

    def test_strict_mode_instruction(self):
        messages = build_prompt("Hello world.", style_profile="", strict_mode=True)
        assert "Strict mode" in messages[0]["content"]

    def test_batch_prompt_segment_count(self):
        messages = build_batch_prompt(["a", "b", "c"], style_profile="")
        assert "3 segments" in messages[0]["content"]
        assert "[1] a" in messages[1]["content"]
        assert "[3] c" in messages[1]["content"]

    def test_glossary_included(self):
        messages = build_prompt("Hello.", style_profile="", glossary=["AES", "RSA", "Paillier"])
        assert "AES" in messages[1]["content"]


class TestUntranslatedEcho:
    def test_identical_english_is_echo(self):
        src = "This is an English sentence that should have been translated."
        assert _is_untranslated_echo(src, src)

    def test_chinese_translation_is_not_echo(self):
        src = "This is an English sentence that should have been translated."
        assert not _is_untranslated_echo(src, "这是应当被翻译的英文句子。")

    def test_short_source_not_flagged(self):
        assert not _is_untranslated_echo("hi", "hi")


class TestShouldSkip:
    def test_pure_url_skipped(self):
        assert should_skip_translation("https://example.com/path")

    def test_citation_skipped(self):
        assert should_skip_translation("[12]")

    def test_english_prose_kept(self):
        assert not should_skip_translation("We propose a novel protocol for secure aggregation.")

    def test_empty_skipped(self):
        assert should_skip_translation("")


def test_provider_runtime_defaults():
    p = ProviderRuntime(id="x", model="m", api_key="k")
    assert p.timeout_sec == 60
    assert p.base_url is None
