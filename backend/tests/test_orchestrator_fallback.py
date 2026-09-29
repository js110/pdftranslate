"""Tests for the per-page translation orchestrator with mocked providers.

These exercise the fallback ladder without any network/LLM access.
"""
from __future__ import annotations

import app.services.orchestrator as orchestrator
from app.services.jobtypes import PageProcessResult, _TextBlockJob
from app.services.orchestrator import _translate_text_jobs
from app.services.translator import ProviderRuntime, TranslationError


def make_job(index: int, text: str) -> _TextBlockJob:
    return _TextBlockJob(
        block_index=index,
        text=text,
        x0=100.0,
        y0=100.0,
        x1=400.0,
        y1=140.0,
        width=300.0,
        height=40.0,
        base_font_size=12.0,
        color=(0, 0, 0),
    )


PRIMARY = ProviderRuntime(id="primary", model="test-model", api_key="k")


def _run(jobs: list[_TextBlockJob]) -> tuple[dict[int, str], PageProcessResult]:
    result = PageProcessResult()
    translated = _translate_text_jobs(
        session_id=None,
        jobs=jobs,
        result=result,
        page_no=1,
        primary_provider=PRIMARY,
        backup_provider=None,
        style_profile="",
        glossary=None,
        max_retries=1,
    )
    return translated, result


class TestNarrativeFallbackPath:
    def test_provider_failure_uses_table_narrative_fallback(self, monkeypatch):
        def fail(*args, **kwargs):
            raise TranslationError("provider down")

        monkeypatch.setattr(orchestrator, "translate_batch_with_fallback", fail)
        monkeypatch.setattr(orchestrator, "translate_with_fallback", fail)

        translated, result = _run([make_job(1, "Table I shows several protocols under given settings.")])

        assert translated[1].startswith("表I")
        assert not result.untranslated_items

    def test_provider_failure_marks_untranslated_without_fallback(self, monkeypatch):
        def fail(*args, **kwargs):
            raise TranslationError("provider down")

        monkeypatch.setattr(orchestrator, "translate_batch_with_fallback", fail)
        monkeypatch.setattr(orchestrator, "translate_with_fallback", fail)

        text = "A regular English sentence that no deterministic fallback can handle at all."
        translated, result = _run([make_job(1, text)])

        assert 1 not in translated
        assert result.untranslated_items
        assert result.untranslated_items[0]["reason"].startswith("translation_failed:")
        assert any("text block translation failed" in w for w in result.warnings)


class TestSuccessPath:
    def test_batch_translation_assigned(self, monkeypatch):
        def fake_batch(texts, primary, backup, style_profile, glossary=None, max_retries=1):
            return ["这是第一个句子的翻译。", "这是第二个句子的翻译。"], primary.id, None

        monkeypatch.setattr(orchestrator, "translate_batch_with_fallback", fake_batch)

        jobs = [
            make_job(1, "This is the first source sentence to translate here."),
            make_job(2, "This is the second source sentence to translate here."),
        ]
        translated, result = _run(jobs)

        assert translated[1] == "这是第一个句子的翻译。"
        assert translated[2] == "这是第二个句子的翻译。"
        assert not result.warnings
        assert not result.untranslated_items

    def test_duplicate_blocks_share_translation(self, monkeypatch):
        def fake_batch(texts, primary, backup, style_profile, glossary=None, max_retries=1):
            assert len(texts) == 1, "identical texts must be deduplicated to one segment"
            return ["重复文本的翻译。"], primary.id, None

        monkeypatch.setattr(orchestrator, "translate_batch_with_fallback", fake_batch)

        jobs = [
            make_job(1, "Identical text appearing twice in the page."),
            make_job(2, "Identical text appearing twice in the page."),
        ]
        translated, _ = _run(jobs)

        assert translated[1] == translated[2] == "重复文本的翻译。"


class TestMemoryDedup:
    def test_identical_texts_translated_once(self, monkeypatch):
        calls: list[list[str]] = []

        def fake_batch(texts, primary, backup, style_profile, glossary=None, max_retries=1):
            calls.append(list(texts))
            return [f"译文 {i}" for i in range(1, len(texts) + 1)], primary.id, None

        monkeypatch.setattr(orchestrator, "translate_batch_with_fallback", fake_batch)

        jobs = [
            make_job(i, f"Unique sentence number {i} for the dedup check.")
            for i in range(1, 5)
        ]
        jobs += [make_job(10, "Unique sentence number 1 for the dedup check.")]  # duplicate of job 1
        translated, _ = _run(jobs)

        assert len(calls) == 1 and len(calls[0]) == 4
        assert translated[10] == translated[1]
