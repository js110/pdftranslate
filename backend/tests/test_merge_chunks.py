"""Tests for block merging and translation chunking."""
from __future__ import annotations

from app.services.jobtypes import _TextBlockJob
from app.services.orchestrator import _chunk_jobs_for_translation
from app.services.pdf_pipeline import _merge_fragmented_text_jobs


def make_job(index: int, x0: float, y0: float, x1: float, y1: float, text: str) -> _TextBlockJob:
    return _TextBlockJob(
        block_index=index,
        text=text,
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        width=x1 - x0,
        height=y1 - y0,
        base_font_size=12.0,
        color=(0, 0, 0),
    )


class TestMergeFragmentedJobs:
    def test_same_column_fragments_merge(self):
        jobs = [
            make_job(1, 100, 100, 400, 130, "first part of a sentence that"),
            make_job(2, 100, 132, 400, 160, "continues here with more words."),
        ]
        merged = _merge_fragmented_text_jobs(jobs, page_width=800)
        assert len(merged) == 1
        assert "continues here" in merged[0].text

    def test_cross_column_never_merges(self):
        jobs = [
            make_job(1, 100, 100, 400, 130, "left column text ends here"),
            make_job(2, 450, 100, 750, 130, "right column text starts."),
        ]
        merged = _merge_fragmented_text_jobs(jobs, page_width=800)
        assert len(merged) == 2

    def test_section_heading_not_merged(self):
        heading = make_job(1, 100, 100, 400, 122, "I. Introduction")
        body = make_job(2, 100, 126, 400, 154, "We present a novel scheme for the task.")
        merged = _merge_fragmented_text_jobs([heading, body], page_width=800)
        assert len(merged) == 2

    def test_merge_group_capped_at_three(self):
        jobs = [
            make_job(1, 100, 100, 400, 124, "part one of the line"),
            make_job(2, 100, 126, 400, 148, "part two of the line"),
            make_job(3, 100, 150, 400, 172, "part three of the line"),
            make_job(4, 100, 174, 400, 196, "part four of the line"),
        ]
        merged = _merge_fragmented_text_jobs(jobs, page_width=800)
        assert all(len([j for j in merged if j.block_index == src.block_index]) <= 1 for src in jobs)
        assert len(merged) == 2  # 3 + 1

    def test_distant_blocks_not_merged(self):
        jobs = [
            make_job(1, 100, 100, 400, 130, "first paragraph ends with a period."),
            make_job(2, 100, 400, 400, 430, "second paragraph far below starts."),
        ]
        merged = _merge_fragmented_text_jobs(jobs, page_width=800)
        assert len(merged) == 2


class TestChunking:
    def _jobs(self, count: int) -> list[_TextBlockJob]:
        return [
            make_job(i, 100, 100 + i * 40, 400, 130 + i * 40, f"segment number {i} with some words")
            for i in range(1, count + 1)
        ]

    def test_respects_max_blocks(self):
        chunks = _chunk_jobs_for_translation(self._jobs(7), max_blocks=3, max_chars=8000)
        assert [len(c) for c in chunks] == [3, 3, 1]

    def test_respects_max_chars(self):
        jobs = [
            make_job(i, 100, 100 + i * 40, 400, 130 + i * 40, "word " * 120)
            for i in range(1, 5)
        ]
        chunks = _chunk_jobs_for_translation(jobs, max_blocks=10, max_chars=800)
        assert all(sum(len(j.text) for j in c) <= 800 + 480 for c in chunks)  # one-job overflow allowed
        assert len(chunks) >= 4

    def test_long_segment_isolated(self):
        long_job = make_job(1, 100, 100, 400, 400, "x " * 900)  # >= 30% of max_chars
        small = make_job(2, 100, 500, 400, 530, "tiny segment")
        chunks = _chunk_jobs_for_translation([long_job, small], max_blocks=10, max_chars=800)
        assert len(chunks) == 2
        assert len(chunks[0]) == 1 and len(chunks[1]) == 1
