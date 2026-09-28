"""Tests for the pure text heuristics (table/algorithm/reference guards)."""
from __future__ import annotations

from app.services.heuristics import (
    _is_algorithm_block_text,
    _is_reference_entry_text,
    _is_running_header_footer_text,
    _is_table_caption_text,
    _is_table_like_text,
    _looks_like_paragraph_block,
    _log_block_skip,
)
from app.services.promptprep import _compact_text_for_prompt


class TestAlgorithmDetection:
    def test_pseudocode_with_io_is_algorithm(self):
        text = "Input: a set S of users\nOutput: the aggregated value\n1. for each x in S do\n2. return result"
        assert _is_algorithm_block_text(text)

    def test_numbered_contributions_are_not_algorithm(self):
        text = (
            "1) We propose a new privacy-preserving protocol for mobile sensing. "
            "2) We design an aggregation scheme with security proofs. "
            "3) We evaluate the scheme on real datasets and show accuracy."
        )
        assert not _is_algorithm_block_text(text)

    def test_theorem_prose_is_not_algorithm(self):
        text = (
            "Theorem 1: The proposed scheme achieves plaintext confidentiality "
            "against semi-honest adversaries under the standard model assumptions."
        )
        assert not _is_algorithm_block_text(text)

    def test_plain_narrative_is_not_algorithm(self):
        text = (
            "We now describe the overall architecture of the system in detail, "
            "including how each component interacts with the others at runtime."
        )
        assert not _is_algorithm_block_text(text)


class TestHeaderFooter:
    def test_running_header(self):
        assert _is_running_header_footer_text(
            text="IEEE TRANSACTIONS ON INFORMATION FORENSICS AND SECURITY, VOL. 14, NO. 8",
            y0=10, y1=40, page_height=800,
        )

    def test_page_number(self):
        assert _is_running_header_footer_text(text="1234", y0=10, y1=40, page_height=800)

    def test_body_text_is_not_header(self):
        assert not _is_running_header_footer_text(
            text="In this section we describe the proposed protocol in detail.",
            y0=300, y1=340, page_height=800,
        )


class TestReferenceEntries:
    def test_reference_entry(self):
        text = "[1] J. Smith, J. Doe, Some Paper Title, IEEE Transactions on Security, 2019."
        assert _is_reference_entry_text(text)

    def test_contribution_list_is_not_reference(self):
        text = "1) We propose the protocol. 2) We design the scheme. 3) We evaluate it."
        assert not _is_reference_entry_text(text)

    def test_plain_sentence_is_not_reference(self):
        assert not _is_reference_entry_text("We propose a new protocol for mobile sensing systems.")


class TestTableText:
    def test_short_caption(self):
        assert _is_table_caption_text("TABLE I\nNotation Summary")

    def test_narrative_table_mention_is_not_caption(self):
        assert not _is_table_caption_text("Table V lists all the important variables in the whole process.")

    def test_numeric_rows_look_like_table(self):
        text = "method_a 0.92 0.88 0.91\nmethod_b 0.90 0.85 0.89\nmethod_c 0.89 0.84 0.88"
        assert _is_table_like_text(text)

    def test_narrative_prose_is_not_table(self):
        text = (
            "The proposed protocol achieves strong accuracy while keeping the communication "
            "cost low, as we demonstrate in the evaluation section of this paper."
        )
        assert not _is_table_like_text(text)


class TestParagraphShape:
    def test_single_sentence_is_paragraph(self):
        assert _looks_like_paragraph_block("This is a complete sentence with enough words to count as prose.")

    def test_short_fragment_is_not_paragraph(self):
        assert not _looks_like_paragraph_block("some fragment")


class TestCompactForPrompt:
    def test_normal_text_keeps_words(self):
        compact = _compact_text_for_prompt("This is a normal paragraph\nspanning multiple lines.")
        assert "This is a normal paragraph" in compact
        assert "\n" not in compact or "spanning" in compact

    def test_structured_lines_preserved(self):
        text = "state1 = init\nnext = state1 + 1\nres = output(state1)"
        compact = _compact_text_for_prompt(text)
        # structured notation keeps line layout
        assert "\n" in compact


def test_log_block_skip_smoke(capsys=None):
    # must not raise for short text (it early-returns)
    _log_block_skip(page_no=1, block_index=0, reason="test", text="tiny")
