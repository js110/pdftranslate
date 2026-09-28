"""Heuristic text classifiers for academic PDF blocks.

Pure functions over extracted text: table / algorithm / reference detection,
running headers & footers, watermarks, prose shape. No I/O, no model calls.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from app.services.textutil import CJK_CHAR_RE, ENGLISH_WORD_RE, _extract_block_text_style

logger = logging.getLogger(__name__)


ABBR_RE = re.compile(r"\b[A-Z]{2,8}s?\b")

TABLE_CAPTION_RE = re.compile(r"^\s*TABLE\s+([IVXLCM]+|\d+)(?:\s*[:.\-]?\s*.*)?$", re.IGNORECASE)

TABLE_NOTE_RE = re.compile(r"^\s*Note\s*[:\-]?\s*\S", re.IGNORECASE)

TABLE_HEADER_HINT_RE = re.compile(
    r"\bnotation\b.*\bdefinition\b|\bdefinition\b.*\bnotation\b|"
    r"\b(method|metric|measured|parameter|prototype|component)\b[^.;:\n]{0,48}\b"
    r"(accuracy|precision|recall|f1|auc|latency|throughput|mean|max|min|std|variance|result)\b",
    re.IGNORECASE,
)

TABLE_ROW_NUMERIC_RE = re.compile(r"^(?:\S+\s+)?(?:\d+(?:\.\d+)?\s+){2,}\d+(?:\.\d+)?$")

TABLE_SYMBOLIC_ROW_RE = re.compile(r"^(?:[A-Za-z][\w\-/*']*\s+){1,6}(?:[×x✓\-]|[0-9]+(?:\.[0-9]+)?)(?:\s+[×x✓\-]|[0-9]+(?:\.[0-9]+)?)+$")

TABLE_SYMBOL_TOKEN_RE = re.compile(r"(?:^|[\s|,;:/()\[\]{}])(?:x|X|×|✓)(?=$|[\s|,;:/()\[\]{}])")

TABLE_NARRATIVE_VERB_RE = re.compile(
    r"\b("
    r"list|lists|show|shows|present|presents|illustrate|illustrates|compare|compares|"
    r"summarize|summarizes|report|reports|give|gives|describe|describes|"
    r"outline|outlines|detail|details|discuss|discusses|analyze|analyzes|"
    r"evaluate|evaluates|investigate|investigates|examine|examines"
    r")\b",
    re.IGNORECASE,
)

REFERENCE_HEADING_RE = re.compile(
    r"^\s*(?:(?:section\s+)?(?:\d+(?:\.\d+)*|[ivxlcdm]+)\s*[\).:\-]?\s*)?"
    r"(references?|bibliography|works\s+cited|literature|参考文献)\s*[:.]?\s*$",
    re.IGNORECASE,
)

REFERENCE_ENTRY_START_RE = re.compile(r"^\s*(\[\d{1,3}\]|\d{1,2}[.)])\s+")

REFERENCE_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")

REFERENCE_VENUE_HINT_RE = re.compile(
    r"\b(ieee|acm|springer|elsevier|wiley|journal|transactions|proc\.?|conference|symposium|doi|arxiv|vol\.?\s*\d+|no\.?\s*\d+|pp\.?\s*\d+)\b",
    re.IGNORECASE,
)

REFERENCE_AUTHOR_HINT_RE = re.compile(
    r"\b(?:[A-Z][a-zA-Z'`-]{1,24},\s*(?:[A-Z]\.\s*){1,3}|[A-Z][a-zA-Z'`-]{1,24}\s+et al\.)",
)

IEEE_LICENSE_WATERMARK_RE = re.compile(
    r"\bauthorized licensed use limited to\b|"
    r"\bdownloaded on\b.*\bieee xplore\b|"
    r"\bieee xplore\b.*\brestrictions apply\b|"
    r"\bpersonal use is permitted\b|"
    r"\brepublication/?redistribution\b",
    re.IGNORECASE,
)

ALGORITHM_HEADING_RE = re.compile(r"^\s*(algorithm|alg\.?|算法)\s*[\divxlcdm一二三四五六七八九十]*", re.IGNORECASE)

ALGORITHM_IO_RE = re.compile(
    r"^\s*(input|output|require|ensure|return|输入|输出|返回|初始化|参数)\b",
    re.IGNORECASE,
)

ALGORITHM_STEP_RE = re.compile(r"^\s*(\d{1,3}|[ivxlcdm]{1,6})[\).]?\s+\S", re.IGNORECASE)

ALGORITHM_CTRL_RE = re.compile(r"\b(if|then|else|while|repeat|until|return)\b|[:=]{1,2}|<-|->", re.IGNORECASE)

ALGORITHM_FOR_LOOP_RE = re.compile(r"\bfor\s+each\b|\bfor\b[^.;:\n]{0,48}\b(in|to|from)\b", re.IGNORECASE)

ALGORITHM_END_RE = re.compile(r"^\s*end\s+(for|while|if|repeat)\b|^\s*结束\b", re.IGNORECASE)

ACADEMIC_LABEL_RE = re.compile(
    r"^\s*(?:the\s+)?(definition|theorem|lemma|corollary|proposition|proof|remark|example)\b",
    re.IGNORECASE,
)

BULLET_SENTENCE_RE = re.compile(r"^\s*(?:[-*•]|(?:\d{1,3}|[ivxlcdm]{1,6})[\).])\s+[A-Za-z]", re.IGNORECASE)

NUMBERED_NARRATIVE_LINE_RE = re.compile(
    r"^\s*(?:\(?\d{1,3}\)?|[ivxlcdm]{1,6})[\).:]\s+[A-Za-z]",
    re.IGNORECASE,
)

NARRATIVE_LIST_LEADIN_RE = re.compile(
    r"\b(contribution|contributions|our main|we propose|we design|we study|we perform)\b",
    re.IGNORECASE,
)

TABLE_NARRATIVE_CONTINUATION_RE = re.compile(
    r"\b("
    r"that|which|where|if|then|means?|mean|known|stored|locally|initially|"
    r"example\d*|suppose|assuming|assume|let|denote|denotes|recording|variable"
    r")\b",
    re.IGNORECASE,
)

FORMULA_NARRATIVE_VERB_RE = re.compile(
    r"\b("
    r"decrypt|decrypts|encrypted|encrypt|encrypts|compute|computes|computed|"
    r"find|finds|get|gets|return|returns|send|sends|receive|receives|"
    r"obtain|obtains|choose|chooses|set|sets|check|checks|compare|compares"
    r")\b",
    re.IGNORECASE,
)

STRUCTURED_LINE_MARKER_RE = re.compile(r"^\s*(?:['`‘’•·\-*]\s*)?(?:r\b|[A-Za-z]\[[^\]]+\]|count\(|state\d*\b|next\b|res\b)")

STRUCTURED_NOTATION_TOKEN_RE = re.compile(
    r"\bcount\s*\(|\bstate\d*\b|\bnext\b|\bres\b|\bminpos\b|"
    r"[A-Za-z]\[[^\]]+\]|\|\||:=|⊕|←",
    re.IGNORECASE,
)

RUNNING_HEADER_HINT_RE = re.compile(
    r"\b(ieee|transactions|journal|vol\.?|no\.?|january|february|march|april|may|june|july|august|"
    r"september|october|november|december)\b",
    re.IGNORECASE,
)

RUNNING_FOOTER_HINT_RE = re.compile(
    r"\b(authorized licensed use limited to|downloaded on|ieee xplore|restrictions apply|copyright|all rights reserved)\b",
    re.IGNORECASE,
)

def _looks_like_overlap_continuation(*, prev_text: str, curr_text: str) -> bool:
    prev_clean = prev_text.strip()
    curr_clean = curr_text.strip()
    if not prev_clean or not curr_clean:
        return False

    # Do not merge distinct list/step items such as "a) ...", "1) ...", "i) ...".
    if re.match(r"^\s*(?:[a-z]|[ivxlcdm]+|\d{1,3})[\).:]\s+", curr_clean, flags=re.IGNORECASE):
        return False

    first = curr_clean[:1]
    if first and (first.islower() or first.isdigit() or first in "=+-*/^_([{⊕←"):
        return True

    if re.search(r"(?:[-=+*/^_⊕←,;:.\(\[])\s*$", prev_clean):
        return True
    if re.search(r"\b(?:for|where|then|that|with|s\.t)\s*$", prev_clean, flags=re.IGNORECASE):
        return True
    return False



def _looks_like_notation_style_line(text: str) -> bool:
    compact = re.sub(r"\s+", " ", text).strip()
    if not compact:
        return False
    if STRUCTURED_LINE_MARKER_RE.match(compact):
        return True
    if len(compact) <= 140 and STRUCTURED_NOTATION_TOKEN_RE.search(compact):
        return True
    return False



def _is_running_header_footer_text(*, text: str, y0: float, y1: float, page_height: float) -> bool:
    if page_height <= 0:
        return False
    compact = re.sub(r"\s+", " ", text).strip()
    if not compact:
        return False

    lower = compact.lower()
    near_top = y1 <= page_height * 0.095
    near_bottom = y0 >= page_height * 0.90
    en_words = len(ENGLISH_WORD_RE.findall(compact))

    if near_top:
        if RUNNING_HEADER_HINT_RE.search(compact):
            return True
        if re.fullmatch(r"\d{1,5}", compact):
            return True
        if len(compact) <= 42 and en_words <= 6 and compact.upper() == compact and any(ch.isalpha() for ch in compact):
            return True
        if len(compact) <= 180 and re.search(r"\bet al\.:", lower) and re.search(r"\b\d{3,5}\b", compact):
            return True
        upper_letters = sum(1 for ch in compact if ch.isupper())
        alpha_letters = sum(1 for ch in compact if ch.isalpha())
        if alpha_letters >= 10 and re.search(r"\b\d{3,5}\b", compact):
            if upper_letters / max(alpha_letters, 1) >= 0.42 and ":" in compact:
                return True

    if near_bottom:
        if _is_ieee_license_watermark_text(compact):
            return True
        if RUNNING_FOOTER_HINT_RE.search(compact):
            return True
        if re.fullmatch(r"\d{1,5}", compact):
            return True
        if len(compact) <= 56 and en_words <= 7 and ("downloaded on" in lower or "ieee xplore" in lower):
            return True

    return False



def _is_ieee_license_watermark_text(text: str) -> bool:
    compact = re.sub(r"\s+", " ", text).strip()
    if not compact:
        return False
    lower = compact.lower()
    # Strong single-signal match used in IEEE watermark/footer text.
    if "authorized licensed use limited to" in lower:
        return True
    # Multi-signal fallback to avoid false positives on regular prose.
    signals = 0
    if "ieee xplore" in lower:
        signals += 1
    if "downloaded on" in lower:
        signals += 1
    if "restrictions apply" in lower:
        signals += 1
    if "personal use is permitted" in lower:
        signals += 1
    if "republication/redistribution" in lower or "republication redistribution" in lower:
        signals += 1
    if IEEE_LICENSE_WATERMARK_RE.search(compact):
        signals += 1
    return signals >= 3 and len(compact) <= 420



def _is_algorithm_block_text(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False

    heading_hit = any(ALGORITHM_HEADING_RE.match(line) for line in lines[:2])

    io_count = 0
    for line in lines:
        if not ALGORITHM_IO_RE.match(line):
            continue
        compact = re.sub(r"\s+", " ", line).strip()
        # Narrative lines may start with words like "input/output" but are not pseudocode I/O headers.
        if len(compact) <= 88 or re.match(
            r"^\s*(input|output|require|ensure|return|输入|输出|返回|初始化|参数)\s*[:：]",
            line,
            flags=re.IGNORECASE,
        ):
            io_count += 1
    step_count = sum(1 for line in lines if ALGORITHM_STEP_RE.match(line))
    ctrl_count = sum(1 for line in lines if ALGORITHM_CTRL_RE.search(line))
    for_loop_count = sum(1 for line in lines if ALGORITHM_FOR_LOOP_RE.search(line))
    ctrl_count += for_loop_count
    end_count = sum(1 for line in lines if ALGORITHM_END_RE.search(line))
    flat = re.sub(r"\s+", " ", " ".join(lines)).strip()
    word_count = len(ENGLISH_WORD_RE.findall(flat))
    sentence_punct = sum(flat.count(mark) for mark in ".?!;:")
    has_academic_label = any(ACADEMIC_LABEL_RE.match(line) for line in lines[:2])
    numbered_prose_count = sum(
        1
        for line in lines
        if re.match(r"^\s*\d{1,3}[\).]?\s+(we|our)\b", line, flags=re.IGNORECASE)
    )
    pseudo_line_count = sum(
        1
        for line in lines
        if len(line) <= 120
        and (
            ALGORITHM_IO_RE.match(line)
            or ALGORITHM_CTRL_RE.search(line)
            or ALGORITHM_FOR_LOOP_RE.search(line)
        )
    )

    # Numbered contribution prose (e.g., "1) We propose ...") should remain translatable.
    if step_count >= 3 and io_count == 0 and end_count == 0 and ctrl_count <= 1 and numbered_prose_count >= 2:
        return False
    # Definitions/lemmas/proofs are often prose with occasional control words.
    if has_academic_label and io_count == 0 and end_count == 0 and word_count >= 24:
        return False
    if has_academic_label and io_count == 0 and end_count == 0 and step_count <= 2:
        return False
    # Long narrative prose should not be treated as pseudocode.
    if word_count >= 34 and sentence_punct >= 2 and io_count == 0 and end_count == 0 and step_count <= 2:
        return False
    # Very long prose blocks are not algorithms, even if they contain a few control words.
    if word_count >= 120 and io_count <= 1 and end_count == 0:
        return False
    # Long narrative blocks around formulas are often split into short lines and may
    # contain tokens that look like pseudocode controls. Keep them translatable.
    if io_count == 0 and end_count == 0 and word_count >= 64 and sentence_punct >= 6 and _looks_like_narrative_prose(text):
        return False

    if heading_hit:
        if io_count >= 1 or step_count >= 1 or ctrl_count >= 1 or pseudo_line_count >= 3:
            return True
        avg_len = sum(len(line) for line in lines) / max(len(lines), 1)
        if len(lines) >= 4 and avg_len <= 70 and sentence_punct <= 1 and not _looks_like_narrative_prose(text):
            return True

    if io_count >= 1 and (step_count >= 1 or ctrl_count >= 1):
        return True
    if io_count >= 2 and (step_count >= 2 or ctrl_count >= 2):
        return True
    if step_count >= 3 and (ctrl_count >= 2 or end_count >= 1):
        return True
    if pseudo_line_count >= 4 and (step_count >= 2 or io_count >= 1):
        return True

    avg_len = sum(len(line) for line in lines) / max(len(lines), 1)
    if step_count >= 5 and ctrl_count >= 2 and avg_len <= 68:
        return True

    # Narrative prose blocks with citations/lists should stay translatable.
    if (
        _looks_like_narrative_prose(text)
        and io_count == 0
        and end_count == 0
        and pseudo_line_count <= max(2, len(lines) // 2)
    ):
        return False

    return False



def _is_reference_entry_text(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False

    first = re.sub(r"\s+", " ", lines[0])
    flat = re.sub(r"\s+", " ", " ".join(lines))
    lower = flat.lower()
    has_year = bool(REFERENCE_YEAR_RE.search(flat))
    has_venue_hint = bool(REFERENCE_VENUE_HINT_RE.search(flat))
    has_author_hint = bool(REFERENCE_AUTHOR_HINT_RE.search(flat))
    start_count = sum(1 for line in lines if REFERENCE_ENTRY_START_RE.match(line))
    word_count = len(ENGLISH_WORD_RE.findall(flat))
    avg_len = sum(len(line) for line in lines) / max(len(lines), 1)
    proposal_like = bool(re.search(r"\b(we|our|propose|design|present|develop|introduce|suggest)\b", lower))

    # Skip detection if this looks like a contributions list (not references).
    # Contributions lists have keywords like "contributions" or "main contributions".
    if "contribution" in lower and start_count >= 1:
        # Check if it's actually a contributions list by looking for proposal verbs
        has_proposal_verbs = any(
            verb in lower for verb in ["propose", "design", "present", "develop", "introduce", "suggest", "we "]
        )
        if has_proposal_verbs:
            return False
    if proposal_like and not has_year and start_count <= 2:
        return False

    # Multiple reference entries in one block:
    # keep this conservative to avoid swallowing long narrative survey paragraphs
    # that contain citation-leading lines and numeric values.
    if start_count >= 2 and word_count <= 120 and len(lines) <= 12:
        if has_venue_hint or (has_year and has_author_hint):
            return True
    if REFERENCE_ENTRY_START_RE.match(first) and (has_year or has_venue_hint or has_author_hint):
        return True
    if has_year and (" doi" in lower or " arxiv" in lower) and (start_count >= 1 or REFERENCE_ENTRY_START_RE.match(first)):
        return True

    if has_author_hint and has_year and (has_venue_hint or start_count >= 1) and len(lines) <= 4 and avg_len <= 120:
        return True
    if start_count >= 1 and has_year and has_venue_hint and word_count <= 48:
        return True
    return False



def _is_reference_page(blocks: list[dict[str, Any]]) -> bool:
    total_text = 0
    entry_blocks = 0
    heading_seen = False
    first_entry_y: float | None = None
    page_bottom = 0.0

    for block in blocks:
        if block.get("type", -1) != 0:
            continue
        bbox = block.get("bbox", [0, 0, 0, 0])
        _, y0, _, y1 = [float(v) for v in bbox]
        page_bottom = max(page_bottom, y1)
        text, _, _ = _extract_block_text_style(block)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            continue
        total_text += 1
        if any(REFERENCE_HEADING_RE.match(re.sub(r"\s+", " ", line).strip()) for line in lines[:2]):
            heading_seen = True
            continue
        if _is_reference_entry_text(text):
            entry_blocks += 1
            if first_entry_y is None:
                first_entry_y = y0

    if total_text == 0:
        return False
    
    # 必须有参考文献标题
    if not heading_seen:
        return False
    
    # 必须有足够的参考文献条目
    if entry_blocks < 2:
        return False
    
    # 增加位置判断：参考文献条目通常不会从页面顶部就开始。
    # 对于真正的参考文献页，首条目一般位于页面中下部。
    if first_entry_y is not None and page_bottom > 0:
        first_entry_ratio = first_entry_y / page_bottom
        # If entries start too high on page, likely a false positive (e.g., contribution list).
        if first_entry_ratio < 0.22:
            return False
    
    return True



def _is_table_caption_text(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    first_line = re.sub(r"\s+", " ", lines[0]).strip()
    if not TABLE_CAPTION_RE.match(first_line):
        return False

    flat = re.sub(r"\s+", " ", " ".join(lines)).strip()
    lower = flat.lower()
    words = ENGLISH_WORD_RE.findall(flat)
    sentence_punct = sum(flat.count(mark) for mark in (".", "?", "!", ";", ":"))

    # Narrative mentions like "Table V outlines/lists ..." should stay translatable.
    if len(words) >= 6 and TABLE_NARRATIVE_VERB_RE.search(lower):
        return False
    # Long sentence-like blocks are prose, not standalone captions.
    if len(words) >= 36:
        return False
    if len(lines) >= 3 and len(words) >= 18:
        return False
    if sentence_punct >= 2 and len(words) >= 14:
        return False
    if len(words) >= 12 and sentence_punct >= 1 and (
        "," in flat or TABLE_NARRATIVE_CONTINUATION_RE.search(lower)
    ):
        return False
    if len(words) >= 10 and (" that " in lower or " since " in lower):
        return False
    return True



def _is_table_header_text(text: str) -> bool:
    flat = re.sub(r"\s+", " ", text).strip().lower()
    if not flat:
        return False
    word_count = len(ENGLISH_WORD_RE.findall(flat))
    sentence_punct = sum(flat.count(mark) for mark in (".", "?", "!", ";", ":"))
    if word_count >= 10 and sentence_punct >= 1 and TABLE_NARRATIVE_CONTINUATION_RE.search(flat):
        return False
    if word_count >= 8 and "example" in flat and sentence_punct >= 1:
        return False
    if _looks_like_numbered_narrative_list(text):
        return False
    if _looks_like_narrative_prose(text):
        return False
    if TABLE_HEADER_HINT_RE.search(flat):
        return True

    tokens = re.findall(r"[a-z][a-z0-9\-]{1,}", flat)
    if len(tokens) < 3:
        return False

    token_set = set(tokens)
    metric_terms = {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auc",
        "latency",
        "throughput",
        "mean",
        "max",
        "min",
        "std",
        "variance",
        "error",
        "mse",
        "mae",
    }
    method_terms = {
        "method",
        "baseline",
        "model",
        "algorithm",
        "metric",
        "parameter",
        "setting",
        "dataset",
        "component",
    }
    metric_hits = sum(1 for term in metric_terms if term in token_set)
    method_hits = sum(1 for term in method_terms if term in token_set)
    has_struct_delim = bool(re.search(r"[|/]| {2,}|\t", text))
    has_digit = bool(re.search(r"\d", text))
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    if metric_hits >= 2 and (method_hits >= 1 or has_struct_delim or has_digit or len(lines) >= 2):
        return True
    if metric_hits >= 3:
        return True
    return False



def _looks_like_narrative_prose(text: str) -> bool:
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False

    if _looks_like_numbered_narrative_list(text):
        return True

    flat = " ".join(lines)
    word_count = len(ENGLISH_WORD_RE.findall(flat))
    if any(ACADEMIC_LABEL_RE.match(line) for line in lines[:2]) and word_count >= 5:
        return True
    if word_count < 10:
        return False

    sentence_punct = sum(flat.count(mark) for mark in ".?!;:")
    alpha_ratio = sum(1 for ch in flat if ch.isalpha()) / max(len(flat), 1)

    bullet_sentence_count = sum(
        1
        for line in lines
        if BULLET_SENTENCE_RE.match(line) and len(ENGLISH_WORD_RE.findall(line)) >= 6
    )
    if bullet_sentence_count >= 1 and word_count >= 16:
        return True

    # Short narrative definition/explanation fragments (common around formulas)
    # should stay translatable and not be treated as table rows.
    if len(lines) <= 3 and word_count >= 10 and sentence_punct >= 2 and alpha_ratio >= 0.35:
        return True

    if word_count >= 26 and sentence_punct >= 2 and alpha_ratio >= 0.45:
        return True

    avg_len = sum(len(line) for line in lines) / max(len(lines), 1)
    if len(lines) <= 3 and word_count >= 14 and sentence_punct >= 1 and avg_len >= 42 and alpha_ratio >= 0.36:
        return True
    if len(lines) >= 3 and word_count >= 24 and avg_len >= 40 and sentence_punct >= 1:
        return True

    return False



def _looks_like_numbered_narrative_list(text: str) -> bool:
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 3:
        return False

    numbered_lines = [line for line in lines if NUMBERED_NARRATIVE_LINE_RE.match(line)]
    if len(numbered_lines) < 2:
        return False

    flat = " ".join(lines)
    word_count = len(ENGLISH_WORD_RE.findall(flat))
    if word_count < 16:
        return False

    # Check if this looks like a reference list (not a contribution list).
    # Reference entries have [N] format and contain publication metadata.
    bracket_numbered = sum(1 for line in lines if re.match(r"^\s*\[\d{1,3}\]\s+", line))
    has_year = bool(REFERENCE_YEAR_RE.search(flat))
    has_venue = bool(REFERENCE_VENUE_HINT_RE.search(flat))
    # If most lines use [N] format and there's publication metadata, it's references.
    if bracket_numbered >= 2 and (has_year or has_venue):
        return False

    if NARRATIVE_LIST_LEADIN_RE.search(flat):
        return True

    numbered_word_count = sum(len(ENGLISH_WORD_RE.findall(line)) for line in numbered_lines)
    avg_numbered_words = numbered_word_count / max(len(numbered_lines), 1)
    return len(numbered_lines) >= 3 and avg_numbered_words >= 4



def _is_table_like_text(text: str) -> bool:
    if not text.strip():
        return False
    flat_all = re.sub(r"\s+", " ", text).strip()
    lower_all = flat_all.lower()
    word_count = len(ENGLISH_WORD_RE.findall(flat_all))
    sentence_punct = sum(flat_all.count(mark) for mark in (".", "?", "!", ";", ":"))
    if word_count >= 10 and sentence_punct >= 1 and TABLE_NARRATIVE_CONTINUATION_RE.search(flat_all):
        return False
    if word_count >= 8 and "example" in lower_all and sentence_punct >= 1:
        return False
    if _is_table_caption_text(text):
        return True
    if _looks_like_numbered_narrative_list(text):
        return False
    # Do not suppress regular prose blocks that happen to contain numbers/citations.
    if _looks_like_narrative_prose(text):
        return False
    if _is_table_header_text(text):
        return True
    if _is_table_body_text(text):
        return True

    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False

    numeric_row_hits = 0
    for line in lines:
        if TABLE_ROW_NUMERIC_RE.match(line):
            numeric_row_hits += 1
            continue
        if TABLE_SYMBOLIC_ROW_RE.match(line):
            numeric_row_hits += 1
            continue
        number_count = len(re.findall(r"\b\d+(?:\.\d+)?\b", line))
        if number_count >= 3 and len(line) <= 96:
            numeric_row_hits += 1

    short_ratio = sum(1 for line in lines if len(line) <= 96) / len(lines)
    return numeric_row_hits >= 2 and short_ratio >= 0.6



def _is_table_body_text(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    flat = re.sub(r"\s+", " ", " ".join(lines)).strip()
    lower_flat = flat.lower()
    word_count = len(ENGLISH_WORD_RE.findall(flat))
    sentence_punct = sum(flat.count(mark) for mark in (".", "?", "!", ";", ":"))

    # Formula-adjacent prose fragments (e.g., "that each round ...", "Example 1 ...")
    # are often split into short lines and can be mistaken as table rows.
    # Keep these translatable by suppressing table-body classification.
    if word_count >= 10 and sentence_punct >= 1 and TABLE_NARRATIVE_CONTINUATION_RE.search(flat):
        return False
    if word_count >= 8 and "example" in lower_flat and sentence_punct >= 1:
        return False

    if _looks_like_numbered_narrative_list(text):
        return False
    if _looks_like_narrative_prose(text):
        return False
    if any(TABLE_ROW_NUMERIC_RE.match(re.sub(r"\s+", " ", line).strip()) for line in lines):
        return True
    if any(TABLE_SYMBOLIC_ROW_RE.match(re.sub(r"\s+", " ", line).strip()) for line in lines):
        return True
    if len(lines) < 4:
        sentence_punct = sum(line.count(".") + line.count("?") + line.count("!") for line in lines)
        if sentence_punct >= 1 and word_count >= 8:
            return False

        numeric_rich = sum(1 for line in lines if len(re.findall(r"\b\d+(?:\.\d+)?\b", line)) >= 3)
        symbolic_rich = sum(1 for line in lines if TABLE_SYMBOL_TOKEN_RE.search(line))
        return (numeric_rich >= 2 and len(lines) >= 2) or (
            symbolic_rich >= 2 and len(lines) >= 3 and numeric_rich >= 1
        )

    short_ratio = sum(1 for line in lines if len(line) <= 78) / len(lines)
    sentence_punct = sum(line.count(".") + line.count("?") + line.count("!") for line in lines)
    if sentence_punct > max(2, len(lines) // 3):
        return False

    lead_like = 0
    row_like = 0
    for line in lines:
        words = line.split()
        if len(words) < 2 or len(words) > 18:
            continue
        lead = words[0].strip(",;:.()[]{}")
        if not lead:
            continue

        has_symbol = bool(re.search(r"[\d_'\-^/]", lead))
        short_symbolic = len(lead) <= 2 or lead.isupper()
        if has_symbol or short_symbolic:
            lead_like += 1
        if (has_symbol or short_symbolic) and len(words) <= 14:
            row_like += 1

    lead_ratio = lead_like / len(lines)
    row_ratio = row_like / len(lines)
    if len(lines) >= 8 and short_ratio >= 0.65 and lead_ratio >= 0.34 and row_ratio >= 0.24:
        return True

    if _is_table_header_text(flat):
        return True
    return False



def _looks_like_paragraph_block(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False

    flat = re.sub(r"\s+", " ", " ".join(lines))
    word_count = len(ENGLISH_WORD_RE.findall(flat))
    punctuation = sum(text.count(mark) for mark in ".?!;")

    # Short paragraphs (1-2 lines) can also be prose, especially table captions/notes.
    # Check if it looks like a sentence (has punctuation and enough words).
    if len(lines) < 3:
        # For 1-2 line blocks, consider it prose if:
        # - It has sentence-ending punctuation (at least 1 period/question/exclamation)
        # - It has enough words (at least 8) suggesting a complete sentence
        return punctuation >= 1 and word_count >= 8

    avg_len = sum(len(line) for line in lines) / max(len(lines), 1)
    return avg_len > 48 or punctuation >= 2



def _looks_like_formula_narrative_fragment(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False

    flat = re.sub(r"\s+", " ", " ".join(lines))
    word_count = len(ENGLISH_WORD_RE.findall(flat))
    if word_count < 4:
        return False
    if not FORMULA_NARRATIVE_VERB_RE.search(flat):
        return False

    symbol_hits = len(re.findall(r"[=<>≤≥+\-*/^_()]", flat))
    if symbol_hits >= 1:
        return True

    return bool(
        re.search(
            r"\b(ciphertext|plaintext|protocol|index|variable|value|random|nonce|message)\b",
            flat,
            flags=re.IGNORECASE,
        )
    )



def _log_block_skip(*, page_no: int, block_index: int, reason: str, text: str) -> None:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) < 24:
        return
    en_words = len(ENGLISH_WORD_RE.findall(compact))
    cjk_chars = len(CJK_CHAR_RE.findall(compact))
    if en_words < 4 and cjk_chars < 8:
        return
    logger.info(
        "page %s block %s skipped: %s | excerpt=%s",
        page_no,
        block_index,
        reason,
        compact[:140],
    )



def _is_continuation_marker_line(line: str) -> bool:
    norm = re.sub(r"\s+", " ", line).strip()
    if not norm:
        return False
    if re.fullmatch(r"[（(]?\s*(?:续上行|接上行|续行|承上|同上|见上行)\s*[）)]?", norm):
        return True
    lower = norm.lower()
    return bool(re.fullmatch(r"[（(]?\s*(?:continued(?:\s+above)?|same as above|ditto)\s*[）)]?", lower))


