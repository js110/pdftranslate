"""Prompt preparation: compaction, fingerprints, translate/skip decisions."""
from __future__ import annotations

import hashlib
import re

from app.services.heuristics import STRUCTURED_LINE_MARKER_RE, STRUCTURED_NOTATION_TOKEN_RE
from app.services.textutil import CJK_CHAR_RE, ENGLISH_WORD_RE, _normalize_common_unicode_text
from app.services.translator import should_skip_translation


def _should_translate_text_block(text: str) -> bool:
    compact = _compact_text_for_prompt(text)
    if not compact:
        return False
    if should_skip_translation(compact):
        return False

    en_words = len(ENGLISH_WORD_RE.findall(compact))
    cjk_count = len(CJK_CHAR_RE.findall(compact))
    latin_chars = len(re.findall(r"[A-Za-z]", compact))

    # Chinese-dominant blocks rarely need EN->ZH translation; skip to save tokens/latency.
    if en_words == 0 and cjk_count > 0:
        return False
    if cjk_count >= 16 and en_words <= 2:
        return False
    if cjk_count >= 48 and en_words <= 5 and latin_chars <= 28:
        return False

    return True



def _should_preserve_source_line_layout(text: str) -> bool:
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 3:
        return False

    marker_lines = sum(1 for line in lines if STRUCTURED_LINE_MARKER_RE.match(line))
    token_hits = len(STRUCTURED_NOTATION_TOKEN_RE.findall(" ".join(lines)))
    short_ratio = sum(1 for line in lines if len(line) <= 92) / max(len(lines), 1)
    return marker_lines >= 2 or (token_hits >= 3 and short_ratio >= 0.45)



def _compact_text_for_prompt(text: str) -> str:
    compact = _normalize_common_unicode_text(text)
    preserve_lines = _should_preserve_source_line_layout(compact)
    compact = compact.replace("-\n", "")
    if preserve_lines:
        lines = [re.sub(r"[ \t]{2,}", " ", line).strip() for line in compact.splitlines() if line.strip()]
        return "\n".join(lines).strip()
    compact = re.sub(r"\s*\n+\s*", " ; ", compact)
    compact = re.sub(r"(?:\s*;\s*){2,}", " ; ", compact)
    compact = re.sub(r"[ \t]{2,}", " ", compact)
    return compact.strip()



def _text_fingerprint(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()



def _should_force_single_retry(text: str) -> bool:
    compact = _compact_text_for_prompt(text)
    if not compact:
        return False
    if should_skip_translation(compact):
        return False

    cjk_count = len(CJK_CHAR_RE.findall(compact))
    en_words = len(ENGLISH_WORD_RE.findall(compact))
    latin_chars = len(re.findall(r"[A-Za-z]", compact))
    if cjk_count > 0:
        # Mixed-language prose still needs retry when English span is substantial.
        if en_words >= 10 and latin_chars >= 40:
            return True
        if cjk_count <= 20 and en_words >= 6 and latin_chars >= 24:
            return True
        return False
    if en_words >= 6:
        return True
    if re.match(r"^\s*(?:section\s+)?\d+(?:\.\d+)*\s+[A-Za-z]", compact, flags=re.IGNORECASE):
        return True
    if len(compact) <= 80 and en_words >= 2:
        return True
    return False


