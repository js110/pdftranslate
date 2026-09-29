"""Translation quality evaluation, repair, and deterministic fallbacks."""
from __future__ import annotations

import logging
import re

from app.services.heuristics import _is_continuation_marker_line
from app.services.promptprep import _compact_text_for_prompt
from app.services.textutil import (
    CJK_CHAR_RE,
    CONTROL_CHAR_RE,
    ENGLISH_WORD_RE,
    _count_numbered_item_markers,
    _first_sentence,
    _normalize_common_unicode_text,
    _trim_extra_numbered_items,
    _trim_overlong_fragment_translation,
    _truncate_by_sentence,
)

logger = logging.getLogger(__name__)



NON_PAPER_META_INLINE_RE = re.compile(
    r"[（(][^()（）]{0,260}(后续内容保持原文|严格遵循要求|翻译要求|输出要求|不要输出|不要翻译|strict retry mode|system prompt|preferred terms|mathseg\d+token)[^()（）]{0,260}[）)]",
    re.IGNORECASE,
)

SHORT_SECTION_HEADING_TRANSLATION_RE = re.compile(
    r"^\s*(?:section\s+)?(?:\d+(?:\.\d+)*|[ivxlcdm]+)\s*[\).:：\-]?\s*"
    r"(?:相关工作|参考文献|引言|摘要|结论|实验(?:评估|结果)?|方法|预备知识|问题定义|系统模型)\s*$",
    re.IGNORECASE,
)

SHORT_SECTION_HEADING_TRANSLATION_SET = {
    "相关工作",
    "参考文献",
    "引言",
    "摘要",
    "结论",
    "实验",
    "实验评估",
    "实验结果",
    "方法",
    "预备知识",
    "问题定义",
    "系统模型",
}

def _normalize_translated_text(source_text: str, translated_text: str) -> str:
    normalized = translated_text.strip()
    normalized = normalized.replace("译文：", "").replace("翻译：", "").strip()
    normalized = _sanitize_formula_render_artifacts(normalized)
    normalized = _strip_non_paper_meta(normalized)
    normalized = _strip_source_echo(source_text, normalized)
    title_repaired = _repair_title_translation(source_text, normalized)
    if title_repaired:
        normalized = title_repaired
    normalized = _trim_extra_numbered_items(source_text, normalized)
    normalized = _trim_overlong_fragment_translation(source_text, normalized)
    normalized = _sanitize_formula_render_artifacts(normalized)
    normalized = _strip_non_paper_meta(normalized)

    source_len = len(source_text.strip())
    source_en = len(ENGLISH_WORD_RE.findall(source_text))
    if source_len <= 80:
        normalized = re.sub(r"\s+", " ", normalized).strip()
        # If model returns only section number, recover common section heading translation.
        if re.fullmatch(r"(?:section\s+)?(?:\d+(?:\.\d+)*|[ivxlcdm]+)\.?", normalized, flags=re.IGNORECASE):
            repaired = _repair_section_heading(source_text, normalized)
            if repaired:
                normalized = repaired
        max_len = max(80, source_len * 3)
        if len(normalized) > max_len:
            normalized = normalized[:max_len].rstrip(" ,;:.-")
    elif source_len <= 220 and source_en <= 42:
        max_len = max(150, int(source_len * 1.8))
        if len(normalized) > max_len:
            normalized = _truncate_by_sentence(normalized, max_len=max_len)

    return normalized or translated_text



def _sanitize_formula_render_artifacts(text: str) -> str:
    if not text:
        return text

    out = _normalize_common_unicode_text(text)
    # Normalize escaped math delimiters first (e.g., "\$ ... \$").
    out = out.replace("\\$", "$")
    # Normalize common LaTeX artifacts that models may inject into prose.
    out = re.sub(r"\\\s*tex\s*t?\s*\{([^{}]{1,80})\}", r"\1", out, flags=re.IGNORECASE)
    out = re.sub(r"\\\s*text\s*\{([^{}]{1,80})\}", r"\1", out, flags=re.IGNORECASE)
    out = re.sub(r"\\\s*mathrm\s*\{([^{}]{1,80})\}", r"\1", out, flags=re.IGNORECASE)
    out = re.sub(r"\\\s*operatorname\s*\{([^{}]{1,80})\}", r"\1", out, flags=re.IGNORECASE)
    # Strip common LaTeX math wrappers while keeping inner formula content.
    out = re.sub(r"\\\(\s*([\s\S]{1,360}?)\s*\\\)", r"\1", out)
    out = re.sub(r"\\\[\s*([\s\S]{1,480}?)\s*\\\]", r"\1", out)
    out = re.sub(r"\$\s*([^$]{1,360}?)\s*\$", r"\1", out)
    out = re.sub(r"\\\s*leftarrow\s*_\s*\{?\s*R\s*\}?", "←R", out, flags=re.IGNORECASE)
    out = re.sub(r"\\\s*rightarrow\s*_\s*\{?\s*R\s*\}?", "→R", out, flags=re.IGNORECASE)
    out = re.sub(r"\bleftarrow\s*_\s*\{?\s*R\s*\}?", "←R", out, flags=re.IGNORECASE)
    out = re.sub(r"\brightarrow\s*_\s*\{?\s*R\s*\}?", "→R", out, flags=re.IGNORECASE)

    latex_map = (
        (r"\\\s*oplus\b", "⊕"),
        (r"\\\s*times\b", "×"),
        (r"\\\s*cdot\b", "·"),
        (r"\\\s*leq\b", "≤"),
        (r"\\\s*geq\b", "≥"),
        (r"\\\s*neq\b", "≠"),
        (r"\\\s*in\b", "∈"),
        (r"\\\s*leftarrow\b", "←"),
        (r"\\\s*rightarrow\b", "→"),
    )
    for pat, rep in latex_map:
        out = re.sub(pat, rep, out, flags=re.IGNORECASE)

    # Convert lightweight LaTeX superscript/subscript braces to plain form.
    out = re.sub(r"([_^])\s*\{([^{}]{1,40})\}", r"\1\2", out)
    out = out.replace("\\_", "_").replace("\\{", "{").replace("\\}", "}")
    # Strip accidental HTML-like superscript/subscript tags emitted by some models.
    # Keep only their inner text to preserve affiliation markers such as a/b/c.
    out = re.sub(r"&lt;\s*/?\s*(?:sup|sub)\b[^&]*&gt;", "", out, flags=re.IGNORECASE)
    out = re.sub(r"<\s*/?\s*(?:sup|sub)\b[^>]*>", "", out, flags=re.IGNORECASE)
    # Remove residual isolated math delimiters after unwrapping.
    out = re.sub(r"(?:(?<=\s)|^)\$(?=\s*[A-Za-z0-9(])", "", out)
    out = re.sub(r"(?<=[A-Za-z0-9\)])\$(?=(?:\s|[，。,:;!?)]|$))", "", out)
    # Remove stray backslashes that are not part of escaped newline or tabs.
    out = re.sub(r"\\(?![nrt])(?=\s|[A-Za-z\u4e00-\u9fff])", "", out)
    # In formula narrative, models may leave an English "by" between a variable and a delta/number.
    out = re.sub(
        r"\b([A-Za-z][A-Za-z0-9*']*)\s+by\s+(?=(?:[+\-]?\d+|at\s+(?:least|most)\b|at\b|least\b|most\b|至少|至多|约))",
        r"\1 ",
        out,
        flags=re.IGNORECASE,
    )
    out = re.sub(
        r"\b([A-Za-z][A-Za-z0-9*']*)\s+by\s+(?=[\u4e00-\u9fff\u0370-\u03FF])",
        r"\1 ",
        out,
        flags=re.IGNORECASE,
    )
    out = re.sub(r"(?<=[=+\-*/<>≤≥])\s*;\s*(?=[A-Za-z0-9\u0370-\u03FF])", " ", out)
    out = _normalize_common_unicode_text(out)
    out = _cleanup_formula_residue_lines(out)
    out = CONTROL_CHAR_RE.sub("", out)
    out = re.sub(r"[ \t]{2,}", " ", out)
    return out.strip()



def _cleanup_formula_residue_lines(text: str) -> str:
    if not text:
        return text

    cleaned_lines: list[str] = []
    for raw in text.splitlines():
        line = re.sub(r"[ \t]{2,}", " ", raw).strip()
        if not line:
            continue
        # Drop orphan symbol-only lines introduced by broken math delimiters.
        if re.fullmatch(r"[+\-*/=·•|]{1,4}", line):
            continue
        # Repair prime/subscript split artifact: "E' s(...)" -> "E_s(...)".
        line = re.sub(r"\b([A-Za-z])'\s+([A-Za-z])(?=\s*\()", r"\1_\2", line)
        # Remove leading operator residue for prose lines (common after "$+ ...$" cleanup).
        if re.match(r"^[+\-]\s*[A-Za-z(]", line):
            has_cjk = bool(CJK_CHAR_RE.search(line))
            has_sentence_punct = bool(re.search(r"[。！？；，,]", line))
            if (has_cjk or has_sentence_punct) and "=" not in line[:24]:
                line = re.sub(r"^[+\-]\s*", "", line)
        # Remove dangling operator at line end.
        line = re.sub(r"\s+[+\-*/=]\s*$", "", line)
        line = line.strip()
        if not line:
            continue
        cleaned_lines.append(line)

    if cleaned_lines:
        return "\n".join(cleaned_lines).strip()
    return ""



def _fallback_translate_short_formula_phrase(source_text: str) -> str | None:
    source = re.sub(r"\s+", " ", source_text).strip()
    if not source:
        return None
    if len(CJK_CHAR_RE.findall(source)) > 0:
        return None

    en_words = ENGLISH_WORD_RE.findall(source)
    if len(en_words) < 2 or len(en_words) > 8 or len(source) > 72:
        return None

    normalized = re.sub(r"[^A-Za-z ]", " ", source).lower()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        return None

    mapped: str | None = None
    if re.fullmatch(r"(then we have|hence we have)", normalized):
        mapped = "则有"
    elif re.fullmatch(r"(easy to see that|it is easy to see that)", normalized):
        mapped = "易得"
    elif re.fullmatch(r"(note that|notice that)", normalized):
        mapped = "注意"
    elif re.fullmatch(r"(for example|as an example)", normalized):
        mapped = "例如"
    elif re.fullmatch(r"(which directly leads to|thus we obtain|thus we have)", normalized):
        mapped = "由此可得"
    elif re.fullmatch(r"we have", normalized):
        mapped = "有"

    if mapped is None:
        return None

    if source.endswith(":") or source.endswith("："):
        return f"{mapped}："
    return mapped



def _fallback_translate_table_narrative(source_text: str) -> str | None:
    source = _compact_text_for_prompt(source_text)
    source = source.replace("＊", "'")
    source = re.sub(r"\s*;\s*", " ", source)
    source = re.sub(r"\s+", " ", source).strip()
    if not source:
        return None
    if len(CJK_CHAR_RE.findall(source)) > 0:
        return None

    table_match = re.match(r"^\s*table\s+([ivxlcdm]+|\d+)\b", source, flags=re.IGNORECASE)
    if not table_match:
        return None

    table_id = table_match.group(1).upper()
    sentences = [seg.strip(" .;") for seg in re.split(r"(?<=[.?!])\s+", source) if seg.strip(" .;")]
    if not sentences:
        return None

    def _norm_tail(text: str) -> str:
        out = text.strip(" .;")
        out = out.replace("’", "'").replace("`", "'")
        out = re.sub(r"\ban\s+d\b", "and", out, flags=re.IGNORECASE)
        replacements = (
            (r"\bsince they are not important regarding how the protocols?\s+works?\b", "因为它们对协议运行方式并不重要"),
            (r"\bthe number of rounds and communication cost of the \[(\d{1,3})\] is an approximation\b", r"[\1]中的轮数与通信开销是近似值"),
            (r"\bthe secret keys are distributed as described above\b", "密钥按上述方式分发"),
            (r"\bthe users could generate pads with arbitrary length\b", "用户可以生成任意长度的填充"),
            (r"\bit doesn't display the value of the public nonce t and the encrypted message mi\b", "它没有展示公共随机数t和加密消息Mi的取值"),
            (r"\bit does not display the value of the public nonce t and the encrypted message mi\b", "它没有展示公共随机数t和加密消息Mi的取值"),
            (r"\bit does(?:n't| not)\s+display the value of the\s+public nonce\s*t\s+and the encrypted message\s*mi\b", "它没有展示公共随机数t和加密消息Mi的取值"),
            (r"\bit\s+doesn.?t\s+display\s+the\s+value\s+of\s+the\s+public\s+nonce\s*t\s+and\s+the\s+encrypted\s+message\s*mi\b", "它没有展示公共随机数t和加密消息Mi的取值"),
            (r"\ba pad for\s+([A-Za-z0-9_]+)\s+with length,\s*e\.g\.,\s*length\b", r"长度为length的\1填充串"),
            (r"\ba pad for\s+([A-Za-z0-9_]+)\s+with length\b", r"长度给定的\1填充串"),
            (r"\bthis pad\b", "该填充串"),
            (r"\bbit-choosing algorithm is executed before aggregation\b", "位选择算法在聚合前执行"),
            (r"\bwhich approximately adds two or three rounds to the whole process\b", "这会使整个过程大约增加2到3轮"),
            (r"\bmincomputing\b", "最小值计算"),
            (r"\bmin-computing\b", "最小值计算"),
            (r"\bprotocols?\b", "协议"),
            (r"\bseveral\b", "若干"),
            (r"\bunder given settings\b", "在给定设置下"),
            (r"\bpublic nonce\b", "公共随机数"),
            (r"\bencrypted message\b", "加密消息"),
            (r"\bthe value of\b", "取值"),
            (r"\bdisplay\b", "展示"),
            (r"\bdoesn.?t\b", ""),
            (r"\bas the\b", "因为"),
            (r"\bsince\b", "因为"),
            (r"\band\b", "和"),
        )
        for pat, rep in replacements:
            out = re.sub(pat, rep, out, flags=re.IGNORECASE)
        out = re.sub(r"\bthe\b", "", out, flags=re.IGNORECASE)
        out = re.sub(r"\bit\b", "", out, flags=re.IGNORECASE)
        out = re.sub(r"\s+", " ", out).strip(" ,;")
        out = out.replace(",", "，")
        out = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", out)
        out = re.sub(r"\s+([，。；：])", r"\1", out)
        out = re.sub(r"([，。；：])\s+", r"\1", out)
        out = re.sub(r"([（\(])\s+", r"\1", out)
        out = re.sub(r"\s+([）\)])", r"\1", out)
        return out

    out_sentences: list[str] = []
    for idx, sentence in enumerate(sentences):
        s = re.sub(r"\s+", " ", sentence).strip()

        if idx == 0:
            match_show = re.match(
                r"^table\s+[ivxlcdm0-9]+\s+shows\s+(.+?)\s+under given settings$",
                s,
                flags=re.IGNORECASE,
            )
            if match_show:
                obj = _norm_tail(match_show.group(1))
                out_sentences.append(f"表{table_id}展示了给定设置下的{obj}。")
                continue

            match_list = re.match(
                r"^table\s+[ivxlcdm0-9]+\s+lists\s+all the important variables in the whole process of the\s+([A-Za-z0-9\-]+)$",
                s,
                flags=re.IGNORECASE,
            )
            if match_list:
                proc = match_list.group(1)
                out_sentences.append(f"表{table_id}列出了{proc}整个过程中的所有重要变量。")
                continue

            match_give = re.match(
                r"^table\s+[ivxlcdm0-9]+\s+gives\s+the comparison between\s+(.+)$",
                s,
                flags=re.IGNORECASE,
            )
            if match_give:
                obj = _norm_tail(match_give.group(1))
                out_sentences.append(f"表{table_id}给出了{obj}的比较。")
                continue

            generic = re.match(
                r"^table\s+[ivxlcdm0-9]+\s+(shows|lists|gives|presents|illustrates|compares|summarizes|reports|describes|provides)\s+(.+)$",
                s,
                flags=re.IGNORECASE,
            )
            if generic:
                verb = generic.group(1).lower()
                rest = _norm_tail(generic.group(2))
                verb_map = {
                    "shows": "展示了",
                    "lists": "列出了",
                    "gives": "给出了",
                    "presents": "给出了",
                    "illustrates": "展示了",
                    "compares": "比较了",
                    "summarizes": "总结了",
                    "reports": "报告了",
                    "describes": "描述了",
                    "provides": "给出了",
                }
                out_sentences.append(f"表{table_id}{verb_map.get(verb, '给出了')}{rest}。")
                continue

            return None

        note_match = re.match(r"^note that\s+(.+)$", s, flags=re.IGNORECASE)
        if note_match:
            out_sentences.append(f"需要注意的是，{_norm_tail(note_match.group(1))}。")
            continue

        below_assume = re.match(r"^below,\s*we\s+(?:will\s+)?assume\s+(.+)$", s, flags=re.IGNORECASE)
        if below_assume:
            out_sentences.append(f"下面，我们假设{_norm_tail(below_assume.group(1))}。")
            continue

        assume_match = re.match(r"^we\s+assume(?:\s+that)?\s+(.+)$", s, flags=re.IGNORECASE)
        if assume_match:
            out_sentences.append(f"我们假设{_norm_tail(assume_match.group(1))}。")
            continue

        when_need = re.match(
            r"^when we need\s+(.+?),\s*we use notation\s+(.+?)\s+to denote\s+(.+)$",
            s,
            flags=re.IGNORECASE,
        )
        if when_need:
            what = _norm_tail(when_need.group(1))
            notation = when_need.group(2).strip()
            target = _norm_tail(when_need.group(3))
            out_sentences.append(f"当我们需要{what}时，使用记号{notation}表示{target}。")
            continue

        # Keep fallback conservative: if any sentence is completely unknown, abort.
        return None

    merged = "".join(seg.strip() for seg in out_sentences if seg.strip())
    return merged or None



def _strip_source_echo(source_text: str, translated_text: str) -> str:
    source = re.sub(r"\s+", " ", source_text).strip()
    translated = translated_text.strip()
    if not source or not translated:
        return translated

    lines = [line.strip() for line in translated.splitlines() if line.strip()]
    if len(lines) >= 2:
        kept: list[str] = []
        removed = False
        for line in lines:
            if _is_source_echo_line(source, line):
                removed = True
                continue
            kept.append(line)
        if removed and kept:
            translated = "\n".join(kept).strip()

    source_lower = source.lower()
    translated_lower = translated.lower()
    if len(source_lower) >= 16 and translated_lower.startswith(source_lower):
        tail = translated[len(source) :].lstrip(" \t\r\n:：-")
        if tail:
            translated = tail

    translated = _drop_untranslated_english_lines_in_mixed_output(source, translated)
    return translated



def _is_source_echo_line(source_text: str, line: str) -> bool:
    clean = re.sub(r"\s+", " ", line).strip()
    if len(clean) < 16:
        return False
    return _is_likely_untranslated_english_line(source_text, clean, strict=True)



def _drop_untranslated_english_lines_in_mixed_output(source_text: str, translated_text: str) -> str:
    lines = [line.strip() for line in translated_text.splitlines() if line.strip()]
    if len(lines) < 2:
        return translated_text

    cjk_lines = sum(1 for line in lines if len(CJK_CHAR_RE.findall(line)) >= 2)
    if cjk_lines == 0:
        return translated_text

    kept: list[str] = []
    removed = 0
    for line in lines:
        # Also detect short English fragments that are likely untranslated residue
        # These often appear at the end of translated paragraphs (e.g., "min protocols, proving...")
        if _is_short_english_fragment(line, source_text):
            removed += 1
            continue
        if _is_likely_untranslated_english_line(source_text, line, strict=False):
            removed += 1
            continue
        kept.append(line)

    if removed > 0 and kept:
        logger.info("removed untranslated english lines from mixed output: %s", removed)
        return "\n".join(kept).strip()
    return translated_text



def _is_short_english_fragment(line: str, source_text: str = "") -> bool:
    """Detect short English fragments that are likely untranslated residue.

    These are typically short English sentences or phrases that appear mixed with Chinese,
    often at the end of translated paragraphs. Examples:
    - "min protocols, proving them being privacy-preserving."
    - "proving them being privacy-preserving."
    """
    clean = re.sub(r"\s+", " ", line).strip()
    # Check if line is mostly English with very few or no Chinese characters
    cjk_count = len(CJK_CHAR_RE.findall(clean))
    if cjk_count > 2:
        return False

    # Must have some English words but be relatively short
    words = ENGLISH_WORD_RE.findall(clean)
    if len(words) < 3 or len(words) > 12:
        return False

    # Length should be relatively short (less than 80 characters)
    if len(clean) > 80:
        return False

    # Check if it's likely a fragment (ends with incomplete punctuation or unusual patterns)
    # Examples: "min protocols, proving them being..." or "...being privacy-preserving."
    if len(clean) >= 15 and len(clean) <= 60:
        # Check if it looks like a fragment that's been left untranslated
        # Pattern: mostly lowercase or mixed case, contains commas or unusual constructions
        if "," in clean or clean.lower() != clean:
            # Additional check: if source text contains this line, it's likely residue
            source_norm = source_text.lower() if source_text else ""
            clean_lower = clean.lower()
            # If the line or significant portion appears in source, it's likely residue
            if source_norm and (clean_lower in source_norm or any(w in source_norm for w in words[:3])):
                return True

    return False



def _contains_untranslated_english_lines_in_mixed_output(source_text: str, translated_text: str) -> bool:
    lines = [line.strip() for line in translated_text.splitlines() if line.strip()]
    if len(lines) < 2:
        return False
    if not any(len(CJK_CHAR_RE.findall(line)) >= 2 for line in lines):
        return False

    residue_lines = 0
    residue_words = 0
    for line in lines:
        if not _is_likely_untranslated_english_line(source_text, line, strict=False):
            continue
        residue_lines += 1
        residue_words += len(ENGLISH_WORD_RE.findall(line))

    if residue_lines == 0:
        return False
    return residue_words >= 8 or residue_lines >= 2



def _is_likely_untranslated_english_line(source_text: str, line: str, *, strict: bool) -> bool:
    clean = re.sub(r"\s+", " ", line).strip()
    min_len = 30 if strict else 22
    if len(clean) < min_len:
        return False

    cjk = len(CJK_CHAR_RE.findall(clean))
    if cjk > (1 if strict else 2):
        return False

    words = ENGLISH_WORD_RE.findall(clean)
    min_words = 6 if strict else 5
    if len(words) < min_words:
        return False

    # Count acronyms/proper nouns (all-caps words 2-8 chars) in the line.
    # These are expected to remain in English and should not count against quality.
    acronyms = re.findall(r"\b[A-Z]{2,8}\b", clean)
    non_acronym_words = len(words) - len(acronyms)

    # If most English words are acronyms/proper nouns and there's Chinese content,
    # this line is likely translated with proper nouns preserved.
    if non_acronym_words <= 2 and cjk >= 2:
        return False

    # Check if line starts with numbered list marker (e.g., "1)", "2.", "3)")
    # Numbered list items with Chinese content should not be flagged as untranslated.
    if re.match(r"^\s*\d{1,3}[\).]\s*", clean) and cjk >= 2:
        return False

    # In mixed-language output, a long pure-English sentence is almost always untranslated residue.
    if cjk <= 1 and len(words) >= 10 and len(clean) >= 56:
        return True

    source_norm = re.sub(r"\s+", " ", source_text).strip().lower()
    if len(source_norm) < 20:
        return False

    lower = clean.lower()
    if lower in source_norm:
        return True

    phrase = " ".join(words[: (12 if strict else 9)]).lower()
    if len(phrase) >= (36 if strict else 26) and phrase in source_norm:
        return True

    source_words = {w.lower() for w in ENGLISH_WORD_RE.findall(source_norm)}
    if not source_words:
        return False
    line_words = {w.lower() for w in words}
    overlap = len(source_words & line_words) / max(1, len(line_words))
    return overlap >= (0.62 if strict else 0.52)



def _strip_non_paper_meta(text: str) -> str:
    if not text:
        return text

    cleaned = NON_PAPER_META_INLINE_RE.sub("", text).strip()
    lines = [line.rstrip() for line in cleaned.splitlines() if line.strip()]
    if not lines:
        return cleaned

    kept: list[str] = []
    removed = False
    for line in lines:
        if _is_non_paper_meta_line(line):
            removed = True
            continue
        kept.append(line)

    if removed and kept:
        cleaned = "\n".join(kept).strip()
    else:
        cleaned = cleaned

    cleaned = re.sub(r"^\s*[.。·…]{2,}\s*", "", cleaned)
    return cleaned



def _is_non_paper_meta_line(line: str) -> bool:
    norm = re.sub(r"\s+", " ", line).strip()
    if len(norm) < 8:
        if re.fullmatch(r"[.。·…]{2,}", norm):
            return True
        if _is_continuation_marker_line(norm):
            return True
        return False

    if _is_continuation_marker_line(norm):
        return True

    lower = norm.lower()
    if re.search(r"mathseg\d+token", lower):
        return True
    meta_keys = (
        "后续内容保持原文",
        "原文未完成状态",
        "严格遵循要求",
        "翻译要求",
        "输出要求",
        "不要输出",
        "不要翻译",
        "术语=",
        "terms=",
        "strict retry mode",
        "system prompt",
        "preferred terms",
        "style:",
    )
    if not any(key in lower for key in meta_keys):
        return False

    cjk_count = len(CJK_CHAR_RE.findall(norm))
    en_words = len(ENGLISH_WORD_RE.findall(norm))
    return cjk_count >= 4 or en_words >= 3



def _contains_non_paper_meta(text: str) -> bool:
    if not text:
        return False
    if NON_PAPER_META_INLINE_RE.search(text):
        return True
    return any(_is_non_paper_meta_line(line) for line in text.splitlines() if line.strip())



def _has_extreme_length_mismatch(source_text: str, translated_text: str) -> bool:
    source = re.sub(r"\s+", " ", source_text).strip()
    translated = re.sub(r"\s+", " ", translated_text).strip()
    if not source or not translated:
        return False

    src_len = len(source)
    tgt_len = len(translated)
    src_en = len(ENGLISH_WORD_RE.findall(source))
    src_cjk = len(CJK_CHAR_RE.findall(source))
    tgt_cjk = len(CJK_CHAR_RE.findall(translated))
    sentence_marks = sum(translated.count(mark) for mark in ("。", "！", "？", ".", ";"))
    src_lines = [line.strip() for line in source_text.splitlines() if line.strip()]
    src_lower = source.lower()
    src_item_count = _count_numbered_item_markers(source_text)
    tgt_item_count = _count_numbered_item_markers(translated_text)

    # Long prose collapsing into a token/number (e.g. a 160-char sentence
    # "translated" as "1") means poisoned cache or a misaligned batch response.
    # Legit EN->ZH output stays well above 18% of the source length.
    if src_en >= 5 and src_len >= 60 and tgt_len < max(8, int(src_len * 0.18)):
        return True

    # Short English source blocks should not expand into a long multi-sentence Chinese passage.
    if src_cjk <= 2 and 4 <= src_en <= 28 and src_len <= 200:
        if tgt_cjk >= max(70, src_en * 5) and tgt_len >= max(170, int(src_len * 2.4)):
            return True

    # 1-2 line source fragments becoming long paragraphs are usually wrong segment mapping/hallucination.
    if len(src_lines) <= 2 and src_len <= 140 and tgt_cjk >= 80 and tgt_len >= 190 and sentence_marks >= 3:
        return True

    # Cross-segment hallucination often appears as extra numbered list items.
    if src_item_count >= 1 and tgt_item_count > src_item_count and src_len <= 240:
        return True
    if src_item_count == 0 and tgt_item_count >= 2 and src_len <= 180 and src_en >= 6:
        return True
    # Numbered list fragments should keep explicit item markers.
    if src_item_count >= 1 and tgt_item_count == 0 and src_len <= 260 and src_en >= 6 and tgt_cjk >= 6:
        return True

    # Under-translation guard: long source blocks collapsing into very short outputs.
    if src_en >= 24 and src_len >= 240:
        if tgt_len <= max(30, int(src_len * 0.20)) and tgt_cjk <= max(20, int(src_en * 0.55)):
            return True
    if src_en >= 55 and src_len >= 520 and tgt_len <= max(42, int(src_len * 0.14)):
        return True

    # Short fragments should not explode into long multi-sentence paragraphs.
    if src_len <= 180 and src_en >= 5 and sentence_marks >= 3 and tgt_len >= int(src_len * 1.65) and tgt_cjk >= 72:
        return True

    # Incomplete tail fragments (common near page/layout boundaries) should remain short in translation.
    if src_len <= 130 and src_en >= 5 and any(
        src_lower.endswith(tail)
        for tail in (
            " for both the",
            " for the",
            " of the",
            " and the",
            " to the",
            " in the",
            ",",
        )
    ):
        if tgt_cjk >= 70 and tgt_len >= 150:
            return True
    if src_len <= 200 and (source.endswith("-") or src_lower.endswith(" and") or src_lower.endswith(" of")):
        if sentence_marks >= 2 and tgt_cjk >= 55 and tgt_len >= int(src_len * 1.45):
            return True
    if src_len <= 170 and src_en >= 8 and source[:1].islower():
        if sentence_marks >= 2 and tgt_cjk >= 42 and tgt_len >= int(src_len * 1.18):
            return True

    return False



def _is_low_quality_translation(source_text: str, translated_text: str) -> bool:
    source = re.sub(r"\s+", " ", source_text).strip()
    translated = re.sub(r"\s+", " ", translated_text).strip()
    if not source:
        return False
    if not translated:
        return True
    if _contains_non_paper_meta(translated):
        return True
    if _contains_untranslated_english_lines_in_mixed_output(source, translated):
        return True
    if _has_extreme_length_mismatch(source, translated):
        return True

    source_lower = source.lower()
    translated_lower = translated.lower()
    if translated_lower == source_lower:
        return True

    src_en = len(ENGLISH_WORD_RE.findall(source))

    tgt_en = len(ENGLISH_WORD_RE.findall(translated))
    tgt_cjk = len(CJK_CHAR_RE.findall(translated))
    
    # Count acronyms/proper nouns (all-caps words 2-8 chars) in translated text.
    # These are expected to remain in English and should not count against quality.
    tgt_acronyms = len(re.findall(r"\b[A-Z]{2,8}\b", translated))
    tgt_en_excluding_acronyms = max(0, tgt_en - tgt_acronyms)

    # Long prose should not degrade into a short section heading.
    if src_en >= 10 and len(source) >= 80 and not _looks_like_title_block(source):
        if _looks_like_short_section_heading_translation(translated):
            return True
    
    if _looks_like_title_block(source):
        if tgt_en >= max(6, int(src_en * 0.45)) and tgt_cjk < max(6, int(src_en * 0.35)):
            return True
        if _contains_large_untranslated_english_segment(source_lower, translated, src_en, for_title=True):
            return True
    if src_en < 5:
        if src_en >= 2 and tgt_cjk == 0 and tgt_en >= max(2, src_en - 1):
            return True
        return False

    # Relax the threshold when most remaining English words are acronyms/proper nouns.
    # If non-acronym English words are few and Chinese content is substantial, it's not low quality.
    if tgt_en_excluding_acronyms <= 3 and tgt_cjk >= 8:
        pass  # Good translation with proper nouns preserved
    elif tgt_en >= max(8, int(src_en * 0.55)) and tgt_cjk < max(8, int(src_en * 0.22)):
        return True

    if _contains_large_untranslated_english_segment(source_lower, translated, src_en, for_title=False):
        return True

    if len(source_lower) >= 24 and source_lower in translated_lower and tgt_cjk >= 4:
        return True

    src_first = _first_sentence(source_lower)
    if len(src_first) >= 28 and src_first in translated_lower:
        return True

    return False



def _contains_large_untranslated_english_segment(
    source_lower: str,
    translated_text: str,
    source_en_words: int,
    *,
    for_title: bool = False,
) -> bool:
    min_line_len = 24 if for_title else 36
    min_words = 6 if for_title else 8
    for raw_line in translated_text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if len(line) < min_line_len:
            continue

        words = ENGLISH_WORD_RE.findall(line)
        if len(words) < min_words:
            continue

        cjk = len(CJK_CHAR_RE.findall(line))
        # Pure or nearly pure English span in translated output.
        if cjk > 2:
            continue

        # Check if line starts with numbered list marker (e.g., "1)", "2.", "3)")
        # Numbered list items with Chinese content should not be flagged as untranslated.
        if re.match(r"^\s*\d{1,3}[\).]\s*", line) and cjk >= 2:
            continue

        # Check if line contains protocol/acronym names (all-caps words 2-8 chars)
        # These are proper nouns that should remain in English.
        acronym_count = len(re.findall(r"\b[A-Z]{2,8}\b", line))
        # If most English words are acronyms/proper nouns, this line is likely translated.
        non_acronym_words = len(words) - acronym_count
        if acronym_count >= 1 and non_acronym_words <= 2 and cjk >= 4:
            continue

        # If source itself is a prose block, long pure-English spans in output are usually untranslated residue.
        if source_en_words >= 12 and len(words) >= 10 and len(line) >= 60:
            return True

        lower = line.lower()
        if lower in source_lower:
            return True

        # Match by long prefix phrase to catch line-wrap differences.
        phrase = " ".join(words[:12]).lower()
        if len(phrase) >= 40 and phrase in source_lower:
            return True

    return False



def _looks_like_title_block(text: str) -> bool:
    compact = re.sub(r"\s+", " ", text).strip()
    if not compact:
        return False
    if any(mark in compact for mark in ("http://", "https://", "doi", "arxiv")):
        return False

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines or len(lines) > 4:
        return False

    en_words = len(ENGLISH_WORD_RE.findall(compact))
    cjk_chars = len(CJK_CHAR_RE.findall(compact))
    if en_words < 6 or en_words > 36:
        return False
    if cjk_chars > 2:
        return False

    punct = sum(compact.count(mark) for mark in ".?!;:")
    if punct > 2:
        return False
    if re.search(r"[=<>\\/\[\]{}_^$]", compact):
        return False
    avg_len = sum(len(line) for line in lines) / max(len(lines), 1)
    return avg_len >= 18



def _looks_like_short_section_heading_translation(text: str) -> bool:
    compact = re.sub(r"\s+", " ", text).strip()
    if not compact:
        return False
    if len(compact) <= 28 and SHORT_SECTION_HEADING_TRANSLATION_RE.match(compact):
        return True
    compact_no_space = compact.replace(" ", "")
    return compact_no_space in SHORT_SECTION_HEADING_TRANSLATION_SET



def _repair_title_translation(source_text: str, translated_text: str) -> str | None:
    source = re.sub(r"\s+", " ", source_text).strip()
    translated = re.sub(r"\s+", " ", translated_text).strip()
    if not source or not translated:
        return None
    source_lower = source.lower()

    # Handle split title fragments first: multi-line English titles are often extracted
    # into 2 blocks (e.g., first line + second line), so full-title matching would miss.
    if re.fullmatch(
        r"revisiting\s+privacy[- ]preserving\s+min\s+and\s+k(?:-|\s)?th",
        source_lower,
    ):
        return "重新审视隐私保护最小值与第k小值"
    if re.fullmatch(
        r"min\s+protocols?\s+for\s+mobile\s+sensing",
        source_lower,
    ):
        return "面向移动感知的计算协议"

    if not _looks_like_title_block(source):
        return None
    if len(CJK_CHAR_RE.findall(source)) > 0:
        return None

    # Common title in this domain that is frequently mistranslated into two loose phrases.
    if re.fullmatch(
        r"revisiting\s+privacy[- ]preserving\s+min\s+and\s+k(?:-|\s)?th\s+min\s+protocols?\s+for\s+mobile\s+sensing",
        source_lower,
    ):
        return "重新审视面向移动感知的隐私保护最小值与第k小值计算协议"

    # If output is already clean and complete, keep it.
    if "第k小值" in translated and "最小值" in translated and ("协议" in translated or "计算" in translated):
        if "面向" in translated or "场景" in translated:
            return None

    # Lightweight structure repair for "Revisiting ... for Mobile Sensing".
    match = re.match(r"^revisiting\s+(.+?)\s+for\s+mobile\s+sensing$", source, flags=re.IGNORECASE)
    if not match:
        return None

    left = match.group(1).strip().lower()
    if re.fullmatch(
        r"privacy[- ]preserving\s+min\s+and\s+k(?:-|\s)?th\s+min\s+protocols?",
        left,
    ):
        return "重新审视面向移动感知的隐私保护最小值与第k小值计算协议"

    return None



def _repair_section_heading(source_text: str, section_no: str) -> str | None:
    src = re.sub(r"\s+", " ", source_text).strip()
    match = re.match(
        r"^(?:Section\s+)?(?P<num>[IVXLCMivxlcm0-9\.]+)\s*[:.\-]?\s*(?P<title>[A-Za-z][A-Za-z \-]+)$",
        src,
    )
    if not match:
        return None

    num = match.group("num")
    title = re.sub(r"\s+", " ", match.group("title")).strip().upper()
    title_map = {
        "ABSTRACT": "摘要",
        "INTRODUCTION": "引言",
        "RELATED WORK": "相关工作",
        "RELATED WORKS": "相关工作",
        "PRELIMINARIES": "预备知识",
        "PROBLEM STATEMENT": "问题定义",
        "SYSTEM MODEL": "系统模型",
        "METHOD": "方法",
        "METHODOLOGY": "方法",
        "EXPERIMENT": "实验",
        "EXPERIMENTS": "实验",
        "EVALUATION": "实验评估",
        "RESULTS": "结果",
        "DISCUSSION": "讨论",
        "CONCLUSION": "结论",
        "CONCLUSIONS": "结论",
    }

    zh = title_map.get(title)
    if zh is None:
        return None
    return f"{num} {zh}"
