from __future__ import annotations

import importlib.util
import json
import logging
import re
from dataclasses import dataclass
from typing import Any

import httpx

from app.core.settings import get_settings


URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
CODE_RE = re.compile(r"[{};<>]|\b(def|class|return|import)\b")
FORMULA_RE = re.compile(r"\b[A-Za-z]\s*=\s*[A-Za-z0-9]|\$[^$]+\$|\\\([^)]*\\\)")
CITATION_RE = re.compile(r"^\s*(\[[0-9,\- ]+\]|\([A-Za-z].*\d{4}.*\))\s*$")
CAPTION_RE = re.compile(r"^\s*(fig(?:ure)?|table)\s*[\.\-: ]*([ivxlcdm]+|\d+)\b", re.IGNORECASE)
LATEX_CMD_RE = re.compile(r"\\(frac|sum|int|alpha|beta|gamma|theta|lambda|mu|sigma|pi|cdot|times|leq|geq)")
UNICODE_MATH_CHARS = "\u2264\u2265\u2248\u2260\u2211\u222b\u221a\u00b1\u00d7\u00f7\u2217\u2212\u2032\u2033"
EN_WORD_RE = re.compile(r"\b[A-Za-z]{3,}\b")
CJK_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")
DISPLAY_EQ_RE = re.compile(r"[A-Za-z0-9\)\]}]\s*=\s*[A-Za-z0-9\(\[{]")
MATH_STRUCT_RE = re.compile(r"[{}_^]|\\[A-Za-z]+|[\u2264\u2265\u2248\u2260\u2211\u222b\u221a\u00b1\u00d7\u00f7\u2217\u2212\u2032\u2033]")
OPT_KEYWORD_RE = re.compile(r"\b(argmax|argmin|min|max|s\.t\.)\b", re.IGNORECASE)
FORMULA_NARRATIVE_HINT_RE = re.compile(
    r"\b("
    r"choose|chooses|compute|computes|computed|send|sends|sent|receive|receives|received|"
    r"encrypt|encrypts|encrypted|decrypt|decryption|let|then|after|before|where|thus|"
    r"assume|assumes|assuming|note|noted|"
    r"count|update|set|sets|check|checks"
    r")\b",
    re.IGNORECASE,
)
FORMULA_PROSE_LEAD_RE = re.compile(
    r"^\s*(where|if|when|let|thus|then|for|assuming|suppose|consider|set|define|denote|given)\b",
    re.IGNORECASE,
)
FORMULA_PROSE_HINT_RE = re.compile(
    r"\b(where|if|when|let|thus|then|assume|assuming|note|denote|denotes|define|defined|represents|subject to|s\.t\.)\b",
    re.IGNORECASE,
)
THEOREM_PREFIX_RE = re.compile(
    r"^\s*(theorem|proof|lemma|corollary|proposition|definition)\b",
    re.IGNORECASE,
)
MATH_LATEX_FRAGMENT_RE = re.compile(r"\\[A-Za-z]+(?:\s*\{[^{}]{0,120}\}){0,2}")
MATH_EQ_FRAGMENT_RE = re.compile(
    r"(?<![A-Za-z])(?:[A-Za-z0-9_().{}\[\]]+)(?:\s*[=+*/^_<>]\s*[A-Za-z0-9_().{}\[\]]+)+(?![A-Za-z])"
)
MATH_UNICODE_FRAGMENT_RE = re.compile(r"[\u2264\u2265\u2248\u2260\u2211\u222b\u221a\u00b1\u00d7\u00f7\u221e\u2202\u2207\u2217\u2212\u2032\u2033]+")
MATH_ASCII_SYMBOL_RE = re.compile(r"(?<!\w)[=+*/^_<>](?!\w)|(?<=\s)[=+*/^_<>](?=\s)")
DEFAULT_STYLE_PROFILE_KEYS = {
    "",
    "academic_conservative",
    "academic",
    "default",
}
BATCH_INDEX_RE = re.compile(r"(?m)^\s*\[(\d+)\]\s*")
_http_client: httpx.Client | None = None
_HTTP2_AVAILABLE = importlib.util.find_spec("h2") is not None
logger = logging.getLogger(__name__)


@dataclass
class ProviderRuntime:
    id: str
    model: str
    api_key: str
    base_url: str | None = None
    timeout_sec: int = 60


class TranslationError(RuntimeError):
    pass


def _math_symbol_count(text: str) -> int:
    count = 0
    for idx, ch in enumerate(text):
        if ch in UNICODE_MATH_CHARS:
            count += 1
            continue
        if ch not in "=+-*/^_<>":
            continue

        left = text[idx - 1] if idx > 0 else " "
        right = text[idx + 1] if idx + 1 < len(text) else " "
        left_alpha = left.isalpha()
        right_alpha = right.isalpha()
        left_digit = left.isdigit()
        right_digit = right.isdigit()

        # Ignore hyphen/slash in normal words like privacy-preserving or and/or.
        if ch in "-/":
            if left_alpha and right_alpha:
                continue
            if not (left_digit or right_digit or left in ")]} _" or right in "([{ _"):
                if left not in "=+-*/^_<>" and right not in "=+-*/^_<>":
                    continue

        count += 1
    return count


def _short_variable_token_count(text: str) -> int:
    return len(re.findall(r"\b[A-Za-z]{1,2}\b", text))


def _looks_like_formula(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if FORMULA_RE.search(stripped):
        return True
    if LATEX_CMD_RE.search(stripped):
        return True

    symbol_count = _math_symbol_count(stripped)
    digit_count = sum(ch.isdigit() for ch in stripped)
    short_tokens = _short_variable_token_count(stripped)
    if symbol_count >= 2 and (digit_count >= 1 or short_tokens >= 3) and len(stripped) <= 220:
        return True
    if symbol_count >= 2 and len(EN_WORD_RE.findall(stripped)) <= 5 and len(stripped) <= 180:
        return True

    if re.search(r"\b[A-Za-z]\s*\([A-Za-z0-9,+\-*/\s]+\)", stripped) and len(stripped) <= 160:
        return True
    return False


def _looks_like_formula_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if _looks_like_formula(stripped):
        return True
    if DISPLAY_EQ_RE.search(stripped) and MATH_STRUCT_RE.search(stripped):
        return True
    if OPT_KEYWORD_RE.search(stripped) and _math_symbol_count(stripped) >= 1:
        return True

    symbol_count = _math_symbol_count(stripped)
    if symbol_count >= 3 and len(stripped) <= 260 and (symbol_count / max(len(stripped), 1)) >= 0.08:
        return True
    return False


def _looks_like_formula_block(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return False

    formula_lines = sum(1 for line in lines if _looks_like_formula_line(line))
    if formula_lines == 0:
        return False

    prose_lines = 0
    prose_words = 0
    total_en_words = 0
    for line in lines:
        line_words = len(EN_WORD_RE.findall(line))
        total_en_words += line_words
        if _looks_like_formula_line(line):
            continue
        word_count = line_words
        if word_count >= 6:
            prose_lines += 1
            prose_words += word_count

    # Mixed blocks (formula + enough prose) should still be translated.
    # Math fragments will be protected/restored by placeholder logic.
    if prose_words >= 8 and prose_lines >= 1:
        return False
    if _looks_like_formula_narrative(text):
        return False
    # If the block has substantial English words overall, it should be translated
    # even if all lines contain math formulas (common in academic papers)
    if total_en_words >= 10:
        return False
    if formula_lines >= 2:
        return True

    return formula_lines == 1 and len(lines) <= 4 and prose_lines <= 1


def _looks_like_formula_narrative(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False

    flat = re.sub(r"\s+", " ", " ".join(lines)).strip()
    en_words = len(EN_WORD_RE.findall(flat))
    word_tokens = re.findall(r"\b[A-Za-z]{1,}\b", flat)
    token_words = len(word_tokens)
    if en_words < 3:
        # Short narrative clauses with math operators are common in正文:
        # e.g. "where p <= 0.05", "if x > 0 then ...".
        if token_words >= 2 and _math_symbol_count(flat) >= 1:
            if FORMULA_PROSE_LEAD_RE.match(flat):
                return True
            if FORMULA_PROSE_HINT_RE.search(flat):
                return True
        return False

    hint_hits = len(FORMULA_NARRATIVE_HINT_RE.findall(flat))
    if hint_hits >= 2 and en_words >= 3:
        return True

    if FORMULA_PROSE_LEAD_RE.match(flat) and _math_symbol_count(flat) >= 1 and en_words >= 3:
        return True
    if FORMULA_PROSE_HINT_RE.search(flat) and _math_symbol_count(flat) >= 1 and en_words >= 4:
        return True

    punct_hits = flat.count(",") + flat.count(";") + flat.count(":")
    if hint_hits >= 1 and punct_hits >= 1 and en_words >= 5:
        return True
    return False


def _looks_like_standalone_caption(text: str) -> bool:
    stripped = re.sub(r"\s+", " ", text).strip()
    if not stripped:
        return False
    if not CAPTION_RE.match(stripped):
        return False

    words = EN_WORD_RE.findall(stripped)
    lower = stripped.lower()
    # Prose references to figures/tables (e.g., "Table V lists ...") should be translated.
    if re.search(r"\b(lists?|shows?|presents?|illustrates?|compares?|summarizes?|reports?|gives?|describes?)\b", lower):
        return False
    if len(words) >= 8 and (" that " in lower or " since " in lower or "," in stripped):
        return False
    if len(words) <= 6:
        return True

    alpha_chars = [ch for ch in stripped if ch.isalpha()]
    if alpha_chars:
        upper_ratio = sum(1 for ch in alpha_chars if ch.isupper()) / len(alpha_chars)
        if upper_ratio >= 0.55:
            return True
    # Short title-like captions should stay un-translated.
    return len(words) <= 12


def should_skip_translation(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if re.fullmatch(r"[\d\W_]+", stripped):
        return True
    if URL_RE.search(stripped):
        return True
    if CITATION_RE.match(stripped):
        return True
    if _looks_like_standalone_caption(stripped):
        return True
    en_words = len(EN_WORD_RE.findall(stripped))
    theorem_like = bool(THEOREM_PREFIX_RE.match(stripped))
    if CODE_RE.search(stripped) and stripped.count(" ") < 40 and en_words < 6:
        if _looks_like_formula_narrative(stripped):
            return False
        return True
    if _looks_like_formula_block(stripped):
        if theorem_like and en_words >= 4:
            return False
        if _looks_like_formula_narrative(stripped):
            return False
        return True

    symbol_ratio = sum(not ch.isalnum() and not ch.isspace() for ch in stripped) / max(len(stripped), 1)
    if symbol_ratio > 0.35 and en_words < 10:
        if _looks_like_formula_narrative(stripped):
            return False
        if en_words >= 4 and FORMULA_PROSE_HINT_RE.search(stripped):
            return False
        return True
    if _looks_like_formula(stripped):
        # Translate prose-rich mixed content; only skip math-dominant text.
        if theorem_like and en_words >= 4:
            return False
        if _looks_like_formula_narrative(stripped):
            return False
        if en_words >= 4 and FORMULA_PROSE_HINT_RE.search(stripped):
            return False
        if en_words >= 6:
            return False
        return True
    return False


def _protect_math_content(text: str) -> tuple[str, list[tuple[str, str]]]:
    protected = text
    replacements: list[tuple[str, str]] = []
    counter = 0

    def _replace(match: re.Match[str]) -> str:
        nonlocal counter
        token = f"MATHSEG{counter}TOKEN"
        replacements.append((token, match.group(0)))
        counter += 1
        return token

    for pattern in (MATH_LATEX_FRAGMENT_RE, MATH_EQ_FRAGMENT_RE, MATH_UNICODE_FRAGMENT_RE, MATH_ASCII_SYMBOL_RE):
        protected = pattern.sub(_replace, protected)

    return protected, replacements


def _restore_math_content(text: str, replacements: list[tuple[str, str]]) -> str:
    restored = text
    for token, original in replacements:
        restored = re.sub(re.escape(token), original, restored, flags=re.IGNORECASE)
    return restored


def _style_instruction(style_profile: str) -> str:
    raw = style_profile.strip()
    if not raw:
        return ""
    key = raw.lower().replace("-", "_").replace(" ", "_")
    if key in DEFAULT_STYLE_PROFILE_KEYS:
        return ""
    return raw


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _is_untranslated_echo(source_text: str, translated_text: str) -> bool:
    source = _normalize_ws(source_text)
    translated = _normalize_ws(translated_text)
    if not source or not translated:
        return False

    source_lower = source.lower()
    translated_lower = translated.lower()
    if translated_lower == source_lower:
        return len(re.findall(r"[A-Za-z]", source)) >= 10

    source_words = EN_WORD_RE.findall(source_lower)
    if len(source_words) < 3:
        return False

    translated_words = EN_WORD_RE.findall(translated_lower)
    translated_cjk = len(CJK_CHAR_RE.findall(translated))
    if len(translated_words) < 4 or translated_cjk >= 4:
        return False

    if translated_lower in source_lower or source_lower in translated_lower:
        return True

    if len(source_words) < 8:
        return False

    src_set = {w.lower() for w in source_words}
    tgt_set = {w.lower() for w in translated_words}
    overlap = len(src_set & tgt_set) / max(1, len(tgt_set))
    return overlap >= 0.82 and len(translated_words) >= max(6, int(len(source_words) * 0.7))


def _has_excessive_untranslated_batch(source_texts: list[str], translated_texts: list[str]) -> bool:
    if len(source_texts) != len(translated_texts):
        return False
    bad_count = 0
    for src, tgt in zip(source_texts, translated_texts, strict=False):
        if _is_untranslated_echo(src, tgt):
            bad_count += 1
    if bad_count == 0:
        return False
    if len(source_texts) == 1:
        return True
    return bad_count >= max(2, int(len(source_texts) * 0.5))


def build_prompt(
    text: str,
    style_profile: str,
    glossary: list[str] | None = None,
    strict_mode: bool = False,
) -> list[dict[str, str]]:
    glossary_text = _compact_glossary_text(glossary)
    style_text = _style_instruction(style_profile)
    system = (
        "Translate EN academic text to Simplified Chinese. "
        "Keep formulas/URLs/citations/code/reference tags unchanged. "
        "Keep placeholders like MATHSEG123TOKEN unchanged. "
        "Preserve list/bullet/line-break structure when present in source. "
        "Do not add, infer, or summarize content not present in source. "
        "Preserve list numbering when present. "
        "Output translation only."
    )
    if strict_mode:
        system += (
            " Strict mode: translate every English sentence into Simplified Chinese, "
            "do not keep full English sentences, do not add extra list items, and output no meta text."
        )
    user_parts: list[str] = []
    if style_text:
        user_parts.append(f"Style={style_text}")
    if glossary_text:
        user_parts.append(f"Terms={glossary_text}")
    user_parts.append(f"Text:\n{text}")
    user = "\n".join(user_parts)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_batch_prompt(texts: list[str], style_profile: str, glossary: list[str] | None = None) -> list[dict[str, str]]:
    glossary_text = _compact_glossary_text(glossary)
    style_text = _style_instruction(style_profile)
    expected_size = len(texts)
    system = (
        f"Translate {expected_size} segments to Simplified Chinese. "
        "Keep formulas/URLs/citations/code/reference tags unchanged. "
        "Keep placeholders like MATHSEG123TOKEN unchanged. "
        "Translate each segment independently: do not copy content across segments, do not invent missing items, "
        "and preserve numbered/bullet list markers and line breaks from the source segment when present. "
        "For each segment, output translation only and do not keep full English sentences. "
        f"Return exactly {expected_size} lines in this exact format: [index] translation."
    )

    lines = "\n".join(f"[{idx}] {text}" for idx, text in enumerate(texts, start=1))
    user_parts: list[str] = []
    if style_text:
        user_parts.append(f"Style={style_text}")
    if glossary_text:
        user_parts.append(f"Terms={glossary_text}")
    user_parts.append("Segments:")
    user_parts.append(lines)
    user = "\n".join(user_parts)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _compact_glossary_text(glossary: list[str] | None) -> str:
    if not glossary:
        return ""
    seen: set[str] = set()
    terms: list[str] = []
    for item in glossary:
        term = item.split(":", 1)[0].strip()
        if not term:
            continue
        if len(term) > 36 and not (term.isupper() and len(term) <= 64):
            continue
        upper = term.upper()
        if upper in seen:
            continue
        seen.add(upper)
        terms.append(term)
        if len(terms) >= 8:
            break
    return ", ".join(terms)


def _is_hunyuan_endpoint(base_url: str) -> bool:
    return "hunyuan.cloud.tencent.com" in base_url


def _is_deepseek_provider(provider: ProviderRuntime, base_url: str) -> bool:
    if "api.deepseek.com" in base_url:
        return True
    if "deepseek" in provider.model.lower():
        return True
    return "deepseek" in provider.id.lower()


def _build_payload(provider: ProviderRuntime, messages: list[dict[str, str]], base_url: str) -> dict[str, Any]:
    settings = get_settings()
    if _is_deepseek_provider(provider, base_url):
        temperature = 1.3
    else:
        temperature = max(0.0, min(2.0, float(settings.translation_temperature)))
    payload: dict[str, Any] = {
        "model": provider.model,
        "messages": messages,
        "temperature": temperature,
    }

    if _is_hunyuan_endpoint(base_url):
        payload["enable_enhancement"] = True

    return payload


def _format_http_error(resp: httpx.Response) -> str:
    detail = ""
    try:
        data = resp.json()
        if isinstance(data, dict):
            err = data.get("error")
            if isinstance(err, dict):
                detail = str(err.get("message") or err.get("code") or "")
            if not detail:
                detail = str(data.get("message") or "")
        elif data is not None:
            detail = str(data)
    except Exception:  # noqa: BLE001
        detail = resp.text.strip()

    detail = detail.strip()
    if detail:
        return f"HTTP {resp.status_code}: {detail}"
    return f"HTTP {resp.status_code}"


def _chat_completion(provider: ProviderRuntime, messages: list[dict[str, str]]) -> str:
    global _http_client

    base_url = provider.base_url.rstrip("/") if provider.base_url else "https://api.openai.com/v1"
    endpoint = f"{base_url}/chat/completions"
    payload = _build_payload(provider, messages, base_url)

    headers = {
        "Authorization": f"Bearer {provider.api_key}",
        "Content-Type": "application/json",
    }
    if _http_client is None:
        _http_client = httpx.Client(
            # httpx requires optional dependency h2 for HTTP/2.
            # Fall back to HTTP/1.1 when h2 is not installed.
            http2=_HTTP2_AVAILABLE,
            limits=httpx.Limits(max_connections=64, max_keepalive_connections=16),
        )
    resp = _http_client.post(endpoint, json=payload, headers=headers, timeout=provider.timeout_sec)

    if resp.status_code >= 400:
        raise TranslationError(_format_http_error(resp))

    data = resp.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        content = "\n".join(parts)

    if not isinstance(content, str) or not content.strip():
        raise TranslationError("empty translation response")
    return content.strip()


def _strip_code_fence(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_json_candidates(raw: str) -> list[str]:
    text = _strip_code_fence(raw)
    candidates = [text]

    object_match = re.search(r"\{[\s\S]*\}", text)
    if object_match:
        candidates.append(object_match.group(0))

    array_match = re.search(r"\[[\s\S]*\]", text)
    if array_match:
        candidates.append(array_match.group(0))

    dedup: list[str] = []
    for item in candidates:
        if item and item not in dedup:
            dedup.append(item)
    return dedup


def _parse_marked_output(raw: str, expected_size: int) -> list[str] | None:
    text = _strip_code_fence(raw)
    matches = list(BATCH_INDEX_RE.finditer(text))
    if not matches:
        return None

    mapping: dict[int, str] = {}
    for idx, match in enumerate(matches):
        seg_no = int(match.group(1))
        if seg_no < 1 or seg_no > expected_size:
            continue
        if seg_no in mapping:
            return None

        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        value = text[start:end].strip()
        if not value:
            return None
        mapping[seg_no] = value

    if not all(i in mapping for i in range(1, expected_size + 1)):
        return None
    return [mapping[i] for i in range(1, expected_size + 1)]


def _parse_batch_output(raw: str, expected_size: int, source_texts: list[str]) -> list[str]:
    parsed: Any = None
    for candidate in _extract_json_candidates(raw):
        try:
            parsed = json.loads(candidate)
            break
        except Exception:  # noqa: BLE001
            continue

    translations: list[str] | None = None
    if parsed is not None:
        if isinstance(parsed, dict):
            maybe = parsed.get("translations")
            if isinstance(maybe, list):
                translations = [str(item).strip() for item in maybe]
        elif isinstance(parsed, list):
            translations = [str(item).strip() for item in parsed]

    if translations is None:
        translations = _parse_marked_output(raw, expected_size=expected_size)

    if translations is None:
        text = _strip_code_fence(raw)
        paragraphs = [chunk.strip() for chunk in re.split(r"\n\s*\n+", text) if chunk.strip()]
        if len(paragraphs) == expected_size:
            translations = paragraphs

    if translations is None:
        lines = [line.strip() for line in _strip_code_fence(raw).splitlines() if line.strip()]
        if len(lines) == expected_size:
            translations = lines

    if translations is None and expected_size == 1:
        single = _strip_code_fence(raw).strip()
        if single:
            translations = [single]

    if translations is None:
        raise TranslationError("batch response parse failed")
    if len(translations) != expected_size:
        raise TranslationError(f"batch size mismatch: expected {expected_size}, got {len(translations)}")

    normalized: list[str] = []
    for idx, item in enumerate(translations):
        value = item.strip()
        normalized.append(value if value else source_texts[idx])
    return normalized


def _translate_batch_once(
    texts: list[str],
    provider: ProviderRuntime,
    style_profile: str,
    glossary: list[str] | None = None,
) -> list[str]:
    protected_texts: list[str] = []
    replacement_maps: list[list[tuple[str, str]]] = []
    for text in texts:
        protected, replacements = _protect_math_content(text)
        protected_texts.append(protected)
        replacement_maps.append(replacements)

    messages = build_batch_prompt(texts=protected_texts, style_profile=style_profile, glossary=glossary)
    raw = _chat_completion(provider, messages)
    translated = _parse_batch_output(raw, expected_size=len(protected_texts), source_texts=protected_texts)

    restored: list[str] = []
    for idx, item in enumerate(translated):
        restored.append(_restore_math_content(item, replacement_maps[idx]))
    return restored


def translate_with_fallback(
    text: str,
    primary: ProviderRuntime,
    backup: ProviderRuntime | None,
    style_profile: str,
    glossary: list[str] | None = None,
    max_retries: int = 1,
    strict_mode: bool = False,
) -> tuple[str, str, str | None]:
    settings = get_settings()
    same_text_guard_enabled = not bool(settings.disable_same_text_retry_guard)
    protected_text, replacements = _protect_math_content(text)
    messages = build_prompt(
        protected_text,
        style_profile=style_profile,
        glossary=glossary,
        strict_mode=strict_mode,
    )
    attempts = max(1, int(max_retries))

    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            translated = _chat_completion(primary, messages)
            restored = _restore_math_content(translated, replacements)
            if same_text_guard_enabled and _is_untranslated_echo(text, restored):
                logger.warning("same-text retry guard triggered for provider=%s", primary.id)
                raise TranslationError("translation appears unchanged from source")
            return restored, primary.id, None
        except Exception as exc:  # noqa: BLE001
            last_error = exc

    if backup is not None:
        for _ in range(attempts):
            try:
                translated = _chat_completion(backup, messages)
                restored = _restore_math_content(translated, replacements)
                if same_text_guard_enabled and _is_untranslated_echo(text, restored):
                    logger.warning("same-text retry guard triggered for provider=%s", backup.id)
                    raise TranslationError("translation appears unchanged from source")
                return restored, backup.id, primary.id
            except Exception as exc:  # noqa: BLE001
                last_error = exc

    raise TranslationError(str(last_error) if last_error else "translation failed")


def translate_batch_with_fallback(
    texts: list[str],
    primary: ProviderRuntime,
    backup: ProviderRuntime | None,
    style_profile: str,
    glossary: list[str] | None = None,
    max_retries: int = 1,
) -> tuple[list[str], str, str | None]:
    if not texts:
        return [], primary.id, None

    settings = get_settings()
    same_text_guard_enabled = not bool(settings.disable_same_text_retry_guard)
    attempts = max(1, int(max_retries))
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            translated = _translate_batch_once(texts, primary, style_profile=style_profile, glossary=glossary)
            if same_text_guard_enabled and _has_excessive_untranslated_batch(texts, translated):
                logger.warning(
                    "same-text retry guard triggered for batch provider=%s count=%s",
                    primary.id,
                    len(texts),
                )
                raise TranslationError("batch translation appears largely unchanged from source")
            return translated, primary.id, None
        except Exception as exc:  # noqa: BLE001
            last_error = exc

    if backup is not None:
        for _ in range(attempts):
            try:
                translated = _translate_batch_once(texts, backup, style_profile=style_profile, glossary=glossary)
                if same_text_guard_enabled and _has_excessive_untranslated_batch(texts, translated):
                    logger.warning(
                        "same-text retry guard triggered for batch provider=%s count=%s",
                        backup.id,
                        len(texts),
                    )
                    raise TranslationError("batch translation appears largely unchanged from source")
                return translated, backup.id, primary.id
            except Exception as exc:  # noqa: BLE001
                last_error = exc

    raise TranslationError(str(last_error) if last_error else "batch translation failed")
