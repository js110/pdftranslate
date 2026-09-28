"""Image-block OCR (local PaddleOCR + remote PaddleOCR cloud API)."""
from __future__ import annotations

import io
import json
import logging
import time
from typing import Any

import httpx
import numpy as np
from PIL import Image, ImageDraw

from app.core.settings import get_settings
from app.services.jobtypes import PageProcessResult
from app.services.promptprep import _should_translate_text_block
from app.services.render import _fit_text_to_box
from app.services.translator import (
    ProviderRuntime,
    TranslationError,
    translate_with_fallback,
)

logger = logging.getLogger(__name__)


try:
    from paddleocr import PaddleOCR
except Exception:  # noqa: BLE001
    PaddleOCR = None  # type: ignore[misc,assignment]


_ocr_instance: Any = None

_remote_ocr_client: httpx.Client | None = None



def get_ocr() -> Any:
    global _ocr_instance
    if not get_settings().enable_image_ocr:
        _ocr_instance = False
        return _ocr_instance
    if _ocr_instance is not None:
        return _ocr_instance
    if PaddleOCR is None:
        _ocr_instance = False
        return _ocr_instance
    try:
        _ocr_instance = PaddleOCR(use_textline_orientation=True, lang="en")
    except Exception:  # noqa: BLE001
        _ocr_instance = False
    return _ocr_instance



def _process_image_block(
    *,
    block: dict[str, Any],
    image: Image.Image,
    result: PageProcessResult,
    zoom: float,
    page_no: int,
    primary_provider: ProviderRuntime,
    backup_provider: ProviderRuntime | None,
    style_profile: str,
    glossary: list[str] | None,
    max_retries: int,
) -> None:
    settings = get_settings()
    if not settings.enable_image_ocr and not settings.enable_remote_image_ocr:
        return

    bbox = block.get("bbox", [0, 0, 0, 0])
    x0, y0, x1, y1 = [int(float(v) * zoom) for v in bbox]
    if x1 <= x0 or y1 <= y0:
        return

    crop = image.crop((x0, y0, x1, y1))
    image_index = int(block.get("number", -1))
    lines: list[dict[str, Any]] = []

    if settings.enable_image_ocr:
        ocr = get_ocr()
        if ocr:
            crop_np = np.array(crop)
            try:
                ocr_result = ocr.predict(crop_np)
                lines = _normalize_ocr_result(ocr_result)
            except Exception as exc:  # noqa: BLE001
                result.image_ocr_failures.append(
                    {
                        "page_no": page_no,
                        "image_index": image_index,
                        "reason": f"ocr_failed: {exc}",
                    }
                )

    if not lines and settings.enable_remote_image_ocr:
        remote_text = _extract_remote_ocr_markdown_text(
            crop=crop,
            page_no=page_no,
            image_index=image_index,
            result=result,
        )
        if remote_text:
            lines = [{"text": remote_text, "rect": [0, 0, crop.width, crop.height]}]

    if not lines:
        return

    overlay = crop.copy()
    overlay_draw = ImageDraw.Draw(overlay)

    for line in lines:
        text = line["text"].strip()
        if not _should_translate_text_block(text):
            continue

        lx0, ly0, lx1, ly1 = line["rect"]
        rect_w = max(1, lx1 - lx0)
        rect_h = max(1, ly1 - ly0)

        try:
            translated, used_provider, switched_from = translate_with_fallback(
                text=text,
                primary=primary_provider,
                backup=backup_provider,
                style_profile=style_profile,
                glossary=glossary,
                max_retries=max_retries,
            )
            if switched_from:
                result.fallback_events.append(
                    {
                        "page_no": page_no,
                        "from_provider": switched_from,
                        "to_provider": used_provider,
                        "reason": "primary provider failed on image text",
                    }
                )
        except TranslationError as exc:
            result.image_ocr_failures.append(
                {
                    "page_no": page_no,
                    "image_index": int(block.get("number", -1)),
                    "reason": f"image_text_translation_failed: {exc}",
                }
            )
            continue

        fit = _fit_text_to_box(translated, width=rect_w, height=rect_h, base_font_size=max(10.0, rect_h * 0.7), min_scale=0.7)
        if fit is None:
            result.image_ocr_failures.append(
                {
                    "page_no": page_no,
                    "image_index": int(block.get("number", -1)),
                    "reason": "image_text_overflow",
                }
            )
            continue

        txt_lines, txt_font, txt_line_height, _ = fit
        overlay_draw.rectangle([(lx0, ly0), (lx1, ly1)], fill=(255, 255, 255))
        ty = ly0
        for item in txt_lines:
            overlay_draw.text((lx0, ty), item, font=txt_font, fill=(0, 0, 0))
            ty += txt_line_height

    image.paste(overlay, (x0, y0))



def _extract_remote_ocr_markdown_text(
    *,
    crop: Image.Image,
    page_no: int,
    image_index: int,
    result: PageProcessResult,
) -> str | None:
    global _remote_ocr_client
    settings = get_settings()
    token = (settings.remote_ocr_token or "").strip()
    if not token:
        result.image_ocr_failures.append(
            {
                "page_no": page_no,
                "image_index": image_index,
                "reason": "remote_ocr_missing_token",
            }
        )
        return None

    optional_payload = {
        "useDocOrientationClassify": bool(settings.remote_ocr_use_doc_orientation_classify),
        "useDocUnwarping": bool(settings.remote_ocr_use_doc_unwarping),
        "useChartRecognition": bool(settings.remote_ocr_use_chart_recognition),
    }

    try:
        with io.BytesIO() as buff:
            crop.save(buff, format="PNG")
            image_bytes = buff.getvalue()

        headers = {"Authorization": f"bearer {token}"}
        timeout = max(5.0, float(settings.remote_ocr_timeout_sec))
        if _remote_ocr_client is None or _remote_ocr_client.is_closed:
            _remote_ocr_client = httpx.Client(
                timeout=timeout,
                limits=httpx.Limits(max_connections=16, max_keepalive_connections=8),
            )
        client = _remote_ocr_client
        job_url = settings.remote_ocr_job_url
        submit = client.post(
            job_url,
            headers=headers,
            data={
                "model": settings.remote_ocr_model,
                "optionalPayload": json.dumps(optional_payload, ensure_ascii=False),
            },
            files={"file": ("crop.png", image_bytes, "image/png")},
        )
        payload = _parse_remote_ocr_response(submit, context="submit_remote_ocr")
        job_id = str((payload.get("data") or {}).get("jobId") or "").strip()
        if not job_id:
            raise RuntimeError(f"remote OCR returned no jobId: {payload}")

        poll_interval = max(1, int(settings.remote_ocr_poll_interval_sec))
        deadline = time.monotonic() + max(10, int(settings.remote_ocr_max_wait_sec))
        jsonl_url = ""
        while True:
            if time.monotonic() > deadline:
                raise TimeoutError("remote OCR polling timed out")
            status_resp = client.get(f"{job_url.rstrip('/')}/{job_id}", headers=headers)
            status_payload = _parse_remote_ocr_response(status_resp, context="poll_remote_ocr")
            data = status_payload.get("data") or {}
            state = str(data.get("state") or "").strip().lower()
            if state == "done":
                jsonl_url = str(((data.get("resultUrl") or {}).get("jsonUrl")) or "").strip()
                break
            if state == "failed":
                raise RuntimeError(str(data.get("errorMsg") or "remote OCR job failed"))
            time.sleep(poll_interval)

        if not jsonl_url:
            raise RuntimeError("remote OCR completed without result URL")
        jsonl_resp = client.get(jsonl_url)
        jsonl_resp.raise_for_status()
        extracted = _extract_markdown_text_from_jsonl(jsonl_resp.text)
        return extracted or None

    except Exception as exc:  # noqa: BLE001
        result.image_ocr_failures.append(
            {
                "page_no": page_no,
                "image_index": image_index,
                "reason": f"remote_ocr_failed: {exc}",
            }
        )
        return None



def _parse_remote_ocr_response(resp: httpx.Response, *, context: str) -> dict[str, Any]:
    if resp.status_code != 200:
        preview = (resp.text or "")[:600]
        raise RuntimeError(f"{context} http_{resp.status_code}: {preview}")
    try:
        payload = resp.json()
    except Exception as exc:  # noqa: BLE001
        preview = (resp.text or "")[:600]
        raise RuntimeError(f"{context} non_json_response: {preview}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{context} response is not a JSON object")
    return payload



def _extract_markdown_text_from_jsonl(raw_jsonl: str) -> str:
    parts: list[str] = []
    for line in raw_jsonl.splitlines():
        item = line.strip()
        if not item:
            continue
        try:
            payload = json.loads(item)
        except Exception:  # noqa: BLE001
            continue
        result = payload.get("result") or {}
        layouts = result.get("layoutParsingResults") or []
        for layout in layouts:
            markdown = (layout.get("markdown") or {}).get("text") or ""
            text = str(markdown).strip()
            if text:
                parts.append(text)
    return "\n\n".join(parts).strip()



def _normalize_ocr_result(raw: Any) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []

    if isinstance(raw, list):
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            rec_texts = entry.get("rec_texts") or []
            rec_polys = entry.get("rec_polys") or []
            for idx, text in enumerate(rec_texts):
                poly = rec_polys[idx] if idx < len(rec_polys) else None
                if poly is None:
                    continue
                xs = [point[0] for point in poly]
                ys = [point[1] for point in poly]
                normalized.append(
                    {
                        "text": str(text),
                        "rect": [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))],
                    }
                )

    return normalized


