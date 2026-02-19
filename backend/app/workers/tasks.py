from __future__ import annotations

from datetime import UTC, datetime

from celery.exceptions import SoftTimeLimitExceeded

from app.core.settings import get_settings
from app.services.events import publish_event
from app.services.pdf_pipeline import PageProcessResult, translate_page_to_image
from app.services.session_store import SessionStore
from app.services.translator import ProviderRuntime
from app.workers.celery_app import celery_app


@celery_app.task(name="app.workers.tasks.translate_page_task")
def translate_page_task(session_id: str, page_no: int) -> dict[str, str | int]:
    store = SessionStore()
    settings = get_settings()

    meta = store.get_meta(session_id)
    if not meta:
        return {"session_id": session_id, "page_no": page_no, "status": "session_not_found"}

    providers = store.get_providers(session_id)
    if not providers:
        store.update_page_state(session_id, page_no, status="failed", error="missing provider runtime")
        publish_event(session_id, "page_failed", {"page_no": page_no, "reason": "missing_provider_runtime"})
        return {"session_id": session_id, "page_no": page_no, "status": "failed"}

    primary = ProviderRuntime(**providers["primary_provider"])
    backup = ProviderRuntime(**providers["backup_provider"]) if providers.get("backup_provider") else None
    style_profile: str = providers.get("style_profile", "academic_conservative")
    glossary = providers.get("glossary", [])

    existing_state = store.get_page_state(session_id, page_no)
    if existing_state and existing_state.status == "ready":
        store.clear_page_enqueued(session_id, page_no)
        return {"session_id": session_id, "page_no": page_no, "status": "already_ready"}

    try:
        store.update_page_state(session_id, page_no, status="processing")
        publish_event(session_id, "progress", {"page_no": page_no, "status": "processing"})

        paths = store.paths(session_id)
        translated_path = paths.translated_dir / f"{page_no}.png"
        page_result = translate_page_to_image(
            session_id=session_id,
            source_pdf=paths.source_pdf,
            output_path=translated_path,
            page_no=page_no,
            primary_provider=primary,
            backup_provider=backup,
            style_profile=style_profile,
            glossary=glossary,
            max_retries=settings.max_retries_per_provider,
        )
        _merge_report(session_id, page_result)

        store.update_page_state(session_id, page_no, status="ready", warnings=page_result.warnings)
        publish_event(
            session_id,
            "page_ready",
            {
                "page_no": page_no,
                "warnings": page_result.warnings,
                "ts": datetime.now(UTC).isoformat(),
            },
        )

    except SoftTimeLimitExceeded:
        reason = f"page timeout: exceeded {settings.task_soft_time_limit_sec}s"
        store.update_page_state(session_id, page_no, status="failed", error=reason)
        publish_event(session_id, "page_failed", {"page_no": page_no, "reason": reason})

    except Exception as exc:  # noqa: BLE001
        store.update_page_state(session_id, page_no, status="failed", error=str(exc))
        publish_event(session_id, "page_failed", {"page_no": page_no, "reason": str(exc)})
    finally:
        store.clear_page_enqueued(session_id, page_no)

    final_status = store.finalize_if_complete(session_id)
    if final_status == "ready":
        publish_event(session_id, "session_ready", {"session_id": session_id})
    elif final_status == "failed":
        publish_event(session_id, "session_failed", {"session_id": session_id})

    latest_state = store.get_page_state(session_id, page_no)
    latest_status = latest_state.status if latest_state else "unknown"
    return {"session_id": session_id, "page_no": page_no, "status": latest_status}


def enqueue_translation_jobs(session_id: str, page_count: int) -> list[int]:
    priority = list(range(1, min(3, page_count) + 1))
    for page_no in priority:
        enqueue_page_if_needed(session_id, page_no)
    for page_no in range(len(priority) + 1, page_count + 1):
        enqueue_page_if_needed(session_id, page_no)

    return priority


def ensure_pages_enqueued(session_id: str, page_count: int, center_page: int, window: int = 1) -> list[int]:
    start = max(1, center_page - window)
    end = min(page_count, center_page + window)
    queued: list[int] = []
    for page_no in range(start, end + 1):
        if enqueue_page_if_needed(session_id, page_no):
            queued.append(page_no)
    return queued


def enqueue_page_if_needed(session_id: str, page_no: int, *, force: bool = False) -> bool:
    store = SessionStore()
    settings = get_settings()
    state = store.get_page_state(session_id, page_no)
    if not state:
        return False

    if state.status in {"ready", "processing"}:
        return False
    if not force and store.is_page_enqueued(session_id, page_no):
        return False

    if state.status == "failed":
        store.update_page_state(session_id, page_no, status="pending", error=None, warnings=state.warnings)
    if not store.mark_page_enqueued(session_id, page_no):
        return False

    queue_name = settings.high_priority_queue if page_no <= 3 else settings.normal_priority_queue
    translate_page_task.apply_async(args=[session_id, page_no], queue=queue_name)
    return True


def _merge_report(session_id: str, page_result: PageProcessResult) -> None:
    store = SessionStore()
    store.append_report_items(
        session_id,
        {
            "layout_overflow": page_result.overflow_items,
            "untranslated_blocks": page_result.untranslated_items,
            "fallback_events": page_result.fallback_events,
            "image_ocr_failures": page_result.image_ocr_failures,
        },
    )
    for item in page_result.fallback_events:
        publish_event(
            session_id,
            "provider_switched",
            {
                "page_no": item["page_no"],
                "from_provider": item["from_provider"],
                "to_provider": item["to_provider"],
                "reason": item["reason"],
            },
        )
