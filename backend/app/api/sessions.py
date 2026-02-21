from __future__ import annotations

from datetime import datetime
from pathlib import Path

import fitz
from fastapi import APIRouter, File, HTTPException, Response, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from PIL import Image

from app.core.settings import get_settings
from app.schemas.session import (
    CreateSessionResponse,
    EnsurePageRequest,
    EnsurePageResponse,
    ErrorResponse,
    QualityReport,
    RetryPageResponse,
    SaveResultPdfResponse,
    SessionState,
    StartSessionRequest,
    StartSessionResponse,
)
from app.services.events import sse_stream
from app.services.pdf_pipeline import extract_glossary_terms, render_original_pages
from app.services.session_store import SessionStore
from app.workers.tasks import enqueue_page_if_needed, enqueue_translation_jobs, ensure_pages_enqueued

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.post("", response_model=CreateSessionResponse, responses={400: {"model": ErrorResponse}})
async def create_session(file: UploadFile = File(...)) -> CreateSessionResponse:
    settings = get_settings()
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only PDF files are supported")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty file")

    size_mb = len(content) / (1024 * 1024)
    if size_mb > settings.max_upload_mb:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File exceeds max size of {settings.max_upload_mb}MB",
        )

    try:
        with fitz.open(stream=content, filetype="pdf") as doc:
            page_count = doc.page_count
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid PDF: {exc}") from exc

    store = SessionStore()
    meta = store.create_session(page_count=page_count)
    session_id = meta["session_id"]
    paths = store.paths(session_id)
    paths.source_pdf.write_bytes(content)

    rendered_count = render_original_pages(paths.source_pdf, paths.original_dir)
    if rendered_count != page_count:
        raise HTTPException(status_code=500, detail="Failed to render original pages")

    return CreateSessionResponse(
        session_id=session_id,
        page_count=page_count,
        expires_at=datetime.fromisoformat(meta["expires_at"]),
    )


@router.post("/{session_id}/start", response_model=StartSessionResponse, responses={404: {"model": ErrorResponse}})
def start_session(session_id: str, request: StartSessionRequest) -> StartSessionResponse:
    settings = get_settings()
    store = SessionStore()
    meta = store.get_meta(session_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Session not found")
    page_count = int(meta["page_count"])
    priority = list(range(1, min(3, page_count) + 1))
    if meta["overall_status"] in {"running", "ready"}:
        return StartSessionResponse(status=meta["overall_status"], priority_pages=priority)

    paths = store.paths(session_id)
    if not paths.source_pdf.exists():
        raise HTTPException(status_code=404, detail="Session file missing")

    glossary = extract_glossary_terms(paths.source_pdf, max_terms=settings.glossary_max_terms)
    store.save_providers(
        session_id,
        {
            "primary_provider": request.primary_provider.model_dump(),
            "backup_provider": request.backup_provider.model_dump() if request.backup_provider else None,
            "style_profile": request.style_profile,
            "glossary": glossary,
        },
    )
    store.mark_running(session_id)

    priority = enqueue_translation_jobs(session_id=session_id, page_count=page_count)
    return StartSessionResponse(status="running", priority_pages=priority)


@router.get("/{session_id}/state", response_model=SessionState, responses={404: {"model": ErrorResponse}})
def get_session_state(session_id: str) -> SessionState:
    store = SessionStore()
    meta = store.get_meta(session_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Session not found")

    pages = store.list_page_states(session_id)
    warnings_count = sum(len(page.warnings) for page in pages)
    return SessionState(
        session_id=session_id,
        overall_status=meta["overall_status"],
        page_count=int(meta["page_count"]),
        page_states=pages,
        first_readable_ready=store.is_first_readable_ready(session_id),
        warnings_count=warnings_count,
        created_at=datetime.fromisoformat(meta["created_at"]),
        expires_at=datetime.fromisoformat(meta["expires_at"]),
    )


@router.get("/{session_id}/events")
async def get_session_events(session_id: str) -> StreamingResponse:
    store = SessionStore()
    if not store.get_meta(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    return StreamingResponse(sse_stream(session_id), media_type="text/event-stream")


@router.get("/{session_id}/pages/{page_no}/original.png")
def get_original_page(session_id: str, page_no: int) -> FileResponse:
    store = SessionStore()
    meta = store.get_meta(session_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Session not found")
    if page_no < 1 or page_no > int(meta["page_count"]):
        raise HTTPException(status_code=404, detail="Page not found")
    file_path = store.paths(session_id).original_dir / f"{page_no}.png"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Page not found")
    return FileResponse(file_path)


@router.get("/{session_id}/pages/{page_no}/translated.png")
def get_translated_page(session_id: str, page_no: int):
    store = SessionStore()
    meta = store.get_meta(session_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Session not found")
    if page_no < 1 or page_no > int(meta["page_count"]):
        raise HTTPException(status_code=404, detail="Page not found")
    file_path = store.paths(session_id).translated_dir / f"{page_no}.png"
    if not file_path.exists():
        return JSONResponse(status_code=202, content={"status": "pending", "page_no": page_no})
    return FileResponse(
        file_path,
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@router.post(
    "/{session_id}/pages/{page_no}/ensure",
    response_model=EnsurePageResponse,
    responses={404: {"model": ErrorResponse}},
)
def ensure_page(
    session_id: str,
    page_no: int,
    request: EnsurePageRequest,
) -> EnsurePageResponse:
    store = SessionStore()
    meta = store.get_meta(session_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Session not found")

    page_count = int(meta["page_count"])
    if page_no < 1 or page_no > page_count:
        raise HTTPException(status_code=404, detail="Page not found")
    if meta["overall_status"] == "created":
        raise HTTPException(status_code=409, detail="Session not started")

    queued = ensure_pages_enqueued(
        session_id=session_id,
        page_count=page_count,
        center_page=page_no,
        window=request.window,
    )
    return EnsurePageResponse(
        session_id=session_id,
        requested_page=page_no,
        window=request.window,
        queued_pages=queued,
    )


@router.get("/{session_id}/report", response_model=QualityReport, responses={404: {"model": ErrorResponse}})
def get_report(session_id: str) -> QualityReport:
    store = SessionStore()
    if not store.get_meta(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return store.get_report(session_id)


@router.post("/{session_id}/pages/{page_no}/retry", response_model=RetryPageResponse)
def retry_page(session_id: str, page_no: int) -> RetryPageResponse:
    settings = get_settings()
    store = SessionStore()
    meta = store.get_meta(session_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Session not found")

    page_count = int(meta["page_count"])
    if page_no < 1 or page_no > page_count:
        raise HTTPException(status_code=404, detail="Page not found")
    if meta["overall_status"] == "created":
        raise HTTPException(status_code=409, detail="Session not started")

    # Remove old diagnostics for this page so report reflects latest retry result.
    store.clear_report_items_for_page(session_id, page_no)
    # Keep translation memory by default for speed/cost; allow explicit opt-out.
    if settings.clear_translation_memory_on_retry:
        store.clear_translation_memory(session_id)

    store.update_page_state(session_id, page_no, status="pending", error=None)
    queued = enqueue_page_if_needed(session_id=session_id, page_no=page_no, force=True)
    return RetryPageResponse(session_id=session_id, page_no=page_no, status="queued" if queued else "skipped")


@router.post("/{session_id}/save-result-pdf", response_model=SaveResultPdfResponse)
def save_result_pdf(session_id: str) -> SaveResultPdfResponse:
    settings = get_settings()
    store = SessionStore()
    meta = store.get_meta(session_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Session not found")

    page_count = int(meta["page_count"])
    paths = store.paths(session_id)
    export_dir = settings.storage_root / "_saved_results"
    export_dir.mkdir(parents=True, exist_ok=True)
    out_path = export_dir / f"{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

    images: list[Image.Image] = []
    translated_pages = 0
    try:
        for page_no in range(1, page_count + 1):
            translated_path = paths.translated_dir / f"{page_no}.png"
            source_path: Path
            if translated_path.exists():
                source_path = translated_path
                translated_pages += 1
            else:
                source_path = paths.original_dir / f"{page_no}.png"
            if not source_path.exists():
                continue
            with Image.open(source_path) as img:
                images.append(img.convert("RGB"))

        if not images:
            raise HTTPException(status_code=500, detail="No page images available for PDF export")

        first, *rest = images
        first.save(out_path, "PDF", resolution=150.0, save_all=True, append_images=rest)
    finally:
        for img in images:
            img.close()

    return SaveResultPdfResponse(
        session_id=session_id,
        saved_path=out_path.as_posix(),
        page_count=page_count,
        translated_pages=translated_pages,
    )


@router.delete("/{session_id}", status_code=204)
def delete_session(session_id: str) -> Response:
    store = SessionStore()
    if not store.get_meta(session_id):
        return Response(status_code=204)
    store.cleanup_session(session_id)
    return Response(status_code=204)
