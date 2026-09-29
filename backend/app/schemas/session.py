from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


PageStatus = Literal["pending", "processing", "ready", "failed"]
SessionStatus = Literal["created", "running", "ready", "failed", "expired", "deleted"]


class ProviderConfigIn(BaseModel):
    id: str = Field(min_length=1, max_length=64)
    model: str = Field(min_length=1, max_length=128)
    api_key: str = Field(min_length=8)
    base_url: str | None = None
    timeout_sec: int = Field(default=60, ge=10, le=300)


class ProviderConfigPublic(BaseModel):
    id: str
    model: str
    base_url: str | None = None
    timeout_sec: int = 60


class CreateSessionResponse(BaseModel):
    session_id: str
    page_count: int
    expires_at: datetime


class StartSessionRequest(BaseModel):
    primary_provider: ProviderConfigIn
    backup_provider: ProviderConfigIn | None = None
    style_profile: str = "academic_conservative"


class StartSessionResponse(BaseModel):
    status: SessionStatus
    priority_pages: list[int]


class PageState(BaseModel):
    page_no: int
    status: PageStatus
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None
    updated_at: datetime


class SessionState(BaseModel):
    session_id: str
    overall_status: SessionStatus
    page_count: int
    page_states: list[PageState]
    first_readable_ready: bool
    warnings_count: int
    created_at: datetime
    expires_at: datetime


class LayoutOverflowItem(BaseModel):
    page_no: int
    bbox: list[float]
    reason: str


class FallbackEventItem(BaseModel):
    page_no: int
    from_provider: str
    to_provider: str
    reason: str


class UntranslatedBlockItem(BaseModel):
    page_no: int
    bbox: list[float]
    source_excerpt: str
    reason: str


class QualityReport(BaseModel):
    layout_overflow: list[LayoutOverflowItem] = Field(default_factory=list)
    fallback_events: list[FallbackEventItem] = Field(default_factory=list)
    untranslated_blocks: list[UntranslatedBlockItem] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    detail: str


class EventMessage(BaseModel):
    event: Literal[
        "page_ready",
        "page_failed",
        "provider_switched",
        "session_expiring",
        "session_ready",
        "session_failed",
        "progress",
        "rewrite_progress",
        "rewrite_ready",
        "rewrite_failed",
    ]
    session_id: str
    payload: dict[str, Any] = Field(default_factory=dict)
    ts: datetime


RewriteStatus = Literal["idle", "running", "ready", "failed"]


class RewriteState(BaseModel):
    session_id: str
    status: RewriteStatus = "idle"
    stage: str | None = None
    done: int = 0
    total: int = 0
    error: str | None = None
    updated_at: float | None = None

    def running_is_stale(self, now: float, limit_sec: float) -> bool:
        """True when status is 'running' but no heartbeat arrived within the
        task time limit — the worker died (hard kill / broker loss) and the
        rewrite must be restartable instead of wedged at 409 forever."""
        if self.status != "running":
            return False
        if self.updated_at is None:
            return True
        return now - self.updated_at > limit_sec


class RetryPageResponse(BaseModel):
    session_id: str
    page_no: int
    status: str


class EnsurePageRequest(BaseModel):
    window: int = Field(default=1, ge=0, le=3)


class EnsurePageResponse(BaseModel):
    session_id: str
    requested_page: int
    window: int
    queued_pages: list[int] = Field(default_factory=list)


class SaveResultPdfResponse(BaseModel):
    session_id: str
    saved_path: str
    page_count: int
    translated_pages: int
