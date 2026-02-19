from __future__ import annotations

import shutil
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import orjson

from app.core.redis_client import get_redis
from app.core.settings import get_settings
from app.schemas.session import PageState, QualityReport


@dataclass(frozen=True)
class SessionPaths:
    root: Path
    source_pdf: Path
    original_dir: Path
    translated_dir: Path


def _now() -> datetime:
    return datetime.now(UTC)


def _dumps(data: Any) -> str:
    return orjson.dumps(data).decode("utf-8")


def _loads(raw: str | None, default: Any) -> Any:
    if not raw:
        return default
    return orjson.loads(raw)


class SessionStore:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.redis = get_redis()

    def new_session_id(self) -> str:
        return uuid.uuid4().hex

    def paths(self, session_id: str) -> SessionPaths:
        root = self.settings.storage_root / session_id
        return SessionPaths(
            root=root,
            source_pdf=root / "source.pdf",
            original_dir=root / "original",
            translated_dir=root / "translated",
        )

    def create_session(self, page_count: int) -> dict[str, Any]:
        session_id = self.new_session_id()
        created_at = _now()
        expires_at = created_at + timedelta(minutes=self.settings.session_ttl_minutes)
        paths = self.paths(session_id)
        paths.original_dir.mkdir(parents=True, exist_ok=True)
        paths.translated_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "session_id": session_id,
            "overall_status": "created",
            "page_count": page_count,
            "created_at": created_at.isoformat(),
            "expires_at": expires_at.isoformat(),
        }
        pages = {
            str(page_no): {
                "page_no": page_no,
                "status": "pending",
                "warnings": [],
                "error": None,
                "updated_at": created_at.isoformat(),
            }
            for page_no in range(1, page_count + 1)
        }
        report = QualityReport().model_dump()

        pipe = self.redis.pipeline()
        pipe.set(self._meta_key(session_id), _dumps(meta), ex=self._ttl_seconds())
        pipe.hset(self._pages_key(session_id), mapping={k: _dumps(v) for k, v in pages.items()})
        pipe.expire(self._pages_key(session_id), self._ttl_seconds())
        pipe.set(self._report_key(session_id), _dumps(report), ex=self._ttl_seconds())
        pipe.execute()
        return meta

    def save_providers(self, session_id: str, providers: dict[str, Any]) -> None:
        self.redis.set(self._providers_key(session_id), _dumps(providers), ex=self._ttl_seconds())

    def get_providers(self, session_id: str) -> dict[str, Any] | None:
        raw = self.redis.get(self._providers_key(session_id))
        return _loads(raw, None)

    def mark_page_enqueued(self, session_id: str, page_no: int) -> bool:
        added = self.redis.sadd(self._enqueued_pages_key(session_id), str(page_no))
        self.redis.expire(self._enqueued_pages_key(session_id), self._ttl_seconds())
        return bool(added)

    def is_page_enqueued(self, session_id: str, page_no: int) -> bool:
        return bool(self.redis.sismember(self._enqueued_pages_key(session_id), str(page_no)))

    def clear_page_enqueued(self, session_id: str, page_no: int) -> None:
        self.redis.srem(self._enqueued_pages_key(session_id), str(page_no))
        self.redis.expire(self._enqueued_pages_key(session_id), self._ttl_seconds())

    def get_translation_memory_bulk(self, session_id: str, keys: list[str]) -> dict[str, str]:
        if not keys:
            return {}
        values = self.redis.hmget(self._translation_memory_key(session_id), keys)
        self.redis.expire(self._translation_memory_key(session_id), self._ttl_seconds())
        return {key: value for key, value in zip(keys, values, strict=False) if isinstance(value, str) and value}

    def set_translation_memory_bulk(self, session_id: str, mapping: dict[str, str]) -> None:
        if not mapping:
            return
        self.redis.hset(self._translation_memory_key(session_id), mapping=mapping)
        self.redis.expire(self._translation_memory_key(session_id), self._ttl_seconds())

    def clear_translation_memory(self, session_id: str) -> None:
        self.redis.delete(self._translation_memory_key(session_id))

    def delete_translation_memory_bulk(self, session_id: str, keys: list[str]) -> int:
        if not keys:
            return 0
        removed = int(self.redis.hdel(self._translation_memory_key(session_id), *keys))
        self.redis.expire(self._translation_memory_key(session_id), self._ttl_seconds())
        return removed

    def get_meta(self, session_id: str) -> dict[str, Any] | None:
        raw = self.redis.get(self._meta_key(session_id))
        return _loads(raw, None)

    def update_meta(self, session_id: str, **patch: Any) -> dict[str, Any]:
        meta = self.get_meta(session_id)
        if not meta:
            raise KeyError(session_id)
        meta.update(patch)
        self.redis.set(self._meta_key(session_id), _dumps(meta), ex=self._ttl_seconds())
        return meta

    def get_page_state(self, session_id: str, page_no: int) -> PageState | None:
        raw = self.redis.hget(self._pages_key(session_id), str(page_no))
        data = _loads(raw, None)
        return PageState.model_validate(data) if data else None

    def list_page_states(self, session_id: str) -> list[PageState]:
        items = self.redis.hgetall(self._pages_key(session_id))
        pages = [PageState.model_validate(orjson.loads(v)) for _, v in items.items()]
        pages.sort(key=lambda p: p.page_no)
        return pages

    def update_page_state(
        self,
        session_id: str,
        page_no: int,
        status: str,
        warnings: list[str] | None = None,
        error: str | None = None,
    ) -> PageState:
        page = self.get_page_state(session_id, page_no)
        if not page:
            raise KeyError(f"{session_id}:{page_no}")
        page.status = status  # type: ignore[misc]
        page.warnings = warnings if warnings is not None else page.warnings
        page.error = error
        page.updated_at = _now()  # type: ignore[misc]
        self.redis.hset(self._pages_key(session_id), str(page_no), _dumps(page.model_dump(mode="json")))
        self.redis.expire(self._pages_key(session_id), self._ttl_seconds())
        return page

    def append_warning(self, session_id: str, page_no: int, warning: str) -> PageState:
        page = self.get_page_state(session_id, page_no)
        if not page:
            raise KeyError(f"{session_id}:{page_no}")
        page.warnings.append(warning)
        page.updated_at = _now()  # type: ignore[misc]
        self.redis.hset(self._pages_key(session_id), str(page_no), _dumps(page.model_dump(mode="json")))
        return page

    def update_report(self, session_id: str, field: str, item: dict[str, Any]) -> None:
        self.append_report_items(session_id, {field: [item]})

    def append_report_items(self, session_id: str, updates: dict[str, list[dict[str, Any]]]) -> None:
        if not updates:
            return
        report = self.get_report(session_id).model_dump()
        changed = False
        for field, items in updates.items():
            if not items:
                continue
            report.setdefault(field, []).extend(items)
            changed = True
        if changed:
            self.redis.set(self._report_key(session_id), _dumps(report), ex=self._ttl_seconds())

    def get_report(self, session_id: str) -> QualityReport:
        raw = self.redis.get(self._report_key(session_id))
        data = _loads(raw, QualityReport().model_dump())
        return QualityReport.model_validate(data)

    def clear_report_items_for_page(self, session_id: str, page_no: int) -> None:
        report = self.get_report(session_id).model_dump()
        for key in ("layout_overflow", "untranslated_blocks", "image_ocr_failures", "fallback_events"):
            report[key] = [item for item in report.get(key, []) if int(item.get("page_no", -1)) != page_no]
        self.redis.set(self._report_key(session_id), _dumps(report), ex=self._ttl_seconds())

    def is_first_readable_ready(self, session_id: str) -> bool:
        meta = self.get_meta(session_id)
        if not meta:
            return False
        page_count = int(meta["page_count"])
        limit = min(3, page_count)
        for page_no in range(1, limit + 1):
            state = self.get_page_state(session_id, page_no)
            if not state or state.status in {"pending", "processing"}:
                return False
        return True

    def finalize_if_complete(self, session_id: str) -> str | None:
        pages = self.list_page_states(session_id)
        statuses = {page.status for page in pages}
        if statuses == {"ready"}:
            self.update_meta(session_id, overall_status="ready")
            return "ready"
        if "failed" in statuses and all(status in {"ready", "failed"} for status in statuses):
            self.update_meta(session_id, overall_status="failed")
            return "failed"
        return None

    def mark_running(self, session_id: str) -> None:
        self.update_meta(session_id, overall_status="running")

    def cleanup_session(self, session_id: str, status: str = "deleted") -> None:
        meta = self.get_meta(session_id)
        if meta:
            meta["overall_status"] = status
            self.redis.set(self._meta_key(session_id), _dumps(meta), ex=120)
        self.redis.delete(self._pages_key(session_id))
        self.redis.delete(self._report_key(session_id))
        self.redis.delete(self._providers_key(session_id))
        self.redis.delete(self._translation_memory_key(session_id))
        self.redis.delete(self._enqueued_pages_key(session_id))
        paths = self.paths(session_id)
        if paths.root.exists():
            shutil.rmtree(paths.root, ignore_errors=True)

    def cleanup_expired_sessions(self) -> list[str]:
        expired: list[str] = []
        now = _now()
        for key in self.redis.scan_iter(match="session:*:meta"):
            raw = self.redis.get(key)
            meta = _loads(raw, None)
            if not meta:
                continue
            expires_at = datetime.fromisoformat(meta["expires_at"])
            if expires_at <= now:
                session_id = meta["session_id"]
                self.cleanup_session(session_id, status="expired")
                expired.append(session_id)
        return expired

    @staticmethod
    def _meta_key(session_id: str) -> str:
        return f"session:{session_id}:meta"

    @staticmethod
    def _pages_key(session_id: str) -> str:
        return f"session:{session_id}:pages"

    @staticmethod
    def _providers_key(session_id: str) -> str:
        return f"session:{session_id}:providers"

    @staticmethod
    def _report_key(session_id: str) -> str:
        return f"session:{session_id}:report"

    @staticmethod
    def _translation_memory_key(session_id: str) -> str:
        return f"session:{session_id}:tm"

    @staticmethod
    def _enqueued_pages_key(session_id: str) -> str:
        return f"session:{session_id}:enqueued_pages"

    def _ttl_seconds(self) -> int:
        return self.settings.session_ttl_minutes * 60
