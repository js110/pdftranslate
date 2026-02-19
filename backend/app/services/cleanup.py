from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import orjson

from app.core.redis_client import get_redis
from app.services.events import publish_event
from app.services.session_store import SessionStore


async def cleanup_loop(stop_event: asyncio.Event) -> None:
    store = SessionStore()
    redis = get_redis()

    while not stop_event.is_set():
        now = datetime.now(UTC)

        for key in redis.scan_iter(match="session:*:meta"):
            raw = redis.get(key)
            if not raw:
                continue
            meta = orjson.loads(raw)
            session_id = meta.get("session_id")
            expires_at_raw = meta.get("expires_at")
            if not session_id or not expires_at_raw:
                continue
            expires_at = datetime.fromisoformat(expires_at_raw)
            if expires_at <= now:
                store.cleanup_session(session_id, status="expired")
                continue

            if expires_at <= now + timedelta(minutes=5):
                notice_key = f"session:{session_id}:expiring_notice"
                if redis.setnx(notice_key, "1"):
                    redis.expire(notice_key, 600)
                    publish_event(
                        session_id,
                        "session_expiring",
                        {"session_id": session_id, "expires_at": expires_at.isoformat()},
                    )

        await asyncio.sleep(30)
