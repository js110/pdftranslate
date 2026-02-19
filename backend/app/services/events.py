from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any, AsyncIterator

import orjson

from app.core.redis_client import get_async_redis, get_redis
from app.schemas.session import EventMessage


def event_channel(session_id: str) -> str:
    return f"session:{session_id}:events"


def publish_event(session_id: str, event: str, payload: dict[str, Any] | None = None) -> None:
    message = EventMessage(
        event=event,  # type: ignore[arg-type]
        session_id=session_id,
        payload=payload or {},
        ts=datetime.now(UTC),
    )
    raw = orjson.dumps(message.model_dump(mode="json")).decode("utf-8")
    redis = get_redis()
    redis.publish(event_channel(session_id), raw)


async def sse_stream(session_id: str) -> AsyncIterator[str]:
    redis = get_async_redis()
    pubsub = redis.pubsub(ignore_subscribe_messages=True)
    await pubsub.subscribe(event_channel(session_id))

    try:
        yield "retry: 3000\n\n"
        while True:
            message = await pubsub.get_message(timeout=5.0)
            if message and message.get("type") == "message":
                yield f"data: {message['data']}\n\n"
            else:
                yield ": ping\n\n"
            await asyncio.sleep(0.2)
    finally:
        await pubsub.unsubscribe(event_channel(session_id))
        await pubsub.close()
