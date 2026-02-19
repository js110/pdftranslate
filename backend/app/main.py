from __future__ import annotations

import asyncio

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.sessions import router as sessions_router
from app.core.settings import get_settings
from app.services.cleanup import cleanup_loop

settings = get_settings()

app = FastAPI(title=settings.app_name)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(sessions_router, prefix=settings.api_prefix)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.on_event("startup")
async def on_startup() -> None:
    app.state.cleanup_stop_event = asyncio.Event()
    app.state.cleanup_task = asyncio.create_task(cleanup_loop(app.state.cleanup_stop_event))


@app.on_event("shutdown")
async def on_shutdown() -> None:
    app.state.cleanup_stop_event.set()
    await app.state.cleanup_task
