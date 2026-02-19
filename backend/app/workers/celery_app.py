from __future__ import annotations

from celery import Celery

from app.core.settings import get_settings

settings = get_settings()

celery_app = Celery(
    "pdf_translate",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["app.workers.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_soft_time_limit=settings.task_soft_time_limit_sec,
    task_time_limit=settings.task_time_limit_sec,
    timezone="UTC",
    task_routes={
        "app.workers.tasks.translate_page_task": {"queue": settings.normal_priority_queue},
    },
)
