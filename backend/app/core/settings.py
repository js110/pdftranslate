from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "pdf-translate-online"
    api_prefix: str = "/v1"
    redis_url: str = "redis://redis:6379/0"
    storage_root: Path = Field(default=Path("/tmp/pdftranslate/sessions"))
    session_ttl_minutes: int = 120
    max_upload_mb: int = 30
    render_dpi: int = 160
    high_priority_queue: str = "page_high"
    normal_priority_queue: str = "page_normal"
    worker_concurrency: int = 4
    max_retries_per_provider: int = 1
    batch_segment_size: int = 36
    batch_segment_char_limit: int = 8000
    translation_temperature: float = 0.2
    enable_single_block_fallback: bool = False
    enable_strict_low_quality_retry: bool = True
    clear_translation_memory_on_retry: bool = True
    enhance_translation_compatibility: bool = False
    compat_batch_segment_size: int = 18
    compat_batch_segment_char_limit: int = 3600
    drop_low_quality_cache_on_read: bool = True
    disable_same_text_retry_guard: bool = False
    glossary_max_terms: int = 12
    glossary_first_chunk_only: bool = True
    task_soft_time_limit_sec: int = 180
    task_time_limit_sec: int = 240
    enable_image_ocr: bool = False
    enable_layout_detection_guard: bool = True
    layout_detection_lang: str = "en"
    enable_grobid_reference_guard: bool = False
    grobid_base_url: str | None = None
    grobid_timeout_sec: int = 30


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    if settings.enhance_translation_compatibility:
        settings.enable_single_block_fallback = True
        settings.enable_strict_low_quality_retry = True
        settings.clear_translation_memory_on_retry = True
        settings.batch_segment_size = max(1, min(settings.batch_segment_size, settings.compat_batch_segment_size))
        settings.batch_segment_char_limit = max(
            1200,
            min(settings.batch_segment_char_limit, settings.compat_batch_segment_char_limit),
        )
    settings.storage_root.mkdir(parents=True, exist_ok=True)
    return settings
