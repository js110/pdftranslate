FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    fonts-noto-cjk \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements-ocr.txt /app/
RUN pip install --upgrade pip && pip install -r requirements.txt
# Optional local PaddleOCR support (~GB-sized). Build with --build-arg INSTALL_OCR=true.
ARG INSTALL_OCR=false
RUN if [ "$INSTALL_OCR" = "true" ]; then pip install --no-cache-dir -r requirements-ocr.txt; fi

COPY app /app/app
COPY celery_worker.py /app/celery_worker.py

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
