# PDF Translate Online (Left/Right Reader)

A local Docker-deployed web app for translating English scientific PDFs into Chinese with a split-screen reader.

## Features

- Single PDF upload session
- Left original page / right translated page (strong scroll sync)
- Priority translation for first 3 pages, rest in background
- BYOK model configuration (OpenAI + OpenAI-compatible)
- Automatic fallback provider switching
- OCR + redraw translation for image/chart text (best effort)
- SSE real-time page progress events
- Session-scoped data with auto cleanup

## Run with Docker

```bash
docker compose up --build
```

Open web UI at `http://localhost:5173`, API at `http://localhost:8000`.

## Hot Reload (Dev)

Use the dev override file to enable hot reload for API, worker, and web:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d --build redis api worker web-dev
```

Behavior in dev mode:

- `api`: runs `uvicorn --reload` with `./backend` mounted into container
- `worker`: restarts Celery automatically when Python files change
- `web-dev`: runs Vite dev server with HMR and polling enabled for Docker Desktop

## API Summary

- `POST /v1/sessions`
- `POST /v1/sessions/{session_id}/start`
- `GET /v1/sessions/{session_id}/state`
- `GET /v1/sessions/{session_id}/events`
- `GET /v1/sessions/{session_id}/pages/{page_no}/original.png`
- `GET /v1/sessions/{session_id}/pages/{page_no}/translated.png`
- `GET /v1/sessions/{session_id}/report`
- `POST /v1/sessions/{session_id}/pages/{page_no}/retry`
- `DELETE /v1/sessions/{session_id}`

## Notes

- The app does not provide file download output by design.
- Session data is temporary and cleaned when expired or manually deleted.
- OCR quality depends on source image clarity and runtime environment.
