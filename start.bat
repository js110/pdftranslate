@echo off
setlocal

docker version >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Docker Desktop is not installed or not running.
  echo Please install/start Docker Desktop, then run this file again.
  pause
  exit /b 1
)

echo Starting PDF Translate services...
docker compose up -d --build
if errorlevel 1 (
  echo [ERROR] Failed to start services.
  pause
  exit /b 1
)

echo.
echo Services are ready.
echo Web UI: http://localhost:5173
echo API:    http://localhost:8000
start "" "http://localhost:5173"
