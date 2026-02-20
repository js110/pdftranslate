@echo off
setlocal

docker version >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Docker Desktop is not installed or not running.
  pause
  exit /b 1
)

echo Stopping PDF Translate services...
docker compose down
if errorlevel 1 (
  echo [ERROR] Failed to stop services.
  pause
  exit /b 1
)

echo Services stopped.
