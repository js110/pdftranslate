import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

IGNORE_DIRS = {
    "__pycache__",
    ".git",
    ".idea",
    ".vscode",
    ".pytest_cache",
    ".mypy_cache",
}


def _parse_csv_env(name: str, default: str) -> list[str]:
    raw = os.getenv(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


def _watch_snapshot(paths: list[Path], extensions: set[str]) -> dict[str, int]:
    snap: dict[str, int] = {}
    for base in paths:
        if not base.exists():
            continue
        if base.is_file():
            if base.suffix in extensions:
                try:
                    snap[str(base)] = base.stat().st_mtime_ns
                except OSError:
                    pass
            continue

        for root, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]
            root_path = Path(root)
            for name in filenames:
                file_path = root_path / name
                if file_path.suffix not in extensions:
                    continue
                try:
                    snap[str(file_path)] = file_path.stat().st_mtime_ns
                except OSError:
                    continue
    return snap


def _start_worker(cmd: list[str]) -> subprocess.Popen:
    print(f"[worker-reload] starting: {' '.join(cmd)}", flush=True)
    return subprocess.Popen(cmd)


def _stop_worker(proc: subprocess.Popen, timeout: float = 12.0) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=timeout)
        return
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def main() -> int:
    command = os.getenv(
        "CELERY_CMD",
        "celery -A celery_worker.celery_app worker -Q page_high,page_normal --concurrency=4 --loglevel=info",
    )
    watch_paths = [Path(p) for p in _parse_csv_env("WATCH_PATHS", "/app/app,/app/celery_worker.py")]
    watch_extensions = set(_parse_csv_env("WATCH_EXTENSIONS", ".py"))
    poll_interval = float(os.getenv("RELOAD_POLL_INTERVAL", "1.0"))

    cmd = shlex.split(command)
    if not cmd:
        print("[worker-reload] CELERY_CMD is empty", flush=True)
        return 1

    worker = _start_worker(cmd)
    last = _watch_snapshot(watch_paths, watch_extensions)

    try:
        while True:
            time.sleep(poll_interval)
            current = _watch_snapshot(watch_paths, watch_extensions)

            if worker.poll() is not None:
                print("[worker-reload] worker exited, restarting", flush=True)
                worker = _start_worker(cmd)
                last = current
                continue

            if current != last:
                print("[worker-reload] source changed, restarting worker", flush=True)
                _stop_worker(worker)
                worker = _start_worker(cmd)
                last = current
    except KeyboardInterrupt:
        pass
    finally:
        _stop_worker(worker)

    return 0


if __name__ == "__main__":
    sys.exit(main())
