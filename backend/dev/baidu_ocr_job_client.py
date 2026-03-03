import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

JOB_URL = "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs"
DEFAULT_MODEL = "PaddleOCR-VL-1.5"
DEFAULT_OPTIONAL_PAYLOAD = {
    "useDocOrientationClassify": False,
    "useDocUnwarping": False,
    "useChartRecognition": False,
}


def build_session(timeout_seconds: int, total_retries: int) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=total_retries,
        read=total_retries,
        connect=total_retries,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET", "POST"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.request = _with_timeout(session.request, timeout_seconds)
    return session


def _with_timeout(request_func, timeout_seconds: int):
    def wrapped(method: str, url: str, **kwargs):
        kwargs.setdefault("timeout", timeout_seconds)
        return request_func(method, url, **kwargs)

    return wrapped


def require_token(cli_token: str | None) -> str:
    token = cli_token or os.getenv("BAIDU_OCR_TOKEN")
    if token:
        return token
    raise RuntimeError(
        "Missing token. Set BAIDU_OCR_TOKEN env var or pass --token."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit OCR job to Baidu PaddleOCR API, poll status, and download markdown/images."
    )
    parser.add_argument(
        "file",
        help="Local file path or HTTP(S) file URL.",
    )
    parser.add_argument("--token", help="API token. Prefer BAIDU_OCR_TOKEN env var.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model name (default: {DEFAULT_MODEL}).")
    parser.add_argument("--job-url", default=JOB_URL, help=f"Job endpoint (default: {JOB_URL}).")
    parser.add_argument("--output-dir", default="output", help="Directory for markdown/images (default: output).")
    parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout in seconds (default: 30).")
    parser.add_argument("--retries", type=int, default=3, help="Retry attempts for transient HTTP errors (default: 3).")
    parser.add_argument("--poll-interval", type=int, default=5, help="Poll interval in seconds (default: 5).")
    parser.add_argument("--max-wait", type=int, default=1800, help="Max wait seconds for job completion (default: 1800).")
    parser.add_argument(
        "--optional-payload",
        default=None,
        help="Optional payload JSON string. Example: '{\"useChartRecognition\":true}'.",
    )
    return parser.parse_args()


def parse_optional_payload(raw: str | None) -> dict[str, Any]:
    if not raw:
        return DEFAULT_OPTIONAL_PAYLOAD
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid --optional-payload JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError("--optional-payload must decode to a JSON object.")
    return data


def request_json_or_raise(resp: requests.Response, context: str) -> dict[str, Any]:
    if resp.status_code != 200:
        body_preview = (resp.text or "")[:1000]
        raise RuntimeError(f"{context} failed. HTTP {resp.status_code}. Body: {body_preview}")
    try:
        return resp.json()
    except ValueError as exc:
        body_preview = (resp.text or "")[:1000]
        raise RuntimeError(f"{context} returned non-JSON body: {body_preview}") from exc


def submit_job(
    session: requests.Session,
    job_url: str,
    token: str,
    file_value: str,
    model: str,
    optional_payload: dict[str, Any],
) -> str:
    headers = {
        "Authorization": f"Bearer {token}",
    }

    print(f"Processing file: {file_value}")

    if file_value.startswith(("http://", "https://")):
        headers["Content-Type"] = "application/json"
        payload = {
            "fileUrl": file_value,
            "model": model,
            "optionalPayload": optional_payload,
        }
        response = session.post(job_url, json=payload, headers=headers)
    else:
        local_path = Path(file_value)
        if not local_path.exists():
            raise RuntimeError(f"Local file not found: {local_path}")
        data = {
            "model": model,
            "optionalPayload": json.dumps(optional_payload, ensure_ascii=False),
        }
        with local_path.open("rb") as file_obj:
            files = {"file": file_obj}
            response = session.post(job_url, headers=headers, data=data, files=files)

    payload = request_json_or_raise(response, "Submit job")
    data = payload.get("data") or {}
    job_id = data.get("jobId")
    if not job_id:
        raise RuntimeError(f"Submit job succeeded but no jobId returned: {payload}")

    print(f"Job submitted successfully. job id: {job_id}")
    return job_id


def poll_job_until_done(
    session: requests.Session,
    job_url: str,
    token: str,
    job_id: str,
    poll_interval: int,
    max_wait_seconds: int,
) -> str:
    headers = {
        "Authorization": f"Bearer {token}",
    }
    deadline = time.monotonic() + max_wait_seconds
    print("Start polling for results")

    while True:
        if time.monotonic() > deadline:
            raise TimeoutError(f"Polling timeout: exceeded {max_wait_seconds} seconds for job {job_id}")

        response = session.get(f"{job_url}/{job_id}", headers=headers)
        payload = request_json_or_raise(response, "Get job status")
        data = payload.get("data") or {}
        state = data.get("state")

        if state == "pending":
            print("The current status of the job is pending")
        elif state == "running":
            progress = data.get("extractProgress") or {}
            total_pages = progress.get("totalPages")
            extracted_pages = progress.get("extractedPages")
            if total_pages is not None and extracted_pages is not None:
                print(
                    "The current status of the job is running, "
                    f"total pages: {total_pages}, extracted pages: {extracted_pages}"
                )
            else:
                print("The current status of the job is running...")
        elif state == "done":
            progress = data.get("extractProgress") or {}
            extracted_pages = progress.get("extractedPages")
            start_time = progress.get("startTime")
            end_time = progress.get("endTime")
            print(
                "Job completed, "
                f"successfully extracted pages: {extracted_pages}, "
                f"start time: {start_time}, end time: {end_time}"
            )
            result_url = (data.get("resultUrl") or {}).get("jsonUrl")
            if not result_url:
                raise RuntimeError(f"Job done but no resultUrl.jsonUrl found: {payload}")
            return result_url
        elif state == "failed":
            error_msg = data.get("errorMsg") or "Unknown error"
            raise RuntimeError(f"Job failed, reason: {error_msg}")
        else:
            print(f"Unknown job state: {state}")

        time.sleep(poll_interval)


def download_results(
    session: requests.Session,
    jsonl_url: str,
    output_dir: Path,
) -> None:
    response = session.get(jsonl_url)
    response.raise_for_status()

    output_dir.mkdir(parents=True, exist_ok=True)
    lines = response.text.splitlines()
    saved_docs = 0

    for line_no, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue

        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            print(f"Skip invalid JSONL line {line_no}: {exc}")
            continue

        result = record.get("result") or {}
        layouts = result.get("layoutParsingResults")
        if not layouts:
            print(f"Line {line_no}: no layoutParsingResults")
            continue

        for layout_idx, layout in enumerate(layouts):
            md_path = output_dir / f"doc_{line_no - 1}_{layout_idx}.md"
            markdown = ((layout.get("markdown") or {}).get("text")) or ""
            md_path.write_text(markdown, encoding="utf-8")
            print(f"Markdown document saved at {md_path}")
            saved_docs += 1

            markdown_images = ((layout.get("markdown") or {}).get("images")) or {}
            for relative_img_path, image_url in markdown_images.items():
                save_path = output_dir / relative_img_path
                save_path.parent.mkdir(parents=True, exist_ok=True)
                img_resp = session.get(image_url)
                img_resp.raise_for_status()
                save_path.write_bytes(img_resp.content)
                print(f"Image saved to: {save_path}")

            output_images = layout.get("outputImages") or {}
            for image_name, image_url in output_images.items():
                img_resp = session.get(image_url)
                if img_resp.status_code == 200:
                    filename = output_dir / f"{image_name}_{line_no - 1}_{layout_idx}.jpg"
                    filename.write_bytes(img_resp.content)
                    print(f"Image saved to: {filename}")
                else:
                    print(
                        "Failed to download image "
                        f"{image_name}, status code: {img_resp.status_code}"
                    )

    print(f"All done. Saved markdown documents: {saved_docs}")


def main() -> int:
    args = parse_args()

    try:
        token = require_token(args.token)
        optional_payload = parse_optional_payload(args.optional_payload)
        session = build_session(timeout_seconds=args.timeout, total_retries=args.retries)

        job_id = submit_job(
            session=session,
            job_url=args.job_url,
            token=token,
            file_value=args.file,
            model=args.model,
            optional_payload=optional_payload,
        )

        jsonl_url = poll_job_until_done(
            session=session,
            job_url=args.job_url,
            token=token,
            job_id=job_id,
            poll_interval=args.poll_interval,
            max_wait_seconds=args.max_wait,
        )

        download_results(
            session=session,
            jsonl_url=jsonl_url,
            output_dir=Path(args.output_dir),
        )
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
