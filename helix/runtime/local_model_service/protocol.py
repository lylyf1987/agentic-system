"""Public protocol constants and request/response helpers for local model service."""

from __future__ import annotations

import base64
import contextlib
import json
import os
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from http import HTTPStatus
from pathlib import Path
from typing import Any


_DEFAULT_IDLE_SECONDS = int(os.environ.get("HELIX_LOCAL_MODEL_SERVICE_IDLE_SECONDS", "300"))
_HTTP_TIMEOUT_SECONDS = int(os.environ.get("HELIX_LOCAL_MODEL_SERVICE_HTTP_TIMEOUT", "30"))
_STARTUP_TIMEOUT_SECONDS = int(os.environ.get("HELIX_LOCAL_MODEL_SERVICE_STARTUP_TIMEOUT", "20"))
_WORKER_REQUEST_TIMEOUT_SECONDS = int(
    os.environ.get("HELIX_LOCAL_MODEL_SERVICE_WORKER_TIMEOUT", "1200")
)
_COORDINATOR_HEALTH_PATH = "/health"
_FAKE_BACKEND_NAME = "fake"
_REAL_BACKEND_NAME = "real"
_DEFAULT_BACKEND_MODE = os.environ.get(
    "HELIX_LOCAL_MODEL_SERVICE_BACKEND",
    _REAL_BACKEND_NAME,
).strip().lower() or _REAL_BACKEND_NAME

_LOCAL_MODEL_SERVICE_NAME = "local-model-service"
_DEFAULT_GENERATION_MODEL_ID = "uqer1244/MLX-z-image"
_TASK_TEXT_TO_IMAGE = "text_to_image"
_TASK_IMAGE_TO_TEXT = "image_to_text"
_TASK_TEXT_TO_VIDEO = "text_to_video"
_TASK_TEXT_IMAGE_TO_VIDEO = "text_image_to_video"
_TASK_TEXT_TO_AUDIO = "text_to_audio"
_BACKEND_PYTORCH = "pytorch"
_BACKEND_MLX = "mlx"
_LEGACY_TASK_TYPE_ALIASES = {
    "image_generation": _TASK_TEXT_TO_IMAGE,
}
_SUPPORTED_TASK_TYPES = (
    _TASK_TEXT_TO_IMAGE,
    _TASK_IMAGE_TO_TEXT,
    _TASK_TEXT_TO_VIDEO,
    _TASK_TEXT_IMAGE_TO_VIDEO,
    _TASK_TEXT_TO_AUDIO,
)
_SUPPORTED_BACKENDS = (_BACKEND_PYTORCH, _BACKEND_MLX)
_SUPPORTED_BACKEND_TASKS = {
    (_TASK_TEXT_TO_IMAGE, _BACKEND_PYTORCH),
    (_TASK_TEXT_TO_IMAGE, _BACKEND_MLX),
    (_TASK_IMAGE_TO_TEXT, _BACKEND_PYTORCH),
    (_TASK_IMAGE_TO_TEXT, _BACKEND_MLX),
    (_TASK_TEXT_TO_VIDEO, _BACKEND_PYTORCH),
    (_TASK_TEXT_IMAGE_TO_VIDEO, _BACKEND_PYTORCH),
    (_TASK_TEXT_TO_AUDIO, _BACKEND_PYTORCH),
}

_DEFAULT_IMAGE_ANALYSIS_QUERY = "Describe this image."
_DEFAULT_AUDIO_SAMPLE_RATE = 24000

_FAKE_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9pob7XUAAAAASUVORK5CYII="
)
_FAKE_WAV_BYTES = base64.b64decode(
    "UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA="
)
_FAKE_MP4_BYTES = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom"


def local_model_service_supported() -> bool:
    return sys_platform_is_darwin_arm64()


def sys_platform_is_darwin_arm64() -> bool:
    import platform
    import sys

    return sys.platform == "darwin" and platform.machine().lower() == "arm64"


def _json_dumps(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=True).encode("utf-8")


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def _kill_process_tree(pid: int, *, grace_seconds: float = 5.0) -> None:
    if pid <= 0:
        return
    deadline = time.time() + max(0.1, grace_seconds)
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pid, signal.SIGTERM)
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        except PermissionError:
            break
        time.sleep(0.1)
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pid, signal.SIGKILL)


def _http_json_request(
    *,
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    token: str | None = None,
    timeout: int = _HTTP_TIMEOUT_SECONDS,
) -> tuple[int, str, dict[str, Any] | None]:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = None if payload is None else _json_dumps(payload)
    req = urllib.request.Request(url, method=method.upper(), data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            parsed = json.loads(body) if body.strip() else None
            return int(getattr(resp, "status", 200)), body, parsed if isinstance(parsed, dict) else None
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        parsed = None
        try:
            candidate = json.loads(body)
        except json.JSONDecodeError:
            candidate = None
        if isinstance(candidate, dict):
            parsed = candidate
        return int(exc.code), body, parsed
    except urllib.error.URLError:
        return 0, "", None


def _normalize_relative_path(path_text: str) -> str:
    return str(path_text or "").strip().replace("\\", "/")


def _resolve_workspace_path(
    workspace_root: Path,
    path_text: str,
    *,
    expect_exists: bool,
) -> Path:
    candidate = _normalize_relative_path(path_text)
    if not candidate:
        raise ValueError("workspace path is required")
    path_obj = Path(candidate)
    if path_obj.is_absolute():
        raise ValueError("absolute paths are not allowed")
    if ".." in path_obj.parts:
        raise ValueError("path traversal is not allowed")
    resolved = (workspace_root / path_obj).resolve(strict=False)
    workspace_resolved = workspace_root.resolve()
    try:
        resolved.relative_to(workspace_resolved)
    except ValueError as exc:
        raise ValueError("path escapes workspace") from exc
    if expect_exists:
        if not resolved.exists() or not resolved.is_file():
            raise ValueError(f"workspace file not found: {candidate}")
    else:
        resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _resolve_service_workspace_root(payload: dict[str, Any]) -> Path:
    raw = str(payload.get("workspace_root", "")).strip()
    if not raw:
        raise ValueError("workspace_root is required")
    root = Path(raw).expanduser()
    if not root.is_absolute():
        raise ValueError("workspace_root must be absolute")
    resolved = root.resolve(strict=False)
    if not resolved.exists() or not resolved.is_dir():
        raise ValueError("workspace_root must exist and be a directory")
    return resolved


def _parse_size(size_text: str) -> tuple[int, int]:
    token = str(size_text or "").strip().lower()
    if "x" not in token:
        raise ValueError(f"invalid size: {size_text}")
    left, right = token.split("x", 1)
    width = int(left)
    height = int(right)
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid size: {size_text}")
    return width, height


def _parse_int(value: Any, *, default: int, minimum: int = 1) -> int:
    if value in (None, ""):
        return default
    parsed = int(value)
    return parsed if parsed >= minimum else minimum


def _parse_float(value: Any, *, default: float, minimum: float = 0.0) -> float:
    if value in (None, ""):
        return default
    parsed = float(value)
    return parsed if parsed >= minimum else minimum


def _request_timeout_seconds(payload: dict[str, Any]) -> int:
    raw = payload.get("request_timeout_seconds")
    if raw in (None, ""):
        return _WORKER_REQUEST_TIMEOUT_SECONDS
    try:
        parsed = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("request_timeout_seconds must be an integer") from exc
    if parsed < 1:
        raise ValueError("request_timeout_seconds must be >= 1")
    return parsed


def _canonical_task_type(task_type: str) -> str:
    raw = str(task_type or "").strip()
    if not raw:
        return ""
    return _LEGACY_TASK_TYPE_ALIASES.get(raw, raw)


def _ok_response(
    *,
    task_type: str,
    backend: str,
    model_id: str,
    outputs: dict[str, Any] | None,
    message: str,
) -> dict[str, Any]:
    return {
        "status": "ok",
        "task_type": task_type,
        "backend": backend,
        "model_id": model_id,
        "outputs": outputs or {},
        "error_code": "",
        "message": message,
    }


def _error_response(
    *,
    task_type: str,
    backend: str,
    model_id: str,
    error_code: str,
    message: str,
    outputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "status": "error",
        "task_type": task_type,
        "backend": backend,
        "model_id": model_id,
        "outputs": outputs or {},
        "error_code": error_code,
        "message": message,
    }


def _extract_generated_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("generated_text", "text", "output_text"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        return json.dumps(value, ensure_ascii=True)
    if isinstance(value, list):
        for item in value:
            text = _extract_generated_text(item)
            if text:
                return text
        return ""
    return str(value).strip()


def _request_inputs(payload: dict[str, Any]) -> dict[str, Any]:
    inputs = payload.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("inputs must be a JSON object")
    return inputs


def _bool_input(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    token = str(value or "").strip().lower()
    return token in {"1", "true", "yes", "on"}


def _supported_backend_task(task_type: str, backend: str) -> bool:
    return (task_type, backend) in _SUPPORTED_BACKEND_TASKS
