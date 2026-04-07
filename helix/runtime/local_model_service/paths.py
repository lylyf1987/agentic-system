"""Path and dependency helpers for local model service."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import venv
from pathlib import Path

from .protocol import _HTTP_TIMEOUT_SECONDS, _LOCAL_MODEL_SERVICE_NAME


_PYTORCH_TEXT_TO_IMAGE_DEPENDENCIES = (
    "accelerate",
    "diffusers>=0.35.0",
    "huggingface_hub",
    "pillow",
    "safetensors",
    "torch",
    "transformers",
)
_PYTORCH_IMAGE_TO_TEXT_DEPENDENCIES = (
    "accelerate",
    "huggingface_hub",
    "pillow",
    "safetensors",
    "torch",
    "transformers",
)
_PYTORCH_VIDEO_DEPENDENCIES = (
    "accelerate",
    "git+https://github.com/huggingface/diffusers",
    "huggingface_hub",
    "imageio",
    "imageio-ffmpeg",
    "numpy",
    "pillow",
    "safetensors",
    "torch",
    "transformers",
)
_PYTORCH_TEXT_TO_AUDIO_DEPENDENCIES = (
    "huggingface_hub",
    "numpy",
    "qwen-tts",
    "soundfile",
    "torch",
    "transformers",
)
_MLX_GENERATION_DEPENDENCIES = (
    "accelerate",
    "diffusers>=0.35.0",
    "hf_transfer",
    "huggingface_hub",
    "mlx>=0.20.0",
    "numpy",
    "pillow",
    "safetensors",
    "torch",
    "transformers",
    "tqdm",
)
_MLX_IMAGE_TO_TEXT_DEPENDENCIES = (
    "huggingface_hub",
    "mlx>=0.20.0",
    "mlx-vlm",
    "pillow",
)
_MLX_Z_IMAGE_COMMIT = "b508c3555cd49b5fb5afd3434053a55d1710c129"
_MLX_Z_IMAGE_FILES = (
    "lora_utils.py",
    "mlx_pipeline.py",
    "mlx_text_encoder.py",
    "mlx_z_image.py",
)


def helix_home() -> Path:
    override = str(os.environ.get("HELIX_HOME", "")).strip()
    if override:
        return Path(override).expanduser().resolve()
    return (Path.home() / ".helix").resolve()


def _runtime_root() -> Path:
    return helix_home() / "runtime"


def _service_runtime_dir(service_name: str) -> Path:
    return _runtime_root() / "services" / service_name


def _service_cache_dir(service_name: str) -> Path:
    return helix_home() / "cache" / service_name


def _active_runtime_dir() -> Path:
    return _runtime_root() / "active-runtimes"


def _runtime_marker_path(pid: int | None = None) -> Path:
    token = int(pid or os.getpid())
    return _active_runtime_dir() / f"{token}.json"


def _prune_stale_runtime_markers() -> None:
    markers = _active_runtime_dir()
    if not markers.exists():
        return
    for marker in markers.glob("*.json"):
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            marker.unlink(missing_ok=True)
            continue
        try:
            pid = int(payload.get("pid"))
        except (TypeError, ValueError):
            marker.unlink(missing_ok=True)
            continue
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            marker.unlink(missing_ok=True)
        except PermissionError:
            continue


def register_active_runtime(*, workspace: Path, session_id: str) -> Path:
    markers = _active_runtime_dir()
    markers.mkdir(parents=True, exist_ok=True)
    _prune_stale_runtime_markers()
    marker = _runtime_marker_path()
    payload = {
        "pid": os.getpid(),
        "workspace": str(Path(workspace).expanduser().resolve()),
        "session_id": str(session_id or "session").strip() or "session",
        "started_at": time.time(),
    }
    marker.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return marker


def unregister_active_runtime(marker_path: Path | None) -> None:
    if marker_path is not None:
        marker_path.unlink(missing_ok=True)


def has_active_runtimes() -> bool:
    _prune_stale_runtime_markers()
    markers = _active_runtime_dir()
    if not markers.exists():
        return False
    return any(markers.glob("*.json"))


def default_cache_root(workspace: Path | None = None) -> Path:
    return _service_cache_dir(_LOCAL_MODEL_SERVICE_NAME).resolve()


def default_runtime_root() -> Path:
    return _service_runtime_dir(_LOCAL_MODEL_SERVICE_NAME).resolve()


def _backend_cache_root(cache_root: Path, backend: str) -> Path:
    return (cache_root / backend).resolve()


def _worker_python(cache_root: Path) -> Path:
    venv_root = cache_root / "venv"
    python_bin = venv_root / "bin" / "python"
    if python_bin.exists():
        return python_bin
    venv_root.parent.mkdir(parents=True, exist_ok=True)
    builder = venv.EnvBuilder(with_pip=True, system_site_packages=True, clear=False)
    builder.create(str(venv_root))
    return python_bin


def _ensure_worker_dependencies(python_bin: Path, dependencies: tuple[str, ...]) -> None:
    if not dependencies:
        return
    completed = subprocess.run(
        [str(python_bin), "-m", "pip", "install", *dependencies],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(detail or "failed installing worker dependencies")


def _download_public_file(url: str, dest: Path) -> None:
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=max(_HTTP_TIMEOUT_SECONDS, 60)) as resp:
        data = resp.read()
    dest.write_bytes(data)


def _safe_model_dir_name(model_id: str) -> str:
    return str(model_id or "model").strip().replace("/", "--")


def _snapshot_download_model(*, repo_id: str, local_dir: Path) -> Path:
    from huggingface_hub import snapshot_download

    local_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    return Path(
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
        )
    ).resolve()


def _download_hub_file(*, repo_id: str, filename: str, local_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    local_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(local_dir),
        )
    ).resolve()


def _is_qwen_tts_custom_voice_model(model_id: str) -> bool:
    normalized = str(model_id or "").strip()
    return normalized.startswith("Qwen/Qwen3-TTS-") and normalized.endswith("-CustomVoice")


def _ensure_mlx_runner_sources(cache_root: Path) -> Path:
    runner_root = cache_root / "sources" / f"mlx_z_image-{_MLX_Z_IMAGE_COMMIT}"
    runner_root.mkdir(parents=True, exist_ok=True)
    for filename in _MLX_Z_IMAGE_FILES:
        target = runner_root / filename
        if target.exists():
            continue
        url = (
            "https://raw.githubusercontent.com/uqer1244/MLX_z-image/"
            f"{_MLX_Z_IMAGE_COMMIT}/{filename}"
        )
        _download_public_file(url, target)
    return runner_root
