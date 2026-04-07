"""Explicit model preparation for spec-driven local model inference."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .model_specs import (
    manifest_matches,
    model_spec_backend_cache_root,
    model_spec_display_id,
    model_spec_model_root,
    normalize_model_spec,
    prepared_marker_path,
)
from .paths import _ensure_worker_dependencies, _worker_python


_HF_CLI_DEPENDENCIES = ("huggingface_hub[cli]",)


class ModelPreparationError(RuntimeError):
    def __init__(self, *, error_code: str, message: str) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.message = message


def _hf_cli_command(python_bin: Path) -> list[str]:
    bin_dir = python_bin.parent
    hf_bin = bin_dir / "hf"
    if hf_bin.exists():
        return [str(hf_bin)]
    legacy_bin = bin_dir / "huggingface-cli"
    if legacy_bin.exists():
        return [str(legacy_bin)]
    return [
        str(python_bin),
        "-m",
        "huggingface_hub.commands.huggingface_cli",
    ]


def _hf_download_command(
    *,
    python_bin: Path,
    repo_id: str,
    local_dir: Path,
    include_patterns: list[str],
    exclude_patterns: list[str],
) -> list[str]:
    cmd = [*_hf_cli_command(python_bin), "download", repo_id]
    # The modern `hf` CLI treats repeated include values differently from the
    # older module entrypoint. For explicit file/glob manifests, positional
    # filenames are the simplest cross-version path.
    if include_patterns and not exclude_patterns:
        cmd.extend(include_patterns)
    else:
        for pattern in include_patterns:
            cmd.extend(["--include", pattern])
        for pattern in exclude_patterns:
            cmd.extend(["--exclude", pattern])
    cmd.extend(["--local-dir", str(local_dir)])
    return cmd


def _check_prerequisites(model_spec: dict[str, Any]) -> None:
    prerequisites = model_spec.get("prerequisites") or {}
    binaries = prerequisites.get("host_binaries")
    if binaries in (None, ""):
        return
    if not isinstance(binaries, list):
        raise ModelPreparationError(
            error_code="invalid_model_spec",
            message="model_spec.prerequisites.host_binaries must be a list of strings",
        )
    missing = [str(name).strip() for name in binaries if str(name).strip() and shutil.which(str(name).strip()) is None]
    if missing:
        install_hint = str(prerequisites.get("install_hint", "")).strip()
        suffix = f" {install_hint}" if install_hint else ""
        raise ModelPreparationError(
            error_code="missing_host_dependency",
            message=f"missing required host binaries: {', '.join(missing)}.{suffix}".strip(),
        )


def _write_prepared_marker(model_root: Path, model_spec: dict[str, Any]) -> None:
    marker_path = prepared_marker_path(model_root)
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(
        json.dumps(
            {
                "id": model_spec["id"],
                "backend": model_spec["backend"],
                "task_type": model_spec["task_type"],
                "family": model_spec["family"],
                "repo_id": model_spec["source"]["repo_id"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def ensure_model_prepared(
    *,
    cache_root: Path,
    model_spec: dict[str, Any],
    backend_mode: str,
) -> tuple[dict[str, Any], Path]:
    normalized = normalize_model_spec(model_spec)
    model_root = model_spec_model_root(cache_root, normalized)
    if backend_mode == "fake":
        marker = prepared_marker_path(model_root)
        if not marker.exists():
            raise ModelPreparationError(
                error_code="model_not_prepared",
                message=f"model {model_spec_display_id(normalized)} has not been prepared",
            )
        return normalized, model_root
    if not manifest_matches(model_root, normalized):
        raise ModelPreparationError(
            error_code="model_not_prepared",
            message=f"model {model_spec_display_id(normalized)} has not been prepared",
        )
    return normalized, model_root


def prepare_model_spec(
    *,
    cache_root: Path,
    model_spec: dict[str, Any],
    backend_mode: str,
    timeout_seconds: int,
    progress_stream: Any | None = None,
) -> tuple[dict[str, Any], Path]:
    normalized = normalize_model_spec(model_spec)
    _check_prerequisites(normalized)
    model_root = model_spec_model_root(cache_root, normalized)
    if backend_mode == "fake":
        model_root.mkdir(parents=True, exist_ok=True)
        _write_prepared_marker(model_root, normalized)
        return normalized, model_root

    backend_cache_root = model_spec_backend_cache_root(cache_root, normalized)
    backend_cache_root.mkdir(parents=True, exist_ok=True)
    python_bin = _worker_python(backend_cache_root)
    _ensure_worker_dependencies(python_bin, _HF_CLI_DEPENDENCIES)

    env = os.environ.copy()
    hub_root = backend_cache_root / "models"
    env.setdefault("HF_HOME", str(hub_root))
    env.setdefault("TRANSFORMERS_CACHE", str(hub_root))
    env.setdefault("HF_HUB_CACHE", str(hub_root))
    env.setdefault("HF_HUB_DISABLE_XET", "1")

    include_patterns = list(normalized["download_manifest"]["include"])
    checkpoint_glob = str(normalized.get("load_config", {}).get("checkpoint_glob", "") or "").strip()
    if checkpoint_glob and any(model_root.glob(checkpoint_glob)):
        existing_checkpoint = next(model_root.glob(checkpoint_glob))
        include_patterns = [
            pattern
            for pattern in include_patterns
            if Path(pattern).name != existing_checkpoint.name
        ]

    exclude_patterns = list(normalized["download_manifest"]["exclude"])
    cmd = _hf_download_command(
        python_bin=python_bin,
        repo_id=normalized["source"]["repo_id"],
        local_dir=model_root,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
    )

    if progress_stream is None:
        completed = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=max(30, int(timeout_seconds)),
            env=env,
        )
    else:
        completed = subprocess.run(
            cmd,
            stdout=progress_stream,
            stderr=progress_stream,
            text=True,
            check=False,
            timeout=max(30, int(timeout_seconds)),
            env=env,
        )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise ModelPreparationError(
            error_code="model_download_failed",
            message=detail or (
                f"failed downloading {normalized['source']['repo_id']}"
                + ("; see terminal output above" if progress_stream is not None else "")
            ),
        )
    if not manifest_matches(model_root, normalized):
        raise ModelPreparationError(
            error_code="model_prepare_validation_failed",
            message=f"prepared files are incomplete for {model_spec_display_id(normalized)}",
        )

    _write_prepared_marker(model_root, normalized)
    return normalized, model_root
