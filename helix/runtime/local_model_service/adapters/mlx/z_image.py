"""Spec-driven MLX Z-Image backend."""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path
from typing import Any

from ..base import _BaseBackend
from ...paths import (
    _MLX_GENERATION_DEPENDENCIES,
    _ensure_mlx_runner_sources,
    _ensure_worker_dependencies,
)
from ...protocol import (
    _parse_int,
    _parse_size,
    _request_inputs,
    _resolve_service_workspace_root,
    _resolve_workspace_path,
)


class _SpecMLXZImageBackend(_BaseBackend):
    def __init__(
        self,
        *,
        task_type: str,
        backend: str,
        model_id: str,
        model_spec: dict[str, Any],
        model_root: Path,
        cache_root: Path,
        python_bin: Path,
    ) -> None:
        super().__init__(
            task_type=task_type,
            backend=backend,
            model_id=model_id,
            cache_root=cache_root,
            python_bin=python_bin,
            model_spec=model_spec,
            model_root=model_root,
        )
        self.pipeline = None

    def _load(self) -> None:
        assert self.cache_root is not None
        assert self.python_bin is not None
        assert self.model_root is not None
        try:
            import mlx  # noqa: F401
        except ImportError:
            _ensure_worker_dependencies(self.python_bin, _MLX_GENERATION_DEPENDENCIES)
            import mlx  # noqa: F401

        source_root = _ensure_mlx_runner_sources(self.cache_root)
        if str(source_root) not in sys.path:
            sys.path.insert(0, str(source_root))
        from mlx_pipeline import ZImagePipeline

        repo_id = str(self.model_spec["source"]["repo_id"])
        with contextlib.redirect_stdout(sys.stderr):
            self.pipeline = ZImagePipeline(
                model_path=str(self.model_root),
                text_encoder_path=str(self.model_root / "text_encoder"),
                repo_id=repo_id,
            )

    def _ensure_loaded(self) -> None:
        if self.pipeline is None:
            self._load()

    def handle(self, payload: dict[str, Any]) -> dict[str, Any]:
        inputs = _request_inputs(payload)
        prompt = str(inputs.get("prompt", "")).strip()
        if not prompt:
            return self._error(error_code="image_prompt_missing", message="prompt is required")
        workspace_root = _resolve_service_workspace_root(payload)
        width, height = _parse_size(str(inputs.get("size", "")).strip())
        output_path = _resolve_workspace_path(
            workspace_root,
            str(inputs.get("output_path", "")).strip(),
            expect_exists=False,
        )
        try:
            self._ensure_loaded()
            assert self.pipeline is not None
            with contextlib.redirect_stdout(sys.stderr):
                image = self.pipeline.generate(
                    prompt=prompt,
                    width=width,
                    height=height,
                    steps=_parse_int(inputs.get("num_inference_steps"), default=9, minimum=1),
                    seed=_parse_int(inputs.get("seed"), default=42, minimum=0),
                )
            image.save(output_path)
        except Exception as exc:
            return self._error(error_code="generation_runtime_error", message=str(exc))
        rel = str(output_path.relative_to(workspace_root))
        return self._ok(outputs={"output_path": rel}, message=f"generated image at {rel}")

