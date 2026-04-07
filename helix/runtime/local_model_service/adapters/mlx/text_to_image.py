"""MLX text-to-image adapter."""

from __future__ import annotations

import contextlib
import os
import sys

from ..base import _BaseBackend
from ...paths import (
    _MLX_GENERATION_DEPENDENCIES,
    _ensure_mlx_runner_sources,
    _ensure_worker_dependencies,
    _safe_model_dir_name,
)
from ...protocol import _bool_input, _parse_int, _parse_size, _request_inputs, _resolve_service_workspace_root, _resolve_workspace_path


class _RealMLXImageGenerationBackend(_BaseBackend):
    def __init__(self, *, model_id, cache_root, python_bin) -> None:
        super().__init__(
            task_type="text_to_image",
            backend="mlx",
            model_id=model_id,
            cache_root=cache_root,
            python_bin=python_bin,
        )
        self.pipeline = None
        self.model_root = cache_root / "mlx-models" / _safe_model_dir_name(model_id)

    def _load(self) -> None:
        assert self.cache_root is not None
        assert self.python_bin is not None
        try:
            import mlx  # noqa: F401
        except ImportError:
            _ensure_worker_dependencies(self.python_bin, _MLX_GENERATION_DEPENDENCIES)
            import mlx  # noqa: F401

        source_root = _ensure_mlx_runner_sources(self.cache_root)
        if str(source_root) not in sys.path:
            sys.path.insert(0, str(source_root))
        from mlx_pipeline import ZImagePipeline

        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        with contextlib.redirect_stdout(sys.stderr):
            self.pipeline = ZImagePipeline(
                model_path=str(self.model_root),
                text_encoder_path=str(self.model_root / "text_encoder"),
                repo_id=self.model_id,
            )

    def _ensure_loaded(self) -> None:
        if self.pipeline is None:
            self._load()

    def handle(self, payload):
        inputs = _request_inputs(payload)
        try:
            if _bool_input(inputs.get("prepare_only")):
                self._ensure_loaded()
                return self._ok(
                    outputs={"prepared": True},
                    message=f"prepared image generation model {self.model_id}",
                )
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
            rel = str(output_path.relative_to(workspace_root))
            return self._ok(outputs={"output_path": rel}, message=f"generated image at {rel}")
        except Exception as exc:
            return self._error(error_code="generation_runtime_error", message=str(exc))
