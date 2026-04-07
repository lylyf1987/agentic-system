"""MLX image-to-text adapter."""

from __future__ import annotations

from ..base import _BaseBackend
from ...paths import _MLX_IMAGE_TO_TEXT_DEPENDENCIES, _ensure_worker_dependencies
from ...protocol import (
    _DEFAULT_IMAGE_ANALYSIS_QUERY,
    _extract_generated_text,
    _parse_int,
    _request_inputs,
    _resolve_service_workspace_root,
    _resolve_workspace_path,
)


class _RealMLXImageToTextBackend(_BaseBackend):
    def __init__(self, *, model_id, cache_root, python_bin) -> None:
        super().__init__(
            task_type="image_to_text",
            backend="mlx",
            model_id=model_id,
            cache_root=cache_root,
            python_bin=python_bin,
        )
        self.model = None
        self.processor = None
        self.config = None
        self.generate_fn = None
        self.apply_chat_template = None

    def _load(self) -> None:
        assert self.python_bin is not None
        try:
            from mlx_vlm import generate, load
            from mlx_vlm.prompt_utils import apply_chat_template
            from mlx_vlm.utils import load_config
        except ImportError:
            _ensure_worker_dependencies(self.python_bin, _MLX_IMAGE_TO_TEXT_DEPENDENCIES)
            from mlx_vlm import generate, load
            from mlx_vlm.prompt_utils import apply_chat_template
            from mlx_vlm.utils import load_config

        self.model, self.processor = load(self.model_id)
        self.config = load_config(self.model_id)
        self.generate_fn = generate
        self.apply_chat_template = apply_chat_template

    def _ensure_loaded(self) -> None:
        if self.model is None or self.processor is None:
            self._load()

    def handle(self, payload):
        inputs = _request_inputs(payload)
        try:
            workspace_root = _resolve_service_workspace_root(payload)
            image_path = _resolve_workspace_path(
                workspace_root,
                str(inputs.get("image_path", "")).strip(),
                expect_exists=True,
            )
            query = str(inputs.get("query", "")).strip() or _DEFAULT_IMAGE_ANALYSIS_QUERY
            max_tokens = _parse_int(inputs.get("max_new_tokens"), default=256, minimum=1)
            self._ensure_loaded()
            assert self.generate_fn is not None
            assert self.apply_chat_template is not None
            assert self.processor is not None
            assert self.config is not None
            formatted_prompt = self.apply_chat_template(
                self.processor,
                self.config,
                query,
                num_images=1,
            )
            try:
                output = self.generate_fn(
                    self.model,
                    self.processor,
                    formatted_prompt,
                    [str(image_path)],
                    max_tokens=max_tokens,
                    verbose=False,
                )
            except TypeError:
                output = self.generate_fn(
                    self.model,
                    self.processor,
                    str(image_path),
                    formatted_prompt,
                    max_tokens=max_tokens,
                    verbose=False,
                )
            text = _extract_generated_text(output)
            return self._ok(outputs={"text": text}, message="generated image analysis")
        except Exception as exc:
            return self._error(error_code="analysis_runtime_error", message=str(exc))
