"""PyTorch text-to-image adapter."""

from __future__ import annotations

from ..base import _BaseBackend
from ...paths import _PYTORCH_TEXT_TO_IMAGE_DEPENDENCIES, _ensure_worker_dependencies
from ...protocol import _bool_input, _request_inputs, _resolve_service_workspace_root, _resolve_workspace_path, _parse_size


class _RealImageGenerationBackend(_BaseBackend):
    def __init__(self, *, model_id, cache_root, python_bin) -> None:
        super().__init__(
            task_type="text_to_image",
            backend="pytorch",
            model_id=model_id,
            cache_root=cache_root,
            python_bin=python_bin,
        )
        self.pipeline = None
        self.device = None

    def _load(self) -> None:
        assert self.cache_root is not None
        assert self.python_bin is not None
        try:
            import torch
            from diffusers import ZImagePipeline
        except ImportError:
            _ensure_worker_dependencies(self.python_bin, _PYTORCH_TEXT_TO_IMAGE_DEPENDENCIES)
            import torch
            from diffusers import ZImagePipeline

        device = "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
        self.device = device
        dtype = torch.float16 if device == "mps" else torch.float32
        hub_cache = self.cache_root / "models"
        hub_cache.mkdir(parents=True, exist_ok=True)
        self.pipeline = ZImagePipeline.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            cache_dir=str(hub_cache),
            low_cpu_mem_usage=False,
        )
        self.pipeline.to(device)

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
            result = self.pipeline(prompt=prompt, width=width, height=height)
            image = result.images[0]
            image.save(output_path)
            rel = str(output_path.relative_to(workspace_root))
            return self._ok(outputs={"output_path": rel}, message=f"generated image at {rel}")
        except Exception as exc:
            return self._error(error_code="generation_runtime_error", message=str(exc))
