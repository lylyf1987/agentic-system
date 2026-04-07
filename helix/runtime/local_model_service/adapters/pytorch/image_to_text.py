"""PyTorch image-to-text adapter."""

from __future__ import annotations

from ..base import _BaseBackend
from ...paths import _PYTORCH_IMAGE_TO_TEXT_DEPENDENCIES, _ensure_worker_dependencies
from ...protocol import (
    _DEFAULT_IMAGE_ANALYSIS_QUERY,
    _extract_generated_text,
    _parse_int,
    _request_inputs,
    _resolve_service_workspace_root,
    _resolve_workspace_path,
)


class _RealPyTorchImageToTextBackend(_BaseBackend):
    def __init__(self, *, model_id, cache_root, python_bin) -> None:
        super().__init__(
            task_type="image_to_text",
            backend="pytorch",
            model_id=model_id,
            cache_root=cache_root,
            python_bin=python_bin,
        )
        self.device = None
        self.processor = None
        self.model = None
        self.image_module = None
        self.torch = None

    def _load(self) -> None:
        assert self.cache_root is not None
        assert self.python_bin is not None
        try:
            import torch
            from PIL import Image
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError:
            _ensure_worker_dependencies(self.python_bin, _PYTORCH_IMAGE_TO_TEXT_DEPENDENCIES)
            import torch
            from PIL import Image
            from transformers import AutoModelForImageTextToText, AutoProcessor

        device = "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.torch = torch
        self.image_module = Image
        dtype = torch.float16 if device == "mps" else torch.float32
        hub_cache = self.cache_root / "models"
        hub_cache.mkdir(parents=True, exist_ok=True)
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            cache_dir=str(hub_cache),
            trust_remote_code=True,
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            cache_dir=str(hub_cache),
            trust_remote_code=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=False,
        )
        self.model.to(device)

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
            max_new_tokens = _parse_int(inputs.get("max_new_tokens"), default=256, minimum=1)
            self._ensure_loaded()
            assert self.image_module is not None
            assert self.processor is not None
            assert self.model is not None
            assert self.torch is not None
            image = self.image_module.open(image_path).convert("RGB")
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": query},
                ],
            }]
            if hasattr(self.processor, "apply_chat_template"):
                prompt_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
                prepared = self.processor(text=prompt_text, images=[image], return_tensors="pt")
            else:
                prepared = self.processor(text=query, images=image, return_tensors="pt")
            if hasattr(prepared, "to"):
                prepared = prepared.to(self.device)
            else:
                prepared = {
                    key: value.to(self.device) if hasattr(value, "to") else value
                    for key, value in prepared.items()
                }
            with self.torch.no_grad():
                generated = self.model.generate(**prepared, max_new_tokens=max_new_tokens)
            if hasattr(self.processor, "post_process_image_text_to_text"):
                texts = self.processor.post_process_image_text_to_text(
                    generated,
                    skip_special_tokens=True,
                )
            else:
                texts = self.processor.batch_decode(generated, skip_special_tokens=True)
            text = _extract_generated_text(texts)
            return self._ok(outputs={"text": text}, message="generated image analysis")
        except Exception as exc:
            return self._error(error_code="analysis_runtime_error", message=str(exc))
