"""Backend registry for local model service."""

from __future__ import annotations

from typing import Protocol

from .adapters.fake import (
    _FakeImageToTextBackend,
    _FakeTextToAudioBackend,
    _FakeTextToImageBackend,
    _FakeVideoGenerationBackend,
)
from .adapters.mlx.z_image import _SpecMLXZImageBackend
from .adapters.mlx.image_to_text import _RealMLXImageToTextBackend
from .adapters.mlx.text_to_image import _RealMLXImageGenerationBackend
from .adapters.pytorch.image_to_text import _RealPyTorchImageToTextBackend
from .adapters.pytorch.qwen_tts_custom_voice import _SpecQwenTTSCustomVoiceBackend
from .adapters.pytorch.text_to_audio import _RealPyTorchTextToAudioBackend
from .adapters.pytorch.text_to_image import _RealImageGenerationBackend
from .adapters.pytorch.video_families import _SpecLTXVideoBackend, _SpecWanVideoBackend
from .adapters.pytorch.video import _RealPyTorchVideoGenerationBackend
from .model_specs import model_spec_display_id
from .protocol import (
    _BACKEND_MLX,
    _BACKEND_PYTORCH,
    _FAKE_BACKEND_NAME,
    _SUPPORTED_BACKENDS,
    _SUPPORTED_TASK_TYPES,
    _TASK_IMAGE_TO_TEXT,
    _TASK_TEXT_IMAGE_TO_VIDEO,
    _TASK_TEXT_TO_AUDIO,
    _TASK_TEXT_TO_IMAGE,
    _TASK_TEXT_TO_VIDEO,
    _supported_backend_task,
)


class _WorkerBackend(Protocol):
    def handle(self, payload: dict) -> dict:
        ...


def _build_backend(
    *,
    task_type: str,
    backend: str,
    cache_root,
    model_id: str,
    backend_mode: str,
    python_bin,
    model_spec: dict | None = None,
    model_root=None,
) -> _WorkerBackend:
    if task_type not in _SUPPORTED_TASK_TYPES:
        raise ValueError(f"unsupported task_type: {task_type}")
    if backend not in _SUPPORTED_BACKENDS:
        raise ValueError(f"unsupported backend: {backend}")
    if not _supported_backend_task(task_type, backend):
        raise ValueError(f"unsupported backend/task combination: {backend}/{task_type}")
    display_model_id = model_spec_display_id(model_spec) if model_spec is not None else model_id
    if backend_mode == _FAKE_BACKEND_NAME:
        if task_type == _TASK_TEXT_TO_IMAGE:
            return _FakeTextToImageBackend(task_type=task_type, backend=backend, model_id=display_model_id)
        if task_type == _TASK_IMAGE_TO_TEXT:
            return _FakeImageToTextBackend(task_type=task_type, backend=backend, model_id=display_model_id)
        if task_type in {_TASK_TEXT_TO_VIDEO, _TASK_TEXT_IMAGE_TO_VIDEO}:
            return _FakeVideoGenerationBackend(task_type=task_type, backend=backend, model_id=display_model_id)
        if task_type == _TASK_TEXT_TO_AUDIO:
            return _FakeTextToAudioBackend(task_type=task_type, backend=backend, model_id=display_model_id)
    if model_spec is not None:
        family = str(model_spec.get("family", "")).strip()
        if family == "mlx.z_image":
            return _SpecMLXZImageBackend(
                task_type=task_type,
                backend=backend,
                model_id=display_model_id,
                model_spec=model_spec,
                model_root=model_root,
                cache_root=cache_root,
                python_bin=python_bin,
            )
        if family == "pytorch.qwen_tts_custom_voice":
            return _SpecQwenTTSCustomVoiceBackend(
                task_type=task_type,
                backend=backend,
                model_id=display_model_id,
                model_spec=model_spec,
                model_root=model_root,
                cache_root=cache_root,
                python_bin=python_bin,
            )
        if family == "pytorch.diffusers_ltx_video":
            return _SpecLTXVideoBackend(
                task_type=task_type,
                backend=backend,
                model_id=display_model_id,
                model_spec=model_spec,
                model_root=model_root,
                cache_root=cache_root,
                python_bin=python_bin,
            )
        if family == "pytorch.diffusers_wan_video":
            return _SpecWanVideoBackend(
                task_type=task_type,
                backend=backend,
                model_id=display_model_id,
                model_spec=model_spec,
                model_root=model_root,
                cache_root=cache_root,
                python_bin=python_bin,
            )
        raise ValueError(f"unsupported model family: {family}")
    if task_type == _TASK_TEXT_TO_IMAGE and backend == _BACKEND_PYTORCH:
        return _RealImageGenerationBackend(model_id=model_id, cache_root=cache_root, python_bin=python_bin)
    if task_type == _TASK_TEXT_TO_IMAGE and backend == _BACKEND_MLX:
        return _RealMLXImageGenerationBackend(model_id=model_id, cache_root=cache_root, python_bin=python_bin)
    if task_type == _TASK_IMAGE_TO_TEXT and backend == _BACKEND_PYTORCH:
        return _RealPyTorchImageToTextBackend(model_id=model_id, cache_root=cache_root, python_bin=python_bin)
    if task_type == _TASK_IMAGE_TO_TEXT and backend == _BACKEND_MLX:
        return _RealMLXImageToTextBackend(model_id=model_id, cache_root=cache_root, python_bin=python_bin)
    if task_type in {_TASK_TEXT_TO_VIDEO, _TASK_TEXT_IMAGE_TO_VIDEO} and backend == _BACKEND_PYTORCH:
        return _RealPyTorchVideoGenerationBackend(
            task_type=task_type,
            model_id=model_id,
            cache_root=cache_root,
            python_bin=python_bin,
        )
    if task_type == _TASK_TEXT_TO_AUDIO and backend == _BACKEND_PYTORCH:
        return _RealPyTorchTextToAudioBackend(model_id=model_id, cache_root=cache_root, python_bin=python_bin)
    raise ValueError(f"unsupported backend/task combination: {backend}/{task_type}")
