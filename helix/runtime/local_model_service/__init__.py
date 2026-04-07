"""Runtime-owned host-native local model inference service."""

from .coordinator import _CoordinatorController, _WorkerState
from .manager import LocalModelServiceManager
from .paths import (
    _ensure_mlx_runner_sources,
    _worker_python,
    default_cache_root,
    default_runtime_root,
)
from .protocol import _http_json_request, _kill_process_tree, local_model_service_supported
from .adapters.pytorch.text_to_image import _RealImageGenerationBackend
from .adapters.mlx.text_to_image import _RealMLXImageGenerationBackend
from .adapters.pytorch.text_to_audio import _RealPyTorchTextToAudioBackend
from .adapters.pytorch.video import _RealPyTorchVideoGenerationBackend

__all__ = [
    "LocalModelServiceManager",
    "_CoordinatorController",
    "_RealImageGenerationBackend",
    "_RealMLXImageGenerationBackend",
    "_RealPyTorchTextToAudioBackend",
    "_RealPyTorchVideoGenerationBackend",
    "_WorkerState",
    "_ensure_mlx_runner_sources",
    "_http_json_request",
    "_kill_process_tree",
    "_worker_python",
    "default_cache_root",
    "default_runtime_root",
    "local_model_service_supported",
]
