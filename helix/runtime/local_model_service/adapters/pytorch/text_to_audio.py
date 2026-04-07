"""PyTorch text-to-audio adapter."""

from __future__ import annotations

import contextlib
import shutil
import sys

from ..base import _BaseBackend
from ...paths import (
    _PYTORCH_TEXT_TO_AUDIO_DEPENDENCIES,
    _ensure_worker_dependencies,
    _is_qwen_tts_custom_voice_model,
    _safe_model_dir_name,
    _snapshot_download_model,
)
from ...protocol import _bool_input, _DEFAULT_AUDIO_SAMPLE_RATE, _request_inputs, _resolve_service_workspace_root, _resolve_workspace_path


class _MissingHostDependencyError(RuntimeError):
    """Raised when a required host binary is missing."""


class _RealPyTorchTextToAudioBackend(_BaseBackend):
    def __init__(self, *, model_id, cache_root, python_bin) -> None:
        super().__init__(
            task_type="text_to_audio",
            backend="pytorch",
            model_id=model_id,
            cache_root=cache_root,
            python_bin=python_bin,
        )
        self.audio_model = None
        self.soundfile = None
        self.device = None
        self.torch = None
        self.audio_mode = "generic"
        self.model_root = cache_root / "models" / _safe_model_dir_name(model_id)

    def _load_qwen_tts(self) -> None:
        assert self.python_bin is not None
        if shutil.which("sox") is None:
            raise _MissingHostDependencyError(
                "SoX is required for Qwen3-TTS host inference. Install it on the host with `brew install sox`."
            )
        try:
            import torch
            import soundfile as sf
            from qwen_tts import Qwen3TTSModel
        except ImportError:
            _ensure_worker_dependencies(self.python_bin, _PYTORCH_TEXT_TO_AUDIO_DEPENDENCIES)
            import torch
            import soundfile as sf
            from qwen_tts import Qwen3TTSModel

        self.soundfile = sf
        self.torch = torch
        local_model_root = _snapshot_download_model(repo_id=self.model_id, local_dir=self.model_root)
        candidate_devices = (
            ["mps", "cpu"]
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
            else ["cpu"]
        )
        last_error = None
        for device_name in candidate_devices:
            dtype = torch.float16 if device_name == "mps" else torch.float32
            try:
                with contextlib.redirect_stdout(sys.stderr):
                    model = Qwen3TTSModel.from_pretrained(
                        str(local_model_root),
                        device_map=device_name,
                        dtype=dtype,
                    )
                self.device = torch.device(device_name)
                self.audio_model = model
                self.audio_mode = "qwen_tts"
                return
            except Exception as exc:
                last_error = exc
                continue
        if last_error is not None:
            raise last_error
        raise RuntimeError(f"failed to load text-to-audio model {self.model_id}")

    def _load_generic_pipeline(self) -> None:
        assert self.python_bin is not None
        try:
            import torch
            import soundfile as sf
            from transformers import pipeline
        except ImportError:
            _ensure_worker_dependencies(self.python_bin, _PYTORCH_TEXT_TO_AUDIO_DEPENDENCIES)
            import torch
            import soundfile as sf
            from transformers import pipeline

        self.soundfile = sf
        self.torch = torch
        self.device = torch.device(
            "mps"
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
            else "cpu"
        )
        try:
            self.audio_model = pipeline(
                task="text-to-audio",
                model=self.model_id,
                device=self.device,
                trust_remote_code=True,
            )
        except TypeError:
            self.audio_model = pipeline(
                task="text-to-audio",
                model=self.model_id,
                trust_remote_code=True,
            )
        self.audio_mode = "generic"

    def _load(self) -> None:
        if _is_qwen_tts_custom_voice_model(self.model_id):
            self._load_qwen_tts()
            return
        self._load_generic_pipeline()

    def _ensure_loaded(self) -> None:
        if self.audio_model is None:
            self._load()

    def handle(self, payload):
        inputs = _request_inputs(payload)
        try:
            if _bool_input(inputs.get("prepare_only")):
                self._ensure_loaded()
                return self._ok(
                    outputs={"prepared": True},
                    message=f"prepared audio model {self.model_id}",
                )
            text = str(inputs.get("text", "")).strip()
            if not text:
                return self._error(error_code="audio_text_missing", message="text is required")
            workspace_root = _resolve_service_workspace_root(payload)
            output_path = _resolve_workspace_path(
                workspace_root,
                str(inputs.get("output_path", "")).strip(),
                expect_exists=False,
            )
            self._ensure_loaded()
            assert self.audio_model is not None
            assert self.soundfile is not None
            sample_rate = _DEFAULT_AUDIO_SAMPLE_RATE
            if self.audio_mode == "qwen_tts":
                language = str(inputs.get("language", "")).strip() or "Auto"
                speaker = str(inputs.get("speaker", "")).strip() or "Vivian"
                instruct = str(inputs.get("instruct", "")).strip()
                if self.torch is not None and inputs.get("seed") not in (None, ""):
                    self.torch.manual_seed(int(inputs.get("seed")))
                generation_kwargs = {
                    key: value
                    for key, value in inputs.items()
                    if key not in {"text", "output_path", "language", "speaker", "instruct", "prepare_only", "seed"}
                    and value not in (None, "")
                }
                call_kwargs = {
                    "text": text,
                    "language": language,
                    "speaker": speaker,
                }
                if instruct:
                    call_kwargs["instruct"] = instruct
                call_kwargs.update(generation_kwargs)
                with contextlib.redirect_stdout(sys.stderr):
                    wavs, sr = self.audio_model.generate_custom_voice(**call_kwargs)
                audio = wavs[0] if isinstance(wavs, (list, tuple)) else wavs
                sample_rate = int(sr or _DEFAULT_AUDIO_SAMPLE_RATE)
            else:
                extra_kwargs = {
                    key: value
                    for key, value in inputs.items()
                    if key not in {"text", "output_path", "prepare_only"} and value not in (None, "")
                }
                result = self.audio_model(text, **extra_kwargs)
                if not isinstance(result, dict):
                    raise RuntimeError("text-to-audio pipeline returned unexpected payload")
                audio = result.get("audio")
                if audio is None:
                    raise RuntimeError("text-to-audio pipeline did not return audio")
                sample_rate = int(result.get("sampling_rate") or _DEFAULT_AUDIO_SAMPLE_RATE)
            self.soundfile.write(str(output_path), audio, sample_rate)
            rel = str(output_path.relative_to(workspace_root))
            return self._ok(
                outputs={"output_path": rel, "sample_rate": sample_rate},
                message=f"generated audio at {rel}",
            )
        except _MissingHostDependencyError as exc:
            return self._error(error_code="missing_host_dependency", message=str(exc))
        except Exception as exc:
            return self._error(error_code="audio_runtime_error", message=str(exc))
