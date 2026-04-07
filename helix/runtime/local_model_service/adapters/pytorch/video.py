"""PyTorch video generation adapter."""

from __future__ import annotations

import inspect

from ..base import _BaseBackend
from ...paths import (
    _PYTORCH_VIDEO_DEPENDENCIES,
    _download_hub_file,
    _ensure_worker_dependencies,
    _safe_model_dir_name,
)
from ...protocol import (
    _bool_input,
    _parse_float,
    _parse_int,
    _parse_size,
    _request_inputs,
    _resolve_service_workspace_root,
    _resolve_workspace_path,
)

_LTX_VIDEO_REPO_ID = "Lightricks/LTX-Video"
_LTX_DEFAULT_CHECKPOINT = "ltxv-13b-0.9.8-dev.safetensors"


def _split_repo_checkpoint(model_id: str) -> tuple[str, str]:
    raw = str(model_id or "").strip()
    if "::" not in raw:
        return raw, ""
    repo_id, checkpoint = raw.split("::", 1)
    return repo_id.strip(), checkpoint.strip()


class _RealPyTorchVideoGenerationBackend(_BaseBackend):
    def __init__(self, *, task_type, model_id, cache_root, python_bin) -> None:
        super().__init__(
            task_type=task_type,
            backend="pytorch",
            model_id=model_id,
            cache_root=cache_root,
            python_bin=python_bin,
        )
        self.pipeline = None
        self.device = None
        self.torch = None
        self.export_to_video = None
        self.load_image = None
        self.call_params: set[str] = set()

    def _load(self) -> None:
        assert self.cache_root is not None
        assert self.python_bin is not None
        try:
            import torch
            from diffusers import (
                AutoencoderKLWan,
                DiffusionPipeline,
                LTXImageToVideoPipeline,
                LTXPipeline,
                WanPipeline,
            )
            from diffusers.utils import export_to_video, load_image
        except ImportError:
            _ensure_worker_dependencies(self.python_bin, _PYTORCH_VIDEO_DEPENDENCIES)
            import torch
            from diffusers import (
                AutoencoderKLWan,
                DiffusionPipeline,
                LTXImageToVideoPipeline,
                LTXPipeline,
                WanPipeline,
            )
            from diffusers.utils import export_to_video, load_image

        device = "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.torch = torch
        self.export_to_video = export_to_video
        self.load_image = load_image
        dtype = torch.float16 if device == "mps" else torch.float32
        hub_cache = self.cache_root / "models"
        hub_cache.mkdir(parents=True, exist_ok=True)
        repo_id, checkpoint_name = _split_repo_checkpoint(self.model_id)
        if repo_id == _LTX_VIDEO_REPO_ID:
            if not checkpoint_name:
                checkpoint_name = _LTX_DEFAULT_CHECKPOINT
            checkpoint_cache = hub_cache / _safe_model_dir_name(repo_id) / "checkpoints"
            checkpoint_path = _download_hub_file(
                repo_id=repo_id,
                filename=checkpoint_name,
                local_dir=checkpoint_cache,
            )
            pipeline_cls = (
                LTXImageToVideoPipeline if self.task_type == "text_image_to_video" else LTXPipeline
            )
            self.pipeline = pipeline_cls.from_single_file(
                str(checkpoint_path),
                config=repo_id,
                cache_dir=str(hub_cache),
                torch_dtype=dtype,
            )
        elif self.model_id == "Wan-AI/Wan2.2-TI2V-5B-Diffusers":
            vae = AutoencoderKLWan.from_pretrained(
                self.model_id,
                subfolder="vae",
                cache_dir=str(hub_cache),
                torch_dtype=torch.float32,
            )
            self.pipeline = WanPipeline.from_pretrained(
                self.model_id,
                vae=vae,
                cache_dir=str(hub_cache),
                torch_dtype=dtype,
            )
        else:
            self.pipeline = DiffusionPipeline.from_pretrained(
                self.model_id,
                cache_dir=str(hub_cache),
                torch_dtype=dtype,
            )
        self.pipeline.to(device)
        self.call_params = set(inspect.signature(self.pipeline.__call__).parameters)

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
                    message=f"prepared video model {self.model_id}",
                )
            prompt = str(inputs.get("prompt", "")).strip()
            if not prompt:
                return self._error(error_code="video_prompt_missing", message="prompt is required")
            workspace_root = _resolve_service_workspace_root(payload)
            output_path = _resolve_workspace_path(
                workspace_root,
                str(inputs.get("output_path", "")).strip(),
                expect_exists=False,
            )
            image_path_text = str(inputs.get("image_path", "")).strip()
            if self.task_type == "text_image_to_video" and not image_path_text:
                return self._error(
                    error_code="video_image_missing",
                    message="image_path is required for text_image_to_video",
                )
            size_text = str(inputs.get("size", "")).strip() or "704x512"
            width, height = _parse_size(size_text)
            num_frames = _parse_int(inputs.get("num_frames"), default=161, minimum=1)
            fps = _parse_int(inputs.get("fps"), default=25, minimum=1)
            num_inference_steps = _parse_int(inputs.get("num_inference_steps"), default=50, minimum=1)
            guidance_scale = _parse_float(inputs.get("guidance_scale"), default=5.0, minimum=0.0)
            decode_timestep = _parse_float(inputs.get("decode_timestep"), default=0.03, minimum=0.0)
            decode_noise_scale = _parse_float(inputs.get("decode_noise_scale"), default=0.025, minimum=0.0)
            guidance_rescale = _parse_float(inputs.get("guidance_rescale"), default=0.0, minimum=0.0)
            max_sequence_length = _parse_int(inputs.get("max_sequence_length"), default=128, minimum=1)
            negative_prompt = str(inputs.get("negative_prompt", "")).strip()
            seed = _parse_int(inputs.get("seed"), default=42, minimum=0)
            self._ensure_loaded()
            assert self.pipeline is not None
            assert self.torch is not None
            assert self.export_to_video is not None
            call_kwargs: dict[str, object] = {}
            if "prompt" in self.call_params:
                call_kwargs["prompt"] = prompt
            if "width" in self.call_params:
                call_kwargs["width"] = width
            if "height" in self.call_params:
                call_kwargs["height"] = height
            if "num_frames" in self.call_params:
                call_kwargs["num_frames"] = num_frames
            if "num_inference_steps" in self.call_params:
                call_kwargs["num_inference_steps"] = num_inference_steps
            if "frame_rate" in self.call_params:
                call_kwargs["frame_rate"] = fps
            if "guidance_scale" in self.call_params:
                call_kwargs["guidance_scale"] = guidance_scale
            if "decode_timestep" in self.call_params:
                call_kwargs["decode_timestep"] = decode_timestep
            if "decode_noise_scale" in self.call_params:
                call_kwargs["decode_noise_scale"] = decode_noise_scale
            if "guidance_rescale" in self.call_params:
                call_kwargs["guidance_rescale"] = guidance_rescale
            if "max_sequence_length" in self.call_params:
                call_kwargs["max_sequence_length"] = max_sequence_length
            if negative_prompt and "negative_prompt" in self.call_params:
                call_kwargs["negative_prompt"] = negative_prompt
            if "generator" in self.call_params:
                call_kwargs["generator"] = self.torch.manual_seed(seed)
            if image_path_text:
                if "image" not in self.call_params:
                    return self._error(
                        error_code="video_conditioning_unsupported",
                        message="this video model does not accept image conditioning",
                    )
                assert self.load_image is not None
                image_path = _resolve_workspace_path(
                    workspace_root,
                    image_path_text,
                    expect_exists=True,
                )
                call_kwargs["image"] = self.load_image(str(image_path))
            result = self.pipeline(**call_kwargs)
            frames = getattr(result, "frames", None)
            if isinstance(frames, list) and frames:
                clip_frames = frames[0] if isinstance(frames[0], list) else frames
            else:
                raise RuntimeError("video pipeline did not return frames")
            self.export_to_video(clip_frames, str(output_path), fps=fps)
            rel = str(output_path.relative_to(workspace_root))
            return self._ok(
                outputs={"output_path": rel, "fps": fps, "num_frames": len(clip_frames)},
                message=f"generated video at {rel}",
            )
        except Exception as exc:
            return self._error(error_code="video_runtime_error", message=str(exc))
