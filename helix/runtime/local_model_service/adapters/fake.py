"""Fake backend implementations for local model service tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import _BaseBackend
from ..protocol import (
    _bool_input,
    _DEFAULT_AUDIO_SAMPLE_RATE,
    _DEFAULT_IMAGE_ANALYSIS_QUERY,
    _FAKE_MP4_BYTES,
    _FAKE_PNG_BYTES,
    _FAKE_WAV_BYTES,
    _parse_int,
    _request_inputs,
    _resolve_service_workspace_root,
    _resolve_workspace_path,
)
class _FakeTextToImageBackend(_BaseBackend):
    def handle(self, payload: dict[str, Any]) -> dict[str, Any]:
        inputs = _request_inputs(payload)
        if _bool_input(inputs.get("prepare_only")):
            return self._ok(
                outputs={"prepared": True},
                message=f"prepared placeholder model state for {self.model_id}",
            )
        prompt = str(inputs.get("prompt", "")).strip()
        if not prompt:
            return self._error(error_code="image_prompt_missing", message="prompt is required")
        workspace_root = _resolve_service_workspace_root(payload)
        resolved = _resolve_workspace_path(
            workspace_root,
            str(inputs.get("output_path", "")).strip(),
            expect_exists=False,
        )
        resolved.write_bytes(_FAKE_PNG_BYTES)
        output_path = str(resolved.relative_to(workspace_root))
        return self._ok(
            outputs={"output_path": output_path},
            message=f"generated placeholder image at {output_path}",
        )


class _FakeImageToTextBackend(_BaseBackend):
    def handle(self, payload: dict[str, Any]) -> dict[str, Any]:
        inputs = _request_inputs(payload)
        workspace_root = _resolve_service_workspace_root(payload)
        image_path = _resolve_workspace_path(
            workspace_root,
            str(inputs.get("image_path", "")).strip(),
            expect_exists=True,
        )
        query = str(inputs.get("query", "")).strip() or _DEFAULT_IMAGE_ANALYSIS_QUERY
        return self._ok(
            outputs={"text": f"placeholder analysis for {image_path.name}: {query}"},
            message="generated placeholder image analysis",
        )


class _FakeVideoGenerationBackend(_BaseBackend):
    def handle(self, payload: dict[str, Any]) -> dict[str, Any]:
        inputs = _request_inputs(payload)
        if _bool_input(inputs.get("prepare_only")):
            return self._ok(
                outputs={"prepared": True},
                message=f"prepared placeholder video model state for {self.model_id}",
            )
        prompt = str(inputs.get("prompt", "")).strip()
        if not prompt:
            return self._error(error_code="video_prompt_missing", message="prompt is required")
        workspace_root = _resolve_service_workspace_root(payload)
        if self.task_type == "text_image_to_video":
            image_path_text = str(inputs.get("image_path", "")).strip()
            if not image_path_text:
                return self._error(
                    error_code="video_image_missing",
                    message="image_path is required for text_image_to_video",
                )
            _resolve_workspace_path(
                workspace_root,
                image_path_text,
                expect_exists=True,
            )
        output_path = _resolve_workspace_path(
            workspace_root,
            str(inputs.get("output_path", "")).strip(),
            expect_exists=False,
        )
        output_path.write_bytes(_FAKE_MP4_BYTES)
        rel = str(output_path.relative_to(workspace_root))
        fps = _parse_int(inputs.get("fps"), default=8, minimum=1)
        frames = _parse_int(inputs.get("num_frames"), default=16, minimum=1)
        return self._ok(
            outputs={"output_path": rel, "fps": fps, "num_frames": frames},
            message=f"generated placeholder video at {rel}",
        )


class _FakeTextToAudioBackend(_BaseBackend):
    def handle(self, payload: dict[str, Any]) -> dict[str, Any]:
        inputs = _request_inputs(payload)
        if _bool_input(inputs.get("prepare_only")):
            return self._ok(
                outputs={"prepared": True},
                message=f"prepared placeholder audio model state for {self.model_id}",
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
        output_path.write_bytes(_FAKE_WAV_BYTES)
        rel = str(output_path.relative_to(workspace_root))
        return self._ok(
            outputs={"output_path": rel, "sample_rate": _DEFAULT_AUDIO_SAMPLE_RATE},
            message=f"generated placeholder audio at {rel}",
        )
