"""Shared adapter base for local model service backends."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..protocol import _error_response, _ok_response


class _BaseBackend:
    def __init__(
        self,
        *,
        task_type: str,
        backend: str,
        model_id: str,
        cache_root: Path | None = None,
        python_bin: Path | None = None,
        model_spec: dict[str, Any] | None = None,
        model_root: Path | None = None,
    ) -> None:
        self.task_type = task_type
        self.backend = backend
        self.model_id = model_id
        self.cache_root = cache_root
        self.python_bin = python_bin
        self.model_spec = model_spec
        self.model_root = model_root

    def _ok(self, *, outputs: dict[str, Any] | None, message: str) -> dict[str, Any]:
        return _ok_response(
            task_type=self.task_type,
            backend=self.backend,
            model_id=self.model_id,
            outputs=outputs,
            message=message,
        )

    def _error(
        self,
        *,
        error_code: str,
        message: str,
        outputs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return _error_response(
            task_type=self.task_type,
            backend=self.backend,
            model_id=self.model_id,
            error_code=error_code,
            message=message,
            outputs=outputs,
        )
