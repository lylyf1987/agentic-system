"""Worker process entrypoint for local model service."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from .protocol import _error_response
from .registry import _build_backend


def _worker_main(args) -> int:
    cache_root = Path(args.cache_root).expanduser().resolve()
    python_bin = Path(sys.executable).resolve()
    raw_model_spec = str(getattr(args, "model_spec_json", "") or "").strip()
    model_spec = json.loads(raw_model_spec) if raw_model_spec else None
    model_root = str(getattr(args, "model_root", "") or "").strip()
    backend = _build_backend(
        task_type=str(args.task_type),
        backend=str(args.backend),
        cache_root=cache_root,
        model_id=str(args.model_id),
        backend_mode=str(args.backend_mode),
        python_bin=python_bin,
        model_spec=model_spec,
        model_root=Path(model_root).expanduser().resolve() if model_root else None,
    )
    print(
        json.dumps(
            {
                "status": "ready",
                "task_type": str(args.task_type),
                "backend": str(args.backend),
                "model_id": str(args.model_id),
                "pid": os.getpid(),
            },
            ensure_ascii=True,
        ),
        flush=True,
    )
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            print(
                json.dumps(
                        _error_response(
                            task_type=str(args.task_type),
                            backend=str(args.backend),
                            model_id=str(args.model_id),
                            error_code="invalid_json",
                        message="worker request must be a JSON object",
                    ),
                    ensure_ascii=True,
                ),
                flush=True,
            )
            continue
        if not isinstance(payload, dict):
            print(
                json.dumps(
                        _error_response(
                            task_type=str(args.task_type),
                            backend=str(args.backend),
                            model_id=str(args.model_id),
                            error_code="invalid_json",
                        message="worker request must be a JSON object",
                    ),
                    ensure_ascii=True,
                ),
                flush=True,
            )
            continue
        try:
            response = backend.handle(payload)
        except Exception as exc:
            response = _error_response(
                task_type=str(args.task_type),
                backend=str(args.backend),
                model_id=str(args.model_id),
                error_code="worker_runtime_error",
                message=str(exc),
            )
        print(json.dumps(response, ensure_ascii=True), flush=True)
    return 0
