from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def execute(executable: dict[str, Any], workspace: str | Path, timeout_seconds: int = 60) -> dict[str, Any]:
    executor = str(executable.get("executor", ""))
    cwd = Path(workspace).expanduser().resolve()
    cwd.mkdir(parents=True, exist_ok=True)

    started_at = datetime.utcnow().isoformat()
    if executor == "Bash":
        command = str(executable.get("command", ""))
        result = subprocess.run(
            command,
            cwd=str(cwd),
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    elif executor == "PythonExec":
        code = str(executable.get("code", ""))
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    else:
        raise RuntimeError(f"Unsupported executor: {executor}")

    ended_at = datetime.utcnow().isoformat()
    status = "success" if result.returncode == 0 else "error"
    return {
        "executor": executor,
        "started_at": started_at,
        "ended_at": ended_at,
        "return_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "status": status,
        "artifacts": [],
    }


def compact_observation(result: dict[str, Any]) -> dict[str, Any]:
    stdout = str(result.get("stdout", ""))
    stderr = str(result.get("stderr", ""))
    return {
        "status": result.get("status", "error"),
        "return_code": result.get("return_code"),
        "stdout_preview": stdout[:500],
        "stderr_preview": stderr[:500],
    }
