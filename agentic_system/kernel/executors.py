"""Execution backend for runtime `exec` actions (async jobs + sync helper)."""

from __future__ import annotations

import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ExecJob:
    """Runtime handle for one launched exec process and its log metadata."""

    job_id: str
    job_name: str
    process: subprocess.Popen[Any]
    cwd: Path
    stdout_path: Path
    stderr_path: Path
    started_at: float
    write_policy_mode: str = "none"
    write_policy_enabled: bool = False
    write_policy_backend: str = "none"
    write_policy_workspace: str = ""
    write_policy_external_roots: list[str] = field(default_factory=list)


def _normalize_external_write_roots(
    roots: list[str] | None,
    *,
    workspace_root: Path,
) -> list[Path]:
    """Resolve and deduplicate external write roots against workspace context."""
    out: list[Path] = []
    if not isinstance(roots, list):
        return out
    seen: set[str] = set()
    for item in roots:
        raw = str(item or "").strip()
        if not raw:
            continue
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = workspace_root / candidate
        normalized = candidate.resolve()
        normalized_text = str(normalized)
        if normalized_text in seen:
            continue
        seen.add(normalized_text)
        out.append(normalized)
    return out


def _escape_sandbox_path(path: Path) -> str:
    value = str(path)
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _build_macos_workspace_write_policy_profile(
    *,
    workspace_root: Path,
    external_write_roots: list[Path],
) -> str:
    """Build sandbox-exec profile that only allows writes under approved roots."""
    allow_write_roots: list[Path] = [workspace_root]
    allow_write_roots.extend(external_write_roots)
    unique: list[Path] = []
    seen: set[str] = set()
    for item in allow_write_roots:
        value = str(item.resolve())
        if value in seen:
            continue
        seen.add(value)
        unique.append(item.resolve())

    lines = [
        "(version 1)",
        "(allow default)",
        "(deny file-write*)",
    ]
    for root in unique:
        lines.append(f'(allow file-write* (subpath "{_escape_sandbox_path(root)}"))')
    return "\n".join(lines)


def _build_exec_environment(
    *,
    workspace_root: Path,
) -> dict[str, str]:
    """Create child env with runtime-local tmp directories inside workspace."""
    env = dict(os.environ)
    runtime_tmp = workspace_root / ".runtime" / "tmp"
    runtime_tmp.mkdir(parents=True, exist_ok=True)
    env["TMPDIR"] = str(runtime_tmp)
    env["TEMP"] = str(runtime_tmp)
    env["TMP"] = str(runtime_tmp)
    return env


def _normalize_exec_input(action_input: dict[str, object]) -> tuple[str, bool, str, str, list[str]]:
    """Validate and normalize exec action_input into command-building primitives."""
    if not isinstance(action_input, dict):
        raise ValueError("exec action requires object action_input")

    code_type = str(action_input.get("code_type", "bash")).strip().lower()
    script_path = str(action_input.get("script_path", "")).strip()
    script = str(action_input.get("script", "")).strip()
    raw_script_args = action_input.get("script_args", [])
    if isinstance(raw_script_args, (list, tuple)):
        script_args = [str(arg) for arg in raw_script_args if str(arg).strip()]
    elif isinstance(raw_script_args, str):
        raw_args_text = raw_script_args.strip()
        if raw_args_text:
            try:
                script_args = [arg for arg in shlex.split(raw_args_text) if arg.strip()]
            except ValueError:
                script_args = [raw_args_text]
        else:
            script_args = []
    else:
        script_args = []

    normalized_code_type = str(code_type).strip().lower()
    path_value = str(script_path or "").strip()
    script_value = str(script or "").strip()
    args_value = [str(arg) for arg in (script_args or []) if str(arg).strip()]

    has_path = bool(path_value)
    has_script = bool(script_value)
    if has_path == has_script:
        raise ValueError("Exactly one of script_path or script must be provided")
    if has_script and args_value:
        raise ValueError("script_args is only supported when script_path is provided")
    return normalized_code_type, has_path, path_value, script_value, args_value


def _build_exec_command(
    *,
    normalized_code_type: str,
    has_path: bool,
    path_value: str,
    script_value: str,
    args_value: list[str],
) -> list[str]:
    """Build subprocess argv for python/bash and inline/path execution modes."""
    if normalized_code_type == "python":
        if has_path:
            return [sys.executable, path_value, *args_value]
        return [sys.executable, "-c", script_value]
    if normalized_code_type == "bash":
        if has_path:
            return ["bash", path_value, *args_value]
        return ["bash", "-lc", script_value]
    raise ValueError(f"Unsupported code_type: {normalized_code_type}")


def start_exec_job(
    *,
    action_input: dict[str, object],
    workspace: str | Path,
    job_id: str,
    job_name: str = "none",
    write_policy_mode: str = "none",
    external_write_roots: list[str] | None = None,
) -> ExecJob:
    """Launch an exec job, writing stdout/stderr to per-job log files."""
    workspace_root = Path(workspace).expanduser().resolve()
    workspace_root.mkdir(parents=True, exist_ok=True)

    normalized_code_type, has_path, path_value, script_value, args_value = _normalize_exec_input(action_input)
    command = _build_exec_command(
        normalized_code_type=normalized_code_type,
        has_path=has_path,
        path_value=path_value,
        script_value=script_value,
        args_value=args_value,
    )

    runtime_logs = workspace_root / ".runtime" / "logs"
    runtime_logs.mkdir(parents=True, exist_ok=True)
    stdout_fd, stdout_name = tempfile.mkstemp(prefix=f"{job_id}_stdout_", suffix=".log", dir=str(runtime_logs))
    stderr_fd, stderr_name = tempfile.mkstemp(prefix=f"{job_id}_stderr_", suffix=".log", dir=str(runtime_logs))
    stdout_path = Path(stdout_name)
    stderr_path = Path(stderr_name)

    resolved_external_roots = _normalize_external_write_roots(
        external_write_roots,
        workspace_root=workspace_root,
    )
    requested_policy_mode = str(write_policy_mode or "none").strip().lower() or "none"
    write_policy_enabled = requested_policy_mode == "workspace_write_only"
    write_policy_backend = "none"
    launch_command = list(command)
    if write_policy_enabled:
        # Strict auto mode writes are enforced with macOS sandbox-exec profile.
        if sys.platform != "darwin":
            raise RuntimeError(
                "[runtime] write policy workspace_write_only is currently supported only on macOS"
            )
        sandbox_exec = shutil.which("sandbox-exec")
        if not sandbox_exec:
            raise RuntimeError(
                "[runtime] write policy workspace_write_only requires sandbox-exec on macOS"
            )
        profile = _build_macos_workspace_write_policy_profile(
            workspace_root=workspace_root,
            external_write_roots=resolved_external_roots,
        )
        launch_command = [sandbox_exec, "-p", profile, *launch_command]
        write_policy_backend = "sandbox-exec"

    env = _build_exec_environment(workspace_root=workspace_root)
    stdout_file = os.fdopen(stdout_fd, "w", encoding="utf-8")
    stderr_file = os.fdopen(stderr_fd, "w", encoding="utf-8")
    try:
        process = subprocess.Popen(
            launch_command,
            cwd=str(workspace_root),
            stdout=stdout_file,
            stderr=stderr_file,
            start_new_session=True,
            env=env,
        )
    finally:
        stdout_file.close()
        stderr_file.close()

    return ExecJob(
        job_id=job_id,
        job_name=str(job_name or "").strip() or "none",
        process=process,
        cwd=workspace_root,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        started_at=time.time(),
        write_policy_mode=requested_policy_mode,
        write_policy_enabled=write_policy_enabled,
        write_policy_backend=write_policy_backend,
        write_policy_workspace=str(workspace_root),
        write_policy_external_roots=[str(path) for path in resolved_external_roots],
    )


def terminate_exec_job(
    job: ExecJob,
    *,
    reason: str,
    sigint_wait_seconds: float = 1.5,
    sigterm_wait_seconds: float = 1.5,
) -> dict[str, Any]:
    """Terminate a running job process group with escalating signals."""
    if job.process.poll() is not None:
        return {"reason": reason, "signals": []}

    signals_sent: list[str] = []

    def _send(group_signal: int, name: str) -> None:
        if job.process.poll() is not None:
            return
        os.killpg(job.process.pid, group_signal)
        signals_sent.append(name)

    try:
        _send(signal.SIGINT, "SIGINT")
        job.process.wait(timeout=max(0.1, float(sigint_wait_seconds)))
    except subprocess.TimeoutExpired:
        try:
            _send(signal.SIGTERM, "SIGTERM")
            job.process.wait(timeout=max(0.1, float(sigterm_wait_seconds)))
        except subprocess.TimeoutExpired:
            _send(signal.SIGKILL, "SIGKILL")
            job.process.wait(timeout=1.0)
    except ProcessLookupError:
        pass

    return {
        "reason": reason,
        "signals": signals_sent,
    }


def collect_exec_job_result(
    job: ExecJob,
    *,
    stderr_append: str = "",
) -> dict[str, Any]:
    """Collect completed job logs and attach optional runtime stderr note."""
    if job.process.poll() is None:
        job.process.wait()

    stdout = ""
    stderr = ""
    if job.stdout_path.exists():
        stdout = job.stdout_path.read_text(encoding="utf-8", errors="replace")
        job.stdout_path.unlink(missing_ok=True)
    if job.stderr_path.exists():
        stderr = job.stderr_path.read_text(encoding="utf-8", errors="replace")
        job.stderr_path.unlink(missing_ok=True)

    note = str(stderr_append or "").strip()
    if note:
        if stderr and not stderr.endswith("\n"):
            stderr += "\n"
        stderr += note + "\n"

    return {
        "stdout": stdout,
        "stderr": stderr,
        "return_code": int(job.process.returncode or 0),
        "write_policy_enabled": bool(job.write_policy_enabled),
        "write_policy_mode": str(job.write_policy_mode),
        "write_policy_backend": str(job.write_policy_backend),
        "write_policy_workspace": str(job.write_policy_workspace),
        "write_policy_external_roots": [str(item) for item in job.write_policy_external_roots],
    }


def execute(
    *,
    action_input: dict[str, object],
    workspace: str | Path,
    timeout_seconds: int | None = None,
    write_policy_mode: str = "none",
    external_write_roots: list[str] | None = None,
) -> dict[str, str]:
    """Synchronous wrapper used by tests/utility callers over async job API."""
    job = start_exec_job(
        action_input=action_input,
        workspace=workspace,
        job_id="exec_job_sync",
        job_name="sync_exec",
        write_policy_mode=write_policy_mode,
        external_write_roots=external_write_roots,
    )
    try:
        if timeout_seconds is not None:
            timeout_value = max(1, int(timeout_seconds))
            job.process.wait(timeout=timeout_value)
        else:
            job.process.wait()
        result = collect_exec_job_result(job)
        return {
            "stdout": str(result.get("stdout", "")),
            "stderr": str(result.get("stderr", "")),
        }
    except subprocess.TimeoutExpired:
        terminate_exec_job(job, reason="timeout")
        result = collect_exec_job_result(
            job,
            stderr_append="[runtime] exec terminated due to timeout",
        )
        return {
            "stdout": str(result.get("stdout", "")),
            "stderr": str(result.get("stderr", "")),
        }
