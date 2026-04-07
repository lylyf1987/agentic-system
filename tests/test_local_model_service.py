"""Local model service tests."""

from __future__ import annotations

import io
import json
import queue
import sys
import tempfile
import threading
import time
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helix.runtime.local_model_service import (
    LocalModelServiceManager,
    _CoordinatorController,
    _RealImageGenerationBackend,
    _RealMLXImageGenerationBackend,
    _RealPyTorchTextToAudioBackend,
    _RealPyTorchVideoGenerationBackend,
    _WorkerState,
    _http_json_request,
    _kill_process_tree,
    default_cache_root,
    default_runtime_root,
)
from helix.runtime.cli import main as runtime_cli_main
from helix.runtime.local_model_service.model_specs import manifest_matches
from helix.runtime.local_model_service.preparer import _hf_cli_command, _hf_download_command


ROOT = Path(__file__).resolve().parent.parent


def _load_model_spec(skill_name: str) -> dict:
    return json.loads(
        (
            ROOT
            / "helix"
            / "builtin_skills"
            / "all-agents"
            / skill_name
            / "model_spec.json"
        ).read_text(encoding="utf-8")
    )


_IMAGE_MODEL_SPEC = _load_model_spec("generate-image")
_AUDIO_MODEL_SPEC = _load_model_spec("generate-audio")
_VIDEO_MODEL_SPEC = _load_model_spec("generate-video")


def _configure_helix_home(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = workspace / ".test-helix-home"
    monkeypatch.setenv("HELIX_HOME", str(home))
    return home


def _start_manager(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    session_id: str = "svc-01",
    idle_seconds: int = 300,
) -> LocalModelServiceManager:
    _configure_helix_home(workspace, monkeypatch)
    manager = LocalModelServiceManager(
        workspace,
        session_id=session_id,
        backend_mode="fake",
        idle_seconds=idle_seconds,
    )
    try:
        manager.start()
    except PermissionError as exc:
        pytest.skip(f"local socket bind is not permitted in this environment: {exc}")
    return manager


def test_local_model_service_start_and_stop(monkeypatch: pytest.MonkeyPatch):
    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        manager = _start_manager(workspace, monkeypatch)
        try:
            status_fields = manager.status_fields()
            status, _, parsed = _http_json_request(
                method="GET",
                url=f"{status_fields['local_model_service']}/health",
            )
            assert status == 200
            assert parsed["status"] == "ok"
            assert parsed["worker_active"] is False
            print("  Local model service start OK")
        finally:
            manager.stop()

        status, _, parsed = _http_json_request(
            method="GET",
            url=f"{status_fields['local_model_service']}/health",
            timeout=2,
        )
        assert status == 0
        assert parsed is None
        print("  Local model service stop OK")


def test_local_model_service_default_cache_root_is_global(monkeypatch: pytest.MonkeyPatch):
    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        helix_home = _configure_helix_home(workspace, monkeypatch)
        expected_cache = helix_home / "cache" / "local-model-service"
        expected_runtime = helix_home / "runtime" / "services" / "local-model-service"
        assert default_cache_root(workspace) == expected_cache.resolve()
        assert default_runtime_root() == expected_runtime.resolve()

        manager = LocalModelServiceManager(
            workspace,
            session_id="svc-default-cache",
            backend_mode="fake",
        )
        try:
            manager.start()
        except PermissionError as exc:
            pytest.skip(f"local socket bind is not permitted in this environment: {exc}")
        try:
            assert manager.cache_root == expected_cache.resolve()
            assert manager.cache_root.exists()
            assert manager.runtime_dir == expected_runtime.resolve()
            assert manager.state_path == expected_runtime.resolve() / "service.json"
            print("  Local model service global cache/runtime root OK")
        finally:
            manager.stop()


def test_local_model_service_auth_and_path_validation(monkeypatch: pytest.MonkeyPatch):
    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        manager = _start_manager(workspace, monkeypatch)
        try:
            env = manager.tool_environment()
            base_url = env["HELIX_LOCAL_MODEL_SERVICE_URL"].replace("host.docker.internal", "127.0.0.1")

            status, _, parsed = _http_json_request(
                method="POST",
                url=f"{base_url}/infer",
                payload={
                    "task_type": "text_to_image",
                    "backend": "pytorch",
                    "model_id": "Tongyi-MAI/Z-Image-Turbo",
                    "workspace_root": str(workspace.resolve()),
                    "inputs": {
                        "prompt": "hello",
                        "size": "1024x1024",
                        "output_path": "generated_images/test.png",
                    },
                },
            )
            assert status == 401
            assert parsed["error_code"] == "unauthorized"

            status, _, parsed = _http_json_request(
                method="POST",
                url=f"{base_url}/infer",
                token=env["HELIX_LOCAL_MODEL_SERVICE_TOKEN"],
                payload={
                    "task_type": "text_to_image",
                    "backend": "pytorch",
                    "workspace_root": str(workspace.resolve()),
                    "model_id": "Tongyi-MAI/Z-Image-Turbo",
                    "inputs": {
                        "prompt": "hello",
                        "size": "1024x1024",
                        "output_path": "../escape.png",
                    },
                },
            )
            assert status == 400
            assert parsed["error_code"] == "invalid_request"
            print("  Local model service auth/path validation OK")
        finally:
            manager.stop()


def test_local_model_service_worker_switch_and_idle_eviction(monkeypatch: pytest.MonkeyPatch):
    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        manager = _start_manager(workspace, monkeypatch, idle_seconds=1)
        try:
            env = manager.tool_environment()
            base_url = env["HELIX_LOCAL_MODEL_SERVICE_URL"].replace("host.docker.internal", "127.0.0.1")
            token = env["HELIX_LOCAL_MODEL_SERVICE_TOKEN"]

            status, _, parsed = _http_json_request(
                method="POST",
                url=f"{base_url}/infer",
                token=token,
                payload={
                    "task_type": "text_to_image",
                    "backend": "pytorch",
                    "workspace_root": str(workspace.resolve()),
                    "model_id": "Tongyi-MAI/Z-Image-Turbo",
                    "inputs": {
                        "prompt": "a bright square",
                        "size": "1024x1024",
                        "output_path": "generated_images/first.png",
                    },
                },
            )
            assert status == 200
            assert parsed["status"] == "ok"
            health_status, _, health = _http_json_request(method="GET", url=f"{base_url}/health")
            assert health_status == 200
            first_pid = int(health["worker_pid"])
            assert health["worker_task_type"] == "text_to_image"
            assert health["worker_backend"] == "pytorch"
            assert health["worker_model_id"] == "Tongyi-MAI/Z-Image-Turbo"

            status, _, parsed = _http_json_request(
                method="POST",
                url=f"{base_url}/infer",
                token=token,
                payload={
                    "task_type": "text_to_image",
                    "backend": "pytorch",
                    "workspace_root": str(workspace.resolve()),
                    "model_id": "Tongyi-MAI/Z-Image-Turbo-q4",
                    "inputs": {
                        "prompt": "a darker square",
                        "size": "1024x1024",
                        "output_path": "generated_images/second.png",
                    },
                },
            )
            assert status == 200
            assert parsed["status"] == "ok"
            health_status, _, health = _http_json_request(method="GET", url=f"{base_url}/health")
            assert health_status == 200
            assert health["worker_task_type"] == "text_to_image"
            assert health["worker_backend"] == "pytorch"
            assert health["worker_model_id"] == "Tongyi-MAI/Z-Image-Turbo-q4"
            assert int(health["worker_pid"]) != first_pid

            deadline = time.time() + 5
            while time.time() < deadline:
                _, _, health = _http_json_request(method="GET", url=f"{base_url}/health")
                if health and health.get("worker_active") is False:
                    break
                time.sleep(0.2)
            assert health["worker_active"] is False
            print("  Local model service worker switch/eviction OK")
        finally:
            manager.stop()


def test_local_model_service_accepts_legacy_task_alias(monkeypatch: pytest.MonkeyPatch):
    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        manager = _start_manager(workspace, monkeypatch)
        try:
            env = manager.tool_environment()
            base_url = env["HELIX_LOCAL_MODEL_SERVICE_URL"].replace("host.docker.internal", "127.0.0.1")
            status, _, parsed = _http_json_request(
                method="POST",
                url=f"{base_url}/infer",
                token=env["HELIX_LOCAL_MODEL_SERVICE_TOKEN"],
                payload={
                    "task_type": "image_generation",
                    "backend": "pytorch",
                    "workspace_root": str(workspace.resolve()),
                    "model_id": "Tongyi-MAI/Z-Image-Turbo",
                    "inputs": {
                        "prompt": "legacy task alias",
                        "size": "1024x1024",
                        "output_path": "generated_images/legacy.png",
                    },
                },
            )
            assert status == 200
            assert parsed["status"] == "ok"
            assert parsed["task_type"] == "text_to_image"
            health_status, _, health = _http_json_request(method="GET", url=f"{base_url}/health")
            assert health_status == 200
            assert health["worker_task_type"] == "text_to_image"
        finally:
            manager.stop()


def test_local_model_service_prepare_only_roundtrip(monkeypatch: pytest.MonkeyPatch):
    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        manager = _start_manager(workspace, monkeypatch)
        try:
            env = manager.tool_environment()
            base_url = env["HELIX_LOCAL_MODEL_SERVICE_URL"].replace("host.docker.internal", "127.0.0.1")
            status, _, parsed = _http_json_request(
                method="POST",
                url=f"{base_url}/infer",
                token=env["HELIX_LOCAL_MODEL_SERVICE_TOKEN"],
                payload={
                    "task_type": "text_to_image",
                    "backend": "mlx",
                    "workspace_root": str(workspace.resolve()),
                    "model_id": "uqer1244/MLX-z-image",
                    "inputs": {
                        "prepare_only": True,
                    },
                },
            )
            assert status == 200
            assert parsed["status"] == "ok"
            assert parsed["outputs"]["prepared"] is True
            assert parsed["backend"] == "mlx"
            assert parsed["task_type"] == "text_to_image"
        finally:
            manager.stop()


def test_local_model_service_prepare_api_roundtrip_for_model_spec(monkeypatch: pytest.MonkeyPatch):
    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        manager = _start_manager(workspace, monkeypatch)
        try:
            env = manager.tool_environment()
            base_url = env["HELIX_LOCAL_MODEL_SERVICE_URL"].replace("host.docker.internal", "127.0.0.1")
            status, _, parsed = _http_json_request(
                method="POST",
                url=f"{base_url}/models/prepare",
                token=env["HELIX_LOCAL_MODEL_SERVICE_TOKEN"],
                payload={
                    "model_spec": _IMAGE_MODEL_SPEC,
                    "request_timeout_seconds": 1200,
                },
            )
            assert status == 200
            assert parsed["status"] == "ok"
            assert parsed["model_id"] == _IMAGE_MODEL_SPEC["id"]
            assert parsed["outputs"]["prepared"] is True
            assert "model_root" in parsed["outputs"]
        finally:
            manager.stop()


def test_local_model_service_model_spec_requires_prepare(monkeypatch: pytest.MonkeyPatch):
    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        manager = _start_manager(workspace, monkeypatch)
        try:
            env = manager.tool_environment()
            base_url = env["HELIX_LOCAL_MODEL_SERVICE_URL"].replace("host.docker.internal", "127.0.0.1")
            status, _, parsed = _http_json_request(
                method="POST",
                url=f"{base_url}/infer",
                token=env["HELIX_LOCAL_MODEL_SERVICE_TOKEN"],
                payload={
                    "task_type": "text_to_image",
                    "workspace_root": str(workspace.resolve()),
                    "model_spec": _IMAGE_MODEL_SPEC,
                    "inputs": {
                        "prompt": "missing prepare",
                        "size": "1024x1024",
                        "output_path": "generated_images/test.png",
                    },
                },
            )
            assert status == 400
            assert parsed["error_code"] == "model_not_prepared"
        finally:
            manager.stop()


def test_local_model_service_model_spec_infer_after_prepare(monkeypatch: pytest.MonkeyPatch):
    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        manager = _start_manager(workspace, monkeypatch)
        try:
            env = manager.tool_environment()
            base_url = env["HELIX_LOCAL_MODEL_SERVICE_URL"].replace("host.docker.internal", "127.0.0.1")
            token = env["HELIX_LOCAL_MODEL_SERVICE_TOKEN"]

            status, _, parsed = _http_json_request(
                method="POST",
                url=f"{base_url}/models/prepare",
                token=token,
                payload={
                    "model_spec": _VIDEO_MODEL_SPEC,
                    "request_timeout_seconds": 1200,
                },
            )
            assert status == 200
            assert parsed["status"] == "ok"

            status, _, parsed = _http_json_request(
                method="POST",
                url=f"{base_url}/infer",
                token=token,
                payload={
                    "task_type": "text_to_video",
                    "workspace_root": str(workspace.resolve()),
                    "model_spec": _VIDEO_MODEL_SPEC,
                    "inputs": {
                        "prompt": "A prepared test clip",
                        "output_path": "generated/demo.mp4",
                    },
                },
            )
            assert status == 200
            assert parsed["status"] == "ok"
            assert parsed["model_id"] == _VIDEO_MODEL_SPEC["id"]
        finally:
            manager.stop()


def test_local_model_service_rejects_unsupported_backend_task(monkeypatch: pytest.MonkeyPatch):
    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        manager = _start_manager(workspace, monkeypatch)
        try:
            env = manager.tool_environment()
            base_url = env["HELIX_LOCAL_MODEL_SERVICE_URL"].replace("host.docker.internal", "127.0.0.1")
            status, _, parsed = _http_json_request(
                method="POST",
                url=f"{base_url}/infer",
                token=env["HELIX_LOCAL_MODEL_SERVICE_TOKEN"],
                payload={
                    "task_type": "text_to_audio",
                    "backend": "mlx",
                    "workspace_root": str(workspace.resolve()),
                    "model_id": "mlx-community/Kokoro-82M-bf16",
                    "inputs": {
                        "text": "Hello from test",
                        "output_path": "generated/audio.wav",
                    },
                },
            )
            assert status == 400
            assert parsed["error_code"] == "unsupported_backend_task"
        finally:
            manager.stop()


def test_local_model_service_shared_across_workspaces(monkeypatch: pytest.MonkeyPatch):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        workspace_one = root / "workspace-one"
        workspace_two = root / "workspace-two"
        workspace_one.mkdir()
        workspace_two.mkdir()
        _configure_helix_home(root, monkeypatch)
        manager_one = LocalModelServiceManager(workspace_one, session_id="svc-one", backend_mode="fake", idle_seconds=1)
        manager_two = LocalModelServiceManager(workspace_two, session_id="svc-two", backend_mode="fake", idle_seconds=1)
        try:
            try:
                manager_one.start()
                manager_two.start()
            except PermissionError as exc:
                pytest.skip(f"local socket bind is not permitted in this environment: {exc}")
            env_one = manager_one.tool_environment()
            env_two = manager_two.tool_environment()
            assert env_one == env_two
            base_url = env_one["HELIX_LOCAL_MODEL_SERVICE_URL"].replace("host.docker.internal", "127.0.0.1")
            status, _, parsed = _http_json_request(method="GET", url=f"{base_url}/health")
            assert status == 200
            assert parsed["status"] == "ok"
            manager_one.stop()
            status, _, parsed = _http_json_request(method="GET", url=f"{base_url}/health")
            assert status == 200
            assert parsed["status"] == "ok"
            print("  Local model service sharing OK")
        finally:
            manager_two.stop()


def test_local_model_service_recovers_stale_coordinator_state(monkeypatch: pytest.MonkeyPatch):
    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        manager = _start_manager(workspace, monkeypatch, session_id="svc-stale")
        state = json.loads(manager.state_path.read_text(encoding="utf-8"))
        stale_pid = int(state["pid"])
        _kill_process_tree(stale_pid)

        replacement = LocalModelServiceManager(
            workspace,
            session_id="svc-stale",
            backend_mode="fake",
            idle_seconds=1,
        )
        try:
            replacement.start()
            fresh_state = json.loads(replacement.state_path.read_text(encoding="utf-8"))
            assert int(fresh_state["pid"]) != stale_pid
            print("  Local model service stale coordinator recovery OK")
        finally:
            replacement.stop()


def test_real_image_generation_backend_loads_lazily(monkeypatch: pytest.MonkeyPatch):
    calls = {"count": 0}

    def fake_load(self) -> None:
        calls["count"] += 1
        self.device = "cpu"

        class FakePipeline:
            def __call__(self, *, prompt: str, width: int, height: int):
                class FakeImage:
                    def save(self, path: Path) -> None:
                        Path(path).write_bytes(b"png")

                class FakeResult:
                    images = [FakeImage()]

                assert prompt == "demo prompt"
                assert width == 512
                assert height == 512
                return FakeResult()

        self.pipeline = FakePipeline()

    monkeypatch.setattr(_RealImageGenerationBackend, "_load", fake_load)

    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td).resolve()
        backend = _RealImageGenerationBackend(
            model_id="Tongyi-MAI/Z-Image-Turbo",
            cache_root=workspace / ".cache",
            python_bin=workspace / "python",
        )
        assert calls["count"] == 0

        result = backend.handle(
            {
                "workspace_root": str(workspace),
                "inputs": {
                    "prompt": "demo prompt",
                    "size": "512x512",
                    "output_path": "generated/demo.png",
                },
            }
        )

        assert calls["count"] == 1
        assert result["status"] == "ok"
        assert result["outputs"]["output_path"] == "generated/demo.png"


def test_real_image_generation_backend_prepare_only_loads_lazily(monkeypatch: pytest.MonkeyPatch):
    calls = {"count": 0}

    def fake_load(self) -> None:
        calls["count"] += 1
        self.device = "cpu"
        self.pipeline = object()

    monkeypatch.setattr(_RealImageGenerationBackend, "_load", fake_load)

    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td).resolve()
        backend = _RealImageGenerationBackend(
            model_id="Tongyi-MAI/Z-Image-Turbo",
            cache_root=workspace / ".cache",
            python_bin=workspace / "python",
        )
        result = backend.handle(
            {
                "workspace_root": str(workspace),
                "inputs": {
                    "prepare_only": True,
                },
            }
        )

        assert calls["count"] == 1
        assert result["status"] == "ok"
        assert result["outputs"]["prepared"] is True


def test_real_mlx_image_generation_backend_loads_lazily(monkeypatch: pytest.MonkeyPatch):
    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td).resolve()
        source_root = workspace / "mlx-source"
        source_root.mkdir(parents=True, exist_ok=True)
        (source_root / "mlx_pipeline.py").write_text(
            "from pathlib import Path\n"
            "class ZImagePipeline:\n"
            "    def __init__(self, model_path, text_encoder_path, repo_id):\n"
            "        self.model_path = model_path\n"
            "        self.text_encoder_path = text_encoder_path\n"
            "        self.repo_id = repo_id\n"
            "    def generate(self, prompt, width, height, steps, seed):\n"
            "        class FakeImage:\n"
            "            def save(self, path):\n"
            "                Path(path).write_bytes(b'mlx-png')\n"
            "        assert prompt == 'mlx prompt'\n"
            "        assert width == 768\n"
            "        assert height == 512\n"
            "        assert steps == 9\n"
            "        assert seed == 42\n"
            "        return FakeImage()\n",
            encoding="utf-8",
        )

        monkeypatch.setattr(
            "helix.runtime.local_model_service.adapters.mlx.text_to_image._ensure_mlx_runner_sources",
            lambda cache_root: source_root,
        )
        monkeypatch.setitem(sys.modules, "mlx", types.ModuleType("mlx"))

        backend = _RealMLXImageGenerationBackend(
            model_id="uqer1244/MLX-z-image",
            cache_root=workspace / ".cache",
            python_bin=workspace / "python",
        )
        assert backend.pipeline is None

        result = backend.handle(
            {
                "workspace_root": str(workspace),
                "inputs": {
                    "prompt": "mlx prompt",
                    "size": "768x512",
                    "output_path": "generated/mlx-demo.png",
                },
            }
        )

        assert backend.pipeline is not None
        assert result["status"] == "ok"
        assert result["outputs"]["output_path"] == "generated/mlx-demo.png"


def test_real_mlx_image_generation_backend_prepare_only_loads_lazily(monkeypatch: pytest.MonkeyPatch):
    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td).resolve()
        source_root = workspace / "mlx-source"
        source_root.mkdir(parents=True, exist_ok=True)
        (source_root / "mlx_pipeline.py").write_text(
            "class ZImagePipeline:\n"
            "    def __init__(self, model_path, text_encoder_path, repo_id):\n"
            "        self.model_path = model_path\n"
            "        self.text_encoder_path = text_encoder_path\n"
            "        self.repo_id = repo_id\n",
            encoding="utf-8",
        )

        monkeypatch.setattr(
            "helix.runtime.local_model_service.adapters.mlx.text_to_image._ensure_mlx_runner_sources",
            lambda cache_root: source_root,
        )
        monkeypatch.setitem(sys.modules, "mlx", types.ModuleType("mlx"))

        backend = _RealMLXImageGenerationBackend(
            model_id="uqer1244/MLX-z-image",
            cache_root=workspace / ".cache",
            python_bin=workspace / "python",
        )
        result = backend.handle(
            {
                "workspace_root": str(workspace),
                "inputs": {
                    "prepare_only": True,
                },
            }
        )

        assert backend.pipeline is not None
        assert result["status"] == "ok"
        assert result["outputs"]["prepared"] is True


def test_real_mlx_image_generation_backend_redirects_pipeline_stdout(capsys: pytest.CaptureFixture[str]):
    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td).resolve()
        backend = _RealMLXImageGenerationBackend(
            model_id="uqer1244/MLX-z-image",
            cache_root=workspace / ".cache",
            python_bin=workspace / "python",
        )

        class FakePipeline:
            def generate(self, *, prompt: str, width: int, height: int, steps: int, seed: int):
                print(f"Pipeline Started | Size: {width}x{height} | Steps: {steps}")

                class FakeImage:
                    def save(self, path: Path) -> None:
                        Path(path).write_bytes(b"mlx-png")

                return FakeImage()

        backend.pipeline = FakePipeline()
        result = backend.handle(
            {
                "workspace_root": str(workspace),
                "inputs": {
                    "prompt": "mlx prompt",
                    "size": "768x512",
                    "output_path": "generated/mlx-demo.png",
                },
            }
        )

        captured = capsys.readouterr()
        assert "Pipeline Started | Size: 768x512 | Steps: 9" not in captured.out
        assert "Pipeline Started | Size: 768x512 | Steps: 9" in captured.err
        assert result["status"] == "ok"
        assert result["outputs"]["output_path"] == "generated/mlx-demo.png"


def test_real_pytorch_text_to_audio_backend_prepare_only_loads_lazily(monkeypatch: pytest.MonkeyPatch):
    calls = {"count": 0}

    def fake_load(self) -> None:
        calls["count"] += 1
        self.audio_model = object()
        self.audio_mode = "qwen_tts"

        class FakeSoundFile:
            @staticmethod
            def write(path: str, audio: object, sample_rate: int) -> None:
                raise AssertionError("prepare_only should not write audio")

        self.soundfile = FakeSoundFile()

    monkeypatch.setattr(_RealPyTorchTextToAudioBackend, "_load", fake_load)

    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td).resolve()
        backend = _RealPyTorchTextToAudioBackend(
            model_id="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            cache_root=workspace / ".cache",
            python_bin=workspace / "python",
        )
        result = backend.handle(
            {
                "workspace_root": str(workspace),
                "inputs": {
                    "prepare_only": True,
                },
            }
        )

        assert calls["count"] == 1
        assert result["status"] == "ok"
        assert result["outputs"]["prepared"] is True


def test_real_pytorch_text_to_audio_backend_reports_missing_sox(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "helix.runtime.local_model_service.adapters.pytorch.text_to_audio.shutil.which",
        lambda _name: None,
    )

    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td).resolve()
        backend = _RealPyTorchTextToAudioBackend(
            model_id="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            cache_root=workspace / ".cache",
            python_bin=workspace / "python",
        )
        result = backend.handle(
            {
                "workspace_root": str(workspace),
                "inputs": {
                    "prepare_only": True,
                },
            }
        )

        assert result["status"] == "error"
        assert result["error_code"] == "missing_host_dependency"
        assert "brew install sox" in result["message"]


def test_real_pytorch_text_to_audio_backend_qwen_generation(monkeypatch: pytest.MonkeyPatch):
    seed_calls: list[int] = []

    def fake_load(self) -> None:
        class FakeAudioModel:
            def generate_custom_voice(self, **kwargs):
                assert kwargs["text"] == "Hello from Qwen"
                assert kwargs["language"] == "English"
                assert kwargs["speaker"] == "Ryan"
                assert kwargs["instruct"] == "Speak calmly."
                assert "seed" not in kwargs
                return [[0.1, -0.1, 0.0]], 24000

        class FakeSoundFile:
            writes: list[tuple[str, object, int]] = []

            @classmethod
            def write(cls, path: str, audio: object, sample_rate: int) -> None:
                cls.writes.append((path, audio, sample_rate))
                Path(path).write_bytes(b"wav")

        self.audio_model = FakeAudioModel()
        self.audio_mode = "qwen_tts"
        self.soundfile = FakeSoundFile
        self.torch = types.SimpleNamespace(manual_seed=lambda value: seed_calls.append(value))

    monkeypatch.setattr(_RealPyTorchTextToAudioBackend, "_load", fake_load)

    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td).resolve()
        backend = _RealPyTorchTextToAudioBackend(
            model_id="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            cache_root=workspace / ".cache",
            python_bin=workspace / "python",
        )
        result = backend.handle(
            {
                "workspace_root": str(workspace),
                "inputs": {
                    "text": "Hello from Qwen",
                    "language": "English",
                    "speaker": "Ryan",
                    "instruct": "Speak calmly.",
                    "seed": 123,
                    "output_path": "generated/demo.wav",
                },
            }
        )

        assert result["status"] == "ok"
        assert result["outputs"]["output_path"] == "generated/demo.wav"
        assert result["outputs"]["sample_rate"] == 24000
        assert seed_calls == [123]
        assert (workspace / "generated" / "demo.wav").exists()


def test_real_pytorch_video_generation_backend_prepare_only_loads_lazily(monkeypatch: pytest.MonkeyPatch):
    calls = {"count": 0}

    def fake_load(self) -> None:
        calls["count"] += 1
        self.pipeline = object()

    monkeypatch.setattr(_RealPyTorchVideoGenerationBackend, "_load", fake_load)

    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td).resolve()
        backend = _RealPyTorchVideoGenerationBackend(
            task_type="text_to_video",
            model_id="Lightricks/LTX-Video::ltxv-13b-0.9.8-dev.safetensors",
            cache_root=workspace / ".cache",
            python_bin=workspace / "python",
        )
        result = backend.handle(
            {
                "workspace_root": str(workspace),
                "inputs": {
                    "prepare_only": True,
                },
            }
        )

        assert calls["count"] == 1
        assert result["status"] == "ok"
        assert result["outputs"]["prepared"] is True


def test_real_pytorch_video_generation_backend_image_conditioned_requires_image():
    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td).resolve()
        backend = _RealPyTorchVideoGenerationBackend(
            task_type="text_image_to_video",
            model_id="Lightricks/LTX-Video::ltxv-13b-0.9.8-dev.safetensors",
            cache_root=workspace / ".cache",
            python_bin=workspace / "python",
        )
        result = backend.handle(
            {
                "workspace_root": str(workspace),
                "inputs": {
                    "prompt": "A cinematic reveal",
                    "output_path": "generated/demo.mp4",
                },
            }
        )

        assert result["status"] == "error"
        assert result["error_code"] == "video_image_missing"


def test_request_worker_ignores_non_json_stdout_until_json_response():
    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td).resolve()
        runtime_dir = workspace / ".runtime"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        log_path = runtime_dir / "worker.log"
        log_path.write_text("", encoding="utf-8")

        controller = _CoordinatorController(
            cache_root=workspace / ".cache",
            token="secret",
            idle_seconds=300,
            backend_mode="fake",
            runtime_dir=runtime_dir,
        )
        try:
            class FakeProcess:
                def __init__(self) -> None:
                    self.stdin = io.StringIO()
                    self.pid = 12345

                def poll(self):
                    return None

                def terminate(self) -> None:
                    return None

                def wait(self, timeout: float | None = None) -> int:
                    return 0

                def kill(self) -> None:
                    return None

            stdout_queue: "queue.Queue[str]" = queue.Queue()
            stdout_queue.put("Pipeline Started | Size: 1536x1024 | Steps: 9")
            stdout_queue.put('{"status":"ok","outputs":{"output_path":"generated/demo.png"}}')

            worker = _WorkerState(
                task_type="text_to_image",
                backend="mlx",
                model_id="uqer1244/MLX-z-image",
                process=FakeProcess(),
                stdout_queue=stdout_queue,
                stdin_lock=threading.Lock(),
                started_at=time.time(),
                log_handle=io.StringIO(),
                log_path=log_path,
            )

            result = controller._request_worker(worker, {"hello": "world"})

            assert result["status"] == "ok"
            assert result["outputs"]["output_path"] == "generated/demo.png"
            assert worker.process.stdin.getvalue().strip() == '{"hello": "world"}'
        finally:
            controller.close()


def test_request_worker_uses_request_timeout_override():
    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td).resolve()
        runtime_dir = workspace / ".runtime"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        log_path = runtime_dir / "worker.log"
        log_path.write_text("", encoding="utf-8")

        controller = _CoordinatorController(
            cache_root=workspace / ".cache",
            token="secret",
            idle_seconds=300,
            backend_mode="fake",
            runtime_dir=runtime_dir,
        )
        try:
            class FakeProcess:
                def __init__(self) -> None:
                    self.stdin = io.StringIO()
                    self.pid = 12346

                def poll(self):
                    return None

                def terminate(self) -> None:
                    return None

                def wait(self, timeout: float | None = None) -> int:
                    return 0

                def kill(self) -> None:
                    return None

            worker = _WorkerState(
                task_type="text_to_image",
                backend="mlx",
                model_id="uqer1244/MLX-z-image",
                process=FakeProcess(),
                stdout_queue=queue.Queue(),
                stdin_lock=threading.Lock(),
                started_at=time.time(),
                log_handle=io.StringIO(),
                log_path=log_path,
            )

            started = time.monotonic()
            with pytest.raises(RuntimeError, match="worker response timed out"):
                controller._request_worker(worker, {"hello": "world", "request_timeout_seconds": 1})
            elapsed = time.monotonic() - started

            assert elapsed < 2.0
            assert worker.process.stdin.getvalue().strip() == '{"hello": "world", "request_timeout_seconds": 1}'
        finally:
            controller.close()


def test_worker_env_enables_hf_xet_high_performance(monkeypatch: pytest.MonkeyPatch):
    captured_env: dict[str, str] = {}

    class FakeProcess:
        def __init__(self, env: dict[str, str]) -> None:
            self.pid = 43210
            self.stdin = io.StringIO()
            self.stdout = io.StringIO('{"status":"ready","task_type":"text_to_image","backend":"pytorch","model_id":"Tongyi-MAI/Z-Image-Turbo","pid":43210}\n')
            self._env = env

        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            return None

        def wait(self, timeout: float | None = None) -> int:
            return 0

        def kill(self) -> None:
            return None

    def fake_popen(*args, **kwargs):
        nonlocal captured_env
        captured_env = dict(kwargs["env"])
        return FakeProcess(captured_env)

    monkeypatch.setattr(
        "helix.runtime.local_model_service.coordinator._worker_python",
        lambda cache_root: Path("/tmp/fake-python"),
    )
    monkeypatch.setattr("helix.runtime.local_model_service.coordinator.subprocess.Popen", fake_popen)

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        controller = _CoordinatorController(
            cache_root=root / "cache",
            token="token",
            idle_seconds=300,
            backend_mode="fake",
            runtime_dir=root / "runtime",
        )
        try:
            worker = controller._start_worker_locked(
                task_type="text_to_image",
                backend="pytorch",
                model_id="Tongyi-MAI/Z-Image-Turbo",
            )
            assert worker.pid == 43210
            assert worker.backend == "pytorch"
            assert captured_env["HF_HUB_DISABLE_XET"] == "1"
            assert captured_env["HF_HOME"].endswith("/cache/pytorch/models")
        finally:
            controller.close()


def test_local_model_service_rejects_invalid_request_timeout(monkeypatch: pytest.MonkeyPatch):
    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        manager = _start_manager(workspace, monkeypatch)
        try:
            env = manager.tool_environment()
            base_url = env["HELIX_LOCAL_MODEL_SERVICE_URL"].replace("host.docker.internal", "127.0.0.1")
            status, _, parsed = _http_json_request(
                method="POST",
                url=f"{base_url}/infer",
                token=env["HELIX_LOCAL_MODEL_SERVICE_TOKEN"],
                payload={
                    "task_type": "text_to_image",
                    "backend": "mlx",
                    "workspace_root": str(workspace.resolve()),
                    "model_id": "uqer1244/MLX-z-image",
                    "request_timeout_seconds": 0,
                    "inputs": {
                        "prepare_only": True,
                    },
                },
            )
            assert status == 400
            assert parsed["error_code"] == "invalid_request"
            assert "request_timeout_seconds" in parsed["message"]
        finally:
            manager.stop()


def test_local_model_service_requires_backend(monkeypatch: pytest.MonkeyPatch):
    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        manager = _start_manager(workspace, monkeypatch)
        try:
            env = manager.tool_environment()
            base_url = env["HELIX_LOCAL_MODEL_SERVICE_URL"].replace("host.docker.internal", "127.0.0.1")
            status, _, parsed = _http_json_request(
                method="POST",
                url=f"{base_url}/infer",
                token=env["HELIX_LOCAL_MODEL_SERVICE_TOKEN"],
                payload={
                    "task_type": "text_to_image",
                    "workspace_root": str(workspace.resolve()),
                    "model_id": "Tongyi-MAI/Z-Image-Turbo",
                    "inputs": {
                        "prompt": "hello",
                        "size": "1024x1024",
                        "output_path": "generated_images/test.png",
                    },
                },
            )
            assert status == 400
            assert parsed["error_code"] == "backend_missing"
        finally:
            manager.stop()


def test_local_model_service_accepts_supported_fake_tasks(monkeypatch: pytest.MonkeyPatch):
    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        manager = _start_manager(workspace, monkeypatch)
        asset = workspace / "assets" / "demo.png"
        asset.parent.mkdir(parents=True, exist_ok=True)
        asset.write_bytes(b"png")
        try:
            env = manager.tool_environment()
            base_url = env["HELIX_LOCAL_MODEL_SERVICE_URL"].replace("host.docker.internal", "127.0.0.1")
            token = env["HELIX_LOCAL_MODEL_SERVICE_TOKEN"]
            scenarios = [
                {
                    "task_type": "image_to_text",
                    "backend": "pytorch",
                    "workspace_root": str(workspace.resolve()),
                    "model_id": "zai-org/GLM-OCR",
                    "inputs": {
                        "image_path": "assets/demo.png",
                        "query": "Describe it",
                    },
                },
                {
                    "task_type": "image_to_text",
                    "backend": "mlx",
                    "workspace_root": str(workspace.resolve()),
                    "model_id": "mlx-community/phi-3.5-vision-instruct-4bit",
                    "inputs": {
                        "image_path": "assets/demo.png",
                        "query": "Describe it",
                    },
                },
                {
                    "task_type": "text_to_video",
                    "backend": "pytorch",
                    "workspace_root": str(workspace.resolve()),
                    "model_id": "Lightricks/LTX-Video::ltxv-13b-0.9.8-dev.safetensors",
                    "inputs": {
                        "prompt": "A cat running on stage",
                        "output_path": "generated/demo-text-only.mp4",
                    },
                },
                {
                    "task_type": "text_image_to_video",
                    "backend": "pytorch",
                    "workspace_root": str(workspace.resolve()),
                    "model_id": "Lightricks/LTX-Video::ltxv-13b-0.9.8-dev.safetensors",
                    "inputs": {
                        "prompt": "A cat running on stage",
                        "image_path": "assets/demo.png",
                        "output_path": "generated/demo.mp4",
                    },
                },
                {
                    "task_type": "text_to_audio",
                    "backend": "pytorch",
                    "workspace_root": str(workspace.resolve()),
                    "model_id": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                    "inputs": {
                        "text": "Hello there",
                        "output_path": "generated/demo.wav",
                    },
                },
            ]
            for payload in scenarios:
                status, _, parsed = _http_json_request(
                    method="POST",
                    url=f"{base_url}/infer",
                    token=token,
                    payload=payload,
                )
                assert status == 200
                assert parsed["status"] == "ok"
                assert isinstance(parsed["outputs"], dict)
        finally:
            manager.stop()


def test_local_model_service_accepts_mlx_backend_in_fake_mode(monkeypatch: pytest.MonkeyPatch):
    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        manager = _start_manager(workspace, monkeypatch)
        try:
            env = manager.tool_environment()
            base_url = env["HELIX_LOCAL_MODEL_SERVICE_URL"].replace("host.docker.internal", "127.0.0.1")
            status, _, parsed = _http_json_request(
                method="POST",
                url=f"{base_url}/infer",
                token=env["HELIX_LOCAL_MODEL_SERVICE_TOKEN"],
                payload={
                    "task_type": "text_to_image",
                    "backend": "mlx",
                    "workspace_root": str(workspace.resolve()),
                    "model_id": "uqer1244/MLX-z-image",
                    "inputs": {
                        "prompt": "hello",
                        "size": "1024x1024",
                        "output_path": "generated_images/mlx-test.png",
                    },
                },
            )
            assert status == 200
            assert parsed["status"] == "ok"
            assert parsed["outputs"]["output_path"] == "generated_images/mlx-test.png"
            health_status, _, health = _http_json_request(method="GET", url=f"{base_url}/health")
            assert health_status == 200
            assert health["worker_backend"] == "mlx"
            assert health["worker_model_id"] == "uqer1244/MLX-z-image"
        finally:
            manager.stop()


def test_runtime_cli_model_download_uses_prepare_subsystem(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    with tempfile.TemporaryDirectory() as td:
        workspace = Path(td)
        spec_path = workspace / "model_spec.json"
        spec_path.write_text(json.dumps(_AUDIO_MODEL_SPEC), encoding="utf-8")

        def fake_prepare_model_spec(*, cache_root, model_spec, backend_mode, timeout_seconds, progress_stream):
            assert backend_mode == "real"
            assert timeout_seconds == 3600
            assert model_spec["id"] == _AUDIO_MODEL_SPEC["id"]
            assert progress_stream is not None
            return model_spec, cache_root / "pytorch" / "models" / "Qwen"

        monkeypatch.setattr(
            "helix.runtime.cli.prepare_model_spec",
            fake_prepare_model_spec,
        )

        code = runtime_cli_main(["model", "download", "--spec", str(spec_path)])
        captured = json.loads(capsys.readouterr().out.strip())

        assert code == 0
        assert captured["status"] == "ok"
        assert captured["backend"] == "pytorch"
        assert captured["task_type"] == "text_to_audio"
        assert captured["model_root"].endswith("/pytorch/models/Qwen")


def test_hf_cli_command_prefers_hf_binary(tmp_path: Path):
    python_bin = tmp_path / "venv" / "bin" / "python"
    hf_bin = python_bin.parent / "hf"
    hf_bin.parent.mkdir(parents=True, exist_ok=True)
    python_bin.write_text("", encoding="utf-8")
    hf_bin.write_text("", encoding="utf-8")

    assert _hf_cli_command(python_bin) == [str(hf_bin)]


def test_hf_cli_command_falls_back_to_legacy_binary(tmp_path: Path):
    python_bin = tmp_path / "venv" / "bin" / "python"
    legacy_bin = python_bin.parent / "huggingface-cli"
    legacy_bin.parent.mkdir(parents=True, exist_ok=True)
    python_bin.write_text("", encoding="utf-8")
    legacy_bin.write_text("", encoding="utf-8")

    assert _hf_cli_command(python_bin) == [str(legacy_bin)]


def test_hf_cli_command_falls_back_to_module_invocation(tmp_path: Path):
    python_bin = tmp_path / "venv" / "bin" / "python"
    python_bin.parent.mkdir(parents=True, exist_ok=True)
    python_bin.write_text("", encoding="utf-8")

    assert _hf_cli_command(python_bin) == [
        str(python_bin),
        "-m",
        "huggingface_hub.commands.huggingface_cli",
    ]


def test_hf_download_command_uses_positional_patterns_when_possible(tmp_path: Path):
    python_bin = tmp_path / "venv" / "bin" / "python"
    hf_bin = python_bin.parent / "hf"
    hf_bin.parent.mkdir(parents=True, exist_ok=True)
    python_bin.write_text("", encoding="utf-8")
    hf_bin.write_text("", encoding="utf-8")

    cmd = _hf_download_command(
        python_bin=python_bin,
        repo_id="uqer1244/MLX-z-image",
        local_dir=tmp_path / "model",
        include_patterns=["model_index.json", "scheduler/*", "vae/*"],
        exclude_patterns=[],
    )

    assert cmd == [
        str(hf_bin),
        "download",
        "uqer1244/MLX-z-image",
        "model_index.json",
        "scheduler/*",
        "vae/*",
        "--local-dir",
        str(tmp_path / "model"),
    ]


def test_hf_download_command_uses_include_flags_when_excludes_are_present(tmp_path: Path):
    python_bin = tmp_path / "venv" / "bin" / "python"
    hf_bin = python_bin.parent / "hf"
    hf_bin.parent.mkdir(parents=True, exist_ok=True)
    python_bin.write_text("", encoding="utf-8")
    hf_bin.write_text("", encoding="utf-8")

    cmd = _hf_download_command(
        python_bin=python_bin,
        repo_id="example/repo",
        local_dir=tmp_path / "model",
        include_patterns=["*.json", "*.safetensors"],
        exclude_patterns=["*.bin"],
    )

    assert cmd == [
        str(hf_bin),
        "download",
        "example/repo",
        "--include",
        "*.json",
        "--include",
        "*.safetensors",
        "--exclude",
        "*.bin",
        "--local-dir",
        str(tmp_path / "model"),
    ]


def test_image_model_spec_matches_realistic_mlx_layout(tmp_path: Path):
    model_root = tmp_path / "mlx-z-image"
    (model_root / "scheduler").mkdir(parents=True, exist_ok=True)
    (model_root / "tokenizer").mkdir(parents=True, exist_ok=True)
    (model_root / "text_encoder").mkdir(parents=True, exist_ok=True)
    (model_root / "transformer").mkdir(parents=True, exist_ok=True)
    (model_root / "vae").mkdir(parents=True, exist_ok=True)
    (model_root / "model_index.json").write_text("{}", encoding="utf-8")
    (model_root / "scheduler" / "scheduler_config.json").write_text("{}", encoding="utf-8")
    (model_root / "tokenizer" / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (model_root / "text_encoder" / "config.json").write_text("{}", encoding="utf-8")
    (model_root / "text_encoder" / "model.safetensors").write_text("", encoding="utf-8")
    (model_root / "transformer" / "config.json").write_text("{}", encoding="utf-8")
    (model_root / "transformer" / "model.safetensors").write_text("", encoding="utf-8")
    (model_root / "vae" / "config.json").write_text("{}", encoding="utf-8")
    (model_root / "vae" / "diffusion_pytorch_model.safetensors").write_text("", encoding="utf-8")

    assert manifest_matches(model_root, _IMAGE_MODEL_SPEC) is True
