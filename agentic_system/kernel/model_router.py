from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass
class ModelResponse:
    provider: str
    model: str
    text: str


def _post_json(url: str, headers: dict[str, str], payload: dict[str, Any], timeout: int = 60) -> dict[str, Any]:
    req = Request(
        url=url,
        method="POST",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
    except URLError as exc:
        raise RuntimeError(f"Network error: {exc}") from exc


class OllamaAdapter:
    provider = "ollama"

    def __init__(self) -> None:
        base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        self.endpoint = f"{base}/api/generate"
        self.timeout_seconds = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "60"))
        keep_alive = os.getenv("OLLAMA_KEEP_ALIVE", "").strip()
        self.keep_alive = keep_alive or None

    def generate(self, model: str, prompt: str) -> ModelResponse:
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2},
        }
        if self.keep_alive:
            payload["keep_alive"] = self.keep_alive
        headers = {"Content-Type": "application/json"}
        data = _post_json(self.endpoint, headers, payload, timeout=self.timeout_seconds)
        text = str(data.get("response", "") or "")
        if not text and isinstance(data.get("message"), dict):
            text = str(data["message"].get("content", "") or "")
        return ModelResponse(provider=self.provider, model=model, text=text)


class ModelRouter:
    def __init__(self, model_name: str | None = None) -> None:
        self.provider = "ollama"
        self.adapter = OllamaAdapter()
        core_model = model_name or os.getenv(
            "OLLAMA_MODEL_CORE_AGENT",
            os.getenv("OLLAMA_MODEL_THINKING", "llama3.1:8b"),
        )
        summarizer_model = os.getenv("OLLAMA_MODEL_WORKFLOW_SUMMARIZER", core_model)
        self.models: dict[str, str] = {
            "core_agent": core_model,
            "workflow_summarizer": summarizer_model,
            "workflow_history_compactor": summarizer_model,
        }

    def _select_model(self, role: str) -> str:
        role_name = str(role).strip()
        if role_name in self.models:
            return self.models[role_name]
        return self.models["core_agent"]

    @staticmethod
    def _parse_json_payload(text: str) -> dict[str, Any] | None:
        raw = text.strip()
        if not raw:
            return None

        match = re.search(r"<output>(.*?)</output>", raw, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            return None
        block = match.group(1).strip()
        if not block:
            return None
        try:
            parsed = json.loads(block)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict):
            return parsed
        return None

    def generate(
        self,
        role: str = "core_agent",
        final_prompt: str | None = None,
    ) -> dict[str, Any]:
        prompt = str(final_prompt or "").strip()
        if not prompt:
            return {}
        model = self._select_model(role)
        response = self.adapter.generate(
            model=model,
            prompt=prompt,
        )
        payload = self._parse_json_payload(response.text or "")
        if isinstance(payload, dict):
            return payload
        return {}
