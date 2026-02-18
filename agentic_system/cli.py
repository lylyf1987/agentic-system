from __future__ import annotations

import argparse
from pathlib import Path

from .runtime import AgentRuntime


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Agentic System (clean runtime kernel)")
    parser.add_argument("--mode", default="safe", choices=["auto", "business", "safe"])
    parser.add_argument("--session-id", default=None)
    parser.add_argument(
        "--workspace",
        required=True,
        help="Runtime workspace path (absolute or relative).",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Ollama model name override (defaults from OLLAMA_MODEL_CORE_AGENT/OLLAMA_MODEL_THINKING).",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    workspace = Path(args.workspace).expanduser().resolve()

    runtime = AgentRuntime(
        workspace=workspace,
        mode=args.mode,
        session_id=args.session_id,
        model_name=args.model_name,
    )
    return runtime.start()


if __name__ == "__main__":
    raise SystemExit(main())
