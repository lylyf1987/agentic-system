# Helix

Helix is an RL-inspired agent framework built around one small control law:

```text
state -> agent -> action -> environment -> state
```

The LLM is the `Agent`. The `Sandbox` is the equipped computer that lets the agent take real actions through bash and python. The `Environment` is everything the agent can inspect or affect through its `bash` and `python` hands. `State` is structured context, not a vague chat log. The loop stays grounded by runtime evidence.

## Motivation

Popular agentic systems have demonstrated that this interaction pattern is powerful, but many production-grade systems, including tools like Codex and Claude Code, have not been available as fully public end-to-end reference implementations.

At the same time, the open community has produced many useful agent design patterns, but not one widely accepted standard architecture.

This project exists to push toward that standardization. Its goal is to express agentic systems in a clean reinforcement-learning-style setting, where `state`, `agent`, `action`, and `environment` are explicit primitives rather than implicit framework conventions. The intent is to make agent design easier to reason about, compare, extend, and teach.

## Elegant Agentic Loop

![Agentic System working loop](design.png)

The illustration above captures the design used throughout the repo:

- `State` is built from `workflow_summary` and `observation`.
- `Agent` reasons over that state and emits exactly one action.
- `Action` can `chat`, `think`, `exec`, or `delegate`.
- `Sandbox` is the agent's computer: it can run bash and python, and it can use skills as installable software.
- `Environment` is everything the agent can inspect or affect through its `bash` and `python` hands: local files, OS resources, databases, APIs, and other external systems.
- `Runtime Evidence` flows back into state through stdout/stderr observations.
- `User` sits outside the loop.
- `Sub-agent Loop` mirrors the same pattern for delegated subtasks over the same runtime and workspace contract.

## Install

```bash
python -m pip install -e .
```

This installs the `Helix` CLI and its required runtime dependencies, including `prompt_toolkit`.
Running the package directly from source without installing dependencies is not a supported setup for the interactive host.

You can also run:

```bash
python -m helix
```

## Quick Start

You need four things:

1. a workspace directory
2. a session id
3. an LLM provider
4. Docker running, because Helix uses the Docker sandbox and managed tool services by default

### Full-Featured Example

```bash
helix \
  --workspace ../test_workspace \
  --provider zai \
  --model glm-5 \
  --mode auto \
  --session-id test_session
```

This launches an agent session with:

- `--workspace` — the project directory the agent works in
- `--provider` / `--model` — the core LLM that drives reasoning
- `--mode auto` — the agent executes without asking for approval (`controlled` prompts before each action)
- `--session-id` — required; names the session so conversation state persists across restarts

### Minimal Local Setup: Ollama

If you want to run the reasoning model locally with Ollama:

```bash
ollama serve
ollama pull llama3.1:8b
```

```bash
helix --workspace . --session-id my-session
```

### Other Providers

```bash
# Z.AI
export ZAI_API_KEY="your-zai-api-key"
helix --workspace . --provider zai --model glm-5 --session-id my-session

# DeepSeek
export DEEPSEEK_API_KEY="your-deepseek-api-key"
helix --workspace . --provider deepseek --model deepseek-chat --session-id my-session

# LM Studio
export LMSTUDIO_BASE_URL="http://localhost:1234/v1"   # only if not using the default
helix --workspace . --provider lmstudio --session-id my-session
```

## Sessions

- `workspace` is the shared project and artifact directory.
- `session_id` is the conversation memory key.
- `--session-id` is required by the current CLI.
- state is saved to `WORKSPACE/sessions/<session_id>/.state/session.json`

Examples:

```bash
# start or resume a named session
helix --workspace . --session-id design-review-01

# start a different session against the same workspace
helix --workspace . --session-id bugfix-02
```

## CLI Essentials

| Argument | Default | Purpose |
|---|---|---|
| `--workspace` | required | workspace root |
| `--session-id` | required | persistent conversation key |
| `--provider` | `ollama` | core model provider |
| `--mode` | `controlled` | `controlled` asks before exec, `auto` does not |
| `--model` | provider default | override the provider's default model |

Provider model defaults when `--model` is omitted:

- `ollama` -> `llama3.1:8b`
- `deepseek` -> `deepseek-chat`
- `zai` -> `glm-5`
- `lmstudio` -> `local-model`
- `openai_compatible` -> `local-model`

## Environment Variables

Precedence is:

1. CLI flags
2. provider-specific environment variables
3. generic OpenAI-compatible environment variables
4. built-in defaults

### Core Model Providers

| Variable | Used by | Default / Notes |
|---|---|---|
| `OLLAMA_BASE_URL` | core provider, `analyze-image` skill | `http://localhost:11434` on the host; Docker execs translate loopback to `host.docker.internal` automatically |
| `OLLAMA_MODEL` | core provider | `llama3.1:8b` |
| `OLLAMA_TIMEOUT_SECONDS` | core provider | `300` |
| `OLLAMA_KEEP_ALIVE` | core provider | optional Ollama keep-alive duration |
| `DEEPSEEK_BASE_URL` | deepseek provider | `https://api.deepseek.com` |
| `DEEPSEEK_API_KEY` | deepseek provider | required for `--provider deepseek` unless `OPENAI_COMPAT_API_KEY` is set |
| `DEEPSEEK_MODEL` | deepseek provider | `deepseek-chat` |
| `LMSTUDIO_BASE_URL` | lmstudio provider | `http://localhost:1234/v1` |
| `LMSTUDIO_API_KEY` | lmstudio provider | optional |
| `LMSTUDIO_MODEL` | lmstudio provider | `local-model` |
| `ZAI_BASE_URL` | zai provider | `https://api.z.ai/api/paas/v4` |
| `ZAI_API_KEY` | zai provider | required for `--provider zai` unless `OPENAI_COMPAT_API_KEY` is set |
| `ZAI_MODEL` | zai provider | `glm-5` |
| `OPENAI_COMPAT_BASE_URL` | generic OpenAI-compatible provider | generic fallback base URL |
| `OPENAI_COMPAT_API_KEY` | generic OpenAI-compatible provider, zai/deepseek fallback | generic fallback API key |
| `OPENAI_COMPAT_MODEL` | generic OpenAI-compatible provider | `local-model` |
| `OPENAI_COMPAT_TIMEOUT_SECONDS` | generic OpenAI-compatible provider | `300` |

### Tool Backends

| Variable | Used by | Default / Notes |
|---|---|---|
| `SEARXNG_BASE_URL` | search-online-context skill | injected automatically by the Docker runtime for sandboxed search execs |
| `HELIX_LOCAL_MODEL_SERVICE_URL` | local MLX/PyTorch skills | injected automatically by the Docker runtime on macOS Apple Silicon; points to the Helix-owned local inference host |
| `HELIX_LOCAL_MODEL_SERVICE_TOKEN` | local MLX/PyTorch skills | injected automatically by the Docker runtime on macOS Apple Silicon |
| `HF_TOKEN` | Hugging Face-backed local MLX/PyTorch skills | recommended before starting Helix so local model downloads/auth work reliably |
| `HELIX_HOME` | shared Helix service state | override the default global Helix home at `~/.helix` |

### Runtime Controls

| Variable | Used by | Default / Notes |
|---|---|---|
| `AGENTIC_DOCKER_SANDBOX_TIMEOUT` | Docker sandbox executor | `1800` seconds |
| `AGENTIC_DOCKER_BUILD_TIMEOUT` | Docker image build / pull steps | `1800` seconds |
| `HELIX_LOCAL_MODEL_SERVICE_WORKER_TIMEOUT` | Helix local inference host worker responses | `1200` seconds |

### Example: Z.AI General API vs Coding API

General API:

```bash
export ZAI_API_KEY="your-zai-api-key"
export ZAI_BASE_URL="https://api.z.ai/api/paas/v4"
helix --workspace . --provider zai --model glm-5 --session-id my-session
```

Coding API:

```bash
export ZAI_API_KEY="your-zai-api-key"
export ZAI_BASE_URL="https://api.z.ai/api/coding/paas/v4"
helix --workspace . --provider zai --model glm-5 --session-id my-session
```

The runtime appends `/chat/completions`, so the coding setup above targets:

```text
https://api.z.ai/api/coding/paas/v4/chat/completions
```

## Runtime Commands

- `/help` — show commands
- `/status` — show runtime configuration and session status
- `/full_history` — open the saved full history view for the current named session
- `/observation` — open the current observation window view
- `/workflow_summary` — open the saved compact summary view
- `/last_prompt` — open the exact last prompt sent to the model
- `/exit` — quit

The inspection commands operate on the active named session.

## Built-In Tool Services

### Image and Audio Skills

The built-in media capabilities are skill-owned rather than CLI-configured:

- `analyze-image` uses `glm-ocr` through a local Ollama daemon
- `generate-image` currently defaults to `uqer1244/MLX-z-image` through the `mlx` backend
- `generate-audio` currently defaults to `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` through the `pytorch` backend
- `generate-video` currently defaults to `Lightricks/LTX-Video::ltxv-13b-0.9.8-dev.safetensors` through the `pytorch` backend

The core Helix CLI does not expose separate media provider/model flags.

`generate-image` is a multi-script skill with two explicit phases:

- first run `scripts/prepare_model.py` to download / warm the model
- then run `scripts/generate_image.py` with `--prompt` plus `--output-dir` or `--output-path`

`generate-audio` is also a multi-script skill with two explicit phases:

- first run `scripts/prepare_model.py` to download / warm the model
- then run `scripts/generate_audio.py` with `--text` plus `--output-dir` or `--output-path`
- the host machine running the local model service must also have `sox` installed on `PATH`
- on macOS, install it with `brew install sox`

`generate-video` is also a multi-script skill with two explicit phases:

- first run `scripts/prepare_model.py` to download / warm the model
- then run `scripts/generate_video.py` with `--prompt` plus `--output-dir` or `--output-path`
- add `--image-path` only when you want image-conditioned video rather than text-only video
- the current LTX integration supports text-to-video and single-image-conditioned video, but not the broader upstream multi-condition or long-duration pipeline variants

`analyze-image` calls Ollama directly from inside the Docker sandbox:

- Helix does not start or stop Ollama for you
- make sure `ollama serve` is running before you launch Helix
- install the OCR model with `ollama pull glm-ocr`
- if you use a non-default Ollama URL, set `OLLAMA_BASE_URL` before launching Helix

On macOS Apple Silicon, the Docker runtime still starts a narrow host-native local inference service for MLX- and PyTorch-backed skills and injects the service URL/token into Docker execs automatically.

That host is stable across skills:

- Helix owns the service lifecycle and worker memory management
- the host exposes a prepare endpoint at `/models/prepare` and an inference endpoint at `/infer`
- skills send a declarative `model_spec` plus the task-specific `task_type`, `workspace_root`, and `inputs`
- adapter selection stays internal to the host; skills do not send an `adapter_type`
- Helix core does not expose image backend flags through the CLI

The `/models/prepare` request envelope is:

- `model_spec`
- `request_timeout_seconds`

The `/infer` request envelope is:

- `task_type`
- `model_spec`
- `workspace_root`
- `request_timeout_seconds`
- `inputs`

The `/infer` response envelope is:

- `status`
- `task_type`
- `backend`
- `model_id`
- `outputs`
- `error_code`
- `message`

Curated v1 support matrix:

- `text_to_image`
  - `pytorch`: supported
  - `mlx`: supported
- `image_to_text`
  - `pytorch`: supported
  - `mlx`: supported
- `text_to_video`
  - `pytorch`: supported
  - `mlx`: returns `unsupported_backend_task`
- `text_image_to_video`
  - `pytorch`: supported
  - `mlx`: returns `unsupported_backend_task`
- `text_to_audio`
  - `pytorch`: supported
  - `mlx`: returns `unsupported_backend_task`

Helix accepts the legacy task name `image_generation` as a temporary alias for `text_to_image`.

So the boundary is:

- Ollama-backed skills call Ollama directly
- MLX- and PyTorch-backed skills call the Helix-owned local inference host

That local inference service is shared across all Helix runtimes on the machine:

- the first active Helix runtime starts it
- later Helix runtimes reuse it
- it is stopped when the last Helix runtime exits

If you use the Hugging Face-backed local MLX/PyTorch skills, export `HF_TOKEN` before launching Helix and restart Helix after changing that token.

### Web Search: SearXNG

The app manages SearXNG for you through Docker.

For normal app usage you do not need to:

- pass a search backend URL
- start a separate local SearXNG container
- wire search execs to a host port

SearXNG is also shared across all Helix runtimes:

- the first active Helix runtime starts the managed SearXNG container
- later Helix runtimes reuse it
- it is stopped when the last Helix runtime exits

At startup the runtime prepares the Docker sandbox image, shared Docker network, workspace-local exec cache, and the managed SearXNG sidecar. Search execs then receive the correct internal `SEARXNG_BASE_URL` automatically.

Manual `SEARXNG_BASE_URL` setup only matters if you run the search scripts outside the Helix runtime.

## Runtime Storage

Workspace-local runtime state:

- `WORKSPACE/sessions/<session_id>/...` for session history and state
- `WORKSPACE/.runtime/docker/cache` for persistent sandbox package/tool cache
- `WORKSPACE/.runtime/tmp` and `WORKSPACE/.runtime/logs` for per-workspace temp files and exec logs

Global shared Helix service state:

- `~/.helix/runtime/services/searxng`
- `~/.helix/runtime/services/local-model-service`
- `~/.helix/cache/local-model-service/pytorch`
- `~/.helix/cache/local-model-service/mlx`
- `~/.helix/runtime/active-runtimes`

Set `HELIX_HOME` if you want those shared global paths rooted somewhere else.

## Troubleshooting

- `Missing API key for provider 'zai'`
  Set `ZAI_API_KEY` or `OPENAI_COMPAT_API_KEY`.
- `Missing API key for provider 'deepseek'`
  Set `DEEPSEEK_API_KEY` or `OPENAI_COMPAT_API_KEY`.
- Ollama connection failures
  Make sure `ollama serve` is running and the model exists locally.
- Search returns `403 Forbidden`
  Your SearXNG `settings.yml` is likely missing `json` under `search.formats`.
- Search returns connection refused
  SearXNG is not running or the configured URL is wrong.

## Repo Layout

```text
helix/
  core/           -> state, action, agent, environment, sandbox
  runtime/        -> loop, approval, host, cli, display, debug
  providers/      -> ollama and OpenAI-compatible providers
  builtin_skills/ -> shipped skills bootstrapped into each workspace
  prompts/        -> system prompt templates
```
