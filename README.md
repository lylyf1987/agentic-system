# Agentic System (Runtime Kernel v1)

## What This Repo Is Now

This repository now implements the new architecture from the updated design/spec docs:
1. LLM-directed step routing (`next_step`)
2. Runtime safety kernel (policy + executors + persistence)
3. Executor-only runtime surface (`Bash`, `PythonExec`)
4. Skill-driven behavior on top of executors (including web research as a skill pattern)

Primary docs:
1. `/Users/yangliu/Projects/Business/AgenticSystem/docs/design/system_design.md`
2. `/Users/yangliu/Projects/Business/AgenticSystem/docs/design/development_plan.md`
3. `/Users/yangliu/Projects/Business/AgenticSystem/docs/specs/architecture.md`
4. `/Users/yangliu/Projects/Business/AgenticSystem/docs/specs/schemas.md`

## Repository Layout

```text
agentic_system/
  kernel/
    constants.py
    state.py
    storage.py
    skills.py
    policy.py
    executors.py
    memory.py
    model_router.py
    prompts.py
    llm.py
    validators.py
    engine.py
  model_router.py
  runtime.py
  cli.py
  skills/
docs/
  design/
  specs/
tests/
```

## Quick Start

```bash
python3 -m agentic_system.cli --mode safe --model-provider openai
```

By default runtime data is stored in:
- `$AGENTIC_RUNTIME_WORKSPACE` (if set), else
- `~/.agentic_system/runtime_workspace`

To force a specific runtime workspace:

```bash
python3 -m agentic_system.cli --workspace /path/to/runtime-workspace
```

Ollama example:

```bash
python3 -m agentic_system.cli \
  --mode safe \
  --workspace . \
  --model-provider ollama \
  --model-name llama3.1:8b
```

Install editable entrypoint:

```bash
python3 -m pip install -e .
agentic-system --mode safe --workspace .
```

## Runtime Commands

- `/help`
- `/status`
- `/exit`

## Execution and Policy

1. Runtime executes only `Bash` and `PythonExec`.
2. Side effects are policy-gated.
3. CLI approval choices:
- deny
- allow-once
- allow-session
- allow-pattern
- allow-always

## Session Persistence

State is written under `<workspace>`:
1. `sessions/<session_id>/state.json`
2. `sessions/<session_id>/events.jsonl`
3. `sessions/<session_id>/full_proc_hist.log`
4. `sessions/<session_id>/llm_hist.log`

Memory paths:
1. `memory/short_term/<session_id>/...`
2. `memory/long_term/docs/...`
3. `memory/index/...`

## Tests

```bash
python3 -m unittest discover -s tests -p "test_*.py"
```
