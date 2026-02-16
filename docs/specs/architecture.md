# Architecture Spec (Runtime v1)

Date: 2026-02-12
Source of truth:
1. `/Users/yangliu/Projects/Business/AgenticSystem/docs/design/system_design.md`
2. `/Users/yangliu/Projects/Business/AgenticSystem/docs/design/development_plan.md`

## 1) Scope

This spec defines runtime architecture for v1:
1. Terminal-first core agent runtime
2. Agent-controlled step loop (`next_step` chosen by LLM, validated by runtime)
3. Runtime execution boundary (`Bash`, `PythonExec` only)
4. Skill metadata injection and scope filtering
5. Session persistence, recovery, and checkpoints
6. Dual text histories (`full_proc_hist`, `llm_hist`) plus structured audit events
7. Subagent orchestration through isolated scoped loops

Out of scope for runtime primitives in v1:
1. Dedicated runtime web-search/web-fetch APIs
2. Dynamic runtime tool catalogs beyond `Bash` and `PythonExec`
3. Nested subagent trees by default

## 2) Runtime Topology

Core components:
1. `TerminalApp`
2. `RuntimeKernel`
3. `LoopEngine`
4. `StepHandlerRegistry`
5. `ModelRouter`
6. `PolicyEngine`
7. `ExecutorBridge` (`Bash`, `PythonExec`)
8. `SkillRegistry`
9. `MemoryService` (STM/LTM/index interfaces)
10. `PersistenceService`
11. `CheckpointManager`
12. `SubagentManager`

## 3) Ownership Boundary

Agent (LLM) owns:
1. semantic reasoning
2. `next_step` proposal
3. plan/task/verification/documentation payload content
4. promotion/skill recommendations

Runtime owns:
1. schema validation
2. step-handler execution
3. side-effect execution via executors only
4. policy + approval enforcement
5. persistence and replay
6. memory write commit and index updates

## 4) Allowed Step Sets

Core step set:
1. `context`
2. `retrieve_ltm`
3. `plan`
4. `do_tasks`
5. `act`
6. `verify`
7. `iterate`
8. `create_sub_agent`
9. `assign_task`
10. `document`
11. `create_skill`
12. `promotion_check`
13. `report`
14. terminal: `None | none | no_next_step | null`

Subagent step set:
1. `context`
2. `retrieve_ltm`
3. `plan`
4. `do_tasks`
5. `act`
6. `verify`
7. `iterate`
8. `document`
9. `create_skill`
10. `promotion_check`
11. `report`
12. terminal: `None | none | no_next_step | null`

## 5) Loop Engine Contract

Per loop iteration:
1. Build step envelope from runtime state.
2. Build prompt = `SYSTEM_PROMPT + STEP_PROMPT[current_step] + envelope`.
3. Call LLM.
4. Validate output contract:
- `next_step`
- `raw_response`
- `structured_info`
5. Execute deterministic handler for `current_step`.
6. Resolve next step with precedence:
- handler-forced step (if provided)
- else LLM-proposed `next_step`
7. Validate resolved step against allowed set (repair path if invalid).
8. Exit on terminal token.

## 6) Execution Boundary

Runtime executes only:
1. `Bash`
2. `PythonExec`

All other capabilities (including web research/fetch) are skill-level behaviors that compile to Bash/Python execution.

## 7) Skill Loading and Exposure

Skill roots:
1. `<workspace>/skills/core-agent/<skill_name>/SKILL.md`
2. `<workspace>/skills/all-agents/<skill_name>/SKILL.md`

Exposure model:
1. Runtime injects metadata snapshot each LLM call.
2. Core agent sees `core-agent + all-agents`.
3. Subagent sees `all-agents`.
4. Runtime validates referenced skill IDs in plan/actions.

## 8) Request Flow

1. User input enters `TerminalApp`.
2. Runtime appends role-prefixed record (`user>`) to:
- `full_proc_hist` (append-only)
- `llm_hist`
3. `LoopEngine` runs core loop from `context`.
4. Runtime executes step handlers and side effects as needed.
5. Runtime appends observations and agent outputs with role prefix.
6. On `report` or terminal token, runtime streams final response.
7. Runtime appends `loop_end` and `turn_result` records.

## 9) Failure and Recovery

1. Every loop step emits structured events.
2. Every side effect emits start/end evidence.
3. On restart:
- load session `state.json`
- load `events.jsonl`
- load `full_proc_hist`
- restore `llm_hist` (or rebuild from STM + recent turns)
4. Resume from unresolved state with latest checkpoints.

## 10) Security Boundary

1. No bypass path around `PolicyEngine` for executor calls.
2. Subagent constraints are enforced by runtime before execution.
3. Skill selection does not bypass policy gates.
4. High-impact changes require configured approval policies.
