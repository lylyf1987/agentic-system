# Agentic System Development Plan

Date: 2026-02-12
Status: Draft v2
Owner: Yang Liu + Codex

## 1) Goal

Deliver a v1 agentic system where:
1. LLM decides loop direction (`next_step`)
2. Runtime enforces execution safety and persistence
3. Skills and LTM evolve and are reusable across sessions
4. Core agent can delegate to scoped subagents

General protocol (reward-free):
1. Observe state (`llm_hist` + memory pack + runtime cursor + capabilities)
2. Decide action (`next_step` + step-structured payload)
3. Runtime executes validated transition and side effects (policy-gated)
4. Record feedback and continue until explicit terminal step

## 2) Build Order

Implementation order:
1. Build runtime foundation
2. Build core agent loop
3. Build skills system
4. Build execution + verification path
5. Build memory pipeline (STM/LTM/index)
6. Build subagent orchestration
7. Harden with observability, recovery, and tests

This order minimizes rework because runtime contracts become stable before higher-level behaviors.

## 3) Phase-by-Phase Plan

### Phase 1: Runtime Foundation

Scope:
1. Define and enforce the reward-free agent-environment loop contract:
- runtime provides observable state
- LLM chooses constrained action from allowed steps
- runtime validates/executes transition and records feedback
2. Define canonical schemas:
- `LLMEnvelope`: `next_step`, `raw_response`, `structured_info`
- step payload schemas (`context`, `plan`, `do_tasks`, `act`, `verify`, `iterate`, `document`, `create_skill`, `promotion_check`, `assign_task`, `report`)
3. Implement runtime dispatcher:
- validates `next_step` against allowed set
- supports terminal tokens: `None`, `none`, `no_next_step`, `null`
- routes to deterministic step handlers
4. Implement procedure records:
- `full_proc_hist` (append-only role-prefixed text)
- `llm_hist` (LLM-facing text history)
- `runtime_cursor` (structured progress pointer)
5. Implement model routing adapter:
- select provider/model per agent role
- normalize request/response interfaces across providers
6. Implement policy gate shell for side effects.
7. Define runtime capability boundary for v1:
- runtime native executors only: `Bash`, `PythonExec`
- web search/fetch is NOT a runtime primitive in v1 (implemented as skills on top of executors)

Done when:
1. Runtime can execute a minimal loop (`context -> report -> terminate`)
2. Invalid `next_step` is rejected/repaired deterministically
3. Every record is appended to `full_proc_hist` and persisted
4. Runtime can switch configured model/provider without changing loop code
5. Action space remains constrained to runtime-allowed steps

### Phase 2: Core Agent Loop

Scope:
1. Implement core loop engine:
- current-step call -> runtime handler -> next-step resolution (`forced > proposed`)
2. Implement step envelope builder:
- `input_context`, `memory_pack`, `capability_snapshot`, `available_next_steps`, `constraints`
3. Ensure loop state is represented through:
- `full_proc_hist` (complete history)
- `llm_hist` (LLM-facing compactable history)
- `runtime_cursor` (structured progress pointer)
4. Add core allowed steps:
- `context`, `retrieve_ltm`, `plan`, `do_tasks`, `act`, `verify`, `iterate`, `create_sub_agent`, `assign_task`, `document`, `create_skill`, `promotion_check`, `report`
5. Implement invalid-step repair call path.

Done when:
1. Core loop can run multi-step turns without hardcoded sequence
2. Forced transitions from runtime handlers override proposed next step
3. Final loop result is appended to histories and returned to user

### Phase 3: Skills System

Scope:
1. Implement skill registry loader from workspace:
- `skills/core-agent/...`
- `skills/all-agents/...`
2. Inject skill metadata snapshot into each LLM step envelope
3. Validate `skills_to_apply` in plan/actions against current metadata and scope
4. Implement skill proposal flow:
- `create_skill` -> validate -> queue/apply via approval policy
5. Add capability snapshot versioning so agents see skill updates next turn.

Done when:
1. Core and subagents receive correct scoped skill metadata
2. Unknown or out-of-scope skill IDs are rejected
3. Approved skill proposals are visible in subsequent turns

### Phase 4: Execution + Verification

Scope:
1. Implement task pipeline:
- `plan` builds task queue
- `do_tasks` selects route (`act` or `assign_task` or done)
2. Implement `act` execution:
- resolve intent to executable draft
- policy-gated `Bash` / `PythonExec` only
- capture stdout/stderr/return code/artifacts/timing
- append compact runtime observation into `llm_hist` and full text into `full_proc_hist`
3. Implement executor governance:
- timeout/kill policy
- working-directory and path constraints
- max-output/resource guardrails
- approval decisions persistence (`once/session/pattern/always`)
4. Implement verification:
- per-action micro verification
- aggregate verification in `verify`
5. Implement `iterate` routing:
- `continue` -> `do_tasks`
- `replan` -> `plan`
- `done` -> `document`
- `ask_user` -> `report`

Done when:
1. One plan can execute end-to-end with evidence-backed verification
2. Policy denials are enforced and auditable
3. Approval and executor limits are enforced consistently
4. Loop can re-enter `plan` or conclude via `document/report`

### Phase 5: Memory Pipeline (STM/LTM/Index)

Scope:
1. Implement context retrieval flow:
- runtime provides `ltm_index_snapshot`
- agent selects `doc_id`s in `context`
- runtime validates IDs and loads selected docs
- runtime builds memory pack for next steps
2. Implement context-window guard:
- token estimate before each LLM call
- STM compaction utility call when overflow
- compact only `llm_hist` (`stm + recent turns`)
3. Implement document commit flow:
- validate memory patch schema
- commit STM by default
- promote to LTM only if reusable + verified + policy-allowed
4. Implement LTM index updates with provenance fields.

Done when:
1. Long sessions continue without context-window failures
2. `full_proc_hist` remains complete while `llm_hist` compacts
3. New LTM entries are retrievable in later sessions

### Phase 6: Subagent Orchestration

Scope:
1. Implement `create_sub_agent` spec validation and runtime registry
2. Implement `assign_task` handoff:
- spawn isolated subagent loop with scoped skills/context
3. Implement subagent allowed steps:
- `context`, `retrieve_ltm`, `plan`, `do_tasks`, `act`, `verify`, `iterate`, `document`, `create_skill`, `promotion_check`, `report`
4. Merge subagent results back to core:
- summary + evidence references + promotion suggestions
5. Enforce v1 boundaries:
- no user-facing subagent output
- no privilege escalation
- no nested subagent creation by default

Done when:
1. Core can assign and consume at least one subagent task in-loop
2. Subagent results are auditable and constrained by policy

### Phase 7: Observability, Recovery, and Hardening

Scope:
1. Persist per-session runtime state:
- `sessions/<id>/state.json`
2. Recovery path:
- reload session state (`full_proc_hist`, `llm_hist`, `runtime_cursor`) and resume safely
3. Add traceability:
- step transitions
- policy decisions
- tool evidence refs
- memory and skill mutation events
4. Add acceptance + regression tests:
- no-tool QA
- tool execution with gate decisions
- long-context compaction
- subagent delegation
- documentation/skill proposal flows
- malformed LLM output handling
5. Add skill-driven web research validation tests:
- `web-research` skill using `Bash/PythonExec`
- ensure no runtime-special web API is required

Done when:
1. Interrupted session resumes with consistent unresolved state
2. All core scenarios pass acceptance tests
3. Runtime invariants are enforced by automated tests
4. Web search/fetch works through skills while runtime remains executor-only

## 4) Suggested First Three Work Items

1. Implement Phase 1 schemas + dispatcher scaffold (`next_step`, terminal tokens, handler routing).
2. Implement dual histories (`full_proc_hist`, `llm_hist`) + `runtime_cursor` update API.
3. Implement minimal Phase 2 core loop with `context -> report` plus invalid-step repair.
