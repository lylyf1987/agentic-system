# Agentic System Design

Date: 2026-02-12
Status: Draft v5 (agent-controlled loop + implementation plan)
Author: Yang Liu + Codex

## 0) Executive Summary

1. LLM as brain, runtime as computer.
- LLM decides what to do next.
- Runtime is the execution environment that lets the LLM observe and interact with the real digital world.

2. Agent-controlled working loop.
- The loop is chosen by the agent (`next_step`) from runtime-provided allowed steps.
- Runtime validates steps, enforces policy, and executes side effects.
- General protocol: observe state, choose action, runtime executes transition, observe feedback.

3. Evolvable system.
- Agents continuously produce documentation updates:
  - STM for session continuity
  - LTM for reusable cross-session knowledge
- Agents can also propose new skills; approved skills become permanently reusable.
- Agents can improve the existing skills

4. Organization-style scaling.
- Core agent decomposes work and assigns tasks to scoped subagents.
- Subagents run isolated loops with constrained permissions and report back to core.

## 1) Design Intent

Build an agentic system that is:
1. Smart in reasoning (`LLM` as brain)
2. Powerful in execution (`runtime` as computer)
3. Safe in autonomy (runtime policy and approval gates)
4. Evolvable over time (shared skill and knowledge growth)

## 2) System Model

Two layers:

1. Brain layer (LLM agents)
- Core agent: user-facing orchestrator
- Subagents: task-scoped executors

2. Runtime layer (control + execution)
- Skill registry loading + refresh
- Prompt/context builder (injects current skill metadata)
- Generic executors (`Bash`, `PythonExec`)
- Memory storage/index/retrieval
- Policy + approval gate
- Persistence, replay, and recovery
- Model routing

Design rule:
1. Intelligence and step decisions stay in agents.
2. Safety, policy, persistence, and execution stay in runtime.

### 2.1 General Agent-Environment Protocol (Reward-Free)

This system follows a reinforcement-learning style interaction pattern without an explicit numeric reward in v1.

Per loop transition:
1. Runtime builds agent-observable state from `llm_hist`, memory pack, capability snapshot, and runtime cursor.
2. LLM chooses an action as `{next_step, structured_info}`.
3. Runtime validates and executes deterministic handlers and side effects (policy-gated).
4. Runtime records environment feedback (`user`, `runtime`, `core_agent`, `sub_agent_*`) into histories.
5. Next decision uses updated state.

S --> LLM --> act info --> runtime execute --> environment update S

Important v1 simplification:
1. Action space is intentionally constrained to runtime-provided allowed steps.
2. Runtime, not the LLM, is the execution trust boundary.

## 3) Roles and Ownership

### 3.1 Core Agent

Owns:
1. user interaction
2. global planning and task decomposition
3. subagent creation/assignment/review
4. final report and high-impact confirmations
5. approval decisions for skill/LTM promotions

Cannot bypass:
1. runtime policy gates
2. approval requirements

### 3.2 Subagents

Own:
1. scoped task execution
2. local decision loop within assigned scope
3. promotion recommendations (skills/LTM) back to core

Cannot:
1. talk directly to end user
2. expand permissions
3. spawn additional subagents by default (v1)

### 3.3 Runtime System

Owns:
1. parsing/loading/injecting skill metadata
2. executor and memory-write policy enforcement
3. persistent state/history
4. indexing and retrieval primitives
5. recovery/replay

Runtime is the trust boundary.

## 4) Agent-Controlled Working Loop

This design does not use a rigid runtime state machine. Runtime offers an allowed step set; the agent chooses `next_step` each turn based on current observations.

Policy constraints:
1. planning is read-only
2. side effects occur only through runtime executors
3. documentation proposal is agent-owned; commit is runtime-owned
4. termination must be explicit (`next_step = null | none | no_next_step | "null"`)

### 4.1 General Loop Protocol

At each decision point:
1. LLM observes state context (`llm_hist` + memory pack + runtime cursor + capability snapshot).
2. LLM decides one next action (`next_step`) and emits step-structured data (`structured_info`).
3. Runtime executes the current step handler and side effects if allowed.
4. Runtime appends feedback to state histories.
5. Loop continues until explicit terminal step.

### 4.2 Allowed Next Steps

Core agent allowed steps:
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
14. `none` (terminate current turn loop)

Subagent allowed steps:
1. `context`
2. `retrieve_ltm`
3. `plan`
4. `do_tasks`
5. `act`
6. `verify`
7. `iterate`
8. `document`
9. `create_skill` (proposal only, routed to core)
10. `promotion_check`
11. `report`
12. `none`

### 4.3 Per-Call Contract (LLM <-> Runtime)

Input to each LLM call:
1. system prompt (role, constraints, allowed steps)
2. current step prompt
3. step envelope (`input_context`, `memory_pack`, `capability_snapshot`, `available_next_steps`, `constraints`)
4. latest runtime observations and active task/action state (when relevant)

Required LLM output:
1. `next_step`
2. `raw_response`
3. `structured_info` (schema depends on current step)

Runtime responsibilities after each call:
1. validate output schema + `next_step`
2. reject/repair invalid next step
3. execute current step handler (deterministic runtime logic for this step)
4. apply policy and approval gates for side effects
5. choose next step using precedence:
- forced step from handler (if any)
- else LLM-proposed `next_step`
6. append role-prefixed text records into:
- `full_proc_hist` (append-only full textual procedure log)
- `llm_hist` (LLM-facing text log, subject to STM compaction)
7. update structured runtime pointer in `runtime_cursor` and persist session state

### 4.4 Plan-to-Execution Handoff

Plan output is executable intent, not immediate execution.

Each planned action should include:
1. `id`
2. `type` (`bash|pythonexec|assign_task`)
3. `skills_to_apply` (skill IDs)
4. `params` (intent params, not raw runtime internals)
5. `risk`
6. referenced verification criteria

Runtime handoff actions:
1. validate plan schema and action constraints
2. validate `skills_to_apply` against current metadata
3. build task queue for `do_tasks`
4. resolve selected task into executable draft (`command`/`code`) or subagent assignment bundle
5. gate and execute in `act` only
6. write observation + evidence back for next LLM decision

### 4.5 Reference Pseudocode (Core Design Idea)

```python
CORE_STEPS = {"context","retrieve_ltm","plan","do_tasks","act","verify","iterate",
              "create_sub_agent","assign_task","document","create_skill","promotion_check",
              "report",None}
SUB_STEPS = {"context","retrieve_ltm","plan","do_tasks","act","verify","iterate",
             "document","create_skill","promotion_check","report",None}
TERMINAL = {None, "none", "no_next_step", "null"}


def append_record(state, role, text, to_llm_hist=True):
    line = f"{role}> : {text}"
    state["full_proc_hist"].append(line)            # never compact/truncate
    persist_full_proc_hist_line(state["session_id"], line)
    if to_llm_hist:
        state["llm_hist"].append(line)


def update_runtime_cursor(state, **updates):
    state["runtime_cursor"].update(updates)


def compact_llm_hist_if_needed(state, caps):
    preview = build_prompt(
        SYSTEM_PROMPT,
        STEP_PROMPTS["context"],
        build_envelope_for_step("context", state, caps, input_hist=state["llm_hist"]),
    )
    if within_context_window(preview):
        return

    # Isolated utility call for STM (runtime-triggered).
    comp = call_llm(
        build_prompt(
            SYSTEM_PROMPT,
            STEP_PROMPTS["stm_compaction"],
            {"llm_hist": state["llm_hist"], "current_stm": state["memory_pack"]["stm"]},
        )
    )
    stm = comp.get("stm", {})
    state["memory_pack"]["stm"] = stm
    recent = state["llm_hist"][-10:]
    state["llm_hist"] = [f"stm> : {stm}"] + recent
    update_runtime_cursor(state, last_event="stm_compaction", retained_turns=len(recent))
    append_record(state, "runtime", "stm_compaction applied", to_llm_hist=False)


def run_agent_loop(state, depth=0):
    current_step = "context"
    while not state["terminated"]:
        caps = load_capability_snapshot(state["agent_kind"])
        compact_llm_hist_if_needed(state, caps)

        envelope = build_envelope_for_step(current_step, state, caps, input_hist=state["llm_hist"])
        prompt = build_prompt(SYSTEM_PROMPT, STEP_PROMPTS[current_step], envelope)
        out = call_llm(prompt)  # {next_step, raw_response, structured_info}
        validate_llm_step_output(current_step, out)

        role = "core_agent" if state["agent_kind"] == "core" else state.get("agent_role", "sub_agent")
        append_record(state, role, out["raw_response"])
        update_runtime_cursor(state, current_step=current_step, proposed_next=out.get("next_step"))

        # Runtime executes deterministic handler for the current step.
        forced_next = None
        info = out["structured_info"]
        if current_step == "context":
            handle_context(state, info)                        # validate selected_doc_ids -> load LTM pack
        elif current_step == "retrieve_ltm":
            handle_retrieve_ltm(state, info)                   # targeted retrieval adds to memory pack
        elif current_step == "plan":
            handle_plan(state, info, caps)                     # validate plan -> build task queue
        elif current_step == "do_tasks":
            forced_next = handle_do_tasks(state, info)         # choose task route: act|assign_task|verify
        elif current_step == "act":
            forced_next = handle_act(state, info, caps)        # policy-gated Bash/PythonExec -> observation
        elif current_step == "verify":
            forced_next = handle_verify(state, info)           # pass/fail -> do_tasks|iterate
        elif current_step == "iterate":
            forced_next = handle_iterate(state, info)          # continue|replan|done|ask_user
        elif current_step == "create_sub_agent":
            forced_next = handle_create_sub_agent(state, info) # runtime registers subagent spec
        elif current_step == "assign_task":
            forced_next = handle_assign_task(state, info)      # run isolated subagent loop
        elif current_step == "document":
            forced_next = handle_document(state, info)         # runtime applies STM/LTM patch
        elif current_step == "create_skill":
            forced_next = handle_create_skill(state, info)     # proposal apply/queue via policy
        elif current_step == "promotion_check":
            forced_next = handle_promotion_check(state, info)  # promotion decision + policy
        elif current_step == "report":
            forced_next = handle_report(state, out["raw_response"], info)

        # Next-step precedence:
        # 1) runtime forced transition from handler
        # 2) LLM-proposed next_step
        proposed = forced_next if forced_next is not None else out.get("next_step")
        next_step = validate_next_step_or_repair(state, proposed)

        if next_step in TERMINAL:
            state["terminated"] = True
            if not state.get("final_report"):
                state["final_report"] = out.get("raw_response", "")
            append_record(state, "runtime", f"loop_end> : {state['final_report']}")
            break
        current_step = next_step

    return state


def run_core_session():
    core = init_agent_state(agent_kind="core", role="core_orchestrator")
    core["full_proc_hist"] = []
    core["llm_hist"] = []
    core["runtime_cursor"] = {}
    while True:
        user_text = get_input()
        if is_exit(user_text):
            break
        append_record(core, "user", user_text)
        update_runtime_cursor(core, last_role="user")
        core["terminated"] = False
        core["final_report"] = None
        result = run_agent_loop(core, depth=0)
        stream(result["final_report"])
        append_record(core, "core_agent", f"turn_result> : {result['final_report']}")
```

Notes:
1. LLM controls loop direction, runtime controls safety and execution.
2. Runtime executes the current step handler first, then resolves next step via `forced > proposed`.
3. `full_proc_hist` is append-only full text history; `llm_hist` is LLM-facing and STM-compacted.
4. Every role message and final loop result is appended with role prefix.
5. `runtime_cursor` is the structured runtime progress pointer.
6. Subagents use the same engine with a restricted allowed-step set.

## 5) Skill-Centric Capability Design

Skill = reusable procedure (how to do work), optionally with scripts.

### 5.1 Skills vs Executors

1. Executors are runtime capabilities.
- `Bash`
- `PythonExec`

2. Skills are agent procedures.
- skills define when/how to use executors safely and repeatably
- skills are not a security boundary
- runtime policy gate remains the security boundary

### 5.2 Scopes

1. `all-agents`
- shared procedures usable by core and subagents

2. `core-agent`
- core-only governance and orchestration procedures

### 5.3 Baseline Prebuilt Skills (v1)

`all-agents`:
1. `web-research` (search + fetch + citation-ready output)
2. `solution-exploration` (resolve unknowns into evidence-backed options)
3. `script-authoring` (author bash/python scripts or one-shot code)
4. `execution-protocol` (safe execution pattern, result capture, retry policy)
5. `workspace-io` (read/write/edit workspace artifacts)
6. `stm-summary` (session compaction summaries for token control)
7. `ltm-documentation` (distill reusable knowledge + metadata proposals)
8. `skill-authoring` (draft/update skill package content)
9. `skill-registration-proposal` (propose new/updated skills for review)

`core-agent`:
1. `subagent-creation` (define role, scope, constraints, handoff)
2. `orchestration` (task decomposition/routing)
3. `subagent-review` (quality/risk review before acceptance)
4. `promotion-approval` (approve/reject skill/LTM promotion proposals)

### 5.4 Folder Contract

```text
<workspace>/skills/
  core-agent/
    <skill_name>/
      SKILL.md
      scripts/
  all-agents/
    <skill_name>/
      SKILL.md
      scripts/
```

`SKILL.md` includes:
1. metadata (`skill_id`, `name`, `scope`, `when_to_use`)
2. required executors (`Bash`, `PythonExec`)
3. input/output schema contract
4. deterministic step procedure
5. script entrypoints
6. safety and policy notes
7. promotion guidance (when to recommend permanent reuse)

### 5.5 Exposure Strategy

1. Runtime injects skill metadata every turn.
2. Agents read full `SKILL.md` on demand.
3. Scope filtering:
- core agent: `core-agent + all-agents`
- subagents: `all-agents`
4. Runtime exposes capability revision so agents know metadata changed.

### 5.6 Evolution and Governance

1. Any agent may propose skill create/update.
2. Subagents may not directly apply global skill changes.
3. Core agent reviews proposals and decides accept/reject.
4. Runtime enforces approval policy and applies accepted changes.
5. Runtime refreshes registry immediately so metadata is visible on next turn.

## 6) Execution Substrate (Simplified)

No separate dynamic tool catalog in v1.

Runtime exposes only:
1. `Bash`
2. `PythonExec`

All reusable behavior lives in skills; scripts run through these two executors.

## 7) Memory Architecture

## 7.1 Memory Layers

1. Session Histories
- `full_proc_hist`: full textual procedure history (append-only)
- `llm_hist`: LLM-facing context history (STM-compacted when needed)

2. Runtime Cursor (session-scoped structured pointer)
- compact structured state for loop progress, active step/task, and last observation summary
- optional STM summary fields used for compaction/re-entry

3. Long-Term Memory (LTM, cross-session)
- reusable knowledge docs (how-to, decisions, failures, patterns)

4. Index Layer
- searchable metadata/chunks for STM/LTM retrieval

## 7.2 Ownership Split (Hybrid)

LLM (documentation skill) owns semantic proposal:
1. query refinement
2. summaries
3. tags/keywords/confidence proposal
4. STM and LTM candidate content

Runtime owns deterministic enforcement:
1. schema validation
2. write destination commit (STM/LTM)
3. dedupe/chunking/index update
4. token budget enforcement for retrieval pack
5. policy gating

## 7.3 Per-Turn Memory Workflow

1. Runtime prepares `input_context` + `ltm_index_snapshot` for current turn.
2. Agent `context` step selects relevant LTM `doc_id`s from the index snapshot.
3. Runtime validates selected IDs and loads selected LTM docs/chunks.
4. Runtime injects compact memory pack (STM context + selected LTM text).
5. Agent runs the rest of its loop (`plan/do_tasks/act/verify/...`) using this memory pack.
6. Agent emits structured documentation patch in `document` step.
7. Runtime validates and commits:
- STM: default
- LTM: only if reusable + verified + policy-allowed
8. Runtime updates LTM index incrementally.

## 7.4 History vs STM Strategy

1. Keep two text histories plus one structured runtime pointer:
- `full_proc_hist`: append-only full text procedure log with explicit role prefix (`user>`, `core_agent>`, `sub_agent_1>`, `runtime>`, etc.)
- `llm_hist`: LLM-facing text log (initially mirrors full text, then compacted by STM when needed)
- `runtime_cursor`: compact structured loop pointer for current progress/state
2. Every new runtime/agent/user record is appended immediately to `full_proc_hist` and persisted in runtime storage.
3. `llm_hist` is used to build prompts:
- use full `llm_hist` when token budget allows
- if overflow, run STM compaction utility and replace older `llm_hist` with `stm + recent turns`
4. Compaction affects only `llm_hist`; `full_proc_hist` is never truncated.
5. Final loop results are appended to both histories so future turns retain closure context.
6. `runtime_cursor` stays compact; detailed chronology remains in histories.

## 7.5 STM Structure

Use one canonical session STM under runtime control. In v1, STM is stored as structured content inside session state (`runtime_cursor` and/or memory pack), not a separate short-term folder.

## 7.6 LTM Storage and Index

Example layout:

```text
<workspace>/knowledge/docs/<doc_id>.md
<workspace>/knowledge/index/catalog.json
```

Minimum LTM index fields:
1. `doc_id`
2. `title`
3. `summary_short`
4. `tags`
5. `keywords`
6. `memory_type` (`ltm`)
7. `scope`
8. `quality_score`
9. `confidence`
10. `source_session_id`
11. `source_event_ids`
12. `agent_role`
13. `created_at`
14. `updated_at`
15. `chunk_count`
16. `content_hash`

LTM metadata generation is produced by documentation skill, then validated/normalized by runtime before commit.

## 7.7 Retrieval Rules

Retrieval order:
1. unresolved STM items
2. recent STM summary
3. top LTM hits

Ranking signals:
1. lexical relevance (FTS/BM25)
2. recency
3. quality/confidence
4. tags/scope match
5. source reliability

Prompt payload shape:
1. active session context
2. relevant long-term knowledge
3. citations/provenance map

## 7.8 Background Maintenance

Keep periodic maintenance as runtime jobs:
1. STM compaction (rollup summaries)
2. LTM dedupe/merge suggestions
3. contradiction marking (“competing hypotheses”)
4. index maintenance/rebuild

This complements, not replaces, per-turn `document` step.

## 8) Safety and Governance

### 8.1 Global User Gate

Only side-effecting executors are gated in v1:
1. `Bash`
2. `PythonExec`

Decision types:
1. deny
2. allow-once
3. allow-session
4. allow-pattern
5. allow-always (risk constrained)

### 8.2 Least Privilege

1. Skills are not a security boundary.
2. Runtime policy gate is the security boundary.
3. Subagents operate under scope + runtime-enforced constraints.

### 8.3 High-Impact Changes

Core agent must confirm with user for materially impactful actions unless standing approval exists.

## 9) Runtime Workspace and Persistence

Runtime workspace is selected by `--workspace`.

Suggested layout:

- `<workspace>/skills/`
- `<workspace>/knowledge/docs/...`
- `<workspace>/knowledge/index/catalog.json`
- `<workspace>/sessions/<session_id>/state.json`

Design intent:
1. skills/LTM are reusable across sessions
2. session records and runtime cursor remain isolated/replayable

## 10) Prompting Contract

### 10.1 Core Agent Prompt Must Include

1. role + authority boundary
2. full loop contract (including `document`)
3. safety/policy constraints
4. skill promotion policy
5. always-current skill metadata
6. memory pack (STM+LTM+citation map)

### 10.2 Subagent Prompt Must Include

1. assigned role/objective
2. same loop contract
3. escalation contract to core
4. promotion recommendation behavior
5. scope-filtered skill metadata
6. scoped memory pack

## 11) Promotion Policy (No Hardcoded Runtime Heuristic)

1. Agent recommends promotion based on repeated successful patterns.
2. Threshold is advisory and prompt-level, not hardcoded runtime trigger.
3. Core agent approves/rejects.
4. Runtime applies only after policy/approval checks.

## 12) v1 Scope and Non-Goals

### 12.1 In Scope

1. terminal-first interaction with core agent
2. subagent delegation
3. always-injected skill metadata
4. execution via `Bash`/`PythonExec`
5. global approval gate for executors
6. hybrid STM/LTM memory pipeline
7. documentation step + LTM index pipeline
8. skill creation/update with approval

### 12.2 Out of Scope

1. recursive subagent trees by default
2. autonomous irreversible actions without approval
3. perfect long-context recall without summarization

## 13) Success Criteria

1. predictable loop behavior across agents
2. clear brain/runtime ownership boundaries
3. token-efficient memory retrieval with provenance
4. safe autonomy via runtime policy gate
5. observable evolution via approved skills and LTM growth

Development plan has been moved to:
- `docs/design/development_plan.md`
