# Agentic System Development Plan

Date: 2026-02-23
Status: Post-Baseline Roadmap (code-aligned)
Owner: Yang Liu

## 1) Scope of This Plan

This plan starts from the implemented baseline in this repository and defines the next delivery stages for publish-quality evolution.

Baseline is already implemented for:

1. Runtime-controlled core agent loop.
2. Strict output contract validation with retry.
3. Executor model (`bash`/`python`) with approval and cancellation.
4. Session persistence and recovery.
5. Workspace-based prompt and skill bootstrap.
6. Skill-based extensibility and knowledge metadata loading.

## 2) Current Baseline Snapshot

### 2.1 Runtime Guarantees Today

1. Deterministic action handling for `chat_with_requester`, `keep_reasoning`, `exec`.
2. Invalid model output is detected and corrected through regenerate loop.
3. Invalid action selection is repaired with bounded retries.
4. Exec side effects are explicit, auditable, and cancellable.
5. State is persisted atomically and resumable by `session_id`.

### 2.2 Known Baseline Gaps

1. No active sub-agent orchestration path yet.
2. Limited automated integration test coverage.
3. Legacy docs/specs still contain pre-baseline architecture concepts.
4. Streaming parser robustness can be improved further for malformed outputs.

## 3) Roadmap Phases

## Phase 1: Publish-Ready Quality and Stability

Goal:
Make current baseline reliable and explainable for external users.

Work:

1. Expand integration tests for:
- invalid output retries
- invalid action retries
- exec approval cache behavior
- cancellation behavior (`Ctrl+C`, `/cancel`)
- session recovery
2. Add regression tests for prompt compaction behavior.
3. Harden runtime error messages for provider failures (`429`, `503`, network timeouts).
4. Align all docs to current code and remove stale architecture statements.

Done when:

1. Core runtime paths are covered by repeatable tests.
2. Publish docs match behavior observed in runtime.
3. Demo scenarios run consistently across providers.

## Phase 2: Sub-Agent Reintroduction (Scoped)

Goal:
Re-enable delegation without destabilizing baseline loop.

Work:

1. Define sub-agent action contract on top of current runtime model.
2. Re-enable `chat_with_sub_agent` behind explicit runtime guardrails.
3. Introduce scoped skill/context visibility for delegated tasks.
4. Merge sub-agent results into core-agent workflow history as evidence blocks.

Done when:

1. Core agent can delegate and consume at least one real task.
2. Delegated execution is auditable and bounded.
3. Failure and cancellation semantics remain deterministic.

## Phase 3: Knowledge System Maturity

Goal:
Improve long-horizon reasoning through better knowledge lifecycle.

Work:

1. Strengthen knowledge index schema and validation.
2. Add quality/confidence update policy during documentation distillation.
3. Improve `load-knowledge-docs` selection guidance based on metadata relevance.
4. Add maintenance tooling for stale/duplicate knowledge entries.

Done when:

1. Knowledge retrieval is predictable and traceable.
2. Distilled docs remain high-signal over long sessions.
3. Index health can be monitored and repaired.

## Phase 4: Skill Ecosystem Growth

Goal:
Make skills the primary capability expansion path.

Work:

1. Add stronger skill authoring and validation templates.
2. Add richer tool skills (deployment, data ops, environment diagnostics).
3. Standardize skill output contracts for better downstream reasoning.
4. Add skill-level test harness for scripts and output schema checks.

Done when:

1. New capabilities are added mostly by skill packages, not kernel changes.
2. Skill behavior is testable and versionable.
3. Runtime history remains readable as skill volume grows.

## Phase 5: Observability and Governance

Goal:
Increase operational confidence for longer autonomous sessions.

Work:

1. Expand `/status` debugging surfaces for deeper turn introspection.
2. Add structured event logs suitable for offline analysis.
3. Add policy profiles for different trust modes.
4. Add execution budget controls (time/turn/tool budgets) per session.

Done when:

1. Long sessions are diagnosable without raw log spelunking.
2. Governance policies are configurable and enforced.
3. Runtime behavior is explainable turn-by-turn.

## 4) Execution Order (Recommended)

1. Phase 1 (stability + docs alignment)
2. Phase 2 (sub-agents)
3. Phase 3 (knowledge maturity)
4. Phase 4 (skill ecosystem)
5. Phase 5 (observability/governance)

Rationale:
Stabilize and publish the baseline before adding major loop complexity.

## 5) Immediate Next Work Items

1. Add end-to-end tests covering invalid-output and invalid-action retry loops.
2. Rewrite `docs/specs/*` to remove legacy `next_step` model artifacts.
3. Add one public demo script and walkthrough for:
- online research using skill loop
- exec approval workflow
- session recovery with `--session-id`
