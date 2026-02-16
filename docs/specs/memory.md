# Memory Spec (Runtime v1)

Date: 2026-02-12

## 1) Memory Layers

1. Full Procedure History (`full_proc_hist`)
- append-only role-prefixed text log
- full human-readable trace for replay and audit

2. LLM History (`llm_hist`)
- LLM-facing text context
- can be compacted by STM when context budget is exceeded

3. Event Log (`runtime_events`)
- structured JSONL audit stream
- authoritative for execution/policy/mutation events

4. Session STM (short-term memory)
- session-scoped summary/open-loops/entities
- used for context compaction and continuity

5. LTM (long-term memory)
- reusable cross-session knowledge docs
- indexed metadata for retrieval

## 2) Storage Layout

Session runtime path:
1. `<workspace>/sessions/<session_id>/state.json`
2. `<workspace>/sessions/<session_id>/events.jsonl`
3. `<workspace>/sessions/<session_id>/full_proc_hist.log`
4. `<workspace>/sessions/<session_id>/llm_hist.log`
5. `<workspace>/sessions/<session_id>/approvals.json`

Checkpoint path:
1. `<workspace>/checkpoints/<session_id>/<checkpoint_id>.json`

STM path:
1. `<workspace>/memory/short_term/<session_id>/session_stm.md`
2. `<workspace>/memory/short_term/<session_id>/open_loops.json`

LTM path:
1. `<workspace>/memory/long_term/docs/<doc_id>.md`
2. `<workspace>/memory/index/catalog.sqlite`

## 3) Write Contract

On every new procedure record:
1. append role-prefixed line to `full_proc_hist.log` (never truncate)
2. append corresponding line to `llm_hist.log` (subject to future compaction)
3. append structured runtime event to `events.jsonl`
4. update `state.json` atomically

Loop completion:
1. append `runtime> : loop_end> : ...`
2. append `core_agent> : turn_result> : ...`

## 4) Context Build Strategy

Per LLM call:
1. prefer full `llm_hist` in envelope input context
2. if token overflow:
- run isolated STM compaction utility call
- update STM snapshot
- replace older `llm_hist` with `stm + recent turns`
3. keep `full_proc_hist` untouched

## 5) Retrieval Strategy

`context` step flow:
1. runtime sends `ltm_index_snapshot`
2. agent selects relevant `doc_id`s
3. runtime validates IDs and loads selected docs
4. runtime builds memory pack for next steps

`retrieve_ltm` step flow:
1. agent requests targeted retrieval
2. runtime fetches and appends retrieved memory observations

## 6) Memory Commit Strategy

`document` step:
1. agent emits structured memory patch proposal
2. runtime validates schema
3. runtime commits STM updates by default
4. runtime commits LTM only if:
- reusable
- verified by evidence
- policy-allowed
5. runtime updates index incrementally

## 7) Recovery Strategy

On restart:
1. load latest session state
2. load full procedure log and event log
3. restore `llm_hist` (or reconstruct from STM + recent full records)
4. load unresolved loops + pending checkpoints
5. resume loop with minimal safe context
