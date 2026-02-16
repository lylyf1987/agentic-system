# Policy Spec (Runtime v1)

Date: 2026-02-12

## 1) Core Principle

Runtime is the enforcement boundary.

Decision order:
1. `deny`
2. `ask`
3. `allow`

First matching rule wins.

## 2) Runtime-Enforced Surface

Runtime-native execution in v1:
1. `Bash`
2. `PythonExec`

Notes:
1. Web search/fetch is not a runtime primitive in v1.
2. Skills can implement web research through `Bash`/`PythonExec`.
3. Policy still evaluates resulting executor calls.

## 3) Approval Scopes

When decision is `ask`, user may choose:
1. `deny`
2. `allow-once`
3. `allow-session`
4. `allow-pattern`
5. `allow-always`

## 4) Rule Matching

Pattern examples:
1. `Bash(*)`
2. `PythonExec(*)`
3. `Bash(curl https://example.com/*)`
4. `PythonExec(import requests; ...)`

Matching baseline (v1):
1. tool name + optional summarized argument signature
2. glob-like matching
3. priority-based first match

## 5) Subagent Constraints

Before policy evaluation, runtime enforces subagent limits:
1. allowed step set by agent role
2. max execution actions
3. max runtime seconds
4. budget/cost limits (if configured)
5. scope isolation (no nested subagent creation by default)

If scope/limit fails, call is denied regardless of grants.

## 6) Governance Gates (Non-Executor Mutations)

Runtime also governs:
1. memory commit (`document`)
2. skill proposal application (`create_skill`)
3. promotion application (`promotion_check`)

These are not direct `Bash`/`PythonExec` calls but still require runtime validation and configured approvals.

## 7) Auditing

For each evaluated side effect or governance mutation, record:
1. request summary
2. matched rule/default rationale
3. user decision (if prompted)
4. final outcome
5. evidence references

Persist to:
1. `events.jsonl` (structured)
2. `full_proc_hist.log` (role-prefixed text summary)

## 8) Reset Controls

Runtime commands/APIs should support:
1. list active grants
2. revoke grant by id
3. clear session grants
4. clear persistent grants
