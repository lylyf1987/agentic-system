# Schema Spec (Runtime v1)

Date: 2026-02-12

## 1) SessionState

```json
{
  "session_id": "string",
  "created_at": "ISO-8601",
  "updated_at": "ISO-8601",
  "mode": "safe|business|auto",
  "status": "idle|running|blocked|failed",
  "active_task_id": "string|null",
  "agent_kind": "core|sub",
  "terminated": false,
  "final_report": "string|null"
}
```

## 2) StepEnvelope (Runtime -> LLM)

```json
{
  "agent_kind": "core|sub",
  "role": "string",
  "objective": "string",
  "current_step": "context|retrieve_ltm|plan|do_tasks|act|verify|iterate|create_sub_agent|assign_task|document|create_skill|promotion_check|report",
  "input_context": {
    "mode": "full_history|stm_compacted",
    "current_input": "string",
    "full_history": ["string"],
    "stm": "string",
    "recent_exact_turns": ["string"]
  },
  "memory_pack": {
    "stm": {},
    "ltm": []
  },
  "capability_snapshot": {
    "skills_meta": [],
    "executors": ["Bash", "PythonExec"],
    "policy_mode": "string",
    "approval_profile": {}
  },
  "available_next_steps": ["string", null],
  "constraints": {
    "allowed_executors": ["Bash", "PythonExec"],
    "planning_is_read_only": true,
    "allow_delegate": true
  }
}
```

## 3) LLMEnvelope (LLM -> Runtime)

```json
{
  "next_step": "string|null",
  "raw_response": "string",
  "structured_info": {}
}
```

Required:
1. `next_step`
2. `raw_response`
3. `structured_info`

## 4) PlanSpec (`structured_info` for `plan`)

```json
{
  "objective": "string",
  "assumptions": ["string"],
  "tasks": [
    {
      "task_id": "string",
      "purpose": "string",
      "route": "act|assign_task",
      "type": "bash|pythonexec|assign_task",
      "skills_to_apply": ["string"],
      "params": {},
      "risk": "low|medium|high",
      "verification_refs": ["string"]
    }
  ],
  "verification_checks": [
    {
      "id": "string",
      "check": "string",
      "type": "exact_match|contains|artifact_exists|numeric_range|manual_review"
    }
  ],
  "missing_skills": ["string"]
}
```

## 5) DoTasksSpec (`structured_info` for `do_tasks`)

```json
{
  "task_id": "string|null",
  "route": "act|assign_task|done"
}
```

## 6) ActSpec (`structured_info` for `act`)

```json
{
  "task_id": "string",
  "refine_params": {},
  "expected_observation": "string"
}
```

Runtime resolves this to executable draft via skill metadata and task params.

## 7) VerifySpec (`structured_info` for `verify`)

```json
{
  "checks": [
    {
      "id": "string",
      "passed": true,
      "evidence_refs": ["string"]
    }
  ],
  "overall_passed": true,
  "gaps": ["string"]
}
```

## 8) IterateSpec (`structured_info` for `iterate`)

```json
{
  "decision": "continue|replan|done|ask_user",
  "reason": "string"
}
```

## 9) Context Selection Spec (`structured_info` for `context`)

```json
{
  "selected_doc_ids": ["doc_123"],
  "selected_reasons": {
    "doc_123": "string"
  }
}
```

## 10) RetrieveLTM Spec (`structured_info` for `retrieve_ltm`)

```json
{
  "ltms": [
    {
      "doc_id": "string",
      "reason": "string"
    }
  ]
}
```

## 11) Document Patch (`structured_info` for `document`)

```json
{
  "stm_update": {
    "summary_delta": "string",
    "open_loops_add": ["string"],
    "open_loops_resolve": ["string"]
  },
  "ltm_candidates": [
    {
      "title": "string",
      "summary_short": "string",
      "text": "string",
      "tags": ["string"],
      "confidence": 0.8,
      "source_event_ids": ["evt_1"]
    }
  ]
}
```

## 12) Skill Proposal (`structured_info` for `create_skill`)

```json
{
  "action": "create|update",
  "skill_id": "string",
  "scope": "core-agent|all-agents",
  "why": "string",
  "artifacts": {
    "skill_md": "string",
    "scripts": [
      {
        "path": "scripts/name.sh",
        "content": "string"
      }
    ]
  }
}
```

## 13) Promotion Proposal (`structured_info` for `promotion_check`)

```json
{
  "propose": true,
  "target": "skill|ltm",
  "name": "string",
  "scope": "core-agent|all-agents",
  "justification": "string",
  "evidence_refs": ["evt_1"]
}
```

## 14) Subagent Specs

Create subagent (`structured_info` for `create_sub_agent`):

```json
{
  "subagent_id": "sub_1",
  "role": "string",
  "objective": "string",
  "constraints": {
    "max_exec_actions": 20,
    "max_runtime_seconds": 300
  }
}
```

Assign task (`structured_info` for `assign_task`):

```json
{
  "subagent_id": "sub_1",
  "task_id": "task_1",
  "task_bundle": {}
}
```

## 15) RuntimeEvent

```json
{
  "event_id": "evt_123",
  "timestamp": "ISO-8601",
  "session_id": "string",
  "kind": "string",
  "payload": {}
}
```

Common kinds:
1. `user_input`
2. `llm_step_output`
3. `tool_result`
4. `policy_block`
5. `verify`
6. `document_patch_applied`
7. `skill_proposal`
8. `promotion_check`
9. `subagent_created`
10. `subagent_result`
11. `stm_compaction`
12. `limit_reached`

## 16) ExecutionRecord (`tool_result` payload)

```json
{
  "executor": "Bash|PythonExec",
  "command_or_code": "string",
  "started_at": "ISO-8601",
  "ended_at": "ISO-8601",
  "return_code": 0,
  "stdout": "string",
  "stderr": "string",
  "artifacts": ["string"]
}
```
