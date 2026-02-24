---
name: Skill Creation
handler: scripts/skill_creation.py
description: Create and improve skills with script-first scaffolding, validation, and structured runtime evidence.
required_tools: exec
recommended_tools: exec
forbidden_tools:
---

# Purpose

Use this skill when skill creation/update work is complex or uncertain and you want deterministic, script-first execution.

# Why Script-First

For uncertain skill tasks, run the helper script immediately so runtime history gets clean, structured evidence in `runtime> stdout`.

# Creation Lifecycle (Recommended)

1. `inspect` target skill status and existing structure.
2. `scaffold` baseline package if missing or outdated.
3. Fill `SKILL.md` with concise procedure and trigger conditions.
4. Implement handler script(s) with explicit runtime logging behavior.
5. `validate` before first use to catch structure/frontmatter/handler issues.
6. Iterate from runtime stderr evidence.

# Runtime Log Contract

1. `stdout` must contain one final JSON object.
2. Reserve `stderr` for unexpected runtime failures only.
3. Keep JSON concise so `workflow_hist` remains readable.

# Helper Script

- Path: `skills/all-agents/skill-creation/scripts/skill_creation.py`
- Supports two actions:
  1. `inspect`: inspect existing skill package status.
  2. `scaffold`: create/update a full skill skeleton (`SKILL.md`, handler script, references/assets dirs).
  3. `validate`: validate skill package quality gates (frontmatter, handler path, runtime logging guidance).

# Preferred Action Input Template

Use `code_type=python` with `script_path` and `script_args` array:

```json
{
  "code_type": "python",
  "script_path": "skills/all-agents/skill-creation/scripts/skill_creation.py",
  "script_args": [
    "--action", "inspect",
    "--skill-id", "search-online-context",
    "--scope", "all-agents"
  ]
}
```

Scaffold example:

```json
{
  "code_type": "python",
  "script_path": "skills/all-agents/skill-creation/scripts/skill_creation.py",
  "script_args": [
    "--action", "scaffold",
    "--skill-id", "new-skill-id",
    "--scope", "all-agents",
    "--description", "One-line purpose of this skill"
  ]
}
```

Validate example:

```json
{
  "code_type": "python",
  "script_path": "skills/all-agents/skill-creation/scripts/skill_creation.py",
  "script_args": [
    "--action", "validate",
    "--skill-id", "new-skill-id",
    "--scope", "all-agents"
  ]
}
```

# Output JSON Shape

```json
{
  "executed_skill": "skill-creation",
  "status": "ok|error",
  "skill_created/updated": "summary string with action result and affected paths"
}
```

# Notes

- Use lowercase hyphenated `skill_id`.
- Use this skill before manually drafting large uncertain skill content.
- For generated scripts, enforce runtime-friendly output:
  - `stdout`: clear, meaningful execution evidence and final result.
  - `stderr`: only actionable failures.
  - final stdout line: one stable JSON object.
