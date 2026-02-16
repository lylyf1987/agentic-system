from __future__ import annotations

from typing import Any
from uuid import uuid4


def validate_llm_step_output(step: str, out: dict[str, Any]) -> None:
    if not isinstance(out, dict):
        raise ValueError(f"{step}: step output must be object")
    for key in ("action", "raw_response"):
        if key not in out:
            raise ValueError(f"{step}: missing {key}")
    if "action_input" not in out:
        if isinstance(out.get("next_step_input"), dict):
            out["action_input"] = out["next_step_input"]
        elif isinstance(out.get("structured_info"), dict):
            out["action_input"] = out["structured_info"]
        else:
            out["action_input"] = {}
    if not isinstance(out.get("action_input"), dict):
        raise ValueError(f"{step}: action_input must be object")


def validate_plan_schema(structured: dict[str, Any], caps: dict[str, Any]) -> dict[str, Any]:
    payload = dict(structured)
    tasks = payload.get("tasks")
    if not isinstance(tasks, list):
        tasks = []
    normalized: list[dict[str, Any]] = []
    allowed = {item.get("skill_id") for item in caps.get("skills_meta", []) if isinstance(item, dict)}
    for item in tasks:
        if not isinstance(item, dict):
            continue
        task_id = str(item.get("task_id") or f"task_{uuid4().hex[:8]}")
        route = str(item.get("route") or "act")
        if route not in {"act", "assign_task"}:
            route = "act"
        task_type = str(item.get("type") or ("assign_task" if route == "assign_task" else "bash"))
        if task_type not in {"bash", "pythonexec", "assign_task"}:
            task_type = "bash"
        skills_to_apply = [str(x) for x in item.get("skills_to_apply", []) if str(x)]
        skills_to_apply = [skill for skill in skills_to_apply if not allowed or skill in allowed]
        normalized.append(
            {
                "task_id": task_id,
                "purpose": str(item.get("purpose", "")),
                "route": route,
                "type": task_type,
                "skills_to_apply": skills_to_apply,
                "params": item.get("params", {}),
                "risk": str(item.get("risk", "medium")),
                "verification_refs": list(item.get("verification_refs", [])),
            }
        )
    payload["tasks"] = normalized
    if not isinstance(payload.get("verification_checks"), list):
        payload["verification_checks"] = []
    return payload


def validate_verify_schema(structured: dict[str, Any]) -> dict[str, Any]:
    checks = structured.get("checks")
    if not isinstance(checks, list):
        checks = []
    overall_passed = bool(structured.get("overall_passed", False))
    gaps = structured.get("gaps")
    if not isinstance(gaps, list):
        gaps = []
    return {
        "checks": checks,
        "overall_passed": overall_passed,
        "gaps": [str(item) for item in gaps],
    }


def validate_subagent_spec(structured: dict[str, Any]) -> dict[str, Any]:
    sub_id = str(structured.get("subagent_id") or f"sub_{uuid4().hex[:6]}")
    return {
        "subagent_id": sub_id,
        "role": str(structured.get("role", "task_executor")),
        "objective": str(structured.get("objective", "Complete assigned task safely.")),
        "constraints": structured.get("constraints", {}),
    }


def validate_assignment(
    structured: dict[str, Any],
    subagents: dict[str, dict[str, Any]],
    active_task: dict[str, Any] | None,
) -> dict[str, Any]:
    sub_id = str(structured.get("subagent_id", "")).strip()
    if not sub_id or sub_id not in subagents:
        if not subagents:
            raise ValueError("No subagent available for assignment")
        sub_id = next(iter(subagents.keys()))
    task_id = str(structured.get("task_id") or (active_task or {}).get("task_id") or "")
    task_bundle = structured.get("task_bundle")
    if not isinstance(task_bundle, dict):
        task_bundle = dict(active_task or {})
    return {
        "subagent_id": sub_id,
        "task_id": task_id,
        "role": subagents[sub_id].get("role", "task_executor"),
        "objective": subagents[sub_id].get("objective", "Complete assigned task."),
        "task_bundle": task_bundle,
        "depth": int(structured.get("depth", 1) or 1),
    }


def validate_memory_patch(structured: dict[str, Any]) -> dict[str, Any]:
    payload = dict(structured)
    if not isinstance(payload.get("stm_update"), dict):
        payload["stm_update"] = {}
    if not isinstance(payload.get("ltm_candidates"), list):
        payload["ltm_candidates"] = []
    return payload


def validate_skill_proposal(structured: dict[str, Any]) -> dict[str, Any]:
    payload = dict(structured)
    payload["action"] = str(payload.get("action", "create"))
    payload["skill_id"] = str(payload.get("skill_id", f"skill_{uuid4().hex[:8]}"))
    payload["scope"] = str(payload.get("scope", "all-agents"))
    payload["why"] = str(payload.get("why", ""))
    if not isinstance(payload.get("artifacts"), dict):
        payload["artifacts"] = {}
    return payload


def validate_promotion_proposal(structured: dict[str, Any]) -> dict[str, Any]:
    payload = dict(structured)
    payload["propose"] = bool(payload.get("propose", False))
    payload["target"] = str(payload.get("target", "skill"))
    payload["name"] = str(payload.get("name", ""))
    payload["scope"] = str(payload.get("scope", "all-agents"))
    if not isinstance(payload.get("evidence_refs"), list):
        payload["evidence_refs"] = []
    return payload
