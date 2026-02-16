from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Any, Callable


@dataclass
class PolicyDecision:
    signature: str
    decision: str  # allow|ask|deny
    reason: str


class PolicyEngine:
    def __init__(self, grants: dict[str, list[dict[str, str]]] | None = None) -> None:
        payload = grants or {"session": [], "persistent": []}
        self.session_grants = list(payload.get("session", []))
        self.persistent_grants = list(payload.get("persistent", []))

    def snapshot(self) -> dict[str, list[dict[str, str]]]:
        return {
            "session": list(self.session_grants),
            "persistent": list(self.persistent_grants),
        }

    def profile(self) -> dict[str, int]:
        return {
            "session_grants": len(self.session_grants),
            "persistent_grants": len(self.persistent_grants),
        }

    def evaluate(self, executable: dict[str, Any]) -> PolicyDecision:
        signature = build_signature(executable)
        if self._is_allowed(signature):
            return PolicyDecision(signature=signature, decision="allow", reason="matched_grant")
        # Runtime v1: Bash/PythonExec default ask.
        return PolicyDecision(signature=signature, decision="ask", reason="side_effect_executor")

    def resolve(
        self,
        executable: dict[str, Any],
        approval_handler: Callable[[str], tuple[bool, str]] | None,
    ) -> tuple[bool, str]:
        decision = self.evaluate(executable)
        if decision.decision == "allow":
            return True, "allow"
        if decision.decision == "deny":
            return False, "deny"
        if approval_handler is None:
            return False, "ask_no_handler"
        handler_result = approval_handler(decision.signature)
        if isinstance(handler_result, tuple):
            if len(handler_result) >= 2:
                allowed = bool(handler_result[0])
                scope = str(handler_result[1])
            elif len(handler_result) == 1:
                allowed = bool(handler_result[0])
                scope = "once"
            else:
                allowed = False
                scope = "deny"
        else:
            allowed = bool(handler_result)
            scope = "once"
        scope_token = (scope or "").strip().lower()
        if not allowed:
            return False, "user_deny"
        if scope_token in {"once", "allow-once", "y"}:
            return True, "allow-once"
        if scope_token in {"session", "allow-session", "s"}:
            self.session_grants.append({"pattern": decision.signature})
            return True, "allow-session"
        if scope_token in {"pattern", "allow-pattern", "p"}:
            wildcard = _to_pattern(decision.signature)
            self.session_grants.append({"pattern": wildcard})
            return True, "allow-pattern"
        if scope_token in {"always", "allow-always", "a"}:
            wildcard = _to_pattern(decision.signature)
            self.persistent_grants.append({"pattern": wildcard})
            return True, "allow-always"
        return True, "allow-once"

    def _is_allowed(self, signature: str) -> bool:
        for grant in self.session_grants + self.persistent_grants:
            pattern = str(grant.get("pattern", "")).strip()
            if not pattern:
                continue
            if fnmatch(signature, pattern):
                return True
        return False


def _to_pattern(signature: str) -> str:
    if "(" not in signature:
        return f"{signature}(*)"
    prefix = signature.split("(", 1)[0]
    return f"{prefix}(*)"


def build_signature(executable: dict[str, Any]) -> str:
    executor = str(executable.get("executor", "unknown"))
    if executor == "Bash":
        command = str(executable.get("command", "")).strip().replace("\n", " ")
        compact = " ".join(command.split())[:120]
        return f"Bash({compact})"
    if executor == "PythonExec":
        code = str(executable.get("code", "")).strip().replace("\n", " ")
        compact = " ".join(code.split())[:120]
        return f"PythonExec({compact})"
    return f"{executor}(*)"
