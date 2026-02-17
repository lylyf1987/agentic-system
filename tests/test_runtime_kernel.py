from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agentic_system.kernel.knowledge import KnowledgeEngine
from agentic_system.kernel.policy import PolicyEngine
from agentic_system.kernel.prompts import PromptEngine
from agentic_system.kernel.skills import SkillEngine
from agentic_system.runtime import AgentRuntime


class RuntimeKernelTests(unittest.TestCase):
    def test_runtime_smoke_turn_persists_histories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            rt = AgentRuntime(workspace=workspace, mode="safe", model_provider="openai")
            with patch("builtins.input", side_effect=["say hello", "/exit"]):
                rt.engine.run_core_session(
                    state=rt.state,
                    command_handler=rt._handle_command,
                )
                out = rt.state.final_report or ""
            rt.shutdown()

            self.assertTrue(isinstance(out, str))
            self.assertTrue(rt.state.full_proc_hist)
            self.assertTrue(rt.state.workflow_hist)
            session_id = rt.state.session_id
            state_path = workspace / "sessions" / session_id / "state.json"
            self.assertTrue(state_path.exists())
            payload = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertTrue(payload.get("full_proc_hist"))
            self.assertTrue(payload.get("workflow_hist"))
            self.assertNotIn("system_prompts", payload)
            self.assertNotIn("agent_role_descriptions", payload)

            system_prompts_path = workspace / "prompts" / "agent_system_prompt.json"
            role_descriptions_path = workspace / "prompts" / "agent_role_description.json"
            step_prompts_path = workspace / "prompts" / "agent_step_prompt.json"
            self.assertTrue(system_prompts_path.exists())
            self.assertTrue(role_descriptions_path.exists())
            self.assertTrue(step_prompts_path.exists())

    def test_policy_engine_grants(self) -> None:
        policy = PolicyEngine()
        executable = {"executor": "Bash", "command": "echo hi"}

        allowed, decision = policy.resolve(executable, approval_handler=lambda _sig: (False, "deny"))
        self.assertFalse(allowed)
        self.assertEqual(decision, "user_deny")

        allowed, decision = policy.resolve(executable, approval_handler=lambda _sig: (True, "session"))
        self.assertTrue(allowed)
        self.assertEqual(decision, "allow-session")
        snap = policy.snapshot()
        self.assertTrue(snap["session"])

    def test_skill_registry_fallbacks_to_packaged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            registry = SkillEngine(workspace=workspace)
            items = registry.load_skill_meta("core+all")
            self.assertTrue(items)
            ids = {item["skill_id"] for item in items}
            self.assertIn("delegate-work", ids)

    def test_knowledge_patch_writes_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            knowledge = KnowledgeEngine(workspace)
            doc = {
                "title": "Doc One",
                "summary_short": "short",
                "text": "Body",
                "tags": ["tag"],
                "confidence": 0.8,
            }
            result = knowledge.create_doc(doc)
            self.assertTrue(result["created"])

            index_path = workspace / "knowledge" / "index" / "catalog.json"
            rows = json.loads(index_path.read_text(encoding="utf-8"))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["title"], "Doc One")

    def test_skill_engine_create_skill_writes_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            skill_engine = SkillEngine(workspace=workspace)
            proposal = {
                "action": "create",
                "skill_id": "demo-skill",
                "scope": "all-agents",
                "artifacts": {
                    "skill_md": "# Demo Skill\n\nUse this skill for demo.\n",
                    "scripts": [
                        {
                            "path": "scripts/run.sh",
                            "content": "echo demo\n",
                        }
                    ],
                },
            }
            result = skill_engine.create_skill(proposal)
            self.assertTrue(result["applied"])
            self.assertTrue((workspace / "skills" / "all-agents" / "demo-skill" / "SKILL.md").exists())
            self.assertTrue((workspace / "skills" / "all-agents" / "demo-skill" / "scripts" / "run.sh").exists())

    def test_prompt_engine_bootstrap_and_lookup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            prompt_engine = PromptEngine(workspace=workspace)
            self.assertTrue((workspace / "prompts" / "agent_system_prompt.json").exists())
            self.assertTrue((workspace / "prompts" / "agent_step_prompt.json").exists())
            self.assertTrue((workspace / "prompts" / "agent_role_description.json").exists())

            core_prompt = prompt_engine.get_system_prompt("core_agent")
            self.assertTrue(isinstance(core_prompt, str))
            self.assertTrue(bool(core_prompt.strip()))

            roles = prompt_engine.list_agent_roles_with_descriptions()
            core_desc = roles.get("core_agent", "")
            self.assertTrue(isinstance(core_desc, str))
            self.assertTrue(bool(core_desc.strip()))


if __name__ == "__main__":
    unittest.main()
