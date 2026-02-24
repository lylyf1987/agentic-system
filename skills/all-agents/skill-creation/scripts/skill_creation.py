from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

_SKILL_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
_EXECUTED_SKILL = "skill-creation"
_REQUIRED_FRONTMATTER_KEYS = [
    "name",
    "handler",
    "description",
    "required_tools",
    "recommended_tools",
    "forbidden_tools",
]


def _ok(skill_target: str) -> dict[str, Any]:
    return {
        "executed_skill": _EXECUTED_SKILL,
        "status": "ok",
        "skill_created/updated": skill_target,
    }


def _err(skill_target: str = "") -> dict[str, Any]:
    return {
        "executed_skill": _EXECUTED_SKILL,
        "status": "error",
        "skill_created/updated": skill_target,
    }


def _read_skill_frontmatter(path: Path) -> dict[str, str]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return {}

    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}

    end = -1
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            end = idx
            break
    if end == -1:
        return {}

    out: dict[str, str] = {}
    for raw in lines[1:end]:
        line = raw.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        out[key.strip()] = value.strip()
    return out


def _skill_template(skill_name: str, description: str, handler_path: str) -> str:
    return "\n".join(
        [
            "---",
            f"name: {skill_name}",
            f"handler: {handler_path}",
            f"description: {description}",
            "required_tools: exec",
            "recommended_tools: exec",
            "forbidden_tools:",
            "---",
            "",
            "# Purpose",
            "",
            "Describe what this skill does.",
            "",
            "# When To Use",
            "",
            "Describe clear trigger conditions.",
            "",
            "# Procedure",
            "",
            "List deterministic steps.",
            "",
            "# Runtime Contract",
            "",
            "Scripts in this skill must produce runtime-friendly output:",
            "1. Print clear, meaningful stdout describing what was attempted and what changed.",
            "2. Use stderr only for errors/failures with actionable messages.",
            "3. End with one final JSON object on stdout containing stable keys for downstream reasoning.",
            "",
            "# Action Input Templates",
            "",
            "Provide concrete action_input examples.",
            "",
            "# Script Requirements",
            "",
            "For script_path handlers, accept explicit CLI args and avoid hidden side effects.",
            "Keep logs concise and objective so runtime history remains readable.",
            "",
            "# Notes",
            "",
            "Add constraints and caveats.",
            "",
        ]
    )


def _normalize_skill_name(skill_id: str) -> str:
    return " ".join(part.capitalize() for part in skill_id.split("-") if part)


def _script_template(skill_id: str) -> str:
    module_name = skill_id.replace("-", "_")
    return "\n".join(
        [
            "#!/usr/bin/env python3",
            "from __future__ import annotations",
            "",
            "import argparse",
            "import json",
            "import sys",
            "from typing import Any",
            "",
            f"_EXECUTED_SKILL = {skill_id!r}",
            "",
            "",
            "def _ok(message: str) -> dict[str, Any]:",
            "    return {",
            '        "executed_skill": _EXECUTED_SKILL,',
            '        "status": "ok",',
            '        "message": message,',
            "    }",
            "",
            "",
            "def _err(message: str) -> dict[str, Any]:",
            "    return {",
            '        "executed_skill": _EXECUTED_SKILL,',
            '        "status": "error",',
            '        "message": message,',
            "    }",
            "",
            "",
            "def parse_args() -> argparse.Namespace:",
            f'    parser = argparse.ArgumentParser(description=\"Runtime handler scaffold for {module_name}\")',
            "    parser.add_argument(\"--dry-run\", action=\"store_true\")",
            "    return parser.parse_args()",
            "",
            "",
            "def main() -> int:",
            "    args = parse_args()",
            "    try:",
            "        if args.dry_run:",
            "            out = _ok(\"dry_run completed\")",
            "            print(json.dumps(out, ensure_ascii=True))",
            "            return 0",
            "",
            "        # TODO: implement deterministic skill logic here.",
            "        # Runtime expectation:",
            "        # - stdout: meaningful evidence of what changed",
            "        # - stderr: only errors/failures",
            "        # - final stdout line: one JSON object with stable keys",
            "        out = _ok(\"scaffold script executed; implementation pending\")",
            "        print(json.dumps(out, ensure_ascii=True))",
            "        return 0",
            "    except Exception as exc:  # unexpected runtime failure",
            "        out = _err(f\"unexpected exception: {exc}\")",
            "        print(json.dumps(out, ensure_ascii=True))",
            "        print(\"unexpected error\", file=sys.stderr)",
            "        return 2",
            "",
            "",
            "if __name__ == \"__main__\":",
            "    raise SystemExit(main())",
            "",
        ]
    )


def _summarize_list(items: list[str]) -> str:
    if not items:
        return "(none)"
    return ",".join(items)


def run_inspect(workspace: Path, skill_id: str, scope: str) -> dict[str, Any]:
    skill_dir = workspace / "skills" / scope / skill_id
    skill_md = skill_dir / "SKILL.md"
    scripts_dir = skill_dir / "scripts"

    exists = skill_dir.exists() and skill_dir.is_dir()
    skill_md_exists = skill_md.exists()
    frontmatter = _read_skill_frontmatter(skill_md) if skill_md_exists else {}

    scripts: list[str] = []
    if scripts_dir.exists() and scripts_dir.is_dir():
        for p in sorted(scripts_dir.rglob("*")):
            if p.is_file():
                scripts.append(str(p.relative_to(workspace)))

    summary = (
        f"skill_id={skill_id}; action=inspect; scope={scope}; exists={exists}; "
        f"skill_md_exists={skill_md_exists}; scripts_count={len(scripts)}"
    )
    if frontmatter:
        name = str(frontmatter.get("name", "")).strip()
        handler = str(frontmatter.get("handler", "")).strip()
        description = str(frontmatter.get("description", "")).strip()
        summary = (
            f"{summary}; name={name or '(empty)'}; "
            f"handler={handler or '(empty)'}; description={description or '(empty)'}"
        )
    _ = workspace
    return _ok(skill_target=summary)


def run_scaffold(workspace: Path, skill_id: str, scope: str, description: str, overwrite: bool) -> dict[str, Any]:
    skill_dir = workspace / "skills" / scope / skill_id
    scripts_dir = skill_dir / "scripts"
    references_dir = skill_dir / "references"
    assets_dir = skill_dir / "assets"
    skill_md = skill_dir / "SKILL.md"
    handler_rel = f"scripts/{skill_id.replace('-', '_')}.py"
    handler_file = skill_dir / handler_rel

    created: list[str] = []
    updated: list[str] = []

    if not skill_dir.exists():
        skill_dir.mkdir(parents=True, exist_ok=True)
        created.append(str(skill_dir.relative_to(workspace)))

    if not scripts_dir.exists():
        scripts_dir.mkdir(parents=True, exist_ok=True)
        created.append(str(scripts_dir.relative_to(workspace)))
    if not references_dir.exists():
        references_dir.mkdir(parents=True, exist_ok=True)
        created.append(str(references_dir.relative_to(workspace)))
    if not assets_dir.exists():
        assets_dir.mkdir(parents=True, exist_ok=True)
        created.append(str(assets_dir.relative_to(workspace)))

    skill_name = _normalize_skill_name(skill_id)
    default_description = description.strip() or f"Describe purpose for {skill_id}."
    template = _skill_template(skill_name, default_description, handler_rel)
    script_template = _script_template(skill_id)

    if not skill_md.exists():
        skill_md.write_text(template, encoding="utf-8")
        created.append(str(skill_md.relative_to(workspace)))
    elif overwrite:
        skill_md.write_text(template, encoding="utf-8")
        updated.append(str(skill_md.relative_to(workspace)))

    if not handler_file.exists():
        handler_file.write_text(script_template, encoding="utf-8")
        created.append(str(handler_file.relative_to(workspace)))
    elif overwrite:
        handler_file.write_text(script_template, encoding="utf-8")
        updated.append(str(handler_file.relative_to(workspace)))

    references_readme = references_dir / "README.md"
    if not references_readme.exists():
        references_readme.write_text(
            "\n".join(
                [
                    "# References",
                    "",
                    "Put large examples and deep technical notes here.",
                    "Keep SKILL.md concise and procedural; load reference files only when needed.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        created.append(str(references_readme.relative_to(workspace)))
    elif overwrite:
        references_readme.write_text(
            "\n".join(
                [
                    "# References",
                    "",
                    "Put large examples and deep technical notes here.",
                    "Keep SKILL.md concise and procedural; load reference files only when needed.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        updated.append(str(references_readme.relative_to(workspace)))

    created_text = _summarize_list(created)
    updated_text = _summarize_list(updated)
    summary = (
        f"skill_id={skill_id}; action=scaffold; scope={scope}; "
        f"created={created_text}; updated={updated_text}"
    )
    _ = workspace
    return _ok(skill_target=summary)


def run_validate(workspace: Path, skill_id: str, scope: str) -> dict[str, Any]:
    skill_dir = workspace / "skills" / scope / skill_id
    skill_md = skill_dir / "SKILL.md"
    scripts_dir = skill_dir / "scripts"

    errors: list[str] = []
    warnings: list[str] = []

    if not skill_dir.exists() or not skill_dir.is_dir():
        errors.append("skill_dir_missing")
    if not skill_md.exists():
        errors.append("skill_md_missing")

    frontmatter: dict[str, str] = {}
    skill_text = ""
    if skill_md.exists():
        try:
            skill_text = skill_md.read_text(encoding="utf-8")
        except OSError:
            errors.append("skill_md_unreadable")
        frontmatter = _read_skill_frontmatter(skill_md)

    if skill_md.exists() and not frontmatter:
        errors.append("frontmatter_missing_or_invalid")

    for key in _REQUIRED_FRONTMATTER_KEYS:
        if key not in frontmatter:
            errors.append(f"frontmatter_missing:{key}")

    handler = str(frontmatter.get("handler", "")).strip()
    if handler:
        handler_path = skill_dir / handler
        if not handler_path.exists():
            errors.append(f"handler_missing:{handler}")
        elif not handler_path.is_file():
            errors.append(f"handler_not_file:{handler}")
    else:
        warnings.append("handler_empty")

    if not scripts_dir.exists():
        warnings.append("scripts_dir_missing")

    script_count = 0
    if scripts_dir.exists() and scripts_dir.is_dir():
        script_count = sum(1 for p in scripts_dir.rglob("*") if p.is_file())
        if script_count == 0:
            warnings.append("scripts_dir_empty")

    lower_skill_text = skill_text.lower()
    if "stdout" not in lower_skill_text:
        warnings.append("skill_md_missing_stdout_guidance")
    if "stderr" not in lower_skill_text:
        warnings.append("skill_md_missing_stderr_guidance")

    summary = (
        f"skill_id={skill_id}; action=validate; scope={scope}; "
        f"errors_count={len(errors)}; warnings_count={len(warnings)}; "
        f"scripts_count={script_count}; "
        f"errors={_summarize_list(errors)}; warnings={_summarize_list(warnings)}"
    )
    if errors:
        return _err(skill_target=summary)
    return _ok(skill_target=summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect or scaffold skill packages with structured JSON output.")
    parser.add_argument("--action", required=True, choices=["inspect", "scaffold", "validate"])
    parser.add_argument("--skill-id", required=True)
    parser.add_argument("--scope", required=True, choices=["all-agents", "core-agent"])
    parser.add_argument("--description", default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workspace", default=".")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    action = str(args.action)
    skill_id = str(args.skill_id).strip()
    scope = str(args.scope).strip()
    workspace = Path(args.workspace).expanduser().resolve()

    if not _SKILL_ID_RE.match(skill_id):
        out = _err(skill_target=f"skill_creation_error: invalid skill_id={skill_id}")
        print(json.dumps(out, ensure_ascii=True))
        return 1

    try:
        if action == "inspect":
            out = run_inspect(workspace=workspace, skill_id=skill_id, scope=scope)
        elif action == "validate":
            out = run_validate(workspace=workspace, skill_id=skill_id, scope=scope)
        else:
            out = run_scaffold(
                workspace=workspace,
                skill_id=skill_id,
                scope=scope,
                description=str(args.description),
                overwrite=bool(args.overwrite),
            )
        print(json.dumps(out, ensure_ascii=True))
        return 0 if out.get("status") == "ok" else 1
    except Exception as exc:  # unexpected runtime failure
        out = _err(skill_target=f"skill_creation_error: unexpected exception: {exc}")
        print(json.dumps(out, ensure_ascii=True))
        print("unexpected error", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
