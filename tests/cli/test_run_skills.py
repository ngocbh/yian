"""`composer run` supplies Claude Code and Codex skill folders to the run."""

from __future__ import annotations

from pathlib import Path

from langchain_core.messages import AIMessage
from typer.testing import CliRunner

import agent_composer.cli.run as run_cli
import agent_composer.llm_clients as llm_clients
from agent_composer.cli import app

_FLOW = """\
id: cli-skills
name: cli-skills
nodes:
  review:
    kind: agent
    prompt: Review.
    skills: [code-review, release-notes]
    output: str
output: ${review.output}
"""


class _Model:
    def __init__(self) -> None:
        self.systems: list[str] = []

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.systems.append(messages[0].content)
        return AIMessage(content="done")


def _skill(root: Path, name: str, description: str) -> None:
    (root / name).mkdir(parents=True)
    (root / name / "SKILL.md").write_text(f"---\ndescription: {description}\n---\nBody.\n", encoding="utf-8")


def test_run_reads_user_and_project_skill_roots(monkeypatch, tmp_path: Path, isolated_cli_home: Path) -> None:
    project = tmp_path / "project"
    (project / ".git").mkdir(parents=True)
    _skill(project / ".claude" / "skills", "code-review", "Project review.")
    _skill(project / ".agents" / "skills", "release-notes", "Project notes.")
    _skill(isolated_cli_home / ".claude" / "skills", "code-review", "Personal review.")
    (project / ".claude" / "skills" / "broken").mkdir()
    (project / ".claude" / "skills" / "broken" / "SKILL.md").write_text("no frontmatter\n", encoding="utf-8")
    flow = project / "flow.yaml"
    flow.write_text(_FLOW, encoding="utf-8")
    model = _Model()
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _config: model)
    monkeypatch.setattr(run_cli, "_ensure_provider_keys", lambda *args: None)
    monkeypatch.chdir(project)

    result = CliRunner().invoke(app, ["run", str(flow), "--quiet"])

    assert result.exit_code == 0, result.output
    assert "warning: skipped skill: " in result.output
    assert "broken" in result.output and "UserWarning" not in result.output
    assert model.systems[0].endswith("- code-review: Personal review.\n- release-notes: Project notes.")


def test_run_reports_a_missing_granted_skill(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    flow = tmp_path / "flow.yaml"
    flow.write_text(_FLOW, encoding="utf-8")
    monkeypatch.setattr(run_cli, "_ensure_provider_keys", lambda *args: None)
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(app, ["run", str(flow), "--quiet"])

    assert result.exit_code == 1
    assert "grants skills not supplied to this run" in result.output


def test_run_without_grants_reads_no_skill_roots(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / ".claude" / "skills" / "broken").mkdir(parents=True)
    (tmp_path / ".claude" / "skills" / "broken" / "SKILL.md").write_text("no frontmatter\n", encoding="utf-8")
    flow = tmp_path / "flow.yaml"
    flow.write_text(_FLOW.replace("    skills: [code-review, release-notes]\n", ""), encoding="utf-8")
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _config: _Model())
    monkeypatch.setattr(run_cli, "_ensure_provider_keys", lambda *args: None)
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(app, ["run", str(flow), "--quiet"])

    assert result.exit_code == 0, result.output
    assert "skipped skill" not in result.output
