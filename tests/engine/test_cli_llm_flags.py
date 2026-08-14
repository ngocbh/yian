"""`ac run --provider/--model` build the outermost cascade layer (`llm_config`)."""

from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

import agent_composer.cli.run as climod
from agent_composer.run import RunResult

_FLOW = "id: f\nname: f\nnodes:\n  a: {kind: agent, prompt: hi}\noutput: ${a.output}\n"


@pytest.fixture(autouse=True)
def _isolated_cli_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep default run logs from reaching the developer's real home."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))


def _app():
    app = typer.Typer()
    app.command()(climod.run)
    return app


def test_cli_builds_llm_config(monkeypatch, tmp_path):
    captured = {}

    def fake_run_flow(loaded, supplied, *, on_event=None, llm_config=None, **kw):
        captured["llm_config"] = llm_config
        return RunResult(
            input={},
            status="succeeded",
            run_id="cli-test",
            log=None,
            output="ok",
        )

    monkeypatch.setattr(climod, "_run_flow_controlled", fake_run_flow)
    monkeypatch.setattr(climod, "_ensure_provider_keys", lambda *a, **k: None)
    f = tmp_path / "f.yaml"
    f.write_text(_FLOW)
    res = CliRunner().invoke(
        _app(), [str(f), "--provider", "openai", "--model", "gpt-5.5"]
    )
    assert res.exit_code == 0
    assert captured["llm_config"] == {"provider": "openai", "model": "gpt-5.5"}


def test_cli_no_flags_passes_none_or_empty(monkeypatch, tmp_path):
    captured = {}

    def fake_run_flow(loaded, supplied, *, on_event=None, llm_config=None, **kw):
        captured["llm_config"] = llm_config
        return RunResult(
            input={},
            status="succeeded",
            run_id="cli-test",
            log=None,
            output="ok",
        )

    monkeypatch.setattr(climod, "_run_flow_controlled", fake_run_flow)
    monkeypatch.setattr(climod, "_ensure_provider_keys", lambda *a, **k: None)
    f = tmp_path / "f.yaml"
    f.write_text(_FLOW)
    CliRunner().invoke(_app(), [str(f)])
    assert not captured["llm_config"]  # None or {}
