"""CLI coverage for durable run-log path selection and recovery options."""

from pathlib import Path
import stat

import pytest
from typer.testing import CliRunner

from agent_composer import load_flow, run_flow
from agent_composer.cli import app
import agent_composer.cli.run as run_cli
from agent_composer.events import NodeStarted
from agent_composer.runtime.log import (
    RunResumedRecord,
    RunStartedRecord,
    decode_record,
)


runner = CliRunner()

_FLOW = """\
id: cli_log
name: cli_log
input:
  topic: str
nodes:
  echo:
    kind: code
    input:
      topic: ${input.topic}
    output: str
    code: |
      return inputs["topic"]
output: ${echo.output}
"""


def _write_flow(tmp_path: Path) -> Path:
    """Write the deterministic code-only flow used by CLI log tests."""
    path = tmp_path / "flow.yaml"
    path.write_text(_FLOW)
    return path


def _envelopes(path: Path):
    """Decode the retained envelopes in one complete JSONL run log."""
    return [decode_record(line) for line in path.read_bytes().splitlines()]


def _records(path: Path):
    """Return the durable records without their common envelopes."""
    return [envelope.record for envelope in _envelopes(path)]


def _fail_if_called(name: str, calls: list[str]):
    """Return a sentinel side effect that makes an early-validation leak visible."""
    def fail(*args, **kwargs):
        calls.append(name)
        raise AssertionError(f"{name} must not run")

    return fail


def test_cli_tests_use_an_isolated_home(isolated_cli_home: Path) -> None:
    """Default CLI state can never escape into the developer's real home."""
    assert Path.home() == isolated_cli_home


def test_fresh_run_uses_private_default_path_and_reports_it(
    isolated_cli_home: Path,
    tmp_path: Path,
) -> None:
    """A fresh CLI run gets one owner-only log named by its recorded run id."""
    state_dir = isolated_cli_home / ".agent_composer"
    runs_dir = state_dir / "runs"
    state_dir.mkdir(mode=0o755)
    runs_dir.mkdir(mode=0o755)
    flow = _write_flow(tmp_path)

    result = runner.invoke(
        app,
        ["run", str(flow), "--input", "topic=clouds", "--quiet"],
    )

    assert result.exit_code == 0, result.output
    logs = list(runs_dir.glob("*.jsonl"))
    assert len(logs) == 1
    path = logs[0]
    started = _envelopes(path)[0]
    assert isinstance(started.record, RunStartedRecord)
    assert started.run_id == path.stem
    assert stat.S_IMODE(state_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(runs_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert f"Run log: {path}" in result.stderr
    assert "clouds" in result.stdout


def test_fresh_run_uses_exact_explicit_log_path(tmp_path: Path) -> None:
    """`--log` selects the new file exactly and does not create default state."""
    flow = _write_flow(tmp_path)
    path = tmp_path / "chosen.jsonl"

    result = runner.invoke(
        app,
        ["run", str(flow), "--input", "topic=rivers", "--log", str(path)],
    )

    assert result.exit_code == 0, result.output
    assert path.is_file()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert f"Run log: {path}" in result.stderr
    assert not (Path.home() / ".agent_composer").exists()


def test_existing_explicit_log_is_rejected_before_input_or_provider_side_effects(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Fresh creation never overwrites and validates before gathering run data."""
    flow = _write_flow(tmp_path)
    path = tmp_path / "existing.jsonl"
    path.write_text("keep")
    calls: list[str] = []
    monkeypatch.setattr(run_cli, "_prompt_missing", _fail_if_called("input", calls))
    monkeypatch.setattr(
        run_cli,
        "_ensure_provider_keys",
        _fail_if_called("provider", calls),
    )

    result = runner.invoke(app, ["run", str(flow), "--log", str(path)])

    assert result.exit_code != 0
    assert "already exists" in result.output
    assert calls == []
    assert path.read_text() == "keep"


@pytest.mark.parametrize(
    ("conflict", "extra"),
    [
        ("--input", ["--input", "topic=new"]),
        ("--inputs", ["--inputs", "inputs.json"]),
        ("--provider", ["--provider", "openai"]),
        ("--model", ["--model", "gpt-5.5"]),
        ("--log", ["--log", "new.jsonl"]),
    ],
)
def test_resume_rejects_fresh_options_before_input_or_provider_side_effects(
    conflict: str,
    extra: list[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Resume accepts only recorded state plus an optional future worker count."""
    flow = _write_flow(tmp_path)
    resume_path = tmp_path / "prior.jsonl"
    resume_path.write_text("not reached\n")
    (tmp_path / "inputs.json").write_text('{"topic": "new"}')
    calls: list[str] = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_cli, "_prompt_missing", _fail_if_called("input", calls))
    monkeypatch.setattr(
        run_cli,
        "_ensure_provider_keys",
        _fail_if_called("provider", calls),
    )

    result = runner.invoke(
        app,
        ["run", str(flow), "--resume", str(resume_path), *extra],
    )

    assert result.exit_code != 0
    assert "--resume cannot be combined" in result.output
    assert conflict in result.output
    assert calls == []


def test_missing_resume_log_is_rejected_before_command_side_effects(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Typer rejects a missing recovery authority before entering the command."""
    flow = _write_flow(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(run_cli, "_prompt_missing", _fail_if_called("input", calls))
    monkeypatch.setattr(
        run_cli,
        "_ensure_provider_keys",
        _fail_if_called("provider", calls),
    )

    result = runner.invoke(
        app,
        ["run", str(flow), "--resume", str(tmp_path / "missing.jsonl")],
    )

    assert result.exit_code != 0
    assert "does not exist" in result.output
    assert calls == []


def test_terminal_resume_is_read_only_and_skips_fresh_side_effects(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A terminal prefix prints its result without prompts, config, or appends."""
    flow_path = _write_flow(tmp_path)
    loaded = load_flow(_FLOW, search_paths=[tmp_path])
    log = tmp_path / "terminal.jsonl"
    run_flow(loaded, {"topic": "mountains"}, log=log, run_id="terminal-run")
    before = log.read_bytes()
    calls: list[str] = []
    monkeypatch.setattr(run_cli, "_prompt_missing", _fail_if_called("input", calls))
    monkeypatch.setattr(
        run_cli,
        "_ensure_provider_keys",
        _fail_if_called("provider", calls),
    )

    result = runner.invoke(
        app,
        ["run", str(flow_path), "--resume", str(log), "--quiet"],
    )

    assert result.exit_code == 0, result.output
    assert calls == []
    assert log.read_bytes() == before
    assert f"Run log: {log}" in result.stderr
    assert "mountains" in result.stdout
    assert not (Path.home() / ".agent_composer").exists()


@pytest.mark.parametrize(
    ("worker_args", "expected_workers"),
    [([], 2), (["--num-workers", "3"], 3)],
)
def test_unfinished_resume_preserves_or_overrides_only_future_workers(
    worker_args: list[str],
    expected_workers: int,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Omission preserves the recorded count; `-w` changes the resumed segment."""
    flow_path = _write_flow(tmp_path)
    loaded = load_flow(_FLOW, search_paths=[tmp_path])
    log = tmp_path / f"unfinished-{expected_workers}.jsonl"

    def interrupt_after_start(event) -> None:
        if isinstance(event, NodeStarted) and event.node_id == "echo":
            raise RuntimeError("host interruption")

    with pytest.raises(RuntimeError, match="host interruption"):
        run_flow(
            loaded,
            {"topic": "recovered"},
            log=log,
            run_id=f"unfinished-{expected_workers}",
            num_workers=2,
            on_event=interrupt_after_start,
        )

    calls: list[str] = []
    monkeypatch.setattr(run_cli, "_prompt_missing", _fail_if_called("input", calls))
    monkeypatch.setattr(
        run_cli,
        "_ensure_provider_keys",
        _fail_if_called("provider", calls),
    )
    result = runner.invoke(
        app,
        ["run", str(flow_path), "--resume", str(log), "--quiet", *worker_args],
    )

    assert result.exit_code == 0, result.output
    assert calls == []
    resumed = [record for record in _records(log) if isinstance(record, RunResumedRecord)]
    assert len(resumed) == 1
    assert resumed[0].num_workers == expected_workers
    assert "recovered" in result.stdout
