"""CLI coverage for durable run-log path selection and recovery options."""

from pathlib import Path
import re
import stat
from typing import Any, Callable

import pytest
from typer.testing import CliRunner

from yian import load_flow, run_flow
from yian.cli import app
import yian.cli.run as run_cli
from yian.events import NodeStarted
from yian.runtime.log import (
    JsonlRunLog,
    RecordEnvelope,
    RunRecord,
    RunResumedRecord,
    RunStartedRecord,
    decode_record,
)


runner = CliRunner()

# CI forces Rich color, which can place style escapes inside rendered option names.
_ANSI_SGR = re.compile(r"\x1b\[[0-9;]*m")

_FLOW = """\
id: cli_log
name: cli_log
input:
  topic: !str
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


def _plain(text: str) -> str:
    """Remove optional Rich styling so CLI contracts are color-environment neutral."""
    return _ANSI_SGR.sub("", text)


def _unwrapped(text: str) -> str:
    """Drop styling and all whitespace, since Rich wraps long paths at any column."""
    return re.sub(r"\s+", "", _plain(text))


def _envelopes(path: Path) -> list[RecordEnvelope]:
    """Decode the retained envelopes in one complete JSONL run log."""
    return [decode_record(line) for line in path.read_bytes().splitlines()]


def _records(path: Path) -> list[RunRecord]:
    """Return the durable records without their common envelopes."""
    return [envelope.record for envelope in _envelopes(path)]


def _fail_if_called(name: str, calls: list[str]) -> Callable[..., None]:
    """Return a sentinel side effect that makes an early-validation leak visible."""
    def fail(*args: Any, **kwargs: Any) -> None:
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
    state_dir = isolated_cli_home / ".yian"
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
    assert f"Run log: {path}" in _plain(result.stderr)
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
    assert f"Run log: {path}" in _plain(result.stderr)
    assert not (Path.home() / ".yian").exists()


def test_existing_explicit_log_is_rejected_before_input_or_provider_side_effects(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Fresh creation never overwrites and validates before gathering run data."""
    flow = _write_flow(tmp_path)
    path = tmp_path / "existing.jsonl"
    path.write_text("keep")
    calls: list[str] = []
    monkeypatch.setattr(
        run_cli, "_gather_missing_inputs", _fail_if_called("input", calls)
    )
    monkeypatch.setattr(
        run_cli,
        "_ensure_provider_keys",
        _fail_if_called("provider", calls),
    )

    result = runner.invoke(app, ["run", str(flow), "--log", str(path)])

    assert result.exit_code == 2
    assert "already exists" in _plain(result.output)
    assert calls == []
    assert path.read_text() == "keep"


def test_dangling_explicit_log_symlink_is_rejected_before_side_effects(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A broken symlink is still an occupied directory entry, never a fresh path."""
    flow = _write_flow(tmp_path)
    path = tmp_path / "dangling.jsonl"
    path.symlink_to(tmp_path / "missing-target.jsonl")
    calls: list[str] = []
    monkeypatch.setattr(
        run_cli, "_gather_missing_inputs", _fail_if_called("input", calls)
    )
    monkeypatch.setattr(
        run_cli,
        "_ensure_provider_keys",
        _fail_if_called("provider", calls),
    )

    result = runner.invoke(app, ["run", str(flow), "--log", str(path)])

    assert result.exit_code == 2
    assert "already exists" in _plain(result.output)
    assert calls == []
    assert path.is_symlink()


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
    monkeypatch.setattr(
        run_cli, "_gather_missing_inputs", _fail_if_called("input", calls)
    )
    monkeypatch.setattr(
        run_cli,
        "_ensure_provider_keys",
        _fail_if_called("provider", calls),
    )

    result = runner.invoke(
        app,
        ["run", str(flow), "--resume", str(resume_path), *extra],
    )

    assert result.exit_code == 2
    plain_output = _plain(result.output)
    assert "--resume cannot be combined" in plain_output
    assert conflict in plain_output
    assert calls == []


def test_missing_resume_log_is_rejected_before_command_side_effects(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Typer rejects a missing recovery authority before entering the command."""
    flow = _write_flow(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(
        run_cli, "_gather_missing_inputs", _fail_if_called("input", calls)
    )
    monkeypatch.setattr(
        run_cli,
        "_ensure_provider_keys",
        _fail_if_called("provider", calls),
    )

    result = runner.invoke(
        app,
        ["run", str(flow), "--resume", str(tmp_path / "missing.jsonl")],
    )

    assert result.exit_code == 2
    assert "does not exist" in _plain(result.output)
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
    monkeypatch.setattr(
        run_cli, "_gather_missing_inputs", _fail_if_called("input", calls)
    )
    monkeypatch.setattr(
        run_cli,
        "_ensure_provider_keys",
        _fail_if_called("provider", calls),
    )
    monkeypatch.setattr(
        run_cli,
        "_check_resume_credentials",
        _fail_if_called("resume credentials", calls),
    )

    result = runner.invoke(
        app,
        ["run", str(flow_path), "--resume", str(log), "--quiet"],
    )

    assert result.exit_code == 0, result.output
    assert calls == []
    assert log.read_bytes() == before
    assert f"Run log: {log}" in _plain(result.stderr)
    assert "mountains" in result.stdout
    assert not (Path.home() / ".yian").exists()


@pytest.mark.parametrize(
    ("trace_args", "has_traceback"),
    [([], False), (["--engine-trace"], True)],
)
def test_corrupt_resume_is_a_clean_cli_error(
    trace_args: list[str],
    has_traceback: bool,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Malformed durable data never leaks a raw exception from the CLI boundary."""
    flow = _write_flow(tmp_path)
    log = tmp_path / "corrupt.jsonl"
    log.write_text("not-json\n")
    calls: list[str] = []
    monkeypatch.setattr(
        run_cli, "_gather_missing_inputs", _fail_if_called("input", calls)
    )
    monkeypatch.setattr(
        run_cli,
        "_ensure_provider_keys",
        _fail_if_called("provider", calls),
    )

    result = runner.invoke(
        app,
        ["run", str(flow), "--resume", str(log), "--quiet", *trace_args],
    )

    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    plain_output = _plain(result.output)
    assert _unwrapped(
        f"resume failed: run log {log} line 1 (seq 0): invalid run-record JSON"
    ) in _unwrapped(result.output)
    assert ("Traceback" in plain_output) is has_traceback
    assert calls == []


def test_locked_resume_names_the_log_and_the_likely_cause(tmp_path: Path) -> None:
    """Resuming a log another run still holds says so instead of a bare errno."""
    flow_path = _write_flow(tmp_path)
    loaded = load_flow(_FLOW, search_paths=[tmp_path])
    log = tmp_path / "held.jsonl"
    run_flow(loaded, {"topic": "lakes"}, log=log, run_id="held-run")

    with JsonlRunLog.resume(log):
        result = runner.invoke(
            app,
            ["run", str(flow_path), "--resume", str(log), "--quiet"],
        )

    assert result.exit_code == 1
    assert _unwrapped(
        f"resume failed: run log {log} is locked by another writer "
        "(is the run still active?)"
    ) in _unwrapped(result.output)
    assert "Errno" not in result.output


def test_default_directory_failure_is_a_clean_cli_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Default-path host failures use the same concise error funnel as recovery."""
    flow = _write_flow(tmp_path)

    def deny_default_path(run_id: str) -> Path:
        raise PermissionError("private run directory is not writable")

    monkeypatch.setattr(run_cli, "_default_run_log_path", deny_default_path)
    result = runner.invoke(
        app,
        ["run", str(flow), "--input", "topic=blocked", "--quiet"],
    )

    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    plain_output = _plain(result.output)
    assert "run failed: private run directory is not writable" in plain_output
    assert "Traceback" not in plain_output


@pytest.mark.parametrize(
    ("worker_args", "expected_workers"),
    [([], 2), (["-w", "0"], 0), (["-w", "3"], 3)],
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
    monkeypatch.setattr(
        run_cli, "_gather_missing_inputs", _fail_if_called("input", calls)
    )
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


def test_keyboard_interrupt_leaves_a_reported_unlocked_recoverable_log(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A controller interrupt closes an unfinished log that `--resume` can finish."""
    flow = _write_flow(tmp_path)
    controlled_run = run_cli._run_flow_controlled

    def interrupting_run(*args: Any, **kwargs: Any):
        observer = kwargs.get("on_event")

        def interrupt_after_start(event: Any) -> None:
            if observer is not None:
                observer(event)
            if isinstance(event, NodeStarted) and event.node_id == "echo":
                raise KeyboardInterrupt

        kwargs["on_event"] = interrupt_after_start
        return controlled_run(*args, **kwargs)

    monkeypatch.setattr(run_cli, "_run_flow_controlled", interrupting_run)
    interrupted = runner.invoke(
        app,
        ["run", str(flow), "--input", "topic=continued", "--quiet"],
    )

    assert interrupted.exit_code == 130
    assert "cancelled by user" in _plain(interrupted.output)
    logs = list((Path.home() / ".yian" / "runs").glob("*.jsonl"))
    assert len(logs) == 1
    log = logs[0]
    assert f"Run log: {log}" in _plain(interrupted.stderr)

    monkeypatch.setattr(run_cli, "_run_flow_controlled", controlled_run)
    resumed = runner.invoke(
        app,
        ["run", str(flow), "--resume", str(log), "--quiet"],
    )

    assert resumed.exit_code == 0, resumed.output
    assert "continued" in resumed.stdout
    assert any(isinstance(record, RunResumedRecord) for record in _records(log))
