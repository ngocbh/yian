"""Tests for the `ac run` CLI — exercised through typer's `CliRunner`.

These use a CODE-only flow (no AGENT) so the suite never hits a network or needs a
provider key. They cover: a successful run, flag-supplied inputs, `--inputs` JSON,
flag-over-JSON precedence, a failing run's non-zero exit, and the `k=v` parse error.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agent_composer.cli import app

runner = CliRunner()


def _write_flow(dir_: Path, body: str) -> Path:
    path = dir_ / "flow.yaml"
    path.write_text(body)
    return path


ECHO_FLOW = """\
id: cli_echo
name: cli_echo
input:
  topic: str
nodes:
  echo:
    kind: code
    input:
      topic: ${input.topic}
    output: str
    code: tests.seeds.fns:echo_value
output:
  topic: ${echo.output}
"""

FAIL_FLOW = """\
id: cli_fail
name: cli_fail
input:
  topic: str
nodes:
  boom:
    kind: code
    input:
      topic: ${input.topic}
    output: str
    code: tests.seeds.fns:fail_always
output:
  topic: ${boom.output}
"""

INLINE_FLOW = """\
id: cli_inline
name: cli_inline
input:
  topic: str
nodes:
  shout:
    kind: code
    input:
      topic: ${input.topic}
    output: str
    code: |
      return inputs["topic"].upper()
output:
  topic: ${shout.output}
"""


def test_run_succeeds_with_input_flag(tmp_path: Path):
    flow = _write_flow(tmp_path, ECHO_FLOW)
    result = runner.invoke(app, ["run", str(flow), "--input", "topic=clouds"])
    assert result.exit_code == 0, result.output
    assert "clouds" in result.stdout


def test_run_inline_code_in_process(tmp_path: Path):
    # an inline `code:` node (bare body, run in-process) works end-to-end through `ac run`.
    flow = _write_flow(tmp_path, INLINE_FLOW)
    result = runner.invoke(app, ["run", str(flow), "--input", "topic=clouds"])
    assert result.exit_code == 0, result.output
    assert "CLOUDS" in result.stdout


def test_run_reads_inputs_json(tmp_path: Path):
    flow = _write_flow(tmp_path, ECHO_FLOW)
    inputs = tmp_path / "in.json"
    inputs.write_text(json.dumps({"topic": "rivers"}))
    result = runner.invoke(app, ["run", str(flow), "--inputs", str(inputs)])
    assert result.exit_code == 0, result.output
    assert "rivers" in result.stdout


def test_flag_overrides_json(tmp_path: Path):
    flow = _write_flow(tmp_path, ECHO_FLOW)
    inputs = tmp_path / "in.json"
    inputs.write_text(json.dumps({"topic": "rivers"}))
    result = runner.invoke(
        app, ["run", str(flow), "--inputs", str(inputs), "--input", "topic=mountains"]
    )
    assert result.exit_code == 0, result.output
    assert "mountains" in result.stdout
    assert "rivers" not in result.stdout


def test_failed_run_exits_nonzero(tmp_path: Path):
    flow = _write_flow(tmp_path, FAIL_FLOW)
    result = runner.invoke(app, ["run", str(flow), "--input", "topic=clouds"])
    assert result.exit_code == 1


def test_bad_kv_is_rejected(tmp_path: Path):
    flow = _write_flow(tmp_path, ECHO_FLOW)
    result = runner.invoke(app, ["run", str(flow), "--input", "no_equals_sign"])
    assert result.exit_code != 0


def test_missing_flow_file_errors(tmp_path: Path):
    result = runner.invoke(app, ["run", str(tmp_path / "nope.yaml")])
    assert result.exit_code != 0


def test_ctrl_c_mid_run_says_cancelled(tmp_path: Path, monkeypatch):
    """Ctrl+C while nodes are executing (a KeyboardInterrupt out of `run_flow`, not a
    prompt cancel) must surface a "cancelled by user" notice and exit 130 — not die
    silently mid-spinner. Regression for "Ctrl+C kills the run but says nothing"."""
    import agent_composer.cli.run as run_mod

    flow = _write_flow(tmp_path, ECHO_FLOW)

    def _boom(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(run_mod, "_run_flow_controlled", _boom)
    result = runner.invoke(app, ["run", str(flow), "--input", "topic=clouds"])
    assert result.exit_code == 130
    assert "cancelled by user" in result.output
