"""Tests for the `ac run` CLI — exercised through typer's `CliRunner`.

These use CODE-only flows (no AGENT) so the suite never hits a network or needs a
provider key. They cover: successful runs, flag-supplied inputs, `--inputs` JSON,
interactive structured input, explicit structured null, flag-over-JSON precedence,
non-TTY missing-input policy and setup ordering, a failing run's non-zero exit, and
the `k=v` parse error.
"""

from __future__ import annotations

from ast import literal_eval
from io import BytesIO
import json
from pathlib import Path
import re
import sys
from types import ModuleType

import pytest
import typer.testing as typer_testing
from typer.testing import CliRunner

import agent_composer.cli.run as run_cli
from agent_composer.cli import app

runner = CliRunner()

_ANSI_SGR = re.compile(r"\x1b\[[0-9;]*m")


class _TTYBytesIO(BytesIO):
    """Byte input stream that makes Click expose an interactive stdin."""

    def isatty(self) -> bool:
        """Report that the synthetic input stream is attached to a terminal."""
        return True


def _tty_input_stream(_input_value, _charset) -> _TTYBytesIO:
    """Return an empty byte stream whose terminal capability is interactive."""
    return _TTYBytesIO()


def _write_flow(dir_: Path, body: str) -> Path:
    path = dir_ / "flow.yaml"
    path.write_text(body)
    return path


def _plain(text: str) -> str:
    """Remove optional Rich styling from CLI output assertions."""
    return _ANSI_SGR.sub("", text)


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

OMITTABLE_INPUTS_FLOW = """\
id: cli_omittable_inputs
name: cli_omittable_inputs
input:
  topic: str
  limit: int = 7
  labels: list[str] = [alpha, beta]
  note: Optional[str]
nodes:
  show:
    kind: code
    input:
      topic: ${input.topic}
      limit: ${input.limit}
      labels: ${input.labels}
      note: ${input.note}
    output: object
    code: |
      return inputs
output: ${show.output}
"""

REQUIRED_INPUTS_FLOW = """\
id: cli_required_inputs
name: cli_required_inputs
input:
  first: str
  optional: Optional[str]
  second: int
nodes:
  show:
    kind: code
    input:
      first: ${input.first}
      optional: ${input.optional}
      second: ${input.second}
    output: object
    code: |
      return inputs
output: ${show.output}
"""

STRUCTURED_INPUT_FLOW = """\
id: cli_structured_input
name: cli_structured_input
input:
  items: list[int]
nodes:
  inspect:
    kind: code
    input:
      items: ${input.items}
    output: object
    code: |
      return {
          "items": inputs["items"],
          "is_list": isinstance(inputs["items"], list),
      }
output: ${inspect.output}
"""

NULLABLE_STRUCTURED_INPUT_FLOW = """\
id: cli_nullable_structured_input
name: cli_nullable_structured_input
input:
  items: Optional[list[int]]
nodes:
  inspect:
    kind: code
    input:
      items: ${input.items}
    output: object
    code: |
      return {
          "items": inputs["items"],
          "is_none": inputs["items"] is None,
      }
output: ${inspect.output}
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


def test_non_tty_run_omits_optional_and_defaulted_inputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A scripted run leaves omittable declarations for START to resolve."""
    flow = _write_flow(tmp_path, OMITTABLE_INPUTS_FLOW)
    log = tmp_path / "run.jsonl"
    monkeypatch.setitem(sys.modules, "questionary", None)

    result = runner.invoke(
        app,
        [
            "run",
            str(flow),
            "--input",
            "topic=clouds",
            "--log",
            str(log),
            "--quiet",
        ],
    )

    assert result.exit_code == 0, result.output
    assert log.is_file()
    assert literal_eval(result.stdout) == {
        "labels": ["alpha", "beta"],
        "limit": 7,
        "note": None,
        "topic": "clouds",
    }
    assert "Input is not a terminal" not in _plain(result.output)


def test_non_tty_run_reports_all_missing_required_inputs_before_setup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A scripted run rejects every missing required name before host setup."""
    flow = _write_flow(tmp_path, REQUIRED_INPUTS_FLOW)
    calls: list[str] = []
    monkeypatch.setitem(sys.modules, "questionary", None)

    def fail_if_called(name: str):
        def fail(*args, **kwargs):
            calls.append(name)
            raise AssertionError(f"{name} must not run")

        return fail

    monkeypatch.setattr(
        run_cli,
        "_ensure_provider_keys",
        fail_if_called("provider"),
    )
    monkeypatch.setattr(
        run_cli,
        "_default_run_log_path",
        fail_if_called("default log path"),
    )
    monkeypatch.setattr(
        run_cli,
        "_report_run_log",
        fail_if_called("log report"),
    )

    result = runner.invoke(app, ["run", str(flow), "--quiet"])

    assert result.exit_code == 2
    assert isinstance(result.exception, SystemExit)
    output = _plain(result.output)
    assert "missing required flow inputs" in output
    assert output.index("first") < output.index("second")
    assert "optional" not in output
    assert "Input is not a terminal" not in output
    assert "Aborted" not in output
    assert "Run log:" not in output
    assert calls == []


def test_tty_run_parses_structured_prompt_as_native_value(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A structured prompt crosses START and CODE as its declared native type."""
    flow = _write_flow(tmp_path, STRUCTURED_INPUT_FLOW)
    log = tmp_path / "run.jsonl"
    raw = "[2, 3, 5]"
    validated: list[str] = []
    questionary = ModuleType("questionary")

    class TextWidget:
        """Return the exact raw text accepted by the production validator."""

        def __init__(self, validate) -> None:
            """Retain the callback that the real widget runs before submission."""
            self._validate = validate

        def ask(self) -> str:
            """Validate and return the scripted answer without converting it."""
            if self._validate is not None:
                validated.append(raw)
                assert self._validate(raw) is True
            return raw

    def style(_rules):
        """Supply the minimal style constructor used by the production prompt."""
        return object()

    def text(_label, *, validate=None, **_options):
        """Build the minimal text widget around production's callback."""
        return TextWidget(validate)

    questionary.Style = style
    questionary.text = text
    monkeypatch.setitem(sys.modules, "questionary", questionary)
    monkeypatch.setattr(run_cli, "_Q_STYLE", None)
    monkeypatch.setattr(typer_testing, "make_input_stream", _tty_input_stream)

    result = runner.invoke(
        app,
        ["run", str(flow), "--log", str(log), "--quiet"],
        input=b"",
    )

    assert result.exit_code == 0, result.output
    assert log.is_file()
    assert validated == [raw]
    assert literal_eval(_plain(result.stdout)) == {
        "is_list": True,
        "items": [2, 3, 5],
    }


def test_non_tty_inputs_json_preserves_explicit_structured_null(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A scripted JSON file can still bind explicit null to a nullable list."""
    flow = _write_flow(tmp_path, NULLABLE_STRUCTURED_INPUT_FLOW)
    inputs = tmp_path / "inputs.json"
    inputs.write_text(json.dumps({"items": None}))
    log = tmp_path / "run.jsonl"
    monkeypatch.setitem(sys.modules, "questionary", None)

    result = runner.invoke(
        app,
        [
            "run",
            str(flow),
            "--inputs",
            str(inputs),
            "--log",
            str(log),
            "--quiet",
        ],
    )

    assert result.exit_code == 0, result.output
    assert log.is_file()
    assert literal_eval(_plain(result.stdout)) == {
        "is_none": True,
        "items": None,
    }


def test_run_forwards_tty_prompt_availability(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The command passes its actual stdin capability to input gathering."""
    flow = _write_flow(tmp_path, ECHO_FLOW)
    log = tmp_path / "run.jsonl"
    prompt_availability: list[bool] = []

    def gather_inputs(_decls, _have, *, prompt_available):
        prompt_availability.append(prompt_available)
        return {"topic": "clouds"}

    monkeypatch.setattr(run_cli, "_gather_missing_inputs", gather_inputs)
    monkeypatch.setattr(typer_testing, "make_input_stream", _tty_input_stream)

    result = runner.invoke(
        app,
        ["run", str(flow), "--log", str(log), "--quiet"],
        input=b"",
    )

    assert result.exit_code == 0, result.output
    assert prompt_availability == [True]
    assert log.is_file()
    assert "clouds" in result.stdout


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
