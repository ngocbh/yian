"""Tests for the `yian run` CLI — exercised through typer's `CliRunner`.

These use CODE-only flows (no AGENT) so the suite never hits a network or needs a
provider key. They cover: successful runs, flag-supplied inputs, `--inputs` JSON,
non-UTF-8, malformed, or non-object `--inputs` files, interactive typed JSON,
interactive `Set null` shadowing a declared default, non-interactive structured null,
flag-over-JSON precedence, non-TTY missing-input policy and setup ordering, a failing
run's non-zero exit, and the `k=v` parse error.
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

import yian.cli.run as run_cli
from yian.cli import app
from yian.tools import TOOL_REGISTRY, register_tool

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
  topic: !str
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
  topic: !str
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
  topic: !str
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
  topic: !str
  limit: !int 7
  labels: !list(str) [alpha, beta]
  note: !Optional(str)
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
  first: !str
  optional: !Optional(str)
  second: !int
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
  items: !list(int)
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
  items: !Optional(list(int))
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

NULLABLE_DEFAULTED_SCALAR_FLOW = """\
id: cli_nullable_defaulted_scalar
name: cli_nullable_defaulted_scalar
input:
  value: !Optional(int) 7
nodes:
  inspect:
    kind: code
    input:
      value: ${input.value}
    output: object
    code: |
      return {
          "value": inputs["value"],
          "is_none": inputs["value"] is None,
          "is_fallback": inputs["value"] == 7,
      }
output: ${inspect.output}
"""


REGISTERED_TOOL_FLOW = """\
id: cli_registered_tool
name: cli_registered_tool
nodes:
  call:
    kind: tool
    tool_id: cli_run_probe
output: ${call.output}
"""


def test_run_succeeds_with_input_flag(tmp_path: Path):
    flow = _write_flow(tmp_path, ECHO_FLOW)
    result = runner.invoke(app, ["run", str(flow), "--input", "topic=clouds"])
    assert result.exit_code == 0, result.output
    assert "clouds" in result.stdout


def test_run_keeps_registered_tools(tmp_path: Path):
    """The CLI passes its ask_user handler per run without hiding registered tools."""
    flow = _write_flow(tmp_path, REGISTERED_TOOL_FLOW)
    register_tool("cli_run_probe")(lambda: "from the registry")
    try:
        result = runner.invoke(app, ["run", str(flow)])
    finally:
        TOOL_REGISTRY.pop("cli_run_probe", None)
    assert result.exit_code == 0, result.output
    assert "from the registry" in result.stdout


def test_run_inline_code_in_process(tmp_path: Path):
    # an inline `code:` node (bare body, run in-process) works end-to-end through `yian run`.
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


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("{bad", "invalid JSON at line 1, column 2"),
        ("null", "must contain a JSON object, got null"),
        ("[1, 2]", "must contain a JSON object, got array"),
        ('[["topic", "rivers"]]', "must contain a JSON object, got array"),
        ('"rivers"', "must contain a JSON object, got string"),
    ],
)
def test_inputs_json_must_be_an_object(tmp_path: Path, text: str, message: str):
    # malformed or non-object `--inputs` is a bad parameter (exit 2), not a traceback;
    # an array of pairs is not quietly read as an object.
    flow = _write_flow(tmp_path, ECHO_FLOW)
    inputs = tmp_path / "in.json"
    inputs.write_text(text)
    result = runner.invoke(app, ["run", str(flow), "--inputs", str(inputs)])
    assert result.exit_code == 2, result.output
    output = " ".join(_plain(result.output).split())
    assert "--inputs" in output
    assert message in output


def test_inputs_file_must_be_utf8(tmp_path: Path):
    # a UTF-16 export is a bad parameter (exit 2), not a UnicodeDecodeError traceback.
    flow = _write_flow(tmp_path, ECHO_FLOW)
    inputs = tmp_path / "in.json"
    inputs.write_bytes(b"\xff\xfe" + '{"topic": "rivers"}'.encode("utf-16-le"))
    result = runner.invoke(app, ["run", str(flow), "--inputs", str(inputs)])
    assert result.exit_code == 2, result.output
    output = " ".join(_plain(result.output).split())
    assert "--inputs" in output
    assert "not UTF-8 text" in output


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


def test_tty_run_binds_explicit_null_instead_of_declared_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The visible null action crosses START as present ``None``, not omission."""
    flow = _write_flow(tmp_path, NULLABLE_DEFAULTED_SCALAR_FLOW)
    log = tmp_path / "run.jsonl"
    choice_titles: list[list[str]] = []
    questionary = ModuleType("questionary")

    class Choice:
        """Questionary-compatible visible title and opaque production value."""

        def __init__(self, title, value=None, **_options) -> None:
            self.title = title
            self.value = title if value is None else value

    class SelectWidget:
        """Return the value carried by the selected production choice."""

        def __init__(self, value) -> None:
            self._value = value

        def ask(self):
            """Return the selected choice's opaque production value."""
            return self._value

    def style(_rules):
        """Supply the minimal style constructor used by the production prompt."""
        return object()

    def select(_label, *, choices, **_options):
        """Select exactly the production choice visibly labelled ``Set null``."""
        titles = [choice.title for choice in choices]
        choice_titles.append(titles)
        matches = [choice for choice in choices if choice.title == "Set null"]
        assert len(matches) == 1
        return SelectWidget(matches[0].value)

    def unexpected_widget(*_args, **_kwargs):
        """Fail if the explicit null action opens a value-entry widget."""
        raise AssertionError("Set null must bypass text and confirm widgets")

    questionary.Style = style
    questionary.Choice = Choice
    questionary.select = select
    questionary.text = unexpected_widget
    questionary.confirm = unexpected_widget
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
    assert choice_titles == [["↩ Use declared default", "Set null", "Enter value"]]
    assert literal_eval(_plain(result.stdout)) == {
        "is_fallback": False,
        "is_none": True,
        "value": None,
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
    import yian.cli.run as run_mod

    flow = _write_flow(tmp_path, ECHO_FLOW)

    def _boom(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(run_mod, "_run_flow_controlled", _boom)
    result = runner.invoke(app, ["run", str(flow), "--input", "topic=clouds"])
    assert result.exit_code == 130
    assert "cancelled by user" in result.output


def test_prompt_request_offers_model_options_as_a_choice(monkeypatch):
    """Options sent with a model ``ask_user`` render as one choice list."""
    from yian.cli.terminal import PromptRequest

    seen = []

    def ask_question(question, index=None, total=None, request=None):
        seen.append(question)
        return "B"

    monkeypatch.setattr(run_cli, "_ask_question", ask_question)
    request = PromptRequest(question="Which?", options=["A", "B"])

    assert run_cli._prompt_request(request) == "B"
    assert seen == [{"question": "Which?", "options": [{"label": "A"}, {"label": "B"}]}]


def test_prompt_request_hides_a_secret_answer(monkeypatch):
    """A secret request asks through a password widget, never plain text."""
    import questionary

    from yian.cli.terminal import PromptRequest

    widget = object()
    built: list[tuple[str, str]] = []
    asked: list[tuple[object, object]] = []

    def password(label, **_options):
        built.append(("password", label))
        return widget

    def text(label, **_options):
        built.append(("text", label))
        return widget

    def ask_widget(given, request=None):
        asked.append((given, request))
        return "sk-x"

    monkeypatch.setattr(questionary, "password", password)
    monkeypatch.setattr(questionary, "text", text)
    monkeypatch.setattr(run_cli, "_ask_widget", ask_widget)
    request = PromptRequest(question="Enter KEY", secret=True)

    assert run_cli._prompt_request(request) == "sk-x"
    assert built == [("password", "Enter KEY")]
    assert asked == [(widget, request)]


def test_undeclared_input_flag_is_a_bad_parameter(tmp_path: Path):
    flow = _write_flow(tmp_path, ECHO_FLOW)
    result = runner.invoke(
        app, ["run", str(flow), "--input", "topic=clouds", "--input", "topik=x"]
    )
    assert result.exit_code == 2
    # Rich wraps the error panel, so compare with the box and line breaks removed.
    text = " ".join(_plain(result.output).replace("│", " ").split())
    assert "unknown input 'topik'; declared inputs: topic" in text


def test_undeclared_inputs_json_key_is_a_bad_parameter(tmp_path: Path):
    flow = _write_flow(tmp_path, ECHO_FLOW)
    inputs = tmp_path / "inputs.json"
    inputs.write_text(json.dumps({"topic": "clouds", "topik": "x"}))
    result = runner.invoke(app, ["run", str(flow), "--inputs", str(inputs)])
    assert result.exit_code == 2
    assert "unknown input 'topik'" in _plain(result.output)


AGENT_FLOW = """\
id: cli_agent
name: cli_agent
input:
  topic: str
nodes:
  write:
    kind: agent
    input:
      topic: ${input.topic}
    prompt: "Write about ${topic}"
    output: str
output: ${write.output}
"""


def _agent_flow(dir_: Path, llm_config: str = "") -> Path:
    """Write the one-agent flow, with ``llm_config`` as a flow-level YAML mapping if given."""
    body = AGENT_FLOW + (f"llm_config: {llm_config}\n" if llm_config else "")
    return _write_flow(dir_, body)


def _fake_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Answer every AGENT locally."""
    from langchain_core.messages import AIMessage
    import yian.llm_clients as llm_clients

    class _Model:
        def invoke(self, messages):
            return AIMessage(content="written offline")

    monkeypatch.setattr(llm_clients, "model_from_config", lambda config: _Model())


def _run_logs(home: Path) -> list[Path]:
    """Return the run logs written under the isolated home."""
    return sorted((home / ".yian" / "runs").glob("**/*.jsonl"))


def test_run_without_any_model_prints_run_start_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    isolated_cli_home: Path,
) -> None:
    """A flow whose agent no layer gives a model fails before any log exists."""
    monkeypatch.delenv("YIAN_DEFAULT_PROVIDER", raising=False)
    monkeypatch.delenv("YIAN_DEFAULT_MODEL", raising=False)
    flow = _agent_flow(tmp_path)

    result = runner.invoke(app, ["run", str(flow), "-q", "--input", "topic=x"])

    assert result.exit_code == 1, result.output
    output = re.sub(r"\s+", " ", _plain(result.output))
    assert "has no provider or model" in output
    assert "yian auth <provider>" in output
    assert "Traceback" not in output
    assert _run_logs(isolated_cli_home) == []


def test_azure_agent_on_other_endpoint_needs_env_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    isolated_cli_home: Path,
) -> None:
    """The saved Azure key covers only its own resource; another one needs the env key."""
    from yian.llm_clients.auth import ApiKeyCredentials, AuthStore

    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    AuthStore().save(
        "azure",
        ApiKeyCredentials("az"),
        "dep",
        endpoint="https://r.openai.azure.com/",
        make_default=False,
    )
    other = tmp_path / "other"
    other.mkdir()
    flow = _agent_flow(
        other,
        "{provider: azure, model: dep, base_url: 'https://other.openai.azure.com/'}",
    )

    result = runner.invoke(app, ["run", str(flow), "-q", "--input", "topic=x"])

    assert result.exit_code == 1, result.output
    assert "AZURE_OPENAI_API_KEY" in re.sub(r"\s+", " ", _plain(result.output))
    assert _run_logs(isolated_cli_home) == []

    _fake_model(monkeypatch)
    saved = tmp_path / "saved"
    saved.mkdir()
    flow = _agent_flow(saved, "{provider: azure, model: dep}")

    result = runner.invoke(app, ["run", str(flow), "-q", "--input", "topic=x"])

    assert result.exit_code == 0, result.output
    assert "written offline" in result.stdout


def test_keyless_vllm_login_runs_off_a_terminal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A vLLM login saved without a key runs with no VLLM_API_KEY and no terminal."""
    from yian.llm_clients.auth import AuthStore

    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    AuthStore().save("vllm", None, "v", make_default=False)
    _fake_model(monkeypatch)
    flow = _agent_flow(tmp_path, "{provider: vllm, model: v}")

    result = runner.invoke(app, ["run", str(flow), "-q", "--input", "topic=x"])

    assert result.exit_code == 0, result.output
    assert "written offline" in result.stdout
