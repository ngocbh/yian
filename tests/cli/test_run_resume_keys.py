"""`yian run --resume` checks the recorded provider keys before it writes.

Every test is offline: ``model_from_config`` is a fake model where the AGENT should
run and a raising stub where the run must stop before any model call.
"""

from __future__ import annotations

from io import BytesIO
import os
from pathlib import Path
import re
from typing import Any

import pytest
import typer.testing as typer_testing
from typer.testing import CliRunner

from yian import load_flow, run_flow
from yian.cli import app
import yian.cli.run as run_cli
from yian.events import NodeStarted
from yian.llm_clients.auth import ApiKeyCredentials, AuthStore

runner = CliRunner()

_ANSI_SGR = re.compile(r"\x1b\[[0-9;]*m")

_FLOW = """\
id: resume_key
name: resume_key
input:
  topic: !str
nodes:
  prep:
    kind: code
    input:
      topic: ${input.topic}
    output: str
    code: |
      return inputs["topic"]
  write:
    kind: agent
    input:
      text: ${prep.output}
    prompt: "Write about ${text}"
    output: str
output: ${write.output}
llm_config: {provider: PROVIDER, model: offline-model}
"""


class _TTYBytesIO(BytesIO):
    """Byte input stream that makes Click expose an interactive stdin."""

    def isatty(self) -> bool:
        """Report that the synthetic input stream is attached to a terminal."""
        return True


def _tty_input_stream(_input_value, _charset) -> _TTYBytesIO:
    """Return an empty byte stream whose terminal capability is interactive."""
    return _TTYBytesIO()


def _plain(text: str) -> str:
    """Remove styling and line wrapping so message assertions see one line."""
    return re.sub(r"\s+", " ", _ANSI_SGR.sub("", text))


@pytest.fixture(autouse=True)
def _isolated_keys(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Start each test with no provider key, auth file override, or saved store."""
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "YIAN_AUTH_FILE"):
        # Set first so teardown removes a key the check exported even if it was absent.
        monkeypatch.setenv(name, "x")
        monkeypatch.delenv(name)
    for name in ("YIAN_DEFAULT_PROVIDER", "YIAN_DEFAULT_MODEL", "OPENAI_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))


def _fake_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Answer the AGENT locally."""
    from langchain_core.messages import AIMessage
    import yian.llm_clients as llm_clients

    class _Model:
        def invoke(self, messages):
            return AIMessage(content="written offline")

    monkeypatch.setattr(llm_clients, "model_from_config", lambda config: _Model())


def _no_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail the test if the run reaches a model."""
    import yian.llm_clients as llm_clients

    def fail(config):
        raise AssertionError("the resumed run must not reach a model")

    monkeypatch.setattr(llm_clients, "model_from_config", fail)


def _flow_file(tmp_path: Path, provider: str = "openai") -> Path:
    path = tmp_path / "flow.yaml"
    path.write_text(_FLOW.replace("PROVIDER", provider))
    return path


def _unfinished_log(
    monkeypatch: pytest.MonkeyPatch,
    flow: Path,
    *,
    env_var: str | None = "OPENAI_API_KEY",
    key: str = "sk-fake",
    **options: Any,
) -> Path:
    """Record a run interrupted at ``prep``, with ``env_var`` set only while it starts."""
    loaded = load_flow(flow.read_text(), search_paths=[flow.parent])
    log = flow.parent / "unfinished.jsonl"

    def interrupt(event: Any) -> None:
        if isinstance(event, NodeStarted) and event.node_id == "prep":
            raise RuntimeError("host interruption")

    with monkeypatch.context() as scoped:
        if env_var is not None:
            scoped.setenv(env_var, key)
        with pytest.raises(RuntimeError, match="host interruption"):
            run_flow(loaded, {"topic": "x"}, log=log, on_event=interrupt, **options)
    return log


def _resume(flow: Path, log: Path):
    return runner.invoke(app, ["run", str(flow), "--resume", str(log), "--quiet"])


def _resume_on_a_tty(monkeypatch: pytest.MonkeyPatch, flow: Path, log: Path, answer):
    """Resume with an interactive stdin whose secret prompt calls ``answer``."""
    requests = []

    def prompt(request):
        requests.append(request)
        return answer()

    monkeypatch.setattr(typer_testing, "make_input_stream", _tty_input_stream)
    monkeypatch.setattr(run_cli, "_prompt_request", prompt)
    result = runner.invoke(
        app, ["run", str(flow), "--resume", str(log), "--quiet"], input=b""
    )
    assert requests and all(request.secret for request in requests)
    return result


def _resumes_with_the_key(monkeypatch: pytest.MonkeyPatch, flow: Path, log: Path) -> None:
    """Assert the log is still recoverable and unlocked."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    _fake_model(monkeypatch)
    result = _resume(flow, log)
    assert result.exit_code == 0, result.output
    assert "written offline" in result.stdout


def test_resume_without_the_key_exits_before_writing(monkeypatch, tmp_path):
    flow = _flow_file(tmp_path)
    log = _unfinished_log(monkeypatch, flow)
    before = log.read_bytes()
    _no_model(monkeypatch)

    result = _resume(flow, log)

    assert result.exit_code == 1, result.output
    output = _plain(result.output)
    assert (
        "provider 'openai' needs an API key but $OPENAI_API_KEY is not set; "
        "set it and resume again" in output
    )
    assert log.read_bytes() == before
    _resumes_with_the_key(monkeypatch, flow, log)


def test_resume_with_a_changed_key_exits_before_writing(monkeypatch, tmp_path):
    flow = _flow_file(tmp_path)
    log = _unfinished_log(monkeypatch, flow)
    before = log.read_bytes()
    monkeypatch.setenv("OPENAI_API_KEY", "sk-other")
    _no_model(monkeypatch)

    result = _resume(flow, log)

    assert result.exit_code == 1, result.output
    assert (
        "$OPENAI_API_KEY is not the key this run started with; set it to that key "
        "and resume again" in _plain(result.output)
    )
    assert log.read_bytes() == before


def test_resume_prompts_for_a_missing_key_on_a_tty(monkeypatch, tmp_path):
    flow = _flow_file(tmp_path)
    log = _unfinished_log(monkeypatch, flow)
    _fake_model(monkeypatch)

    result = _resume_on_a_tty(monkeypatch, flow, log, lambda: "sk-fake")

    assert result.exit_code == 0, result.output
    assert "written offline" in result.stdout
    assert os.environ["OPENAI_API_KEY"] == "sk-fake"


def test_resume_rejects_a_typed_key_that_does_not_match(monkeypatch, tmp_path):
    flow = _flow_file(tmp_path)
    log = _unfinished_log(monkeypatch, flow)
    before = log.read_bytes()
    _no_model(monkeypatch)

    result = _resume_on_a_tty(monkeypatch, flow, log, lambda: "sk-other")

    assert result.exit_code == 1, result.output
    assert (
        "the key entered is not the one this run started with; set $OPENAI_API_KEY "
        "to that key and resume again" in _plain(result.output)
    )
    assert log.read_bytes() == before


def test_resume_reprompts_on_empty_answers_then_gives_up(monkeypatch, tmp_path):
    flow = _flow_file(tmp_path)
    log = _unfinished_log(monkeypatch, flow)
    before = log.read_bytes()
    _no_model(monkeypatch)
    answers = iter(["", "  ", ""])

    result = _resume_on_a_tty(monkeypatch, flow, log, lambda: next(answers))

    assert result.exit_code == 1, result.output
    assert (
        "no API key provided for 'openai' ($OPENAI_API_KEY); set it and resume again"
        in _plain(result.output)
    )
    assert next(answers, None) is None
    assert log.read_bytes() == before


def test_resume_checks_other_providers_by_env_var(monkeypatch, tmp_path):
    flow = _flow_file(tmp_path, provider="anthropic")
    log = _unfinished_log(monkeypatch, flow, env_var="ANTHROPIC_API_KEY", key="sk-ant")
    before = log.read_bytes()
    _no_model(monkeypatch)

    result = _resume(flow, log)

    assert result.exit_code == 1, result.output
    assert (
        "provider 'anthropic' needs an API key but $ANTHROPIC_API_KEY is not set; "
        "set it and resume again" in _plain(result.output)
    )
    assert log.read_bytes() == before


def test_resume_on_a_saved_api_key_login_needs_no_environment_key(monkeypatch, tmp_path):
    flow = _flow_file(tmp_path)
    AuthStore().save("openai", ApiKeyCredentials("saved-key"), "gpt-5.5")
    log = _unfinished_log(monkeypatch, flow, env_var=None)
    _fake_model(monkeypatch)

    def no_prompt(request):
        raise AssertionError("a saved login must not prompt")

    monkeypatch.setattr(run_cli, "_prompt_request", no_prompt)
    result = _resume(flow, log)

    assert result.exit_code == 0, result.output
    assert "written offline" in result.stdout


def test_agents_sharing_one_connection_get_one_prompt(monkeypatch, tmp_path):
    flow = tmp_path / "flow.yaml"
    flow.write_text(
        _FLOW.replace("PROVIDER", "openai").replace(
            "output: ${write.output}",
            "  again:\n"
            "    kind: agent\n"
            "    input:\n"
            "      text: ${write.output}\n"
            '    prompt: "Rewrite ${text}"\n'
            "    output: str\n"
            "output: ${again.output}",
        )
    )
    log = _unfinished_log(monkeypatch, flow)
    _fake_model(monkeypatch)
    prompts = []

    def answer():
        prompts.append(None)
        return "sk-fake"

    result = _resume_on_a_tty(monkeypatch, flow, log, answer)

    assert result.exit_code == 0, result.output
    assert "written offline" in result.stdout
    assert len(prompts) == 1


def test_resume_relays_a_reconnected_store_credential(monkeypatch, tmp_path):
    flow = _flow_file(tmp_path)
    AuthStore().save("openai", ApiKeyCredentials("saved-key"), "gpt-5.5")
    log = _unfinished_log(monkeypatch, flow, env_var=None)
    # A reconnect mints a new credential id, so the pinned one is gone.
    AuthStore().save("openai", ApiKeyCredentials("saved-key"), "gpt-5.5")
    before = log.read_bytes()
    _no_model(monkeypatch)

    result = _resume(flow, log)

    assert result.exit_code == 1, result.output
    output = _plain(result.output)
    assert "yian auth openai" in output
    assert "start a new run" in output
    assert log.read_bytes() == before


def test_resume_key_prompt_eof_leaves_the_log_unchanged(monkeypatch, tmp_path):
    flow = _flow_file(tmp_path)
    log = _unfinished_log(monkeypatch, flow)
    before = log.read_bytes()
    _no_model(monkeypatch)

    def eof():
        raise EOFError

    result = _resume_on_a_tty(monkeypatch, flow, log, eof)

    assert result.exit_code == 1, result.output
    assert "run cancelled" in _plain(result.output)
    assert log.read_bytes() == before
    _resumes_with_the_key(monkeypatch, flow, log)


def test_resume_key_prompt_ctrl_c_leaves_the_log_unchanged(monkeypatch, tmp_path):
    flow = _flow_file(tmp_path)
    log = _unfinished_log(monkeypatch, flow)
    before = log.read_bytes()
    _no_model(monkeypatch)

    def interrupt():
        raise KeyboardInterrupt

    result = _resume_on_a_tty(monkeypatch, flow, log, interrupt)

    assert result.exit_code == 130, result.output
    assert "cancelled by user" in _plain(result.output)
    assert log.read_bytes() == before
    _resumes_with_the_key(monkeypatch, flow, log)


def test_resume_of_a_host_key_run_points_to_python(monkeypatch, tmp_path):
    flow = _flow_file(tmp_path)
    log = _unfinished_log(
        monkeypatch,
        flow,
        env_var=None,
        llm_config={"provider": "openai", "model": "m", "api_key": "sk-h"},
    )
    before = log.read_bytes()
    monkeypatch.setenv("OPENAI_API_KEY", "sk-h")
    _no_model(monkeypatch)

    result = _resume(flow, log)

    assert result.exit_code == 1, result.output
    output = _plain(result.output)
    assert "resume_flow(api_key=" in output
    assert "cannot supply" in output
    assert log.read_bytes() == before


def test_terminal_resume_of_an_agent_run_skips_the_key_check(monkeypatch, tmp_path):
    flow = _flow_file(tmp_path)
    loaded = load_flow(flow.read_text(), search_paths=[tmp_path])
    log = tmp_path / "finished.jsonl"
    _fake_model(monkeypatch)
    with monkeypatch.context() as scoped:
        scoped.setenv("OPENAI_API_KEY", "sk-fake")
        run_flow(loaded, {"topic": "x"}, log=log)
    before = log.read_bytes()
    _no_model(monkeypatch)

    result = _resume(flow, log)

    assert result.exit_code == 0, result.output
    assert "written offline" in result.stdout
    assert log.read_bytes() == before
