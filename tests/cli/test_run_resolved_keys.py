"""`yian run` never prints or logs a key from the environment or a saved login a provider echoes."""

from __future__ import annotations

from pathlib import Path
import re

import httpx
import pytest
from typer.testing import CliRunner

from yian.cli import app
from yian.llm_clients.auth import ApiKeyCredentials, AuthStore
from yian.llm_clients.openai_client import OpenAIClient

runner = CliRunner()

_KEY = "sk-cli-401-echo-wvbn"
_FRAGMENTS = (_KEY, _KEY[:8], "*" + _KEY[-4:])
_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")

_FLOW = """\
id: keyed
name: keyed
nodes:
  agent: {kind: agent, prompt: hi}
output: ${agent.output}
llm_config: {provider: vllm, model: served-model}
"""


@pytest.fixture(autouse=True)
def _isolated_keys(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Start each test with no provider key or default, and an empty store in ``tmp_path``."""
    for name in ("OPENAI_API_KEY", "VLLM_API_KEY"):
        # Set first so teardown removes a key the run exported even if it was absent.
        monkeypatch.setenv(name, "x")
        monkeypatch.delenv(name)
    for name in ("YIAN_DEFAULT_PROVIDER", "YIAN_DEFAULT_MODEL", "OPENAI_BASE_URL", "VLLM_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("YIAN_AUTH_FILE", str(tmp_path / "auth" / "auth.json"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))


def _rejecting(request: httpx.Request) -> httpx.Response:
    """Echo the key masked, as OpenAI's 401 does."""
    masked = f"{_KEY[:8]}****{_KEY[-4:]}"
    return httpx.Response(401, json={"error": {
        "message": f"Incorrect API key provided: {masked}.",
        "type": "invalid_request_error", "code": "invalid_api_key",
    }})


def _route_openai_client(monkeypatch: pytest.MonkeyPatch, transport: httpx.Client) -> None:
    original = OpenAIClient.get_llm

    def construct(wrapper):
        wrapper.kwargs.update(http_client=transport, max_retries=0)
        return original(wrapper)

    monkeypatch.setattr(OpenAIClient, "get_llm", construct)


def _compact(text: str) -> str:
    """Remove styling, box-drawing borders, and whitespace, so Rich wrapping cannot split a fragment."""
    return re.sub(r"[\s\u2500-\u257f]+", "", _ANSI.sub("", text))


@pytest.mark.parametrize("source", ["environment", "saved login"])
@pytest.mark.parametrize("options", [[], ["--engine-trace"]], ids=["plain", "engine-trace"])
def test_a_rejected_key_is_neither_printed_nor_logged(monkeypatch, tmp_path, source, options):
    if source == "environment":
        monkeypatch.setenv("VLLM_API_KEY", _KEY)
    else:
        AuthStore().save("vllm", ApiKeyCredentials(_KEY), "served-model")
    flow = tmp_path / "flow.yaml"
    flow.write_text(_FLOW)
    log = tmp_path / "run.jsonl"
    with httpx.Client(transport=httpx.MockTransport(_rejecting)) as transport:
        _route_openai_client(monkeypatch, transport)
        result = runner.invoke(app, ["run", str(flow), "--log", str(log), *options])
    output = _compact(result.output)
    assert result.exit_code != 0
    assert "[REDACTED]" in output
    for fragment in _FRAGMENTS:
        assert fragment not in output
        assert fragment not in log.read_text()


_INPUT_PROVIDER_FLOW = """\
id: chosen
name: chosen
input:
  provider: !str openai
nodes:
  agent: {kind: agent, prompt: hi}
output: ${agent.output}
llm_config: {provider: '${input.provider}', model: chosen-model}
"""


def test_the_key_check_asks_for_the_provider_an_input_selects(monkeypatch, tmp_path):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    flow = tmp_path / "flow.yaml"
    flow.write_text(_INPUT_PROVIDER_FLOW)
    log = tmp_path / "run.jsonl"
    result = runner.invoke(
        app, ["run", str(flow), "--input", "provider=anthropic", "--log", str(log)]
    )
    output = _compact(result.output)
    assert result.exit_code == 1
    assert "ANTHROPIC_API_KEY" in output
    assert "OPENAI_API_KEY" not in output
    assert not log.exists()


def test_a_bad_provider_input_fails_before_the_log_exists(tmp_path):
    flow = tmp_path / "flow.yaml"
    flow.write_text(_INPUT_PROVIDER_FLOW)
    log = tmp_path / "run.jsonl"
    result = runner.invoke(app, ["run", str(flow), "--input", "provider=nope", "--log", str(log)])
    output = _compact(result.output)
    assert result.exit_code == 1
    assert _compact("llm_config.provider: not a known provider") in output
    assert not log.exists()


def _call_time_flow(chat_llm: str, *, pick: str = "str", value: str = "'m1'", typedefs="") -> str:
    """A flow whose ``chat`` agent's config takes ``pick``'s output, below an Anthropic flow."""
    return f"""id: picked
name: picked
{typedefs}nodes:
  pick: {{kind: code, output: {pick}, code: "return {value}"}}
  chat:
    kind: agent
    prompt: hi
    llm_config: {chat_llm}
output: ${{chat.output}}
llm_config: {{provider: anthropic}}
"""


def _answer_locally(monkeypatch: pytest.MonkeyPatch) -> list:
    from langchain_core.messages import AIMessage
    import yian.llm_clients as llm_clients

    built = []

    class _Model:
        def invoke(self, messages):
            return AIMessage(content="answered")

    def model_from_config(config):
        built.append(config)
        return _Model()

    monkeypatch.setattr(llm_clients, "model_from_config", model_from_config)
    return built


def test_the_key_check_covers_a_call_time_agent_with_a_known_provider(monkeypatch, tmp_path):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    built = _answer_locally(monkeypatch)
    flow = tmp_path / "flow.yaml"
    flow.write_text(_call_time_flow("{model: '${pick.output}'}"))
    log = tmp_path / "run.jsonl"
    result = runner.invoke(app, ["run", str(flow), "--log", str(log)])
    output = _compact(result.output)
    assert result.exit_code == 1, result.output
    assert "ANTHROPIC_API_KEY" in output
    assert not log.exists()
    assert built == []


@pytest.mark.parametrize("text", [
    _call_time_flow("{provider: '${pick.output}'}", pick="Optional(str)", value="'openai'"),
    _call_time_flow(
        "!Cfg ${pick.output}", pick="Cfg", value="{'provider': 'openai', 'model': 'm1'}",
        typedefs="typedefs:\n  Cfg: {provider: Optional(str), model: str}\n",
    ),
], ids=["nullable provider", "whole record"])
def test_a_provider_a_value_may_still_set_is_not_checked_up_front(monkeypatch, tmp_path, text):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    built = _answer_locally(monkeypatch)
    flow = tmp_path / "flow.yaml"
    flow.write_text(text.replace("llm_config: {provider: anthropic}", (
        "llm_config: {provider: anthropic, model: claude-x}"
    )))
    log = tmp_path / "run.jsonl"
    result = runner.invoke(app, ["run", str(flow), "--log", str(log)])
    assert result.exit_code == 0, result.output
    assert log.exists()
    [config] = built
    assert config["provider"] == "openai"
