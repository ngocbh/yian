"""A resumed or delayed run keeps each provider's recorded connection."""

import json

from langchain_core.messages import AIMessage
import pytest

from yian.compose import load_flow
from yian.events import NodeStarted
from yian.llm_clients import connection as connection_module
from yian.llm_clients import factory
from yian.llm_clients.auth import ApiKeyCredentials, AuthStore
from yian.run import resume_flow, run_flow

_NO_FEEDBACK = json.dumps({"input_feedback": {}, "self_feedback": [], "variable_feedback": {}})


def _flow(provider, model):
    return load_flow(
        "id: pinned\nname: pinned\nnodes:\n"
        f"  agent: {{kind: agent, prompt: hi, llm_config: {{provider: {provider}, model: {model}}}}}\n"
        "output: ${agent.output}\n"
    )


@pytest.fixture(autouse=True)
def no_provider_keys(monkeypatch):
    for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL"):
        monkeypatch.delenv(name, raising=False)


def _fake_clients(monkeypatch, *replies):
    """Record each built client's provider and kwargs; models answer ``replies`` in order."""
    built, calls = [], []
    replies = replies or ("ok",)

    class Model:
        def invoke(self, messages):
            calls.append(messages)
            return AIMessage(content=replies[min(len(calls), len(replies)) - 1])

    def create(provider, model, **kwargs):
        built.append((provider, kwargs))
        return type("Client", (), {"get_llm": lambda self: Model()})()

    monkeypatch.setattr(factory, "create_llm_client", create)
    return built


def _plain(key):
    return key.get_secret_value() if hasattr(key, "get_secret_value") else key


def _interrupt(flow, path, **options):
    def stop(event):
        if isinstance(event, NodeStarted) and event.node_id == "agent":
            raise RuntimeError("host stopped before model")

    with pytest.raises(RuntimeError, match="host stopped"):
        run_flow(flow, {}, log=path, on_event=stop, **options)


def _recorded(path):
    first = json.loads(path.read_text().splitlines()[0])
    return first["payload"]["effective_llm_by_static_path"]["agent"]


def test_store_connection_is_pinned_across_resume(tmp_path, monkeypatch):
    saved = AuthStore().save("anthropic", ApiKeyCredentials("old-key"), "claude-x")
    flow = _flow("anthropic", "claude-x")
    path = tmp_path / "run.jsonl"
    _interrupt(flow, path)
    connection = _recorded(path)["_connection"]
    assert connection["source"] == "store"
    assert connection["credential_id"] == saved.credential_id
    assert b"old-key" not in path.read_bytes()

    AuthStore().save("anthropic", ApiKeyCredentials("new-key"), "claude-x")
    built = _fake_clients(monkeypatch)
    result = resume_flow(flow, log=path)

    assert result.status == "failed"
    assert "yian auth anthropic" in result.failure.message
    assert all(_plain(kwargs.get("api_key")) != "new-key" for _, kwargs in built)
    assert b"new-key" not in path.read_bytes()


def test_environment_connection_never_switches_to_a_later_login(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
    flow = _flow("anthropic", "claude-x")
    path = tmp_path / "run.jsonl"
    _interrupt(flow, path)
    assert _recorded(path)["_connection"]["source"] == "environment"

    AuthStore().save("anthropic", ApiKeyCredentials("later-key"), "claude-x")
    built = _fake_clients(monkeypatch)
    result = resume_flow(flow, log=path)

    assert result.status == "succeeded", result.failure
    assert built and all("api_key" not in kwargs for _, kwargs in built)


def test_removed_login_fails_cleanly(tmp_path, monkeypatch):
    AuthStore().save("anthropic", ApiKeyCredentials("old-key"), "claude-x")
    flow = _flow("anthropic", "claude-x")
    path = tmp_path / "run.jsonl"
    _interrupt(flow, path)

    AuthStore().remove("anthropic")
    built = _fake_clients(monkeypatch)
    result = resume_flow(flow, log=path)

    assert result.status == "failed"
    assert "yian auth anthropic" in result.failure.message
    assert built == []


def test_ollama_resume_never_reads_the_snapshot(tmp_path, monkeypatch):
    import yian.run as boundary

    flow = _flow("ollama", "llama3")
    path = tmp_path / "run.jsonl"
    _interrupt(flow, path)
    assert "_connection" not in _recorded(path)

    def no_snapshot():
        pytest.fail("read the connection snapshot")

    monkeypatch.setattr(connection_module, "connection_snapshot", no_snapshot)
    monkeypatch.setattr(boundary, "connection_snapshot", no_snapshot)
    built = _fake_clients(monkeypatch)
    result = resume_flow(flow, log=path)

    assert result.status == "succeeded", result.failure
    assert built[0][0] == "ollama"


def test_host_key_resume_and_backward(tmp_path, monkeypatch):
    AuthStore().save("anthropic", ApiKeyCredentials("saved-key"), "claude-x")
    flow = _flow("anthropic", "claude-x")
    path = tmp_path / "run.jsonl"
    _interrupt(flow, path, llm_config={"provider": "anthropic", "api_key": "host-key"}, learnable=True)
    recorded = _recorded(path)
    assert "_connection" not in recorded
    assert b"host-key" not in path.read_bytes()

    def no_store(*args, **kwargs):
        pytest.fail("resolved a saved login")

    monkeypatch.setattr(AuthStore, "credentials", no_store)
    built = _fake_clients(monkeypatch, "ok", _NO_FEEDBACK)
    result = resume_flow(flow, log=path, api_key="host-key")
    assert result.status == "succeeded", result.failure
    outcome = result.backward("Be brief.", api_key="host-key")
    assert outcome.status == "succeeded", outcome.failure

    assert len(built) == 2
    for provider, kwargs in built:
        assert provider == "anthropic"
        assert _plain(kwargs["api_key"]) == "host-key"
        assert "base_url" not in kwargs
