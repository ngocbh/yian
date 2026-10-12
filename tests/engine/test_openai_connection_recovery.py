"""Recorded OpenAI connection identity governs resume and delayed backward."""

import json
import time

import httpx
import pytest

from yian.compose import load_flow
from yian.events import NodeStarted
from yian.llm_clients.auth import (
    ApiKeyCredentials,
    AuthStore,
    OAuthCredentials,
)
from yian.run import resume_flow, run_flow
from tests.fixtures.openai_responses import ResponsesServer

_FLOW = """id: pinned-openai
name: pinned_openai
nodes:
  agent: {kind: agent, prompt: hi, llm_config: {provider: openai, model: gpt-5.5}}
output: ${agent.output}
"""


def _session_flow():
    flow = _FLOW.replace("{kind: agent,", "{kind: agent, long_term_memory: session,")
    assert flow != _FLOW
    return flow


@pytest.fixture(autouse=True)
def isolated_auth(tmp_path, monkeypatch):
    monkeypatch.setenv("YIAN_AUTH_FILE", str(tmp_path / "auth.json"))
    for key in (
        "YIAN_DEFAULT_PROVIDER",
        "YIAN_DEFAULT_MODEL",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
    ):
        monkeypatch.delenv(key, raising=False)


def _connect():
    return AuthStore().save(
        "openai",
        OAuthCredentials(
            "access-secret", "refresh-secret", time.time() + 3600, "account"
        ),
        "gpt-5.5",
    )


def _interrupt(flow, path):
    def stop(event):
        if isinstance(event, NodeStarted) and event.node_id == "agent":
            raise RuntimeError("host stopped before model")

    with pytest.raises(RuntimeError, match="host stopped"):
        run_flow(flow, {}, log=path, on_event=stop)


def test_saved_identity_is_recorded_and_resume_refuses_reconnect(tmp_path):
    selected = _connect()
    flow = load_flow(_FLOW)
    path = tmp_path / "run.jsonl"
    _interrupt(flow, path)
    payload = json.loads(path.read_text().splitlines()[0])["payload"]
    config = payload["effective_llm_by_static_path"]["agent"]
    assert config["_connection"]["credential_id"] == selected.credential_id
    assert "secret" not in path.read_text()
    AuthStore().save("openai", ApiKeyCredentials("new-key"), "different-model")
    result = resume_flow(flow, log=path)
    assert result.status == "failed"
    assert "changed" in result.failure.message


def test_old_api_record_resumes_via_api_despite_saved_subscription(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("OPENAI_API_KEY", "old-api-key")
    flow = load_flow(_FLOW)
    path = tmp_path / "old-run.jsonl"
    _interrupt(flow, path)
    lines = path.read_text().splitlines()
    first = json.loads(lines[0])
    del first["payload"]["effective_llm_by_static_path"]["agent"]["_connection"]
    lines[0] = json.dumps(first)
    path.write_text("\n".join(lines) + "\n")
    _connect()
    server = ResponsesServer(tool_first=False)
    from yian.llm_clients.openai_client import OpenAIClient

    original = OpenAIClient.get_llm

    def model(client):
        client.kwargs["http_client"] = transport
        client.kwargs["max_retries"] = 0
        return original(client)

    with httpx.Client(transport=httpx.MockTransport(server)) as transport:
        monkeypatch.setattr(OpenAIClient, "get_llm", model)
        result = resume_flow(flow, log=path)
    assert result.status == "succeeded"
    assert server.requests[0].url.host == "api.openai.com"
    assert server.requests[0].headers["Authorization"] == "Bearer old-api-key"


def test_backward_refuses_changed_selected_connection(tmp_path, monkeypatch):
    _connect()
    from yian.llm_clients import openai_subscription

    original = openai_subscription.create_subscription_model
    server = ResponsesServer(tool_first=False)
    with httpx.Client(transport=httpx.MockTransport(server)) as client:
        monkeypatch.setattr(
            openai_subscription,
            "create_subscription_model",
            lambda *args, **kwargs: original(
                *args, **kwargs, http_client=client, max_retries=0
            ),
        )
        result = run_flow(
            load_flow(_FLOW), {}, learnable=True, log=tmp_path / "run.jsonl"
        )
        assert result.status == "succeeded"
        _connect()
        outcome = result.backward("Review")
    assert outcome.status == "failed"
    assert len(server.requests) == 1


def test_old_backward_config_gets_api_semantics_before_factory(monkeypatch):
    from langchain_core.messages import AIMessage
    import yian.llm_clients as clients
    from yian.nodes.agent.backward import backward_agent
    from tests.engine.test_agent_backward import _context

    data = _context().node_data
    data["effective_llm_config"] = {"provider": "openai", "model": "old-model"}
    context = _context(node_data=data)
    _connect()
    configs = []

    class Model:
        def invoke(self, messages):
            return AIMessage(
                content='{"input_feedback":{},"self_feedback":[],"variable_feedback":{}}'
            )

    def factory(config):
        configs.append(config)
        return Model()

    monkeypatch.setattr(clients, "model_from_config", factory)
    backward_agent(context, ({"answer": "Review"},))
    assert configs[0]["_connection"]["method"] == "api_key"
    assert configs[0]["_connection"]["source"] == "legacy_environment"


def test_old_api_run_resumes_shared_message_history(tmp_path, monkeypatch):
    from langchain_core.messages import AIMessage
    import yian.run as boundary
    import yian.llm_clients as clients
    from yian.llm_clients.connection import (
        ConnectionSnapshot,
        connection_snapshot,
    )
    from yian.memory.session import SessionMemory

    flow = load_flow(_session_flow())
    history = SessionMemory()
    snapshot = connection_snapshot()
    monkeypatch.setattr(
        boundary,
        "connection_snapshot",
        lambda: ConnectionSnapshot(snapshot.defaults, snapshot.provider_models, {}),
    )
    calls = []
    configs = []

    class Model:
        def invoke(self, messages):
            # Snapshot: the AGENT loop appends the reply to this same list after invoke.
            calls.append(list(messages))
            return AIMessage(content="reply")

    def factory(config):
        configs.append(config)
        return Model()

    monkeypatch.setattr(clients, "model_from_config", factory)
    assert run_flow(flow, {}, memory={"session": history}).status == "succeeded"
    path = tmp_path / "old-session.jsonl"

    def stop(event):
        if isinstance(event, NodeStarted) and event.node_id == "agent":
            raise RuntimeError("stop")

    with pytest.raises(RuntimeError, match="stop"):
        run_flow(flow, {}, memory={"session": history}, log=path, on_event=stop)
    _connect()
    monkeypatch.setattr(
        boundary,
        "connection_snapshot",
        lambda: pytest.fail("resume read saved selection"),
    )
    resumed = resume_flow(flow, log=path, memory={"session": history})
    assert resumed.status == "succeeded"
    assert [message.content for message in calls[-1]][-3:] == ["hi", "reply", "hi"]
    assert configs[-1]["_connection"]["source"] == "legacy_environment"


def test_fresh_connected_run_reuses_pre_cutover_api_message_history(monkeypatch):
    from langchain_core.messages import AIMessage
    import yian.run as boundary
    import yian.llm_clients as clients
    from yian.llm_clients import openai_subscription
    from yian.llm_clients.connection import (
        ConnectionSnapshot,
        connection_snapshot,
    )
    from yian.memory.session import SessionMemory

    flow = load_flow(_session_flow())
    history = SessionMemory()
    snapshot = connection_snapshot()

    class OldModel:
        def invoke(self, messages):
            return AIMessage(content="old API reply")

    with monkeypatch.context() as old:
        old.setattr(
            boundary,
            "connection_snapshot",
            lambda: ConnectionSnapshot(snapshot.defaults, snapshot.provider_models, {}),
        )
        old.setattr(clients, "model_from_config", lambda config: OldModel())
        assert run_flow(flow, {}, memory={"session": history}).status == "succeeded"

    original = openai_subscription.create_subscription_model
    server = ResponsesServer(tool_first=False)
    with httpx.Client(transport=httpx.MockTransport(server)) as client:
        monkeypatch.setattr(
            openai_subscription,
            "create_subscription_model",
            lambda *a, **kw: original(*a, **kw, http_client=client, max_retries=0),
        )
        # Each explicit reconnect changes the stored credential identity. Both
        # new runs retain ordinary compatible history without a memory migration.
        for _ in range(2):
            _connect()
            result = run_flow(flow, {}, memory={"session": history})
            assert result.status == "succeeded", result.failure
            assert result.output == "6"
    assert len(server.bodies) == 2
    for request in server.bodies:
        assert any(
            item.get("role") == "assistant"
            and any(
                block.get("text") == "old API reply"
                for block in item.get("content", [])
            )
            for item in request["input"]
        )
