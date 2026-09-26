"""Connection identity, provider defaults, and factory dispatch without network."""

import json
import time

import httpx
import pytest
from langchain_core.messages import HumanMessage

from agent_composer.llm_clients.auth import (
    ApiKeyCredentials,
    AuthStore,
    OAuthCredentials,
)
from agent_composer.llm_clients.connection import (
    connection_snapshot,
    legacy_connection_config,
    resolve_model_config,
)
from agent_composer.llm_clients.factory import model_from_config
from agent_composer.compile.llm_cascade import resolve_llm_cascade
from agent_composer.compose.loader import load_flow
from tests.fixtures.openai_responses import ResponsesServer


@pytest.fixture(autouse=True)
def isolated_auth(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_COMPOSER_AUTH_FILE", str(tmp_path / "auth.json"))
    for name in (
        "AGENT_COMPOSER_DEFAULT_PROVIDER",
        "AGENT_COMPOSER_DEFAULT_MODEL",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)


def subscription():
    return AuthStore().save(
        "openai",
        OAuthCredentials(
            "access-secret", "refresh-secret", time.time() + 3600, "account"
        ),
        "gpt-selected",
    )


def test_default_openai_and_provider_scoped_models():
    assert resolve_model_config({})["provider"] == "openai"
    assert resolve_model_config({})["model"] == "gpt-5.5"
    assert (
        resolve_model_config({"provider": "anthropic"})["model"] == "claude-sonnet-4-5"
    )
    with pytest.raises(ValueError, match="explicit model"):
        resolve_model_config({"provider": "google"})


def test_saved_selection_wins_incidental_key_and_is_nonsecret(monkeypatch):
    selected = subscription()
    monkeypatch.setenv("OPENAI_API_KEY", "incidental-secret")
    config = resolve_model_config({})
    assert config["model"] == "gpt-selected"
    assert config["_connection"]["method"] == "subscription"
    assert config["_connection"]["credential_id"] == selected.credential_id
    assert "secret" not in json.dumps(config)


def test_authored_model_wins_saved_default():
    subscription()
    assert resolve_model_config({"model": "explicit"})["model"] == "explicit"


def test_provider_only_cascade_uses_matching_fallback():
    snap = connection_snapshot()
    flow = load_flow(
        "id: test\nname: test\nnodes:\n  a: {kind: agent, prompt: hi, llm_config: {provider: anthropic}}\noutput: ${a.output}\n"
    )
    effective = resolve_llm_cascade(
        flow.compiled, {}, defaults=snap.defaults, provider_models=snap.provider_models
    )
    assert effective["a"]["model"] == "claude-sonnet-4-5"


def test_authored_model_still_inherits_across_provider_fields():
    snap = connection_snapshot()
    flow = load_flow(
        "id: test\nname: test\nllm_config: {model: authored}\nnodes:\n  a: {kind: agent, prompt: hi, llm_config: {provider: anthropic}}\noutput: ${a.output}\n"
    )
    effective = resolve_llm_cascade(
        flow.compiled, {}, defaults=snap.defaults, provider_models=snap.provider_models
    )
    assert effective["a"]["model"] == "authored"


def test_snapshot_does_not_follow_reconnect():
    subscription()
    config = resolve_model_config({})
    AuthStore().save("openai", ApiKeyCredentials("new-secret"), "other")
    model = model_from_config(config)
    with pytest.raises(RuntimeError, match="changed"):
        model.invoke([HumanMessage(content="hi")])


def test_api_environment_identity_checked_on_every_request(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "first-secret")
    config = resolve_model_config({})
    model = model_from_config(config)
    monkeypatch.setenv("OPENAI_API_KEY", "second-secret")
    with pytest.raises(RuntimeError, match="changed"):
        model.invoke([HumanMessage(content="hi")])


def test_missing_identity_does_not_adopt_later_environment_key(monkeypatch):
    config = resolve_model_config({})
    monkeypatch.setenv("OPENAI_API_KEY", "new-secret")
    model = model_from_config(config)
    with pytest.raises(RuntimeError, match="ac auth openai"):
        model.invoke([HumanMessage(content="hi")])


def test_legacy_openai_cannot_select_subscription(monkeypatch):
    subscription()
    monkeypatch.setenv("OPENAI_API_KEY", "legacy-secret")
    old = {"provider": "openai", "model": "old"}
    config = legacy_connection_config(old)
    assert old == {"provider": "openai", "model": "old"}
    assert config["_connection"]["method"] == "api_key"
    assert config["_connection"]["source"] == "legacy_environment"
    assert legacy_connection_config(config) == config


def test_internal_metadata_is_not_yaml_configuration():
    from agent_composer.llm_clients.config import LLMConfig

    config = resolve_model_config({})
    with pytest.raises(ValueError, match="Extra inputs"):
        LLMConfig(**config)
    config["_connection"]["token"] = "bad"
    with pytest.raises(ValueError, match="connection"):
        model_from_config(config)


def test_factory_subscription_supports_structured_output_on_responses(monkeypatch):
    subscription()
    from agent_composer.llm_clients import openai_subscription
    from pydantic import BaseModel
    import tests.fixtures.openai_responses as fixture

    class Answer(BaseModel):
        value: int

    monkeypatch.setattr(
        fixture,
        "TOOL_CALL",
        {**fixture.TOOL_CALL, "name": "Answer", "arguments": '{"value": 3}'},
    )
    server = ResponsesServer()
    original = openai_subscription.create_subscription_model
    with httpx.Client(transport=httpx.MockTransport(server)) as client:
        monkeypatch.setattr(
            openai_subscription,
            "create_subscription_model",
            lambda *a, **kw: original(*a, **kw, http_client=client, max_retries=0),
        )
        model = model_from_config(resolve_model_config({}))
        result = model.with_structured_output(Answer).invoke("Give an integer")
    assert result.value == 3
    assert server.bodies[0]["tools"][0]["name"] == "Answer"
    assert server.requests[0].headers["Authorization"] == "Bearer access-secret"


def test_saved_api_key_and_preflight_ignore_incidental_environment(monkeypatch):
    from agent_composer.cli.utils import ensure_api_key
    from agent_composer.llm_clients.openai_client import OpenAIClient

    AuthStore().save("openai", ApiKeyCredentials("saved-key"), "gpt-5.5")
    monkeypatch.setenv("OPENAI_API_KEY", "incidental-key")
    assert ensure_api_key("openai", interactive=False) == "saved-key"
    original = OpenAIClient.get_llm
    server = ResponsesServer(tool_first=False)
    with httpx.Client(transport=httpx.MockTransport(server)) as client:

        def construct(wrapper):
            wrapper.kwargs.update(http_client=client, max_retries=0)
            return original(wrapper)

        monkeypatch.setattr(OpenAIClient, "get_llm", construct)
        model_from_config({}).invoke("hi")
    assert server.requests[0].url.host == "api.openai.com"
    assert server.requests[0].headers["Authorization"] == "Bearer saved-key"


def test_subscription_preflight_needs_no_api_key():
    from agent_composer.cli.utils import ensure_api_key

    subscription()
    assert ensure_api_key("openai", interactive=False) is None


def test_refresh_retains_connection_identity_on_model_requests(monkeypatch):
    from agent_composer.llm_clients import openai_subscription
    from agent_composer.llm_clients.auth import OpenAIAuth

    selected = AuthStore().save(
        "openai",
        OAuthCredentials("expired-token", "refresh-token", time.time() + 1, "account"),
        "gpt-5.5",
    )
    refreshed = []

    def refresh(service, credentials):
        refreshed.append(credentials)
        return OAuthCredentials(
            "fresh-token", "new-refresh", time.time() + 3600, "account"
        )

    monkeypatch.setattr(OpenAIAuth, "refresh", refresh)
    server = ResponsesServer(tool_first=False)
    original = openai_subscription.create_subscription_model
    with httpx.Client(transport=httpx.MockTransport(server)) as client:
        monkeypatch.setattr(
            openai_subscription,
            "create_subscription_model",
            lambda *a, **kw: original(*a, **kw, http_client=client, max_retries=0),
        )
        config = resolve_model_config({})
        model = model_from_config(config)
        model.invoke("hi")
        model.invoke("again")
    assert len(refreshed) == 1
    assert (
        config["_connection"]["credential_id"]
        == selected.credential_id
        == AuthStore().selection("openai").credential_id
    )
    assert [request.headers["Authorization"] for request in server.requests] == [
        "Bearer fresh-token"
    ] * 2


def test_endpoint_metadata_cannot_record_embedded_secrets(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.example/v1?key=hidden-secret")
    with pytest.raises(ValueError, match="without credentials") as failure:
        connection_snapshot()
    assert "hidden-secret" not in str(failure.value)
