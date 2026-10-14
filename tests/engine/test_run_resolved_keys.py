"""A key from the environment or a saved login stays out of every record when a provider echoes it."""

import dataclasses
import enum
import json
import time
from collections.abc import Mapping

import httpx
import pytest

import yian.llm_clients as clients
from yian.compose import load_flow
from yian.llm_clients.auth import ApiKeyCredentials, AuthStore, OAuthCredentials
from yian.llm_clients.openai_client import OpenAIClient
from yian.run import run_flow

_KEY = "sk-resolved-401-qvjm"

_FLOW = """id: keyed
name: keyed
nodes:
  agent: {kind: agent, prompt: hi}
output: ${agent.output}
llm_config: {provider: PROVIDER, model: MODEL}
"""


@pytest.fixture(autouse=True)
def clean_env(tmp_path, monkeypatch):
    monkeypatch.setenv("YIAN_AUTH_FILE", str(tmp_path / "auth.json"))
    for name in (
        "YIAN_DEFAULT_PROVIDER",
        "YIAN_DEFAULT_MODEL",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "VLLM_API_KEY",
        "VLLM_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)


def _flow(provider="vllm", model="served-model"):
    return load_flow(_FLOW.replace("PROVIDER", provider).replace("MODEL", model))


def _chat_completions(requests, *replies):
    """Answer chat-completions requests with ``replies`` in order, repeating the last."""
    replies = replies or ("ok",)

    def handler(request):
        requests.append(request)
        content = replies[min(len(requests), len(replies)) - 1]
        return httpx.Response(200, json={
            "id": "c", "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })

    return httpx.MockTransport(handler)


def _rejecting(requests, *, accept=0):
    """Answer the first ``accept`` requests, then echo the key masked, as OpenAI's 401 does."""
    ok = _chat_completions([])

    def handler(request):
        requests.append(request)
        if len(requests) <= accept:
            return ok.handle_request(request)
        masked = f"{_KEY[:8]}****{_KEY[-4:]}"
        return httpx.Response(401, json={"error": {
            "message": f"Incorrect API key provided: {masked}.",
            "type": "invalid_request_error", "code": "invalid_api_key",
        }})

    return httpx.MockTransport(handler)


def _route_openai_client(monkeypatch, transport):
    original = OpenAIClient.get_llm

    def construct(wrapper):
        wrapper.kwargs.update(http_client=transport, max_retries=0)
        return original(wrapper)

    monkeypatch.setattr(OpenAIClient, "get_llm", construct)


def _plain(value, seen=None):
    """Serialize events and results field by field, so a JSON scan sees every value."""
    seen = set() if seen is None else seen
    if id(value) in seen:
        return None
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        seen = seen | {id(value)}
        return {
            field.name: _plain(getattr(value, field.name), seen)
            for field in dataclasses.fields(value)
            if not field.name.startswith("_")
        }
    if isinstance(value, Mapping):
        return {str(key): _plain(item, seen | {id(value)}) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_plain(item, seen | {id(value)}) for item in value]
    if isinstance(value, enum.Enum):
        return value.value
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _assert_hidden(path, *values, fragments=(_KEY, _KEY[:8], "*" + _KEY[-4:])):
    texts = [path.read_text()]
    for value in values:
        texts += [repr(value), json.dumps(_plain(value))]
    for text in texts:
        for fragment in fragments:
            assert fragment not in text


def _run_rejected(tmp_path, monkeypatch):
    requests, events = [], []
    path = tmp_path / "run.jsonl"
    with httpx.Client(transport=_rejecting(requests)) as transport:
        _route_openai_client(monkeypatch, transport)
        result = run_flow(_flow(), {}, log=path, on_event=events.append)
    assert result.status == "failed"
    assert requests[0].headers["Authorization"] == f"Bearer {_KEY}"
    assert "[REDACTED]" in path.read_text()
    _assert_hidden(path, events, result)


def test_a_401_with_the_key_variable_records_no_part_of_it(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_API_KEY", _KEY)
    _run_rejected(tmp_path, monkeypatch)


def test_a_401_with_a_saved_login_records_no_part_of_it(tmp_path, monkeypatch):
    AuthStore().save("vllm", ApiKeyCredentials(_KEY), "served-model")
    _run_rejected(tmp_path, monkeypatch)


def test_a_backward_401_with_a_saved_login_records_no_part_of_it(tmp_path, monkeypatch):
    AuthStore().save("vllm", ApiKeyCredentials(_KEY), "served-model")
    requests = []
    path = tmp_path / "run.jsonl"
    with httpx.Client(transport=_rejecting(requests, accept=1)) as transport:
        _route_openai_client(monkeypatch, transport)
        result = run_flow(_flow(), {}, log=path, learnable=True)
        assert result.status == "succeeded", result.failure
        outcome = result.backward("Be brief.")
    assert outcome.status == "failed"
    assert len(requests) == 2
    assert requests[1].headers["Authorization"] == f"Bearer {_KEY}"
    assert "[REDACTED]" in path.read_text()
    _assert_hidden(path, result, outcome)


def test_a_subscription_token_echoed_by_the_provider_is_not_recorded(tmp_path, monkeypatch):
    access, refresh = "atok47-echo-7h3k", "rtok91-echo-p9d2"
    AuthStore().save(
        "openai", OAuthCredentials(access, refresh, time.time() + 3600, "account"), "gpt-5.5"
    )

    def rejecting(config):
        raise RuntimeError(f"401 for Bearer {access}; refresh {refresh[:6]}****{refresh[-4:]}")

    monkeypatch.setattr(clients, "model_from_config", rejecting)
    events = []
    path = tmp_path / "run.jsonl"
    result = run_flow(_flow("openai", "gpt-5.5"), {}, log=path, on_event=events.append)
    assert result.status == "failed"
    assert "[REDACTED]" in path.read_text()
    _assert_hidden(
        path, events, result,
        fragments=(access, access[:8], refresh, refresh[:6], "*" + refresh[-4:]),
    )
