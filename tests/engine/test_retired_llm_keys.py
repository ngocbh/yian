"""Retired `llm_config` keys fail everywhere with their provider-block form."""

import json

import httpx
import pytest

from yian.compile.llm_cascade import resolve_llm_cascade
from yian.compose import load_flow
from yian.compose.helpers.errors import LoadError
from yian.events import NodeStarted
from yian.llm_clients.config import LLMConfig, retired_key_error
from yian.llm_clients.connection import legacy_connection_config
from yian.llm_clients.openai_client import OpenAIClient
from yian.memory.contracts import MemoryOperationError
from yian.memory.reflective import ReflectiveMemory
from yian.run import resume_flow, run_flow

SENTINEL = "sentinel-3f9a1c"
SENTINEL_NUMBER = 0.734519

# Retired key -> (sentinel value, text the message must show).
RETIRED = {
    "anthropic_effort": (SENTINEL, "`anthropic: {output_config: {effort: <level>}}`"),
    "openai_reasoning_effort": (SENTINEL, "`openai: {reasoning: {effort: <level>}}`"),
    "google_thinking_level": (
        SENTINEL,
        "`google: {generationConfig: {thinkingConfig: {thinkingLevel: <LEVEL>}}}`",
    ),
    "temperature": (SENTINEL_NUMBER, "`<provider>: {temperature: <value>}`"),
}


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
        "ANTHROPIC_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


def _assert_migration(message: str, key: str, provider: str | None = None) -> None:
    value, form = RETIRED[key]
    if key == "temperature" and provider is not None:
        form = f"`{provider}: {{temperature: <value>}}`"
    assert f"`{key}`" in message
    assert form in message
    assert str(value) not in message


@pytest.mark.parametrize("key", RETIRED)
def test_llm_config_rejects_each_retired_key_with_its_new_form(key):
    with pytest.raises(ValueError) as caught:
        LLMConfig(**{key: RETIRED[key][0]})
    _assert_migration(str(caught.value), key)


@pytest.mark.parametrize(
    ("provider", "form"),
    [
        ("anthropic", "`anthropic: {temperature: <value>}`"),
        ("qwen-cn", "`qwen-cn: {temperature: <value>}`"),
        ("google", "`google: {generationConfig: {temperature: <value>}}`"),
        ("ollama", "`ollama: {options: {temperature: <value>}}`"),
    ],
)
def test_temperature_message_names_the_layer_provider(provider, form):
    message = retired_key_error({"provider": provider, "temperature": SENTINEL_NUMBER})
    assert message == f"`temperature` moved into the provider block: write {form}"


def test_temperature_message_never_echoes_an_unknown_provider():
    message = retired_key_error({"provider": SENTINEL, "temperature": SENTINEL_NUMBER})
    assert SENTINEL not in message
    assert "`<provider>: {temperature: <value>}`" in message


def _yaml_value(key):
    value = RETIRED[key][0]
    return json.dumps(value)


def _node_source(key):
    return f"""id: retired
name: retired
nodes:
  agent:
    kind: agent
    prompt: hi
    llm_config: {{{key}: {_yaml_value(key)}}}
output: ${{agent.output}}
"""


def _flow_source(key):
    return f"""id: retired
name: retired
llm_config: {{{key}: {_yaml_value(key)}}}
nodes:
  agent: {{kind: agent, prompt: hi}}
output: ${{agent.output}}
"""


def _def_source(key):
    return f"""id: retired
name: retired
defs:
  body:
    llm_config: {{{key}: {_yaml_value(key)}}}
    nodes:
      agent: {{kind: agent, prompt: hi}}
    output: ${{agent.output}}
nodes:
  k: {{kind: call, call: body}}
output: ${{k.output}}
"""


@pytest.mark.parametrize("key", RETIRED)
@pytest.mark.parametrize(
    ("source", "where"),
    [
        (_node_source, "node 'agent': llm_config:"),
        (_flow_source, "flow: llm_config:"),
        (_def_source, "defs entry 'body': llm_config:"),
    ],
    ids=["node", "flow", "def"],
)
def test_a_retired_key_in_yaml_is_a_load_error(key, source, where):
    with pytest.raises(LoadError) as caught:
        load_flow(source(key))
    assert where in str(caught.value)
    _assert_migration(str(caught.value), key)


_FLOW = """id: plain
name: plain
nodes:
  agent: {kind: agent, prompt: hi}
output: ${agent.output}
"""


@pytest.mark.parametrize("key", RETIRED)
def test_a_retired_key_in_a_host_layer_fails_the_cascade(key):
    with pytest.raises(ValueError) as caught:
        resolve_llm_cascade(
            load_flow(_FLOW).compiled,
            {key: RETIRED[key][0]},
            defaults={"provider": "vllm", "model": "served-model"},
        )
    _assert_migration(str(caught.value), key)


@pytest.mark.parametrize("key", RETIRED)
def test_a_retired_key_in_the_run_layer_fails_before_the_log(tmp_path, key):
    log = tmp_path / "run.jsonl"
    with pytest.raises(ValueError) as caught:
        run_flow(
            load_flow(_FLOW),
            {},
            log=log,
            llm_config={"provider": "vllm", "model": "served-model", key: RETIRED[key][0]},
        )
    _assert_migration(str(caught.value), key, provider="vllm")
    assert not log.exists()


def _old_snapshot():
    """Return a reflective snapshot shaped like one recorded before the keys retired."""
    with ReflectiveMemory({"provider": "openai", "model": "old-model"}).open_run() as session:
        old = session.snapshot
    old["llm_config"].pop("_connection")
    old["llm_config"].update({key: None for key in RETIRED})
    return old


def test_a_snapshot_with_null_retired_keys_replays_unchanged():
    old = _old_snapshot()
    with ReflectiveMemory().open_run(snapshot=old) as session:
        assert session.snapshot == old
    assert not set(RETIRED) & set(legacy_connection_config(old["llm_config"]))


@pytest.mark.parametrize("key", RETIRED)
def test_a_snapshot_with_a_set_retired_key_fails(key):
    old = _old_snapshot()
    old["llm_config"][key] = RETIRED[key][0]
    with pytest.raises(MemoryOperationError) as caught:
        ReflectiveMemory().open_run(snapshot=old)
    _assert_migration(str(caught.value.__cause__), key, provider="openai")


def _stop_before_model(event):
    if isinstance(event, NodeStarted) and event.node_id == "agent":
        raise RuntimeError("host stopped before model")


def test_resuming_a_record_with_a_set_retired_key_fails_before_appending(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_API_KEY", "sk-environment-key")
    log = tmp_path / "run.jsonl"
    with pytest.raises(RuntimeError, match="host stopped"):
        run_flow(
            load_flow(_FLOW),
            {},
            log=log,
            on_event=_stop_before_model,
            llm_config={"provider": "vllm", "model": "served-model"},
        )
    lines = log.read_text().splitlines()
    started = json.loads(lines[0])
    assert started["kind"] == "run_started"
    started["payload"]["effective_llm_by_static_path"]["agent"]["temperature"] = 0.25
    lines[0] = json.dumps(started, sort_keys=True, separators=(",", ":"))
    log.write_text("\n".join(lines) + "\n")
    before = log.read_bytes()

    with pytest.raises(ValueError) as caught:
        resume_flow(load_flow(_FLOW), log=log)

    message = str(caught.value)
    assert message.startswith("Agent 'agent': ")
    assert "`temperature` moved into the provider block" in message
    assert "`vllm: {temperature: <value>}`" in message
    assert "0.25" not in message
    assert log.read_bytes() == before


def test_a_finished_record_with_a_set_retired_key_still_replays(tmp_path, monkeypatch):
    def reply(request):
        return httpx.Response(200, json={
            "id": "c", "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })

    original = OpenAIClient.get_llm

    def construct(wrapper):
        wrapper.kwargs.update(http_client=transport, max_retries=0)
        return original(wrapper)

    monkeypatch.setattr(OpenAIClient, "get_llm", construct)
    monkeypatch.setenv("VLLM_API_KEY", "sk-environment-key")
    log = tmp_path / "run.jsonl"
    with httpx.Client(transport=httpx.MockTransport(reply)) as transport:
        result = run_flow(
            load_flow(_FLOW), {}, log=log, llm_config={"provider": "vllm", "model": "served-model"}
        )
    assert result.status == "succeeded", result.failure
    lines = log.read_text().splitlines()
    started = json.loads(lines[0])
    started["payload"]["effective_llm_by_static_path"]["agent"]["temperature"] = 0.25
    lines[0] = json.dumps(started, sort_keys=True, separators=(",", ":"))
    log.write_text("\n".join(lines) + "\n")
    before = log.read_bytes()

    replayed = resume_flow(load_flow(_FLOW), log=log)

    assert replayed.status == "succeeded"
    assert log.read_bytes() == before
