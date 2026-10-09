"""A run-level `llm_config.api_key` authenticates agents and stays out of every record."""

import dataclasses
import enum
import json
from collections.abc import Mapping

import httpx
import pytest

import yian.llm_clients as clients
from yian.compose import load_flow
from yian.compose.helpers.errors import LoadError
from yian.events import NodeStarted
from yian.llm_clients.openai_client import OpenAIClient
from yian.memory.profile import ProfileMemory
from yian.memory.reflective import ReflectiveMemory
from yian.memory.storage import FileMarkdownStore
from yian.run import resume_flow, run_flow
from tests.fixtures.openai_responses import ResponsesServer

SENTINEL = "sk-sentinel-5b2e8d41"

_FLOW = """id: keyed
name: keyed
nodes:
  agent: {kind: agent, mode: plain, prompt: hi}
output: ${agent.output}
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
        "ANTHROPIC_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


_NO_FEEDBACK = json.dumps({"input_feedback": {}, "self_feedback": [], "variable_feedback": {}})


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


def _record_calls(requests, *replies):
    """Answer record-schema requests with a tool call and others with ``replies`` in order."""
    texts = _chat_completions([], *replies)

    def handler(request):
        requests.append(request)
        body = json.loads(request.content)
        if "tools" not in body:
            return texts.handle_request(request)
        name = body["tools"][0]["function"]["name"]
        return httpx.Response(200, json={
            "id": "c", "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": None,
                "tool_calls": [{"id": "t", "type": "function", "function": {
                    "name": name, "arguments": json.dumps({"text": "ok"}),
                }}],
            }}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })

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


def _assert_hidden(path, *values, fragments=(SENTINEL,)):
    texts = [] if path is None else [path.read_text()]
    for value in values:
        texts += [repr(value), json.dumps(_plain(value))]
    for text in texts:
        for fragment in fragments:
            assert fragment not in text


def _stop_before_model(event):
    if isinstance(event, NodeStarted) and event.node_id == "agent":
        raise RuntimeError("host stopped before model")


def _interrupted_run(path, llm_config):
    with pytest.raises(RuntimeError, match="host stopped"):
        run_flow(load_flow(_FLOW), {}, log=path, on_event=_stop_before_model, llm_config=llm_config)


_VLLM = {"provider": "vllm", "model": "served-model", "api_key": SENTINEL}


def test_openai_compatible_run_sends_the_key_and_records_none_of_it(tmp_path, monkeypatch):
    requests, events = [], []
    path = tmp_path / "run.jsonl"
    with httpx.Client(transport=_chat_completions(requests)) as transport:
        _route_openai_client(monkeypatch, transport)
        result = run_flow(
            load_flow(_FLOW),
            {},
            log=path,
            on_event=events.append,
            llm_config={"provider": "vllm", "model": "served-model", "api_key": SENTINEL},
        )
    assert result.status == "succeeded", result.failure
    assert requests[0].headers["Authorization"] == f"Bearer {SENTINEL}"
    effective = json.loads(path.read_text().splitlines()[0])["payload"][
        "effective_llm_by_static_path"
    ]
    # A nonsecret marker records that the agent authenticates with a host key.
    assert effective == {"agent": {"provider": "vllm", "model": "served-model", "_host_key": True}}
    _assert_hidden(path, events, result)


def test_openai_run_records_a_host_connection(tmp_path, monkeypatch):
    server, events = ResponsesServer(tool_first=False), []
    path = tmp_path / "run.jsonl"
    llm_config = {"provider": "openai", "model": "gpt-5.5", "api_key": SENTINEL}
    with httpx.Client(transport=httpx.MockTransport(server)) as transport:
        _route_openai_client(monkeypatch, transport)
        result = run_flow(
            load_flow(_FLOW), {}, log=path, on_event=events.append, llm_config=llm_config
        )
    assert result.status == "succeeded", result.failure
    assert server.requests[0].headers["Authorization"] == f"Bearer {SENTINEL}"
    effective = json.loads(path.read_text().splitlines()[0])["payload"][
        "effective_llm_by_static_path"
    ]
    assert effective["agent"]["_connection"]["source"] == "host"
    assert effective["agent"]["_host_key"] is True
    _assert_hidden(path, events, result)


def test_resume_sends_the_key_passed_again_and_records_none_of_it(tmp_path, monkeypatch):
    requests, events = [], []
    path = tmp_path / "run.jsonl"
    _interrupted_run(path, _VLLM)
    with httpx.Client(transport=_chat_completions(requests)) as transport:
        _route_openai_client(monkeypatch, transport)
        result = resume_flow(load_flow(_FLOW), log=path, on_event=events.append, api_key=SENTINEL)
    assert result.status == "succeeded", result.failure
    assert requests[0].headers["Authorization"] == f"Bearer {SENTINEL}"
    _assert_hidden(path, events, result)


@pytest.mark.parametrize(
    "llm_config, env",
    [
        (_VLLM, "VLLM_API_KEY"),
        ({"provider": "anthropic", "model": "claude-sonnet-4-5", "api_key": SENTINEL},
         "ANTHROPIC_API_KEY"),
        ({"provider": "openai", "model": "gpt-5.5", "api_key": SENTINEL}, "OPENAI_API_KEY"),
    ],
)
def test_resume_without_the_key_fails_instead_of_reading_the_environment(
    tmp_path, monkeypatch, llm_config, env
):
    requests = []
    path = tmp_path / "run.jsonl"
    _interrupted_run(path, llm_config)
    monkeypatch.setenv(env, "sk-environment-key")
    with httpx.Client(transport=_chat_completions(requests)) as transport:
        _route_openai_client(monkeypatch, transport)
        result = resume_flow(load_flow(_FLOW), log=path)
    assert result.status == "failed"
    assert "api_key" in result.failure.message
    assert requests == []
    assert SENTINEL not in path.read_text()


def test_resume_ignores_a_key_for_a_run_started_without_one(tmp_path, monkeypatch):
    requests, events = [], []
    path = tmp_path / "run.jsonl"
    monkeypatch.setenv("VLLM_API_KEY", "sk-environment-key")
    _interrupted_run(path, {"provider": "vllm", "model": "served-model"})
    with httpx.Client(transport=_chat_completions(requests)) as transport:
        _route_openai_client(monkeypatch, transport)
        result = resume_flow(load_flow(_FLOW), log=path, on_event=events.append, api_key=SENTINEL)
    assert result.status == "succeeded", result.failure
    # The run recorded no marked agent, so its agents keep their own credentials.
    assert requests[0].headers["Authorization"] == "Bearer sk-environment-key"
    _assert_hidden(path, events, result)


_UNMARKED_FLOWS = {
    "other-provider": """id: other
name: other
nodes:
  agent:
    kind: agent
    mode: plain
    prompt: hi
    llm_config: {provider: anthropic, model: claude-sonnet-4-5}
output: ${agent.output}
""",
    "no-agents": """id: plain
name: plain
nodes:
  step:
    kind: code
    output: str
    code: |
      return "ok"
output: ${step.output}
""",
}


@pytest.mark.parametrize("source", _UNMARKED_FLOWS.values(), ids=_UNMARKED_FLOWS.keys())
def test_a_key_that_marks_no_agent_does_not_block_replay(tmp_path, monkeypatch, source):
    from langchain_core.messages import AIMessage

    keys = []

    class Model:
        def invoke(self, messages):
            return AIMessage(content="ok" if len(keys) == 1 else _NO_FEEDBACK)

    def factory(config):
        keys.append(config.get("api_key"))
        return Model()

    monkeypatch.setattr(clients, "model_from_config", factory)
    events = []
    path = tmp_path / "run.jsonl"
    result = run_flow(
        load_flow(source), {}, log=path, on_event=events.append, learnable=True, llm_config=_VLLM
    )
    assert result.status == "succeeded", result.failure
    retained = result.backward("Be brief.")
    assert retained.status == "succeeded", retained.failure
    recovered = resume_flow(load_flow(source), log=path, on_event=events.append, api_key=SENTINEL)
    assert recovered.status == "succeeded", recovered.failure
    outcome = recovered.backward("Be brief.")
    assert outcome.status == "succeeded", outcome.failure
    assert keys == [None] * len(keys)
    assert "_host_key" not in path.read_text()
    _assert_hidden(path, events, result, retained, recovered, outcome)


def test_backward_sends_the_key_and_records_none_of_it(tmp_path, monkeypatch):
    requests, events = [], []
    path = tmp_path / "run.jsonl"
    with httpx.Client(transport=_chat_completions(requests, "ok", _NO_FEEDBACK)) as transport:
        _route_openai_client(monkeypatch, transport)
        result = run_flow(
            load_flow(_FLOW), {}, log=path, on_event=events.append, learnable=True, llm_config=_VLLM
        )
        assert result.status == "succeeded", result.failure
        # The result keeps the run's key in memory for its own backward calls.
        retained = result.backward("Be brief.")
        assert retained.status == "succeeded", retained.failure
        # After a restart, the key comes back through resume_flow or backward.
        for recovered, key in (
            (resume_flow(load_flow(_FLOW), log=path, on_event=events.append, api_key=SENTINEL), None),
            (resume_flow(load_flow(_FLOW), log=path, on_event=events.append), SENTINEL),
        ):
            outcome = recovered.backward("Be brief.", api_key=key)
            assert outcome.status == "succeeded", outcome.failure
            _assert_hidden(path, recovered, outcome)
    assert len(requests) == 4
    assert {request.headers["Authorization"] for request in requests} == {f"Bearer {SENTINEL}"}
    assert events
    _assert_hidden(path, events, result, retained)


def test_backward_without_the_key_fails_clearly(tmp_path, monkeypatch):
    requests, events = [], []
    path = tmp_path / "run.jsonl"
    with httpx.Client(transport=_chat_completions(requests, "ok", _NO_FEEDBACK)) as transport:
        _route_openai_client(monkeypatch, transport)
        run_flow(load_flow(_FLOW), {}, log=path, on_event=events.append, learnable=True, llm_config=_VLLM)
        monkeypatch.setenv("VLLM_API_KEY", "sk-environment-key")
        recovered = resume_flow(load_flow(_FLOW), log=path, on_event=events.append)
        outcome = recovered.backward("Be brief.")
    assert outcome.status == "failed"
    assert "api_key" in repr(outcome.failure)
    assert len(requests) == 1
    _assert_hidden(path, events, recovered, outcome)


def test_invalid_config_with_a_key_fails_without_echoing_it(tmp_path):
    events = []
    path = tmp_path / "run.jsonl"
    result = run_flow(
        load_flow(_FLOW),
        {},
        log=path,
        on_event=events.append,
        llm_config={
            "provider": "vllm",
            "model": "served-model",
            "api_key": SENTINEL,
            "temperature": "hot",
        },
    )
    assert result.status == "failed"
    assert "temperature" in result.failure.message
    _assert_hidden(path, events, result)


def test_key_goes_only_to_agents_of_its_provider(tmp_path, monkeypatch):
    flow = load_flow("""id: two
name: two
defs:
  body:
    nodes:
      nested: {kind: agent, mode: plain, prompt: hi, llm_config: {model: nested-model}}
    output: ${nested.output}
nodes:
  inherits: {kind: agent, mode: plain, prompt: hi}
  child: {kind: call, call: body, depends_on: [inherits]}
  other:
    kind: agent
    mode: plain
    prompt: hi
    depends_on: [inherits]
    llm_config: {provider: anthropic, model: claude-sonnet-4-5}
  optout:
    kind: agent
    mode: plain
    prompt: hi
    depends_on: [other]
    llm_config: {inherit: false, provider: vllm, model: other-model}
output: ${optout.output}
""")
    from langchain_core.messages import AIMessage

    keys = {}

    class Model:
        def invoke(self, messages):
            return AIMessage(content="ok")

    def factory(config):
        key = config.get("api_key")
        keys[config["model"]] = None if key is None else key.get_secret_value()
        return Model()

    monkeypatch.setattr(clients, "model_from_config", factory)
    events = []
    path = tmp_path / "run.jsonl"
    result = run_flow(
        flow, {}, log=path, on_event=events.append,
        llm_config={"provider": "vllm", "model": "served-model", "api_key": SENTINEL},
    )
    assert result.status == "succeeded", result.failure
    _assert_hidden(path, events, result)
    assert keys == {
        "served-model": SENTINEL,
        "nested-model": SENTINEL,
        "claude-sonnet-4-5": None,
        "other-model": SENTINEL,
    }


def test_key_without_a_provider_is_rejected():
    with pytest.raises(ValueError, match="provider") as caught:
        run_flow(load_flow(_FLOW), {}, llm_config={"api_key": SENTINEL})
    assert SENTINEL not in str(caught.value)


def test_an_empty_key_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_API_KEY", "sk-environment-key")
    with pytest.raises(ValueError, match="api_key"):
        run_flow(load_flow(_FLOW), {}, llm_config={**_VLLM, "api_key": ""})
    path = tmp_path / "run.jsonl"
    _interrupted_run(path, _VLLM)
    with pytest.raises(ValueError, match="api_key"):
        resume_flow(load_flow(_FLOW), log=path, api_key="")


def test_authored_key_is_rejected_without_echoing_it():
    with pytest.raises(LoadError) as caught:
        load_flow(_FLOW.replace("prompt: hi}", f"prompt: hi, llm_config: {{api_key: {SENTINEL}}}}}"))
    assert "api_key" in str(caught.value)
    assert SENTINEL not in str(caught.value)
    flow_level = _FLOW.replace("nodes:", f"llm_config: {{api_key: {SENTINEL}}}\nnodes:")
    def_level = f"""id: d
name: d
defs:
  body:
    llm_config: {{api_key: {SENTINEL}}}
    nodes:
      nested: {{kind: agent, mode: plain, prompt: hi}}
    output: ${{nested.output}}
nodes:
  child: {{kind: call, call: body}}
output: ${{child.output}}
"""
    for source in (flow_level, def_level):
        with pytest.raises(LoadError) as caught:
            load_flow(source)
        assert "api_key" in str(caught.value)
        assert SENTINEL not in str(caught.value)


def test_host_key_marker_is_not_an_authored_option():
    with pytest.raises(ValueError, match="_host_key"):
        run_flow(load_flow(_FLOW), {}, llm_config={"provider": "vllm", "model": "m", "_host_key": True})
    flow = load_flow(_FLOW.replace("nodes:", "llm_config: {_host_key: true}\nnodes:"))
    with pytest.raises(ValueError, match="_host_key"):
        run_flow(flow, {}, llm_config={"provider": "vllm", "model": "m"})


def test_reflective_memory_refuses_a_key_it_would_record():
    with pytest.raises(ValueError, match="api_key") as caught:
        ReflectiveMemory({"provider": "vllm", "model": "m", "api_key": SENTINEL})
    assert SENTINEL not in str(caught.value)


def test_profile_memory_needs_a_provider_with_its_key(tmp_path):
    store = FileMarkdownStore(tmp_path / "ada.md")
    for llm_config in ({"api_key": SENTINEL}, {"model": "profiler", "api_key": SENTINEL}):
        with pytest.raises(ValueError, match="provider") as caught:
            ProfileMemory(store, llm_config=llm_config)
        assert SENTINEL not in str(caught.value)
    with pytest.raises(ValueError, match="api_key"):
        ProfileMemory(store, llm_config={"provider": "vllm", "api_key": ""})
    memory = ProfileMemory(store, llm_config={"provider": "vllm", "api_key": SENTINEL})
    assert SENTINEL not in repr(vars(memory))


def test_profile_memory_learns_with_its_key_and_records_none_of_it(tmp_path, monkeypatch):
    flow = load_flow(_FLOW.replace("nodes:", "long_term_memory: profile\nnodes:"))
    (tmp_path / "profile").mkdir()
    store = FileMarkdownStore(tmp_path / "profile" / "ada.md")
    store.write("# Ada\n", base_version=store.read()[1])
    memory = ProfileMemory(
        store, llm_config={"provider": "vllm", "model": "profiler", "api_key": SENTINEL}
    )
    requests, events = [], []
    path = tmp_path / "run.jsonl"
    replies = ("ok", _NO_FEEDBACK, "<profile>\n# Ada\n- Prefers brief replies.\n</profile>")
    monkeypatch.setenv("VLLM_API_KEY", "sk-environment-key")
    with httpx.Client(transport=_chat_completions(requests, *replies)) as transport:
        _route_openai_client(monkeypatch, transport)
        result = run_flow(
            flow, {}, log=path, on_event=events.append, learnable=True,
            memory={"profile": memory},
            llm_config={"provider": "vllm", "model": "served-model"},
        )
        outcome = result.backward("Be brief.")
    assert outcome.status == "succeeded", outcome.failure
    learner = [r for r in requests if json.loads(r.content)["model"] == "profiler"]
    assert [r.headers["Authorization"] for r in learner] == [f"Bearer {SENTINEL}"]
    assert memory.read()[0] == "# Ada\n- Prefers brief replies.\n"
    _assert_hidden(path, events, result, outcome)
    for written in (tmp_path / "profile").rglob("*"):
        if written.is_file():
            assert SENTINEL not in written.read_text(errors="replace")


_REJECTED_KEY = "sk-sentinel-401-zqxw"


def _rejecting(requests, *, accept=0, replies=("ok",)):
    """Answer the first ``accept`` requests, then echo the key masked, as OpenAI's 401 does."""
    ok = _chat_completions([], *replies)

    def handler(request):
        requests.append(request)
        if len(requests) <= accept:
            return ok.handle_request(request)
        masked = f"{_REJECTED_KEY[:8]}****{_REJECTED_KEY[-4:]}"
        return httpx.Response(401, json={"error": {
            "message": f"Incorrect API key provided: {masked}.",
            "type": "invalid_request_error", "code": "invalid_api_key",
        }})

    return httpx.MockTransport(handler)


def _assert_rejected_key_hidden(path, *values):
    _assert_hidden(
        path, *values, fragments=(_REJECTED_KEY, _REJECTED_KEY[:8], "*" + _REJECTED_KEY[-4:])
    )


def test_a_provider_401_records_no_part_of_the_key(tmp_path, monkeypatch):
    requests, events = [], []
    path = tmp_path / "run.jsonl"
    with httpx.Client(transport=_rejecting(requests)) as transport:
        _route_openai_client(monkeypatch, transport)
        result = run_flow(
            load_flow(_FLOW), {}, log=path, on_event=events.append,
            llm_config={**_VLLM, "api_key": _REJECTED_KEY},
        )
    assert result.status == "failed"
    assert requests[0].headers["Authorization"] == f"Bearer {_REJECTED_KEY}"
    assert "Incorrect API key provided: [REDACTED]" in path.read_text()
    _assert_rejected_key_hidden(path, events, result)


def test_a_provider_401_in_backward_records_no_part_of_the_key(tmp_path, monkeypatch):
    requests = []
    path = tmp_path / "run.jsonl"
    with httpx.Client(transport=_rejecting(requests, accept=1)) as transport:
        _route_openai_client(monkeypatch, transport)
        result = run_flow(
            load_flow(_FLOW), {}, log=path, learnable=True,
            llm_config={**_VLLM, "api_key": _REJECTED_KEY},
        )
        assert result.status == "succeeded", result.failure
        outcome = result.backward("Be brief.")
    assert outcome.status == "failed"
    assert len(requests) == 2
    assert "[REDACTED]" in path.read_text()
    _assert_rejected_key_hidden(path, result, outcome)


def test_a_profile_learner_401_records_no_part_of_the_key(tmp_path, monkeypatch):
    flow = load_flow(_FLOW.replace("nodes:", "long_term_memory: profile\nnodes:"))
    (tmp_path / "profile").mkdir()
    store = FileMarkdownStore(tmp_path / "profile" / "ada.md")
    store.write("# Ada\n", base_version=store.read()[1])
    memory = ProfileMemory(
        store, llm_config={"provider": "vllm", "model": "profiler", "api_key": _REJECTED_KEY}
    )
    requests, events = [], []
    path = tmp_path / "run.jsonl"
    monkeypatch.setenv("VLLM_API_KEY", "sk-environment-key")
    with httpx.Client(transport=_rejecting(requests, accept=2, replies=("ok", _NO_FEEDBACK))) as transport:
        _route_openai_client(monkeypatch, transport)
        result = run_flow(
            flow, {}, log=path, on_event=events.append, learnable=True,
            memory={"profile": memory},
            llm_config={"provider": "vllm", "model": "served-model"},
        )
        outcome = result.backward("Be brief.")
    assert outcome.status == "failed"
    assert json.loads(requests[-1].content)["model"] == "profiler"
    _assert_rejected_key_hidden(path, events, result, outcome)


_MAP_FLOW = """id: fan
name: fan
input:
  values: list[str]
defs:
  body:
    input:
      value: str
    nodes:
      inner:
        kind: agent
        mode: plain
        input:
          value: ${input.value}
        prompt: ${value}
    output: ${inner.output}
nodes:
  fan:
    kind: map
    call: body
    over: ${input.values}
    input:
      value: ${item}
output: ${fan.output}
"""

_LOOP_FLOW = """id: turns
name: turns
defs:
  body:
    input:
      text: str
    nodes:
      inner:
        kind: agent
        mode: plain
        input:
          text: ${input.text}
        output:
          text: str
        prompt: ${text}
    output: ${inner.output}
nodes:
  turns:
    kind: loop
    call: body
    input:
      text: start
    times: 2
output: ${turns.output.text}
"""

_GROWN = {"map": (_MAP_FLOW, {"values": ["a", "b"]}), "loop": (_LOOP_FLOW, {})}


def _stop_in_grown_body(event):
    if isinstance(event, NodeStarted) and "inner" in event.node_id:
        raise RuntimeError("host stopped inside a grown body")


@pytest.mark.parametrize("kind", _GROWN)
def test_grown_agents_resume_with_the_key_and_record_none_of_it(tmp_path, monkeypatch, kind):
    source, inputs = _GROWN[kind]
    requests, events = [], []
    path = tmp_path / "run.jsonl"
    with pytest.raises(RuntimeError, match="grown body"):
        run_flow(load_flow(source), inputs, log=path, on_event=_stop_in_grown_body, llm_config=_VLLM)
    monkeypatch.setenv("VLLM_API_KEY", "sk-environment-key")
    with httpx.Client(transport=_record_calls(requests)) as transport:
        _route_openai_client(monkeypatch, transport)
        result = resume_flow(load_flow(source), log=path, on_event=events.append, api_key=SENTINEL)
    assert result.status == "succeeded", result.failure
    assert len(requests) == 2
    assert {request.headers["Authorization"] for request in requests} == {f"Bearer {SENTINEL}"}
    _assert_hidden(path, events, result)


@pytest.mark.parametrize("kind", _GROWN)
def test_grown_agents_backward_with_the_key_and_record_none_of_it(tmp_path, monkeypatch, kind):
    source, inputs = _GROWN[kind]
    requests, events = [], []
    path = tmp_path / "run.jsonl"
    with httpx.Client(transport=_record_calls(requests, _NO_FEEDBACK)) as transport:
        _route_openai_client(monkeypatch, transport)
        result = run_flow(
            load_flow(source), inputs, log=path, on_event=events.append, learnable=True,
            llm_config=_VLLM,
        )
        assert result.status == "succeeded", result.failure
        recovered = resume_flow(load_flow(source), log=path, on_event=events.append, api_key=SENTINEL)
        outcome = recovered.backward("Review.")
        assert outcome.status == "succeeded", outcome.failure
        # Without the key, the grown agents fail instead of reading the environment.
        monkeypatch.setenv("VLLM_API_KEY", "sk-environment-key")
        sent = len(requests)
        keyless = resume_flow(load_flow(source), log=path).backward("Review.")
        assert keyless.status == "failed"
        assert len(requests) == sent
    assert len(requests) > 2
    assert {request.headers["Authorization"] for request in requests} == {f"Bearer {SENTINEL}"}
    _assert_hidden(path, events, result, recovered, outcome, keyless)
