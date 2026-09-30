"""A run-level `llm_config.api_key` authenticates agents and stays out of every record."""

import json

import httpx
import pytest

import agent_composer.llm_clients as clients
from agent_composer.compose import load_flow
from agent_composer.compose.helpers.errors import LoadError
from agent_composer.events import NodeStarted
from agent_composer.llm_clients.openai_client import OpenAIClient
from agent_composer.memory.profile import ProfileMemory
from agent_composer.memory.reflective import ReflectiveMemory
from agent_composer.memory.storage import FileMarkdownStore
from agent_composer.run import resume_flow, run_flow
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
    monkeypatch.setenv("AGENT_COMPOSER_AUTH_FILE", str(tmp_path / "auth.json"))
    for name in (
        "AGENT_COMPOSER_DEFAULT_PROVIDER",
        "AGENT_COMPOSER_DEFAULT_MODEL",
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


def _route_openai_client(monkeypatch, transport):
    original = OpenAIClient.get_llm

    def construct(wrapper):
        wrapper.kwargs.update(http_client=transport, max_retries=0)
        return original(wrapper)

    monkeypatch.setattr(OpenAIClient, "get_llm", construct)


def _assert_hidden(path, *values):
    assert SENTINEL not in path.read_text()
    for value in values:
        assert SENTINEL not in repr(value)


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
    requests = []
    path = tmp_path / "run.jsonl"
    with httpx.Client(transport=_chat_completions(requests, "ok", _NO_FEEDBACK)) as transport:
        _route_openai_client(monkeypatch, transport)
        result = run_flow(load_flow(_FLOW), {}, log=path, learnable=True, llm_config=_VLLM)
        assert result.status == "succeeded", result.failure
        # The result keeps the run's key in memory for its own backward calls.
        retained = result.backward("Be brief.")
        assert retained.status == "succeeded", retained.failure
        # After a restart, the key comes back through resume_flow or backward.
        for recovered, key in (
            (resume_flow(load_flow(_FLOW), log=path, api_key=SENTINEL), None),
            (resume_flow(load_flow(_FLOW), log=path), SENTINEL),
        ):
            outcome = recovered.backward("Be brief.", api_key=key)
            assert outcome.status == "succeeded", outcome.failure
            _assert_hidden(path, recovered, outcome)
    assert len(requests) == 4
    assert {request.headers["Authorization"] for request in requests} == {f"Bearer {SENTINEL}"}
    _assert_hidden(path, result, retained)


def test_backward_without_the_key_fails_clearly(tmp_path, monkeypatch):
    requests = []
    path = tmp_path / "run.jsonl"
    with httpx.Client(transport=_chat_completions(requests, "ok", _NO_FEEDBACK)) as transport:
        _route_openai_client(monkeypatch, transport)
        run_flow(load_flow(_FLOW), {}, log=path, learnable=True, llm_config=_VLLM)
        monkeypatch.setenv("VLLM_API_KEY", "sk-environment-key")
        outcome = resume_flow(load_flow(_FLOW), log=path).backward("Be brief.")
    assert outcome.status == "failed"
    assert "api_key" in repr(outcome.failure)
    assert len(requests) == 1
    assert SENTINEL not in path.read_text()


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


def test_key_goes_only_to_agents_of_its_provider(monkeypatch):
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
    result = run_flow(
        flow, {}, llm_config={"provider": "vllm", "model": "served-model", "api_key": SENTINEL}
    )
    assert result.status == "succeeded", result.failure
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


def test_authored_key_is_rejected_without_echoing_it():
    with pytest.raises(LoadError) as caught:
        load_flow(_FLOW.replace("prompt: hi}", f"prompt: hi, llm_config: {{api_key: {SENTINEL}}}}}"))
    assert "api_key" in str(caught.value)
    assert SENTINEL not in str(caught.value)
    flow = load_flow(_FLOW.replace("nodes:", f"llm_config: {{api_key: {SENTINEL}}}\nnodes:"))
    with pytest.raises(ValueError, match="api_key") as caught:
        run_flow(flow, {}, llm_config={"provider": "vllm", "model": "served-model"})
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
