"""The ``anthropic`` block reaches the Messages request body as written; a host's HTTP
clients carry the requests and never reach the body."""

import asyncio
import importlib
import inspect
import json
import warnings

import anthropic
import pytest
from anthropic.resources.messages import Messages
from langchain_anthropic import convert_to_anthropic_tool
from langchain_anthropic.chat_models import _supports_forced_tool_choice
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

from tests.engine._session_fakes import harness, request
from tests.engine.test_agent_session import _view
from yian.llm_clients.anthropic_client import (
    AnthropicClient,
    NormalizedChatAnthropic,
    _forces_tool,
)
from yian.llm_clients.base_client import missing_tool_call
from yian.llm_clients.capabilities import has_open_schema
from yian.llm_clients.factory import model_from_config
from yian.llm_clients.provider_blocks import AnthropicMessagesBlock
from yian.memory import MemoryContext
from yian.memory.session import SessionMemory
from yian.nodes.agent.structured import generate_structured, type_to_schema
from yian.typesys.types import read_typedefs, type_for
from yian.typesys.values import Type, ValueKind

_ENV_KEYS = (
    "ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "YIAN_DEFAULT_PROVIDER", "YIAN_DEFAULT_MODEL",
)
_SAMPLING = {"temperature", "top_k", "top_p"}


def _anthropic_httpx():
    """Return the httpx package this Anthropic SDK is built on (newer SDKs use `httpx2`)."""
    client = next(c for c in anthropic.DefaultHttpxClient.__mro__ if c.__name__ == "Client")
    return importlib.import_module(client.__module__.split(".")[0])


class _Server:
    """Record Messages request bodies and answer each with a reply.

    The reply is the next of `replies` (lists of content blocks) while any are queued,
    else `content` when set, else one text block holding `text`.
    """

    def __init__(self):
        self.text = "ok"
        self.content = None
        self.replies = []
        self.requests = []
        self.bodies = []

    def __call__(self, request):
        self.requests.append(request)
        self.bodies.append(json.loads(request.content))
        if self.replies:
            content = self.replies.pop(0)
        else:
            content = self.content or [{"type": "text", "text": self.text}]
        return _anthropic_httpx().Response(200, json={
            "id": "msg_1", "type": "message", "role": "assistant", "model": "m",
            "content": content,
            "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        })


@pytest.fixture
def server(tmp_path, monkeypatch):
    """Route every chat model `AnthropicClient` builds to a recording server."""
    monkeypatch.setenv("YIAN_AUTH_FILE", str(tmp_path / "auth.json"))
    for name in _ENV_KEYS:
        monkeypatch.delenv(name, raising=False)
    recorder = _Server()
    httpx = _anthropic_httpx()
    with httpx.Client(transport=httpx.MockTransport(recorder)) as http_client:
        original = AnthropicClient.get_llm

        def construct(wrapper):
            chat = original(wrapper)
            # ChatAnthropic builds its SDK client lazily in a cached property.
            chat.__dict__["_client"] = anthropic.Anthropic(
                api_key="k", http_client=http_client, max_retries=0, timeout=30
            )
            return chat

        monkeypatch.setattr(AnthropicClient, "get_llm", construct)
        yield recorder


def _model(block=None, model="claude-sonnet-4-5"):
    config = {"provider": "anthropic", "model": model, "api_key": "k"}
    if block is not None:
        config["anthropic"] = block
    return model_from_config(config)


class Answer(BaseModel):
    answer: str


def test_block_effort_sits_beside_the_structured_output_format(server):
    server.text = '{"answer": "ok"}'
    model = _model({"output_config": {"effort": "high"}})
    result = model.with_structured_output(Answer, method="json_schema").invoke("hi")
    assert result == Answer(answer="ok")
    output_config = server.bodies[0]["output_config"]
    assert output_config["effort"] == "high"
    assert output_config["format"]["type"] == "json_schema"


def test_block_does_not_leak_into_bound_call_options(server):
    server.text = '{"answer": "ok"}'
    structured = _model({"output_config": {"effort": "high"}}).with_structured_output(
        Answer, method="json_schema"
    )
    structured.invoke("hi")
    bound = structured.first.kwargs["output_config"]
    assert set(bound) == {"format"}


def test_sampling_and_length_reach_the_body_unchanged(server):
    _model({"temperature": 0.2, "top_k": 5, "max_tokens": 8000}).invoke([HumanMessage("hi")])
    body = server.bodies[0]
    assert (body["temperature"], body["top_k"], body["max_tokens"]) == (0.2, 5, 8000)


def test_thinking_reaches_the_body_as_written(server):
    block = {"thinking": {"type": "enabled", "budget_tokens": 2048}, "max_tokens": 4096}
    _model(block).invoke([HumanMessage("hi")])
    body = server.bodies[0]
    assert body["thinking"] == {"type": "enabled", "budget_tokens": 2048}
    assert body["max_tokens"] == 4096


@pytest.mark.parametrize("model", ["claude-opus-4-5-20251101", "claude-haiku-4-5"])
def test_block_effort_is_sent_for_any_model_id(server, model):
    _model({"output_config": {"effort": "low"}}, model=model).invoke([HumanMessage("hi")])
    assert server.bodies[0]["output_config"] == {"effort": "low"}


FULL_BLOCK = {
    "cache_control": {"type": "ephemeral", "ttl": "5m"},
    "inference_geo": "us",
    "max_tokens": 4096,
    "metadata": {"user_id": "user-1"},
    "output_config": {"effort": "medium"},
    "service_tier": "auto",
    "stop_sequences": ["END"],
    "temperature": 0.5,
    "thinking": {"type": "enabled", "budget_tokens": 1024},
    "top_k": 10,
    "top_p": 0.9,
}


def test_full_block_covers_every_field():
    assert set(FULL_BLOCK) == set(AnthropicMessagesBlock.model_fields)
    AnthropicMessagesBlock.model_validate(FULL_BLOCK)


# provider -> a model id and a block whose every field must reach the request body.
BODY_CASES = {"anthropic": {"model": "claude-sonnet-4-5", "block": FULL_BLOCK}}


@pytest.mark.parametrize("provider", sorted(BODY_CASES))
def test_every_block_field_is_sent(server, provider):
    case = BODY_CASES[provider]
    _model(case["block"], model=case["model"]).invoke([HumanMessage("hi")])
    body = server.bodies[0]
    for key, value in case["block"].items():
        assert body[key] == value, key


def test_every_block_field_is_a_messages_keyword_or_a_relocated_sampling_field():
    # langchain-anthropic moves the sampling fields into extra_body when the SDK lacks them.
    keywords = set(inspect.signature(Messages.create).parameters)
    assert set(AnthropicMessagesBlock.model_fields) <= keywords | _SAMPLING


def test_key_outside_the_sdk_signature_goes_through_extra_body(server):
    chat = AnthropicClient("claude-sonnet-4-5", api_key="k", block={"speed": "fast"}).get_llm()
    payload = chat._get_request_payload([HumanMessage("hi")])
    assert payload["extra_body"] == {"speed": "fast"}
    chat.invoke([HumanMessage("hi")])
    assert server.bodies[0]["speed"] == "fast"


def test_block_merges_into_a_caller_extra_body(server):
    chat = AnthropicClient("claude-sonnet-4-5", api_key="k", block={"speed": "fast"}).get_llm()
    payload = chat._get_request_payload([HumanMessage("hi")], extra_body={"x": 1})
    assert payload["extra_body"] == {"x": 1, "speed": "fast"}


def test_block_field_is_hidden_from_serialization():
    chat = NormalizedChatAnthropic(model="claude-sonnet-4-5", api_key="k", block={"top_k": 5})
    assert "block" not in chat.model_dump()
    assert "top_k" not in repr(chat)


def test_without_a_block_the_body_is_unchanged(server):
    _model().invoke([HumanMessage("hi")])
    assert server.requests[0].url.path == "/v1/messages"
    assert server.bodies[0] == {
        "max_tokens": 64000,
        "messages": [{"content": "hi", "role": "user"}],
        "model": "claude-sonnet-4-5",
    }


@pytest.mark.parametrize("thinking", [
    {"type": "adaptive"},
    {"type": "enabled", "budget_tokens": 2048},
])
@pytest.mark.parametrize("include_raw", [False, True])
def test_structured_output_with_block_thinking_uses_the_output_format(
    server, thinking, include_raw
):
    # The Messages API rejects thinking beside a forced tool call.
    server.text = '{"answer": "ok"}'
    model = _model({"thinking": thinking, "max_tokens": 4096})
    result = model.with_structured_output(Answer, include_raw=include_raw).invoke("hi")
    assert (result["parsed"] if include_raw else result) == Answer(answer="ok")
    body = server.bodies[0]
    assert body["thinking"] == thinking
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert "tool_choice" not in body
    assert "tools" not in body


def test_structured_output_with_block_thinking_keeps_an_explicit_method(server):
    model = _model({"thinking": {"type": "adaptive"}})
    model.with_structured_output(Answer, method="function_calling").invoke("hi")
    assert "output_config" not in server.bodies[0]
    assert server.bodies[0]["tools"][0]["name"] == "Answer"


@pytest.mark.parametrize("block", [None, {"top_k": 5}, {"thinking": {"type": "disabled"}}])
def test_structured_output_without_block_thinking_forces_the_tool(server, block):
    _model(block).with_structured_output(Answer).invoke("hi")
    body = server.bodies[0]
    assert body["tool_choice"] == {"type": "tool", "name": "Answer"}
    assert "output_config" not in body


_AGENT_TYPES = read_typedefs({"R": {"a": "int", "m": "object"}, "C": {"a": "int", "b": "list(str)"}})
_THINKING = {"type": "enabled", "budget_tokens": 1024}
# AGENT output type -> tool arguments of a valid reply.
_OPEN_CASES = {
    "object": {"value": {"k": 1}},
    "Any": {"value": [1, {"a": None}]},
    "R": {"a": 1, "m": {"q": "x"}},
}


def _agent_schema(text):
    return type_to_schema(type_for(text, _AGENT_TYPES))


def _tool_reply(name, args):
    return [
        {"type": "thinking", "thinking": "h", "signature": "s"},
        {"type": "tool_use", "id": "toolu_1", "name": name, "input": args},
    ]


_NO_TOOL_REPLIES = {
    "text": [
        {"type": "thinking", "thinking": "h", "signature": "s"},
        {"type": "text", "text": '{"value": {"k": 1}}'},
    ],
    "other tool": _tool_reply("Other", {"value": {"k": 1}}),
}


@pytest.mark.parametrize("thinking", [_THINKING, {"type": "adaptive"}])
@pytest.mark.parametrize("text", sorted(_OPEN_CASES))
@pytest.mark.parametrize("include_raw", [False, True])
def test_open_schema_with_block_thinking_goes_as_an_unforced_tool(
    server, thinking, text, include_raw
):
    # The output format and a forced tool both fail here: the Messages API rejects
    # thinking beside a forced tool, and its grammar rejects an open schema.
    schema = _agent_schema(text)
    name = convert_to_anthropic_tool(schema)["name"]
    server.content = _tool_reply(name, _OPEN_CASES[text])
    model = _model({"thinking": thinking, "max_tokens": 4096})
    result = model.with_structured_output(schema, include_raw=include_raw).invoke("hi")
    expected = schema.model_validate(_OPEN_CASES[text])
    if include_raw:
        assert result["parsed"] == expected and result["parsing_error"] is None
        assert result["raw"].tool_calls[0]["name"] == name
    else:
        assert result == expected
    body = server.bodies[0]
    assert body["thinking"] == thinking
    assert [tool["name"] for tool in body["tools"]] == [name]
    assert has_open_schema(body["tools"][0]["input_schema"])
    assert "tool_choice" not in body
    assert "output_config" not in body


_HOLDER = {
    "title": "Holder", "description": "Holds a value.", "type": "object",
    "properties": {"value": {"type": "object", "additionalProperties": True}},
    "required": ["value"],
}


def test_open_dict_schema_with_block_thinking_parses_the_tool_arguments(server):
    server.content = _tool_reply("Holder", {"value": {"k": 1}})
    model = _model({"thinking": _THINKING, "max_tokens": 4096})
    assert model.with_structured_output(_HOLDER).invoke("hi") == {"value": {"k": 1}}
    assert "tool_choice" not in server.bodies[0]


@pytest.mark.parametrize("include_raw", [False, True])
@pytest.mark.parametrize("kind", ["pydantic", "dict"])
def test_open_schema_reply_calling_another_tool_first_parses_the_schemas_call(
    server, kind, include_raw
):
    schema = _agent_schema("object") if kind == "pydantic" else _HOLDER
    name = convert_to_anthropic_tool(schema)["name"]
    other, call = _tool_reply("Other", {"x": 1}), _tool_reply(name, {"value": {"k": 1}})
    server.content = [*other, {**call[1], "id": "toolu_2"}]
    model = _model({"thinking": _THINKING, "max_tokens": 4096})
    result = model.with_structured_output(schema, include_raw=include_raw).invoke("hi")
    value = {"value": {"k": 1}}
    expected = schema.model_validate(value) if kind == "pydantic" else value
    if include_raw:
        assert result["parsed"] == expected and result["parsing_error"] is None
        assert [c["name"] for c in result["raw"].tool_calls] == ["Other", name]
    else:
        assert result == expected


def test_open_record_keeps_its_open_part_in_the_tool(server):
    schema = _agent_schema("R")
    server.content = _tool_reply(schema.__name__, _OPEN_CASES["R"])
    _model({"thinking": _THINKING, "max_tokens": 4096}).with_structured_output(schema).invoke("hi")
    input_schema = server.bodies[0]["tools"][0]["input_schema"]
    assert input_schema["properties"]["m"]["additionalProperties"] is True


@pytest.mark.parametrize("reply", sorted(_NO_TOOL_REPLIES))
def test_open_schema_reply_without_the_tool_call_raises_naming_the_tool(server, reply):
    schema = _agent_schema("object")
    name = convert_to_anthropic_tool(schema)["name"]
    server.content = _NO_TOOL_REPLIES[reply]
    model = _model({"thinking": _THINKING, "max_tokens": 4096})
    with pytest.raises(OutputParserException) as caught:
        model.with_structured_output(schema).invoke("hi")
    assert str(caught.value) == str(missing_tool_call(name))


@pytest.mark.parametrize("reply", sorted(_NO_TOOL_REPLIES))
def test_open_schema_reply_without_the_tool_call_is_a_parsing_error_with_raw(server, reply):
    schema = _agent_schema("object")
    name = convert_to_anthropic_tool(schema)["name"]
    server.content = _NO_TOOL_REPLIES[reply]
    model = _model({"thinking": _THINKING, "max_tokens": 4096})
    result = model.with_structured_output(schema, include_raw=True).invoke("hi")
    assert set(result) == {"raw", "parsed", "parsing_error"}
    assert isinstance(result["raw"], AIMessage)
    assert result["raw"].response_metadata["id"] == "msg_1"
    assert result["parsed"] is None
    assert isinstance(result["parsing_error"], OutputParserException)
    assert str(result["parsing_error"]) == str(missing_tool_call(name))


def test_open_schema_binding_leaves_warnings_alone(server):
    schema = _agent_schema("object")
    server.content = _tool_reply(schema.__name__, _OPEN_CASES["object"])
    model = _model({"thinking": _THINKING, "max_tokens": 4096})
    filters = list(warnings.filters)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.with_structured_output(schema).invoke("hi")
        model.with_structured_output(schema, include_raw=True).invoke("hi")
    assert caught == []
    model.with_structured_output(schema)
    assert warnings.filters == filters


def test_closed_agent_record_with_block_thinking_uses_the_output_format(server):
    schema = _agent_schema("C")
    server.text = '{"a": 1, "b": ["x"]}'
    model = _model({"thinking": _THINKING, "max_tokens": 4096})
    assert model.with_structured_output(schema).invoke("hi") == schema(a=1, b=["x"])
    body = server.bodies[0]
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert "tools" not in body


@pytest.mark.parametrize("block", [None, {"thinking": {"type": "disabled"}}])
def test_open_schema_without_block_thinking_forces_the_tool(server, block):
    schema = _agent_schema("object")
    name = convert_to_anthropic_tool(schema)["name"]
    server.content = [{"type": "tool_use", "id": "toolu_1", "name": name,
                       "input": _OPEN_CASES["object"]}]
    result = _model(block).with_structured_output(schema).invoke("hi")
    assert result == schema.model_validate(_OPEN_CASES["object"])
    assert server.bodies[0]["tool_choice"] == {"type": "tool", "name": name}
    assert "output_config" not in server.bodies[0]


@pytest.mark.parametrize("method", ["json_schema", "function_calling"])
def test_open_schema_with_block_thinking_keeps_an_explicit_method(server, method):
    schema = _agent_schema("object")
    name = convert_to_anthropic_tool(schema)["name"]
    server.text = '{"value": {"k": 1}}'
    model = _model({"thinking": _THINKING, "max_tokens": 4096})
    model.with_structured_output(schema, method=method).invoke("hi")
    body = server.bodies[0]
    if method == "json_schema":
        assert body["output_config"]["format"]["type"] == "json_schema"
        assert "tools" not in body
    else:
        assert body["tool_choice"] == {"type": "tool", "name": name}
        assert "output_config" not in body


def test_open_schema_run_with_a_transcript_corrects_a_reply_without_the_tool_call(server):
    typ = type_for("object", _AGENT_TYPES)
    name = convert_to_anthropic_tool(type_to_schema(typ))["name"]
    server.replies = [_NO_TOOL_REPLIES["text"], _tool_reply(name, _OPEN_CASES["object"])]
    model = _model({"thinking": _THINKING, "max_tokens": 4096})
    with harness() as opened:
        with opened.frame.open_turn(request(), check_stopped=lambda: None) as turn:
            value = generate_structured(
                model, [HumanMessage(content="question")], typ,
                llm_config={"provider": "anthropic", "model": "claude-sonnet-4-5"},
                memory_context=MemoryContext(transcript=turn),
            )
    assert value == {"k": 1}
    assert len(server.bodies) == 2
    correction = json.dumps(server.bodies[1]["messages"][-1])
    assert str(missing_tool_call(name)) in correction


# Models langchain-anthropic lists as refusing a forced tool choice.
_LANGCHAIN_NO_FORCE = ["claude-opus-5-5", "claude-sonnet-5-5", "claude-fable-5-1"]
# Plus those Anthropic's forced-tool table names that langchain-anthropic does not list.
_NO_FORCE = [*_LANGCHAIN_NO_FORCE, "claude-mythos-5-1"]
_NO_FORCE_SCHEMAS = {
    "Answer": (lambda: Answer, {"answer": "ok"}),
    "C": (lambda: _agent_schema("C"), {"a": 1, "b": ["x"]}),
    "object": (lambda: _agent_schema("object"), {"value": {"k": 1}}),
}


def _no_force_case(server, model, case, block=None, method=None, include_raw=False):
    """Bind `case`'s schema on `model` and check it goes as an unforced tool whose call parses."""
    make, args = _NO_FORCE_SCHEMAS[case]
    schema = make()
    name = convert_to_anthropic_tool(schema)["name"]
    server.content = [{"type": "tool_use", "id": "toolu_1", "name": name, "input": args}]
    kwargs = {} if method is None else {"method": method}
    structured = _model(block, model=model).with_structured_output(
        schema, include_raw=include_raw, **kwargs
    )
    result = structured.invoke("hi")
    expected = schema.model_validate(args)
    if include_raw:
        assert result["parsed"] == expected and result["parsing_error"] is None
        assert isinstance(result["raw"], AIMessage)
    else:
        assert result == expected
    body = server.bodies[0]
    assert body["tools"] == [convert_to_anthropic_tool(schema)]
    assert "tool_choice" not in body
    assert "output_config" not in body


@pytest.mark.parametrize("model", _NO_FORCE)
@pytest.mark.parametrize("case", sorted(_NO_FORCE_SCHEMAS))
@pytest.mark.parametrize("include_raw", [False, True])
def test_models_that_refuse_a_forced_tool_get_an_unforced_tool(server, model, case, include_raw):
    _no_force_case(server, model, case, include_raw=include_raw)


@pytest.mark.parametrize("model", _NO_FORCE)
def test_those_models_get_an_unforced_tool_for_an_explicit_function_calling(server, model):
    _no_force_case(server, model, "Answer", method="function_calling")


def test_those_models_get_an_unforced_tool_with_block_thinking_disabled(server):
    # langchain-anthropic's model check refuses `disabled` on the other two before any
    # request, so only claude-opus-5-5 can exercise this branch here.
    _no_force_case(server, "claude-opus-5-5", "Answer", block={"thinking": {"type": "disabled"}})


@pytest.mark.parametrize("reply", sorted(_NO_TOOL_REPLIES))
def test_a_reply_without_the_tool_call_on_those_models_raises_naming_the_tool(server, reply):
    server.content = _NO_TOOL_REPLIES[reply]
    with pytest.raises(OutputParserException) as caught:
        _model(model="claude-opus-5-5").with_structured_output(Answer).invoke("hi")
    assert str(caught.value) == str(missing_tool_call("Answer"))


@pytest.mark.parametrize("reply", sorted(_NO_TOOL_REPLIES))
def test_a_reply_without_the_tool_call_on_those_models_is_a_parsing_error_with_raw(server, reply):
    server.content = _NO_TOOL_REPLIES[reply]
    model = _model(model="claude-opus-5-5")
    result = model.with_structured_output(Answer, include_raw=True).invoke("hi")
    assert isinstance(result["raw"], AIMessage)
    assert result["parsed"] is None
    assert isinstance(result["parsing_error"], OutputParserException)
    assert str(result["parsing_error"]) == str(missing_tool_call("Answer"))


@pytest.mark.parametrize("model", _NO_FORCE)
def test_binding_on_those_models_emits_no_warning(server, model):
    server.content = _tool_reply("Answer", {"answer": "ok"})
    chat = _model(model=model)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        chat.with_structured_output(Answer).invoke("hi")
        chat.with_structured_output(Answer, include_raw=True).invoke("hi")
    assert caught == []


@pytest.mark.parametrize("model", _NO_FORCE)
def test_those_models_with_block_thinking_and_a_closed_schema_keep_the_output_format(
    server, model
):
    server.text = '{"answer": "ok"}'
    # These models take only adaptive thinking.
    model = _model({"thinking": {"type": "adaptive"}, "max_tokens": 4096}, model=model)
    assert model.with_structured_output(Answer).invoke("hi") == Answer(answer="ok")
    body = server.bodies[0]
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert "tools" not in body


@pytest.mark.parametrize("model", _NO_FORCE)
def test_those_models_keep_an_explicit_json_schema_method(server, model):
    server.text = '{"answer": "ok"}'
    model = _model(model=model)
    assert model.with_structured_output(Answer, method="json_schema").invoke("hi") == Answer(
        answer="ok"
    )
    body = server.bodies[0]
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert "tools" not in body


def test_forced_tool_support_matches_langchain():
    for model in [*_LANGCHAIN_NO_FORCE, "claude-sonnet-4-5"]:
        assert _forces_tool(model) == _supports_forced_tool_choice(model), model
    assert not any(_forces_tool(model) for model in _NO_FORCE)
    assert _forces_tool("claude-sonnet-4-5")
    assert _forces_tool("claude-mythos-5-1") is False
    assert _forces_tool("claude-mythos-5-1-20261001") is False
    # Anthropic's table names only Mythos 5.1.
    assert _forces_tool("claude-mythos-5")


def test_a_transcript_run_on_those_models_stores_the_reply_and_corrects_it(server):
    typ = Type.scalar(ValueKind.INTEGER)
    name = convert_to_anthropic_tool(type_to_schema(typ))["name"]
    server.replies = [
        [{"type": "text", "text": "42"}],
        [{"type": "tool_use", "id": "toolu_1", "name": name, "input": {"value": 42}}],
    ]
    model = _model(model="claude-opus-5-5")
    memory = SessionMemory()
    with harness(memory) as opened:
        with opened.frame.open_turn(request(), check_stopped=lambda: None) as turn:
            value = generate_structured(
                model, [HumanMessage(content="question")], typ,
                llm_config={"provider": "anthropic", "model": "claude-opus-5-5"},
                memory_context=MemoryContext(transcript=turn),
            )
    assert value == 42
    assert len(server.bodies) == 2
    types = [item.payload["message"]["type"] for item in _view(memory).items]
    assert types[-3:] == ["ai", "human", "ai"]
    messages = server.bodies[1]["messages"]
    assert [message["role"] for message in messages] == ["user"]
    last = messages[0]["content"][-1]["text"]
    assert str(missing_tool_call(name)) in last
    assert "provider error (" not in last


@pytest.fixture
def clean_env(tmp_path, monkeypatch):
    monkeypatch.setenv("YIAN_AUTH_FILE", str(tmp_path / "auth.json"))
    for name in (*_ENV_KEYS, "ANTHROPIC_PROXY"):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


@pytest.fixture
def host_clients():
    """A recorder and the SDK's own sync and async httpx clients routed to it."""
    httpx = _anthropic_httpx()
    recorder = _Server()
    transport = httpx.MockTransport(recorder)
    sync, async_ = httpx.Client(transport=transport), httpx.AsyncClient(transport=transport)
    yield recorder, sync, async_
    sync.close()
    asyncio.run(async_.aclose())


def test_a_host_http_client_carries_every_request(clean_env, host_clients):
    recorder, sync, async_ = host_clients
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # langchain warns when a key goes to model_kwargs
        chat = AnthropicClient(
            "claude-sonnet-4-5", api_key="k", max_retries=0,
            http_client=sync, http_async_client=async_,
        ).get_llm()
    assert chat.model_kwargs == {}
    assert chat.invoke("hi").content == "ok"
    assert asyncio.run(chat.ainvoke("hi")).content == "ok"
    assert [str(r.url) for r in recorder.requests] == [
        "https://api.anthropic.com/v1/messages"
    ] * 2
    for body in recorder.bodies:
        assert "http_client" not in body and "http_async_client" not in body
    assert not {"http_client", "http_async_client"} & set(chat.model_dump())


def test_a_non_redirecting_host_client_is_used_with_a_base_url(clean_env, host_clients):
    recorder, sync, _ = host_clients
    chat = AnthropicClient(
        "claude-sonnet-4-5", base_url="https://gateway.test", api_key="k",
        max_retries=0, http_client=sync,
    ).get_llm()
    assert chat.invoke("hi").content == "ok"
    assert [str(r.url) for r in recorder.requests] == ["https://gateway.test/v1/messages"]


@pytest.mark.parametrize(
    ("key", "factory"),
    [
        ("http_client", lambda: anthropic.DefaultHttpxClient()),
        ("http_async_client", lambda: anthropic.DefaultAsyncHttpxClient()),
    ],
)
def test_a_redirecting_host_client_is_rejected_beside_a_base_url(clean_env, key, factory):
    client = factory()
    try:
        with pytest.raises(ValueError, match=f"{key} follows redirects.*follow_redirects=False"):
            AnthropicClient(
                "claude-sonnet-4-5", base_url="https://gateway.test", api_key="k",
                **{key: client},
            ).get_llm()
    finally:
        if key == "http_client":
            client.close()
        else:
            asyncio.run(client.aclose())


def test_a_directly_built_model_rejects_a_redirecting_client_beside_a_base_url():
    with anthropic.DefaultHttpxClient() as client:
        chat = NormalizedChatAnthropic(
            model="claude-sonnet-4-5", api_key="k", base_url="https://gateway.test",
            follow_redirects=False, http_client=client,
        )
        with pytest.raises(ValueError, match="http_client follows redirects"):
            chat._client


def test_a_redirecting_host_client_is_allowed_without_a_base_url(clean_env):
    with anthropic.DefaultHttpxClient() as client:
        chat = AnthropicClient("claude-sonnet-4-5", api_key="k", http_client=client).get_llm()
        assert chat._client._client is client


def test_a_proxy_beside_a_host_client_is_rejected(clean_env, host_clients):
    _, sync, _ = host_clients
    clean_env.setenv("ANTHROPIC_PROXY", "http://proxy.test:3128")
    with pytest.raises(ValueError, match="ANTHROPIC_PROXY can't be used with http_client"):
        AnthropicClient("claude-sonnet-4-5", api_key="k", http_client=sync).get_llm()


def test_a_host_client_of_the_wrong_type_fails_when_the_model_is_built(clean_env):
    with pytest.raises(TypeError, match="Invalid `http_client` argument; Expected an instance"):
        AnthropicClient("claude-sonnet-4-5", api_key="k", http_client=object()).get_llm()


@pytest.mark.skipif(
    _anthropic_httpx().__name__ == "httpx", reason="the SDK is built on httpx itself"
)
def test_a_plain_httpx_client_on_an_httpx2_sdk_fails_when_the_model_is_built(clean_env):
    import httpx

    with httpx.Client() as client, pytest.raises(TypeError, match="is from the `httpx` package"):
        AnthropicClient("claude-sonnet-4-5", api_key="k", http_client=client).get_llm()
