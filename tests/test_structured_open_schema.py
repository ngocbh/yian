"""Open AGENT output schemas (`object`, `Any`, elementless lists) under strict structured output."""

import json
from typing import TypedDict

import httpx
import pytest
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableBinding, RunnableParallel, RunnableSequence
from langchain_openai import AzureChatOpenAI, ChatOpenAI

from yian.llm_clients.azure_client import AzureOpenAIClient
from yian.llm_clients.capabilities import has_open_schema
from yian.llm_clients.openai_client import OpenAIClient
from yian.llm_clients.openai_subscription import SubscriptionAccess, create_subscription_model
from yian.nodes.agent.structured import type_to_schema
from yian.typesys.types import read_typedefs, type_for
from yian.typesys.values import Type, ValueKind

REGISTRY = read_typedefs({
    "R": {"a": "int", "m": "object"},
    "Buy": {"kind": "Literal(buy)", "x": "Any"},
    "Sell": {"kind": "Literal(sell)", "n": "int"},
    "D": "Union(Buy, Sell)",
    "Outer": {"inner": "R", "n": "int"},
    "C": {"a": "int", "b": "list(str)"},
    "C2": {"kind": "Literal(c2)", "a": "int"},
    "C3": {"kind": "Literal(c3)", "b": "str"},
    "CU": "Union(C2, C3)",
})


def _schema(text):
    return type_to_schema(type_for(text, REGISTRY))


@pytest.mark.parametrize("text", [
    "object", "Any", "Optional(Any)", "list(object)", "list(Any)", "list(Optional(Any))",
    "Optional(object)", "list(list(object))", "Optional(R)", "list(R)", "R", "D", "Outer",
])
def test_open_types_have_an_open_schema(text):
    assert has_open_schema(_schema(text))


def test_an_elementless_list_has_an_open_schema():
    assert has_open_schema(type_to_schema(Type(kind=ValueKind.LIST_ANY)))


@pytest.mark.parametrize("text", [
    "int", "float(0.0..1.0)", "bool", "date", "Literal(a, b)", "list(str)", "list(int)",
    "C", "CU", "Optional(int)",
])
def test_closed_types_have_no_open_schema(text):
    assert not has_open_schema(_schema(text))


@pytest.mark.parametrize(("schema", "expected"), [
    ({"type": "object", "additionalProperties": True}, True),
    ({"type": "object", "properties": {}, "additionalProperties": False}, False),
    ({"type": "array", "items": {}}, True),
])
def test_dict_schemas_are_read_as_given(schema, expected):
    assert has_open_schema(schema) is expected


class _Answer(TypedDict):
    value: dict


@pytest.mark.parametrize("schema", [_Answer, None])
def test_other_inputs_have_no_open_schema(schema):
    assert has_open_schema(schema) is False


_AZURE_ENDPOINT = "https://x.openai.azure.com/"


@pytest.fixture
def azure_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_VERSION", "2025-03-01-preview")
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)


def _azure(deployment="gpt-4o", **kwargs):
    return AzureOpenAIClient(deployment, base_url=_AZURE_ENDPOINT, api_key="k", **kwargs).get_llm()


def _binding(runnable):
    """Return the `RunnableBinding` holding the model call inside a structured runnable."""
    # Walks langchain's runnable internals, as built by the pinned langchain version.
    stack = [runnable]
    while stack:
        node = stack.pop()
        if isinstance(node, RunnableBinding):
            return node
        if isinstance(node, RunnableSequence):
            stack.extend(reversed(node.steps))
        elif isinstance(node, RunnableParallel):
            stack.extend(node.steps__.values())
        else:
            stack.extend(
                child for name in ("first", "bound", "runnable")
                if (child := getattr(node, name, None)) is not None
            )
    raise AssertionError(f"no model binding in {runnable!r}")


def _payload(llm, runnable):
    """Return the request body keywords `llm` builds for the bound structured call."""
    kwargs = dict(_binding(runnable).kwargs)
    # Tracing metadata langchain pops before building a real request.
    kwargs.pop("ls_structured_output_format", None)
    return llm._get_request_payload([HumanMessage("hi")], **kwargs)


def _only_function(payload):
    (tool,) = payload["tools"]
    assert tool["type"] == "function"
    return tool.get("function", tool)


@pytest.mark.parametrize("text", ["object", "Any", "R"])
def test_azure_sends_an_open_schema_as_a_non_strict_function_tool(azure_env, text):
    llm = _azure()
    schema = _schema(text)
    payload = _payload(llm, llm.with_structured_output(schema))
    function = _only_function(payload)
    assert "strict" not in function
    assert "strict" not in payload["tools"][0]
    assert payload["tool_choice"] == {"type": "function", "function": {"name": function["name"]}}
    assert "response_format" not in payload
    assert has_open_schema(function["parameters"])
    if text == "Any":
        assert {"additionalProperties": True, "type": "object"} in (
            function["parameters"]["properties"]["value"]["anyOf"]
        )
        assert {"items": {}, "type": "array"} in function["parameters"]["properties"]["value"]["anyOf"]


@pytest.mark.parametrize("text", ["C", "int"])
def test_azure_closed_schema_payload_is_unchanged(azure_env, text):
    llm = _azure()
    schema = _schema(text)
    payload = _payload(llm, llm.with_structured_output(schema))
    assert payload == _payload(llm, AzureChatOpenAI.with_structured_output(llm, schema))
    assert payload["response_format"] is schema
    assert "tools" not in payload


def test_azure_explicit_method_is_kept_for_an_open_schema(azure_env):
    llm = _azure()
    schema = _schema("object")
    payload = _payload(llm, llm.with_structured_output(schema, method="json_schema"))
    assert payload["response_format"] is schema
    assert "tools" not in payload


def test_azure_responses_deployment_sends_strict_false_for_an_open_schema(azure_env):
    llm = _azure("gpt-5-codex")
    assert llm._use_responses_api({})
    payload = _payload(llm, llm.with_structured_output(_schema("object")))
    (tool,) = payload["tools"]
    assert tool["type"] == "function"
    assert tool["strict"] is False
    assert has_open_schema(tool["parameters"])


def test_azure_responses_deployment_explicit_function_calling_sends_strict_false(azure_env):
    llm = _azure("gpt-5-codex")
    runnable = llm.with_structured_output(_schema("object"), method="function_calling")
    (tool,) = _payload(llm, runnable)["tools"]
    assert tool["type"] == "function"
    assert tool["strict"] is False


def test_azure_json_mode_without_a_schema_keeps_langchain_behavior(azure_env):
    llm = _azure()
    payload = _payload(llm, llm.with_structured_output(None, method="json_mode"))
    expected = _payload(llm, AzureChatOpenAI.with_structured_output(llm, None, method="json_mode"))
    assert payload == expected
    assert payload["response_format"] == {"type": "json_object"}


def test_azure_responses_deployment_closed_schema_payload_is_unchanged(azure_env):
    llm = _azure("gpt-5-codex")
    schema = _schema("C")
    payload = _payload(llm, llm.with_structured_output(schema))
    assert payload == _payload(llm, AzureChatOpenAI.with_structured_output(llm, schema))
    assert "tools" not in payload


class _ToolCallServer:
    """Record Chat Completions request bodies; answer with a call to the first tool."""

    def __init__(self, arguments):
        self.arguments = arguments
        self.bodies = []

    def __call__(self, request):
        body = json.loads(request.content)
        self.bodies.append(body)
        if "tools" not in body:
            return httpx.Response(200, json={
                "id": "c", "object": "chat.completion", "created": 0, "model": "m",
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": "{}"}}],
            })
        name = body["tools"][0]["function"]["name"]
        return httpx.Response(200, json={
            "id": "c", "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": None,
                "tool_calls": [{"id": "call_1", "type": "function", "function": {
                    "name": name, "arguments": json.dumps(self.arguments)}}],
            }}],
        })


def test_azure_open_schema_with_include_raw_parses_the_tool_call(azure_env):
    server = _ToolCallServer({"value": {"k": 1}})
    with httpx.Client(transport=httpx.MockTransport(server)) as http_client:
        llm = _azure(http_client=http_client, max_retries=0)
        result = llm.with_structured_output(_schema("object"), include_raw=True).invoke(
            [HumanMessage("hi")]
        )
    assert set(result) == {"raw", "parsed", "parsing_error"}
    assert result["parsing_error"] is None
    assert result["parsed"].value == {"k": 1}
    (body,) = server.bodies
    assert "tools" in body
    assert "response_format" not in body


def _responses_tool(payload):
    (tool,) = payload["tools"]
    assert tool["type"] == "function"
    assert "function" not in tool
    return tool


def _openai(model="gpt-5", provider="openai"):
    return OpenAIClient(model, provider=provider, api_key="k").get_llm()


def _subscription():
    return create_subscription_model(
        "gpt-5.5", access_source=lambda: SubscriptionAccess("synthetic-access", "synthetic-account"),
    )


def _today(llm, schema):
    """Return langchain's own function-calling request payload for `schema` on `llm`."""
    return _payload(llm, ChatOpenAI.with_structured_output(llm, schema, method="function_calling"))


@pytest.mark.parametrize("text", ["object", "R"])
def test_openai_responses_sends_strict_false_for_an_open_schema(text):
    llm = _openai()
    assert llm.use_responses_api is True
    tool = _responses_tool(_payload(llm, llm.with_structured_output(_schema(text))))
    assert tool["strict"] is False
    assert has_open_schema(tool["parameters"])


def test_openai_responses_closed_schema_payload_is_unchanged():
    llm = _openai()
    schema = _schema("C")
    payload = _payload(llm, llm.with_structured_output(schema))
    assert payload == _today(llm, schema)
    assert "strict" not in _responses_tool(payload)


def test_openai_compatible_chat_completions_open_schema_payload_is_unchanged():
    llm = _openai("openai/gpt-4o", provider="openrouter")
    assert not llm._use_responses_api({})
    schema = _schema("object")
    payload = _payload(llm, llm.with_structured_output(schema))
    assert payload == _today(llm, schema)
    assert "strict" not in _only_function(payload)
    assert "strict" not in payload["tools"][0]


def test_openai_compatible_responses_routed_model_sends_strict_false_for_an_open_schema():
    llm = _openai("openai/gpt-5.1-codex", provider="openrouter")
    assert llm._use_responses_api({})
    tool = _responses_tool(_payload(llm, llm.with_structured_output(_schema("object"))))
    assert tool["strict"] is False


def test_openai_explicit_strict_is_passed_through_for_an_open_schema():
    llm = _openai()
    tool = _responses_tool(_payload(llm, llm.with_structured_output(_schema("object"), strict=True)))
    assert tool["strict"] is True


def test_subscription_sends_strict_false_for_an_open_schema():
    model = _subscription()
    tool = _responses_tool(_payload(model, model.with_structured_output(_schema("object"))))
    assert tool["strict"] is False
    assert has_open_schema(tool["parameters"])


def test_subscription_closed_schema_payload_is_unchanged():
    model = _subscription()
    schema = _schema("C")
    payload = _payload(model, model.with_structured_output(schema))
    assert payload == _today(model, schema)
    assert "strict" not in _responses_tool(payload)


def test_subscription_explicit_strict_is_passed_through_for_an_open_schema():
    model = _subscription()
    tool = _responses_tool(
        _payload(model, model.with_structured_output(_schema("object"), strict=True))
    )
    assert tool["strict"] is True
