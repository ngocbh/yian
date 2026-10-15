"""Azure strict structured output without the JSON-schema keywords its strict mode rejects."""

import json

import httpx
import pytest
from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field

from tests.test_structured_open_schema import _ToolCallServer, _azure, _payload, azure_env
from yian.llm_clients.azure_client import _strict_schema_without
from yian.memory.reflective import _ReflectionResponse
from yian.nodes.agent.structured import type_to_schema
from yian.typesys.types import read_typedefs, type_for

__all__ = ["azure_env"]

REGISTRY = read_typedefs({
    "C": {"a": "int", "b": "list(str)"},
    "M": {"minimum": "int(0..3)"},
    "Rec": {"n": "int(0..5)", "s": "str"},
    "C2": {"kind": "Literal(c2)", "a": "int(0..4)"},
    "C3": {"kind": "Literal(c3)", "b": "str"},
    "CU": "Union(C2, C3)",
    "OpenRec": {"n": "int(0..5)", "m": "object"},
})


def _schema(text):
    return type_to_schema(type_for(text, REGISTRY))


def _value(fmt):
    return fmt["json_schema"]["schema"]["properties"]["value"]


def test_a_bounded_int_moves_its_bounds_into_the_description():
    assert _strict_schema_without(_schema("int(0..10)")) == {
        "type": "json_schema",
        "json_schema": {
            "name": "ScalarWrapper",
            "strict": True,
            "schema": {
                "properties": {"value": {
                    "title": "Value", "type": "integer",
                    "description": "{minimum: 0, maximum: 10}",
                }},
                "required": ["value"],
                "title": "ScalarWrapper",
                "type": "object",
                "additionalProperties": False,
            },
        },
    }


def test_a_bound_on_an_any_of_member_moves_on_that_member():
    assert _value(_strict_schema_without(_schema("Optional(int(0..3))"))) == {
        "anyOf": [
            {"type": "integer", "description": "{minimum: 0, maximum: 3}"},
            {"type": "null"},
        ],
        "title": "Value",
    }


def test_float_bounds_keep_their_spelling():
    value = _value(_strict_schema_without(_schema("float(0.0..1.0)")))
    assert value["description"] == "{minimum: 0.0, maximum: 1.0}"
    assert "minimum" not in value and "maximum" not in value


def test_an_open_end_gives_only_the_given_bound():
    value = _value(_strict_schema_without(_schema("int(0..)")))
    assert value["description"] == "{minimum: 0}"
    assert "minimum" not in value


class _Described(BaseModel):
    count: int = Field(ge=1, description="How many.")


def test_an_existing_description_is_kept_before_the_note():
    count = _strict_schema_without(_Described)["json_schema"]["schema"]["properties"]["count"]
    assert count["description"] == "How many.\n\n{minimum: 1}"
    assert "minimum" not in count


def test_a_field_named_like_a_keyword_is_kept():
    properties = _strict_schema_without(_schema("M"))["json_schema"]["schema"]["properties"]
    assert properties == {"minimum": {
        "title": "Minimum", "type": "integer", "description": "{minimum: 0, maximum: 3}",
    }}


def test_reflection_response_min_items_moves_into_the_description():
    schema = _strict_schema_without(_ReflectionResponse)["json_schema"]["schema"]
    actions = schema["properties"]["actions"]
    assert "minItems" not in actions
    assert actions["description"].endswith("{minItems: 1}")


def test_closed_unbounded_schemas_need_no_change():
    assert _strict_schema_without(_schema("C")) is None
    assert _strict_schema_without(_schema("int")) is None


_BOUNDED = {
    "int(0..10)": lambda: _schema("int(0..10)"),
    "list(int(1..5))": lambda: _schema("list(int(1..5))"),
    "Rec": lambda: _schema("Rec"),
    "CU": lambda: _schema("CU"),
    "_ReflectionResponse": lambda: _ReflectionResponse,
}
_MOVED = {"minimum", "maximum", "minItems"}


def _keys(value):
    """Yield every dict key anywhere in `value`."""
    stack = [value]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            yield from node
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)


@pytest.mark.parametrize("name", _BOUNDED)
def test_azure_sends_a_bounded_schema_strict_without_the_bounds(azure_env, name):
    llm = _azure()
    schema = _BOUNDED[name]()
    payload = _payload(llm, llm.with_structured_output(schema))
    assert payload["response_format"] == _strict_schema_without(schema)
    assert "tools" not in payload
    assert not _MOVED & set(_keys(payload))


@pytest.mark.parametrize("name", _BOUNDED)
def test_azure_responses_deployment_sends_a_bounded_schema_strict_without_the_bounds(
    azure_env, name,
):
    llm = _azure("gpt-5-codex")
    assert llm._use_responses_api({})
    schema = _BOUNDED[name]()
    payload = _payload(llm, llm.with_structured_output(schema))
    fmt = _strict_schema_without(schema)["json_schema"]
    assert payload["text"]["format"] == {"type": "json_schema", **fmt}
    assert payload["text"]["format"]["strict"] is True
    assert "text_format" not in payload
    assert not _MOVED & set(_keys(payload))


@pytest.mark.parametrize("text", ["int", "C"])
def test_azure_closed_unbounded_schema_payload_is_unchanged(azure_env, text):
    llm = _azure()
    schema = _schema(text)
    payload = _payload(llm, llm.with_structured_output(schema))
    assert payload == _payload(llm, AzureChatOpenAI.with_structured_output(llm, schema))
    assert payload["response_format"] is schema


@pytest.mark.parametrize("text", ["object", "OpenRec"])
def test_azure_open_schema_payload_is_unchanged(azure_env, text):
    llm = _azure()
    schema = _schema(text)
    payload = _payload(llm, llm.with_structured_output(schema))
    expected = AzureChatOpenAI.with_structured_output(llm, schema, method="function_calling")
    assert payload == _payload(llm, expected)
    if text == "OpenRec":
        (tool,) = payload["tools"]
        assert tool["function"]["parameters"]["properties"]["n"]["maximum"] == 5


@pytest.mark.parametrize("kwargs", [{"method": "json_schema"}, {"strict": True}])
def test_azure_explicit_method_or_strict_is_unchanged(azure_env, kwargs):
    llm = _azure()
    schema = _schema("int(0..10)")
    payload = _payload(llm, llm.with_structured_output(schema, **kwargs))
    assert payload == _payload(llm, AzureChatOpenAI.with_structured_output(llm, schema, **kwargs))
    assert payload["response_format"] is schema


def test_azure_dict_schema_is_unchanged(azure_env):
    llm = _azure()
    schema = {
        "title": "Count", "type": "object", "additionalProperties": False,
        "properties": {"n": {"type": "integer", "minimum": 1}}, "required": ["n"],
    }
    payload = _payload(llm, llm.with_structured_output(schema))
    assert payload == _payload(llm, AzureChatOpenAI.with_structured_output(llm, schema))
    assert payload["response_format"]["json_schema"]["schema"]["properties"]["n"]["minimum"] == 1


class _ContentServer:
    """Record Chat Completions request bodies; answer each with the next reply message."""

    def __init__(self, *messages):
        self.messages = list(messages)
        self.bodies = []

    def __call__(self, request):
        self.bodies.append(json.loads(request.content))
        message = self.messages.pop(0)
        if isinstance(message, str):
            message = {"role": "assistant", "content": message}
        return httpx.Response(200, json={
            "id": "c", "object": "chat.completion", "created": 0, "model": "gpt-4o",
            "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })


def test_azure_include_raw_is_forwarded_for_a_closed_schema(azure_env):
    server = _ContentServer(json.dumps({"a": 1, "b": ["x"]}))
    with httpx.Client(transport=httpx.MockTransport(server)) as http_client:
        llm = _azure(http_client=http_client, max_retries=0)
        result = llm.with_structured_output(_schema("C"), include_raw=True).invoke(
            [HumanMessage("hi")]
        )
    assert set(result) == {"raw", "parsed", "parsing_error"}
    assert result["parsing_error"] is None
    assert (result["parsed"].a, result["parsed"].b) == (1, ["x"])


def test_azure_include_raw_is_forwarded_for_an_open_schema(azure_env):
    server = _ToolCallServer({"value": {"k": 1}})
    with httpx.Client(transport=httpx.MockTransport(server)) as http_client:
        llm = _azure(http_client=http_client, max_retries=0)
        result = llm.with_structured_output(_schema("object"), include_raw=True).invoke(
            [HumanMessage("hi")]
        )
    assert set(result) == {"raw", "parsed", "parsing_error"}
    assert result["parsed"].value == {"k": 1}
