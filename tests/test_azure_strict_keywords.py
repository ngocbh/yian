"""Azure strict structured output without the JSON-schema keywords its strict mode rejects."""

import json
from dataclasses import dataclass
from typing import Annotated, TypedDict

import annotated_types
import httpx
import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    Strict,
    ValidationError,
    WithJsonSchema,
)
from pydantic import v1 as pydantic_v1

from tests.test_structured_open_schema import _ToolCallServer, _azure, _payload, azure_env
from yian.llm_clients.azure_client import _langchain_sends_a_tool, _strict_schema_without
from yian.memory.reflective import _ReflectionResponse
from yian.nodes.agent.structured import (
    _structured_validation_path,
    generate_structured,
    type_to_schema,
)
from yian.typesys.types import read_typedefs, type_for

# Names the imported fixture as used, so pytest finds it here and linters keep the import.
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
    "Optional(int(0..3))": lambda: _schema("Optional(int(0..3))"),
    "float(0.0..1.0)": lambda: _schema("float(0.0..1.0)"),
    "list(Rec)": lambda: _schema("list(Rec)"),
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


def test_azure_extra_kwargs_reach_the_payload(azure_env):
    llm = _azure()
    schema = _schema("int(0..10)")
    payload = _payload(llm, llm.with_structured_output(schema, temperature=0.1))
    assert payload["temperature"] == 0.1
    assert payload["response_format"] == _strict_schema_without(schema)


def _lookup(query: str) -> str:
    """Look `query` up."""
    return query


def test_azure_extra_tools_are_sent_as_langchain_sends_them(azure_env):
    llm = _azure()
    schema = _schema("int(0..10)")
    payload = _payload(llm, llm.with_structured_output(schema, tools=[_lookup]))
    expected = AzureChatOpenAI.with_structured_output(llm, schema, tools=[_lookup])
    assert payload["tools"] == _payload(llm, expected)["tools"]
    assert payload["response_format"] == _strict_schema_without(schema)


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


class _HandWrittenFormat(BaseModel):
    n: int = Field(ge=0)
    x: str = Field(json_schema_extra={"format": "email"})


class _HandWrittenInner(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [{"y": "a"}]})

    y: str


class _HandWrittenNested(BaseModel):
    n: int = Field(ge=0)
    inner: list[_HandWrittenInner | None]


class _WithJsonSchema(BaseModel):
    n: int = Field(ge=0)
    p: Annotated[str, WithJsonSchema({"type": "string", "pattern": "^a+$"})]


class _SchemaHook(BaseModel):
    n: int = Field(ge=0)

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, handler):
        schema = handler(core_schema)
        schema["properties"]["n"]["maximum"] = 5
        return schema


class _PatternDict(TypedDict):
    p: Annotated[str, Field(json_schema_extra={"pattern": "^a$"})]


class _HasTypedDict(BaseModel):
    n: int = Field(ge=0)
    d: _PatternDict


@dataclass
class _PatternData:
    p: Annotated[str, Field(json_schema_extra={"pattern": "^a$"})]


class _HasDataclass(BaseModel):
    n: int = Field(ge=0)
    d: _PatternData


class _ExtraInList(BaseModel):
    n: int = Field(ge=0)
    xs: list[Annotated[str, Field(json_schema_extra={"pattern": "^a$"})]]


@pytest.mark.parametrize("schema", [
    _HandWrittenFormat, _HandWrittenNested, _WithJsonSchema, _SchemaHook, _HasTypedDict,
    _HasDataclass, _ExtraInList,
])
def test_azure_hand_written_schema_keywords_keep_the_unchanged_path(azure_env, schema):
    llm = _azure()
    payload = _payload(llm, llm.with_structured_output(schema))
    assert payload == _payload(llm, AzureChatOpenAI.with_structured_output(llm, schema))
    assert payload["response_format"] is schema


class _Constrained(BaseModel):
    a: float = Field(ge=0, le=1, allow_inf_nan=False)
    b: str = Field(min_length=1, max_length=3, pattern="^a")
    c: list[Annotated[int, Field(gt=0, multiple_of=2)]] = Field(min_length=1)
    d: int | None = Field(lt=9, strict=True)
    e: list[Annotated[int, annotated_types.Ge(1), Strict()]]


def test_azure_pydantic_constraints_take_the_stripped_path(azure_env):
    llm = _azure()
    payload = _payload(llm, llm.with_structured_output(_Constrained))
    assert payload["response_format"] == _strict_schema_without(_Constrained)


_LEGACY_DEPLOYMENTS = ["gpt-4", "gpt-4-turbo", "gpt-35-turbo"]


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("deployment", _LEGACY_DEPLOYMENTS)
@pytest.mark.parametrize("name", ["int(0..10)", "_ReflectionResponse"])
def test_azure_legacy_deployment_keeps_langchains_request(azure_env, deployment, name):
    llm = _azure(deployment)
    schema = _BOUNDED[name]()
    payload = _payload(llm, llm.with_structured_output(schema))
    assert payload == _payload(llm, AzureChatOpenAI.with_structured_output(llm, schema))
    assert "tools" in payload
    assert "response_format" not in payload


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("deployment", [
    *_LEGACY_DEPLOYMENTS, "gpt-3.5-turbo", "gpt-4o", "gpt-4.1", "gpt-4o-mini", "o3", "gpt-5",
])
def test_legacy_deployment_check_matches_langchain(azure_env, deployment):
    llm = _azure(deployment)
    schema = _schema("int")
    payload = _payload(llm, AzureChatOpenAI.with_structured_output(llm, schema))
    assert _langchain_sends_a_tool(deployment) is ("tools" in payload)


class _V1Count(pydantic_v1.BaseModel):
    n: int = pydantic_v1.Field(ge=0)


@pytest.mark.filterwarnings("ignore")
def test_azure_pydantic_v1_schema_is_unchanged(azure_env):
    llm = _azure()
    payload = _payload(llm, llm.with_structured_output(_V1Count))
    assert payload == _payload(llm, AzureChatOpenAI.with_structured_output(llm, _V1Count))


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


_AZURE_CONFIG = {"provider": "azure", "model": "gpt-4o"}


def _ask(server, schema, deployment="gpt-4o", **kwargs):
    """Invoke Azure's structured output for `schema` against `server`'s replies."""
    with httpx.Client(transport=httpx.MockTransport(server)) as http_client:
        llm = _azure(deployment, http_client=http_client, max_retries=0)
        return llm.with_structured_output(schema, **kwargs).invoke([HumanMessage("q")])


def test_a_reply_within_the_bounds_parses(azure_env):
    schema = _schema("int(0..10)")
    assert _ask(_ContentServer('{"value": 7}'), schema).value == 7
    result = _ask(_ContentServer('{"value": 7}'), schema, include_raw=True)
    assert result["parsing_error"] is None
    assert result["parsed"].value == 7


def test_a_reply_past_a_moved_bound_is_a_parse_error(azure_env):
    with pytest.raises(OutputParserException) as caught:
        _ask(_ContentServer('{"value": 11}'), _schema("int(0..10)"))
    cause = caught.value.__cause__
    assert isinstance(cause, ValidationError)
    assert cause.errors()[0]["loc"] == ("value",)


def test_a_reply_past_a_moved_bound_is_a_parsing_error_with_include_raw(azure_env):
    result = _ask(_ContentServer('{"value": 11}'), _schema("int(0..10)"), include_raw=True)
    assert result["parsed"] is None
    assert isinstance(result["parsing_error"], OutputParserException)
    assert isinstance(result["parsing_error"].__cause__, ValidationError)


@pytest.mark.parametrize(("text", "reply", "path"), [
    ("int(0..10)", {"value": 11}, ()),
    ("Rec", {"n": 6, "s": "x"}, ("n",)),
])
def test_a_reply_past_a_moved_bound_is_owned_by_the_output_contract(
    azure_env, text, reply, path,
):
    typ = type_for(text, REGISTRY)
    server = _ContentServer(json.dumps(reply))
    with httpx.Client(transport=httpx.MockTransport(server)) as http_client:
        model = _azure(http_client=http_client, max_retries=0)
        with pytest.raises(OutputParserException) as caught:
            generate_structured(
                model, [HumanMessage("q")], typ, llm_config=_AZURE_CONFIG, max_retries=0,
            )
    assert _structured_validation_path(caught.value, typ) == path


def test_reflection_response_with_no_actions_is_a_parse_error(azure_env):
    with pytest.raises(OutputParserException):
        _ask(_ContentServer('{"actions": []}'), _ReflectionResponse)


def test_a_valid_reflection_response_parses(azure_env):
    reply = {"actions": [{"action": "noop", "lesson_id": None, "text": None}]}
    result = _ask(_ContentServer(json.dumps(reply)), _ReflectionResponse)
    assert [action.action for action in result.actions] == ["noop"]


def _responses_server(content, status="completed"):
    """Answer Responses API requests with one message holding `content` blocks."""
    def handler(request):
        return httpx.Response(200, json={
            "id": "resp_1", "object": "response", "created_at": 0, "status": status,
            "incomplete_details": (
                {"reason": "max_output_tokens"} if status == "incomplete" else None
            ),
            "model": "gpt-5-codex", "error": None, "instructions": None, "metadata": {},
            "parallel_tool_calls": False, "tool_choice": "auto", "tools": [],
            "output": [{"type": "message", "id": "m", "status": status, "role": "assistant",
                        "content": content}],
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2,
                      "input_tokens_details": {"cached_tokens": 0},
                      "output_tokens_details": {"reasoning_tokens": 0}},
        })
    return handler


_TRUNCATED = '{"items": [1, 2, 3'


def test_a_truncated_reply_is_a_parse_error(azure_env):
    with pytest.raises(OutputParserException):
        _ask(_ContentServer(_TRUNCATED), _schema("list(int(1..5))"))


def test_a_truncated_responses_reply_is_a_parse_error(azure_env):
    text = [{"type": "output_text", "text": _TRUNCATED, "annotations": []}]
    with pytest.raises(OutputParserException):
        _ask(_responses_server(text, "incomplete"), _schema("list(int(1..5))"), "gpt-5-codex")


def test_a_valid_responses_reply_parses(azure_env):
    text = [{"type": "output_text", "text": '{"items": [1, 5]}', "annotations": []}]
    result = _ask(_responses_server(text), _schema("list(int(1..5))"), "gpt-5-codex")
    assert result.items == [1, 5]


def test_a_refusal_is_a_parse_error_naming_it(azure_env):
    refusal = {"role": "assistant", "content": None, "refusal": "I can't help with that."}
    with pytest.raises(OutputParserException, match="I can't help with that."):
        _ask(_ContentServer(refusal), _schema("int(0..10)"))


def test_a_responses_refusal_is_a_parse_error_naming_it(azure_env):
    refusal = [{"type": "refusal", "refusal": "I can't help with that."}]
    with pytest.raises(OutputParserException, match="I can't help with that."):
        _ask(_responses_server(refusal), _schema("int(0..10)"), "gpt-5-codex")


def test_a_reply_past_a_moved_bound_is_corrected_and_retried(azure_env):
    typ = type_for("int(0..10)", REGISTRY)
    server = _ContentServer('{"value": 11}', '{"value": 7}')
    with httpx.Client(transport=httpx.MockTransport(server)) as http_client:
        model = _azure(http_client=http_client, max_retries=0)
        value = generate_structured(
            model, [HumanMessage("q")], typ, llm_config=_AZURE_CONFIG, max_retries=1,
        )
    assert value == 7
    assert len(server.bodies) == 2
    assert "less than or equal to 10" in json.dumps(server.bodies[1]["messages"][-1])
