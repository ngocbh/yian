"""The ``google`` block reaches the Gemini generateContent request body as written.

A structured output schema's ``const`` keywords are sent as one-value ``enum``s, which
Gemini's ``responseJsonSchema`` subset supports.
"""

import json

import httpx
import pytest
from google.genai._api_client import SyncHttpxClient
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, ValidationError

from yian.llm_clients.factory import model_from_config
from yian.llm_clients.google_client import _const_as_enum
from yian.nodes.agent.structured import type_to_schema
from yian.typesys.types import read_typedefs, type_for


def _reply(text):
    """Return a generateContent response whose one candidate says `text`."""
    return {
        "candidates": [{
            "content": {"role": "model", "parts": [{"text": text}]},
            "finishReason": "STOP",
            "index": 0,
        }],
        "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2},
    }


class _Bodies(list):
    """The request bodies recorded so far; every reply's text is `text`."""

    text = '{"answer": "ok"}'


@pytest.fixture
def bodies(tmp_path, monkeypatch):
    """Answer every Gemini request with one reply and record each request body."""
    monkeypatch.setenv("YIAN_AUTH_FILE", str(tmp_path / "auth.json"))
    for name in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "YIAN_DEFAULT_PROVIDER", "YIAN_DEFAULT_MODEL"):
        monkeypatch.delenv(name, raising=False)
    recorded = _Bodies()

    def send(self, request, **kwargs):
        recorded.append(json.loads(request.content))
        reply = _reply(recorded.text)
        if ":streamGenerateContent" in request.url.path:
            return httpx.Response(200, text=f"data: {json.dumps(reply)}\n\n", request=request)
        return httpx.Response(200, json=reply, request=request)

    monkeypatch.setattr(SyncHttpxClient, "send", send)
    return recorded


def _model(model, block=None):
    config = {"provider": "google", "model": model, "api_key": "k"}
    if block is not None:
        config["google"] = block
    return model_from_config(config)


@pytest.mark.parametrize(
    "model, thinking",
    [
        ("gemini-3-pro-preview", {"thinkingLevel": "HIGH"}),
        ("gemini-2.5-flash", {"thinkingBudget": 1024}),
    ],
)
def test_thinking_config_is_sent_as_written(bodies, model, thinking):
    _model(model, {"generationConfig": {"thinkingConfig": thinking}}).invoke([HumanMessage("hi")])

    assert bodies[-1]["generationConfig"]["thinkingConfig"] == thinking


def test_no_temperature_is_sent_without_one(bodies):
    _model("gemini-2.5-flash").invoke([HumanMessage("hi")])

    assert "temperature" not in bodies[-1].get("generationConfig", {})


# provider -> a model id and a block whose fields must reach the request body.
BODY_CASES = {
    "google": {
        "model": "gemini-2.5-flash",
        "block": {
            "safetySettings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"}
            ],
            "generationConfig": {"topK": 5, "maxOutputTokens": 64},
        },
    },
}


@pytest.mark.parametrize("provider", sorted(BODY_CASES))
def test_block_reaches_the_body_as_written(bodies, provider):
    case = BODY_CASES[provider]

    _model(case["model"], case["block"]).invoke([HumanMessage("hi")])

    body = bodies[-1]
    assert body["safetySettings"] == case["block"]["safetySettings"]
    for key, value in case["block"]["generationConfig"].items():
        assert body["generationConfig"][key] == value, key


def test_streaming_carries_the_block(bodies):
    block = {"generationConfig": {"thinkingConfig": {"thinkingLevel": "LOW"}, "maxOutputTokens": 64}}

    chunks = list(_model("gemini-3-flash-preview", block).stream([HumanMessage("hi")]))

    assert chunks
    assert bodies[-1]["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "LOW"}
    assert bodies[-1]["generationConfig"]["maxOutputTokens"] == 64


class _Answer(BaseModel):
    answer: str


def test_structured_output_keeps_the_schema_beside_block_fields(bodies):
    model = _model("gemini-2.5-flash", {"generationConfig": {"temperature": 0.2}})

    result = model.with_structured_output(_Answer).invoke([HumanMessage("hi")])

    assert result == _Answer(answer="ok")
    config = bodies[-1]["generationConfig"]
    assert config["temperature"] == 0.2
    assert config["responseMimeType"] == "application/json"
    assert config["responseJsonSchema"]["properties"] == {"answer": {"title": "Answer", "type": "string"}}


def test_block_merges_into_a_callers_extra_body(bodies):
    model = _model("gemini-2.5-flash", {"generationConfig": {"topK": 5}})
    caller = {"extra_body": {"generationConfig": {"seed": 7}}}

    model.invoke([HumanMessage("hi")], http_options=caller)

    assert bodies[-1]["generationConfig"]["topK"] == 5
    assert bodies[-1]["generationConfig"]["seed"] == 7
    assert caller == {"extra_body": {"generationConfig": {"seed": 7}}}


_TRADES = read_typedefs({
    "Buy": {"kind": "Literal(buy)", "x": "int"},
    "Sell": {"kind": "Literal(sell)", "n": "int"},
    "D": "Union(Buy, Sell)",
})


def _keywords(schema):
    """Return every key of every dict in `schema`, property names included."""
    keys, stack = set(), [schema]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            keys.update(node)
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    return keys


def test_union_tags_are_sent_as_one_value_enums(bodies):
    bodies.text = '{"value": {"kind": "buy", "x": 1}}'

    _model("gemini-2.5-flash").with_structured_output(
        type_to_schema(type_for("D", _TRADES))
    ).invoke([HumanMessage("hi")])

    schema = bodies[-1]["generationConfig"]["responseJsonSchema"]
    assert "const" not in _keywords(schema)
    kinds = {name: d["properties"]["kind"] for name, d in schema["$defs"].items()}
    assert kinds == {
        "buy": {"enum": ["buy"], "type": "string", "title": "Kind"},
        "sell": {"enum": ["sell"], "type": "string", "title": "Kind"},
    }


def test_a_one_value_literal_is_sent_as_an_enum(bodies):
    bodies.text = '{"value": "only"}'

    result = _model("gemini-2.5-flash").with_structured_output(
        type_to_schema(type_for("Literal(only)", {}))
    ).invoke([HumanMessage("hi")])

    assert result.value == "only"
    value = bodies[-1]["generationConfig"]["responseJsonSchema"]["properties"]["value"]
    assert value == {"enum": ["only"], "type": "string", "title": "Value"}


def test_a_schema_without_const_is_sent_unchanged(bodies):
    _model("gemini-2.5-flash").with_structured_output(_Answer).invoke([HumanMessage("hi")])

    assert bodies[-1]["generationConfig"]["responseJsonSchema"] == _Answer.model_json_schema()


def test_a_property_named_const_keeps_its_name(bodies):
    bodies.text = '{"const": "x"}'
    schema = {
        "title": "Pinned",
        "type": "object",
        "properties": {"const": {"type": "string", "const": "x"}},
        "required": ["const"],
    }

    _model("gemini-2.5-flash").with_structured_output(schema).invoke([HumanMessage("hi")])

    sent = bodies[-1]["generationConfig"]["responseJsonSchema"]
    assert sent["properties"] == {"const": {"type": "string", "enum": ["x"]}}
    assert schema["properties"]["const"] == {"type": "string", "const": "x"}


def test_the_bound_schema_is_not_rewritten(bodies):
    bodies.text = '{"value": {"kind": "buy", "x": 1}}'
    output = type_to_schema(type_for("D", _TRADES))
    before = output.model_json_schema()
    structured = _model("gemini-2.5-flash").with_structured_output(output)

    structured.invoke([HumanMessage("hi")])
    structured.invoke([HumanMessage("hi")])

    assert bodies[-1] == bodies[-2]
    assert output.model_json_schema() == before
    assert structured.first.kwargs["response_json_schema"] == before


def test_streaming_sends_the_union_tags_as_enums(bodies):
    bodies.text = '{"value": {"kind": "sell", "n": 2}}'

    chunks = list(
        _model("gemini-2.5-flash").with_structured_output(
            type_to_schema(type_for("D", _TRADES))
        ).stream([HumanMessage("hi")])
    )

    assert chunks
    schema = bodies[-1]["generationConfig"]["responseJsonSchema"]
    assert "const" not in _keywords(schema)
    assert schema["$defs"]["sell"]["properties"]["kind"]["enum"] == ["sell"]


def test_parsing_still_checks_the_tag(bodies):
    structured = _model("gemini-2.5-flash").with_structured_output(
        type_to_schema(type_for("D", _TRADES)), include_raw=True
    )

    bodies.text = '{"value": {"kind": "hold", "x": 1}}'
    wrong = structured.invoke([HumanMessage("hi")])
    bodies.text = '{"value": {"kind": "buy", "x": 1}}'
    right = structured.invoke([HumanMessage("hi")])

    assert wrong["parsed"] is None
    assert isinstance(wrong["parsing_error"], OutputParserException)
    assert isinstance(wrong["parsing_error"].__cause__, ValidationError)
    assert right["parsing_error"] is None
    assert right["parsed"].value.kind == "buy"
    assert right["parsed"].value.x == 1


@pytest.mark.parametrize(
    "schema, sent",
    [
        ({"const": "a", "enum": ["a", "b"]}, {"enum": ["a"]}),
        ({"const": "a", "enum": ["b"]}, {"const": "a", "enum": ["b"]}),
        ({"const": 1, "enum": [True, "b"]}, {"const": 1, "enum": [True, "b"]}),
        ({"const": True, "enum": [1]}, {"const": True, "enum": [1]}),
        ({"const": 3, "type": "integer"}, {"enum": [3], "type": "integer"}),
        ({"type": "array", "items": {"const": "a"}}, {"type": "array", "items": {"enum": ["a"]}}),
        ({"prefixItems": [{"const": "a"}]}, {"prefixItems": [{"enum": ["a"]}]}),
        ({"additionalProperties": {"const": 1}}, {"additionalProperties": {"enum": [1]}}),
        ({"allOf": [{"const": "a"}], "oneOf": [{"const": "b"}]},
         {"allOf": [{"enum": ["a"]}], "oneOf": [{"enum": ["b"]}]}),
        ({"anyOf": [{"const": "a"}, {"type": "null"}]},
         {"anyOf": [{"enum": ["a"]}, {"type": "null"}]}),
        ({"$defs": {"A": {"const": "a"}}}, {"$defs": {"A": {"enum": ["a"]}}}),
        ({"properties": {"const": {"const": 1}}}, {"properties": {"const": {"enum": [1]}}}),
        ({"default": {"const": 1}}, {"default": {"const": 1}}),
    ],
)
def test_const_as_enum(schema, sent):
    original = json.loads(json.dumps(schema))

    assert _const_as_enum(schema) == sent
    assert schema == original
