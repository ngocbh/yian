"""Messages: the stable JSON shape, strict decoding, and construction checks."""

import json
import math

import pytest

from yian.llm.messages import (
    MESSAGE_FORMAT,
    MediaPart,
    Message,
    MessageFormatError,
    NamedTool,
    ReasoningPart,
    Request,
    Response,
    TextPart,
    ToolCallPart,
    ToolResultPart,
    ToolSpec,
    Usage,
)


def full_message():
    return Message(
        role="assistant",
        parts=(
            ReasoningPart("thinking", provider_payload={"signature": "abc", "n": [1, 2.5, None]}),
            TextPart("hello ", provider_payload={"id": "msg_1"}),
            TextPart("world"),
            MediaPart(
                "image/png", data="aGVsbG8=", name="pic.png", provider_payload={"detail": "low"}
            ),
            MediaPart("image/jpeg", url="https://example.com/a.jpg"),
            ToolCallPart("call_1", "search", {"q": "x", "deep": {"k": [True, False]}}),
            ToolCallPart("call_2", "broken", None, raw_arguments="{oops"),
            ToolCallPart("call_3", "both", {"a": 1}, raw_arguments='{"a": 1}'),
            ToolResultPart("call_1", "search", "result", is_error=True, provider_payload={"x": 1}),
            ToolResultPart("call_2", "broken", "ok"),
        ),
        protocol="anthropic_messages",
        provider_payload={"stop_sequence": None, "meta": {"k": "v"}},
    )


def canonical():
    return {
        "format": MESSAGE_FORMAT,
        "role": "assistant",
        "parts": [
            {"type": "reasoning", "text": "t", "provider_payload": {"signature": "s"}},
            {"type": "text", "text": "hi"},
            {"type": "media", "mime": "image/png", "url": "https://e.com/x.png"},
            {"type": "media", "mime": "image/png", "data": "AAAA", "name": "x"},
            {"type": "tool_call", "id": "c1", "name": "f", "arguments": {"a": [1, 2]}},
            {"type": "tool_call", "id": "c2", "name": "g", "raw_arguments": "{bad"},
            {
                "type": "tool_result", "call_id": "c1", "name": "f", "content": "r",
                "is_error": True,
            },
            {"type": "tool_result", "call_id": "c2", "name": "g", "content": "r"},
        ],
        "protocol": "openai_responses",
        "provider_payload": {"id": "resp_1"},
    }


def error_path(data):
    with pytest.raises(MessageFormatError) as info:
        Message.from_dict(data)
    return info.value.path


class TestRoundTrip:
    def test_every_part_kind_round_trips_through_json(self):
        message = full_message()
        data = message.to_dict()
        assert data["format"] == "yian.message.v1"
        assert Message.from_dict(data) == message
        assert Message.from_dict(json.loads(json.dumps(data))) == message

    def test_canonical_dict_round_trips(self):
        data = canonical()
        assert Message.from_dict(data).to_dict() == data

    def test_defaults_are_omitted(self):
        data = Message("user", [TextPart("hi"), ToolResultPart("c", "f", "r")]).to_dict()
        assert data == {
            "format": MESSAGE_FORMAT,
            "role": "user",
            "parts": [
                {"type": "text", "text": "hi"},
                {"type": "tool_result", "call_id": "c", "name": "f", "content": "r"},
            ],
        }

    def test_protocol_round_trips(self):
        message = Message("assistant", (TextPart("x"),), protocol="catalog:minimax")
        assert message.to_dict()["protocol"] == "catalog:minimax"
        assert Message.from_dict(message.to_dict()).protocol == "catalog:minimax"


class TestStrictDecode:
    def test_unknown_top_level_key(self):
        data = canonical()
        data["extra"] = 1
        assert error_path(data) == "extra"

    def test_unknown_part_key(self):
        data = canonical()
        data["parts"][1]["bold"] = True
        assert error_path(data) == "parts[1].bold"

    def test_unknown_part_type(self):
        data = canonical()
        data["parts"][2]["type"] = "video"
        assert error_path(data) == "parts[2].type"

    def test_missing_format(self):
        data = canonical()
        del data["format"]
        assert error_path(data) == "format"

    def test_old_format(self):
        data = canonical()
        data["format"] = "yian.message.v0"
        assert error_path(data) == "format"

    def test_media_with_data_and_url(self):
        data = canonical()
        data["parts"][2]["data"] = "AAAA"
        assert error_path(data) == "parts[2]"

    def test_media_with_neither(self):
        data = canonical()
        del data["parts"][2]["url"]
        assert error_path(data) == "parts[2]"

    def test_tool_call_without_arguments(self):
        data = canonical()
        del data["parts"][5]["raw_arguments"]
        assert error_path(data) == "parts[5].arguments"

    def test_payload_with_tuple(self):
        data = canonical()
        data["parts"][0]["provider_payload"] = {"a": [1, (2, 3)]}
        assert error_path(data) == "parts[0].provider_payload.a[1]"

    def test_payload_with_nan(self):
        data = canonical()
        data["provider_payload"] = {"x": math.nan}
        assert error_path(data) == "provider_payload.x"

    def test_payload_with_non_str_key(self):
        data = canonical()
        data["parts"][1]["provider_payload"] = {"ok": {1: "v"}}
        assert error_path(data) == "parts[1].provider_payload.ok"

    def test_arguments_with_infinity(self):
        data = canonical()
        data["parts"][4]["arguments"] = {"a": [1, math.inf]}
        assert error_path(data) == "parts[4].arguments.a[1]"

    def test_unknown_role(self):
        data = canonical()
        data["role"] = "bot"
        assert error_path(data) == "role"

    def test_wrong_type(self):
        data = canonical()
        data["parts"][1]["text"] = 3
        assert error_path(data) == "parts[1].text"

    def test_bool_is_not_str(self):
        data = canonical()
        data["parts"][6]["is_error"] = "yes"
        assert error_path(data) == "parts[6].is_error"

    def test_missing_part_field(self):
        data = canonical()
        del data["parts"][4]["id"]
        assert error_path(data) == "parts[4].id"

    def test_parts_not_a_list(self):
        data = canonical()
        data["parts"] = {"type": "text"}
        assert error_path(data) == "parts"

    def test_not_a_mapping(self):
        assert error_path([]) == ""

    def test_message_names_the_path(self):
        data = canonical()
        data["parts"][4]["arguments"] = {"a": (1,)}
        with pytest.raises(MessageFormatError, match=r"^parts\[4\]\.arguments\.a: "):
            Message.from_dict(data)

    def test_is_a_value_error(self):
        assert issubclass(MessageFormatError, ValueError)


class TestConstruction:
    def test_invalid_parts_raise(self):
        with pytest.raises(MessageFormatError):
            MediaPart("image/png", data="a", url="https://e.com")
        with pytest.raises(MessageFormatError):
            ToolCallPart("c", "f", None)
        with pytest.raises(MessageFormatError):
            TextPart(3)
        with pytest.raises(MessageFormatError):
            TextPart("x", provider_payload={"a": (1,)})
        with pytest.raises(MessageFormatError):
            ToolCallPart("c", "f", [1, 2])
        with pytest.raises(MessageFormatError):
            Message("bot", ())
        with pytest.raises(MessageFormatError):
            Message("user", ("text",))
        with pytest.raises(MessageFormatError):
            ToolSpec("f", "d", {"a": math.nan})

    def test_payload_is_copied_on_construction(self):
        payload = {"a": [1, {"b": 2}]}
        part = TextPart("x", provider_payload=payload)
        payload["a"][1]["b"] = 3
        payload["c"] = 4
        assert part.provider_payload == {"a": [1, {"b": 2}]}

    def test_arguments_and_parameters_are_copied(self):
        arguments = {"q": ["x"]}
        call = ToolCallPart("c", "f", arguments)
        arguments["q"].append("y")
        assert call.arguments == {"q": ["x"]}
        parameters = {"type": "object", "properties": {}}
        spec = ToolSpec("f", "d", parameters)
        parameters["properties"]["x"] = {}
        assert spec.parameters == {"type": "object", "properties": {}}

    def test_to_dict_output_is_a_copy(self):
        message = full_message()
        data = message.to_dict()
        data["provider_payload"]["meta"]["k"] = "changed"
        data["parts"][0]["provider_payload"]["n"].append(9)
        data["parts"][5]["arguments"]["deep"]["k"].clear()
        assert message == full_message()

    def test_parts_list_is_stored_as_tuple(self):
        message = Message("user", [TextPart("a")])
        assert message.parts == (TextPart("a"),)
        assert isinstance(message.parts, tuple)

    def test_messages_are_unhashable(self):
        with pytest.raises(TypeError):
            hash(full_message())
        with pytest.raises(TypeError):
            hash(TextPart("x"))
        with pytest.raises(TypeError):
            hash(ToolSpec("f", "d", {}))

    def test_compare_by_value(self):
        assert full_message() == full_message()
        assert TextPart("a") != TextPart("b")

    def test_deep_payload_raises_without_recursion_error(self):
        deep = {}
        node = deep
        for _ in range(10_000):
            child = {}
            node["k"] = child
            node = child
        with pytest.raises(MessageFormatError) as info:
            TextPart("x", provider_payload=deep)
        assert not isinstance(info.value.__context__, RecursionError)

    def test_nesting_limit(self, set_defaults):
        set_defaults(MAX_VALUE_NESTING=3)
        TextPart("x", provider_payload={"a": {"b": {"c": 1}}})
        with pytest.raises(MessageFormatError) as info:
            TextPart("x", provider_payload={"a": {"b": {"c": []}}})
        assert info.value.path == "provider_payload.a.b.c"
        ToolCallPart("c", "f", {"a": [[1]]})
        with pytest.raises(MessageFormatError):
            ToolCallPart("c", "f", {"a": [[[1]]]})
        with pytest.raises(MessageFormatError):
            ToolSpec("f", "d", {"a": [[{}]]})

    def test_int_digit_limit(self, set_defaults):
        set_defaults(MAX_INT_DIGITS=5)
        TextPart("x", provider_payload={"n": 99999, "m": -99999})
        with pytest.raises(MessageFormatError) as info:
            TextPart("x", provider_payload={"n": [100000]})
        assert info.value.path == "provider_payload.n[0]"
        with pytest.raises(MessageFormatError):
            ToolCallPart("c", "f", {"n": -100000})

    def test_key_paths_quote_odd_keys(self):
        with pytest.raises(MessageFormatError) as info:
            TextPart("x", provider_payload={"a b": (1,)})
        assert info.value.path == 'provider_payload["a b"]'


class TestHelpers:
    def test_text_joins_text_parts_in_order(self):
        assert full_message().text() == "hello world"
        assert Message("user", ()).text() == ""

    def test_tool_calls(self):
        calls = full_message().tool_calls()
        assert [call.id for call in calls] == ["call_1", "call_2", "call_3"]


class TestOtherTypes:
    def test_request_defaults(self):
        message = Message("user", (TextPart("hi"),))
        request = Request([message])
        assert request.messages == (message,)
        assert request.tools == ()
        assert request.tool_choice is None
        assert request.stream is True

    def test_request_with_tools(self):
        spec = ToolSpec("f", "does f", {"type": "object"}, strict=True)
        request = Request((), tools=[spec], tool_choice=NamedTool("f"))
        assert request.tools == (spec,)
        assert request.tool_choice == NamedTool("f")

    def test_response_and_usage(self):
        usage = Usage(1, 2, 3, cache_read_tokens=1)
        response = Response(Message("assistant", ()), usage=usage, finish_reason="stop")
        assert response.usage.reasoning_tokens is None
        assert response.response_id is None
        assert hash(usage) == hash(Usage(1, 2, 3, cache_read_tokens=1))
