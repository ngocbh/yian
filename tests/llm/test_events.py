"""Stream events: argument parsing, collecting a reply, and the owning stream."""

import pytest

from yian.llm.errors import ErrorKind, ProviderError
from yian.llm.events import (
    EventStream,
    Finish,
    PartDone,
    ReasoningDelta,
    TextDelta,
    ToolCallDelta,
    UsageReport,
    collect,
    parse_arguments,
)
from yian.llm.messages import Message, ReasoningPart, TextPart, ToolCallPart, Usage


class TestParseArguments:
    def test_empty_text_is_an_empty_object(self):
        assert parse_arguments("") == ({}, None)

    def test_object(self):
        text = '{"a": [1, 2.5, null, true]}'
        assert parse_arguments(text) == ({"a": [1, 2.5, None, True]}, None)

    @pytest.mark.parametrize(
        "text",
        ["[1, 2]", "{bad", '{"a": NaN}', '{"a": Infinity}', '{"a": -Infinity}', '"s"', "1e400"],
    )
    def test_rejected(self, text):
        assert parse_arguments(text) == (None, text)

    def test_int_digit_limit(self, set_defaults):
        set_defaults(MAX_INT_DIGITS=5)
        assert parse_arguments('{"n": -99999}') == ({"n": -99999}, None)
        assert parse_arguments('{"n": 100000}') == (None, '{"n": 100000}')

    def test_nesting_limit(self, set_defaults):
        set_defaults(MAX_VALUE_NESTING=3)
        assert parse_arguments('{"a": {"b": [1]}}') == ({"a": {"b": [1]}}, None)
        assert parse_arguments('{"a": {"b": [[1]]}}') == (None, '{"a": {"b": [[1]]}}')

    @pytest.mark.parametrize("text", ['{"a": "\\ud800"}', '{"\\udfff": 1}', '{"a": "\ud800"}'])
    def test_unpaired_surrogate_is_a_failure(self, text):
        assert parse_arguments(text) == (None, text)

    def test_escaped_surrogate_pair_parses(self):
        assert parse_arguments('{"a": "\\ud83d\\ude00"}') == ({"a": "\U0001f600"}, None)

    def test_deep_brackets_without_recursion_error(self):
        text = "[" * 100_000
        assert parse_arguments(text) == (None, text)
        text = '{"a":' * 100_000
        assert parse_arguments(text) == (None, text)


class TestCollect:
    def test_interleaved_text_and_tool_call(self):
        response = collect([
            TextDelta(0, "Hel"),
            ToolCallDelta(1, "call_1", None, '{"q": '),
            TextDelta(0, "lo"),
            ToolCallDelta(1, None, "search", '"x"}'),
            ToolCallDelta(1, "ignored", "ignored", ""),
            Finish("tool_calls"),
        ])
        assert response.message == Message(
            "assistant",
            (TextPart("Hello"), ToolCallPart("call_1", "search", {"q": "x"})),
        )
        assert response.finish_reason == "tool_calls"

    def test_parts_ordered_by_index(self):
        response = collect([ReasoningDelta(2, "b"), TextDelta(0, "a"), ReasoningDelta(2, "c")])
        assert response.message.parts == (TextPart("a"), ReasoningPart("bc"))

    def test_part_done_wins(self):
        final = ReasoningPart("full", provider_payload={"signature": "s"})
        response = collect([ReasoningDelta(0, "fu"), PartDone(0, final), ReasoningDelta(0, "ll")])
        assert response.message.parts == (final,)

    def test_part_done_supplies_tool_call_id(self):
        call = ToolCallPart("assigned", "f", {})
        response = collect([ToolCallDelta(0, None, "f", "{}"), PartDone(0, call)])
        assert response.message.tool_calls() == (call,)

    def test_unparsable_arguments_keep_raw_text(self):
        response = collect([ToolCallDelta(0, "c", "f", "{bad")])
        assert response.message.parts == (ToolCallPart("c", "f", None, raw_arguments="{bad"),)

    def test_empty_arguments_are_an_empty_object(self):
        response = collect([ToolCallDelta(0, "c", "f", "")])
        assert response.message.parts == (ToolCallPart("c", "f", {}),)

    def test_last_usage_and_finish_win(self):
        response = collect([
            UsageReport(Usage(1, 1, 2)),
            Finish("length", response_id="r1"),
            TextDelta(0, "x"),
            UsageReport(Usage(3, 4, 7)),
            Finish("stop", response_id="r2"),
        ])
        assert response.usage == Usage(3, 4, 7)
        assert response.finish_reason == "stop"
        assert response.response_id == "r2"

    def test_finish_sets_protocol_and_payload(self):
        response = collect([
            TextDelta(0, "x"),
            Finish("stop", protocol="openai_chat", provider_payload={"system_fingerprint": "fp"}),
        ])
        assert response.message.protocol == "openai_chat"
        assert response.message.provider_payload == {"system_fingerprint": "fp"}

    def test_mixed_delta_kinds_raise(self):
        with pytest.raises(ValueError, match="index 0"):
            collect([TextDelta(0, "a"), ReasoningDelta(0, "b")])
        with pytest.raises(ValueError):
            collect([ToolCallDelta(0, "c", "f", ""), TextDelta(0, "a")])

    def test_tool_call_without_id_raises(self):
        with pytest.raises(ProviderError) as info:
            collect([ToolCallDelta(3, None, "f", "{}")])
        assert info.value.kind is ErrorKind.PROTOCOL
        assert str(info.value) == "tool call 3 has no id"

    def test_tool_call_without_name_raises(self):
        with pytest.raises(ProviderError) as info:
            collect([ToolCallDelta(0, "c", None, "{}")])
        assert info.value.kind is ErrorKind.PROTOCOL

    def test_non_event_raises_and_closes_the_input(self):
        state = []

        def events():
            try:
                yield TextDelta(0, "a")
                yield "not an event"
                yield TextDelta(0, "b")
            finally:
                state.append("closed")

        # Held here so garbage collection cannot be what closes it.
        stream = events()
        with pytest.raises(TypeError, match="not a stream event: str"):
            collect(stream)
        assert state == ["closed"]

    def test_failure_after_exhaustion_closes_the_input(self):
        closes = []

        class Events:
            def __init__(self):
                self._items = iter([ToolCallDelta(0, None, "f", "{}")])

            def __iter__(self):
                return self

            def __next__(self):
                return next(self._items)

            def close(self):
                closes.append(1)

        with pytest.raises(ProviderError):
            collect(Events())
        assert closes == [1]

    def test_empty_stream(self):
        response = collect([])
        assert response.message == Message("assistant", ())
        assert response.usage is None
        assert response.finish_reason is None


class TestEventStream:
    def test_close_is_idempotent(self):
        closes = []
        stream = EventStream(iter([TextDelta(0, "a")]), close=lambda: closes.append(1))
        stream.close()
        stream.close()
        assert closes == [1]
        assert list(stream) == []

    def test_closes_when_exhausted(self):
        closes = []
        events = iter([TextDelta(0, "a"), TextDelta(0, "b")])
        stream = EventStream(events, close=lambda: closes.append(1))
        assert list(stream) == [TextDelta(0, "a"), TextDelta(0, "b")]
        assert closes == [1]
        stream.close()
        assert closes == [1]

    def test_closes_on_context_exit(self):
        closes = []
        events = iter([TextDelta(0, "a"), TextDelta(0, "b")])
        with EventStream(events, close=lambda: closes.append(1)) as stream:
            assert next(stream) == TextDelta(0, "a")
        assert closes == [1]

    def test_closes_the_inner_generator(self):
        state = []

        def events():
            try:
                yield TextDelta(0, "a")
                yield TextDelta(0, "b")
            finally:
                state.append("closed")

        stream = EventStream(events())
        next(stream)
        stream.close()
        assert state == ["closed"]

    def test_closes_when_the_inner_iterator_raises(self):
        closes = []

        def events():
            yield TextDelta(0, "a")
            raise ProviderError(ErrorKind.CONNECTION, "dropped")

        stream = EventStream(events(), close=lambda: closes.append(1))
        assert next(stream) == TextDelta(0, "a")
        with pytest.raises(ProviderError, match="dropped"):
            next(stream)
        assert closes == [1]
        assert list(stream) == []
        assert closes == [1]

    def test_collect_reads_an_event_stream(self):
        stream = EventStream(iter([TextDelta(0, "a"), Finish("stop")]))
        assert collect(stream).message.text() == "a"
