"""SSE and NDJSON framing over byte chunks."""

import time

import pytest

from yian.llm.errors import ErrorKind, ProviderError
from yian.llm.framing import SSEEvent, iter_ndjson, iter_sse


def sse(*chunks):
    return list(iter_sse(chunks))


class TestSSE:
    def test_event_split_at_every_offset(self):
        raw = 'event: delta\nid: 7\ndata: {"a": "é"}\n\n'.encode()
        expected = [SSEEvent(data='{"a": "é"}', event="delta", id="7")]
        for cut in range(len(raw) + 1):
            assert sse(raw[:cut], raw[cut:]) == expected, cut

    def test_one_byte_chunks(self):
        raw = "data: one\r\ndata: two\r\n\r\ndata: ü\n\n".encode()
        assert sse(*(raw[i:i + 1] for i in range(len(raw)))) == [
            SSEEvent(data="one\ntwo"), SSEEvent(data="ü"),
        ]

    def test_multi_line_data(self):
        assert sse(b"data: a\ndata:b\ndata:  c\n\n") == [SSEEvent(data="a\nb\n c")]

    def test_comments_are_ignored(self):
        assert sse(b": keep-alive\n\n: ping\ndata: x\n: mid\n\n") == [SSEEvent(data="x")]

    def test_event_and_id_fields(self):
        events = sse(b"event: start\nid: 1\ndata: a\n\ndata: b\n\n")
        assert events[0] == SSEEvent(data="a", event="start", id="1")
        # The event type resets after a dispatch; the last event id carries over (WHATWG).
        assert events[1] == SSEEvent(data="b", event=None, id="1")

    def test_retry_and_unknown_fields_ignored(self):
        assert sse(b"retry: 100\nfoo: bar\ndata: x\nnocolon\n\n") == [SSEEvent(data="x")]

    def test_field_without_value(self):
        assert sse(b"data\n\ndata:\n\n") == [SSEEvent(data=""), SSEEvent(data="")]

    def test_crlf_split_across_chunks(self):
        assert sse(b"data: a\r", b"\ndata: b\r", b"\n\r", b"\n") == [SSEEvent(data="a\nb")]

    def test_lf_alone_after_split_crlf_is_a_blank_line(self):
        assert sse(b"data: a\r", b"\n", b"\n") == [SSEEvent(data="a")]

    def test_bare_cr(self):
        assert sse(b"data: a\rdata: b\r\rdata: c\r\r") == [
            SSEEvent(data="a\nb"), SSEEvent(data="c"),
        ]

    def test_bom_is_stripped(self):
        assert sse(b"\xef\xbb", b"\xbfdata: x\n\n") == [SSEEvent(data="x")]

    def test_only_one_bom_is_stripped(self):
        assert sse(b"\xef\xbb\xbf\xef\xbb\xbfdata: x\n\n") == []

    def test_utf8_character_split_across_chunks(self):
        raw = "data: 日本\n\n".encode()
        cut = raw.index("本".encode()) + 1
        assert sse(raw[:cut], raw[cut:]) == [SSEEvent(data="日本")]

    def test_invalid_utf8_is_replaced(self):
        assert sse(b"data: \xff\n\n") == [SSEEvent(data="�")]

    def test_event_without_data_is_not_dispatched(self):
        assert sse(b"event: ping\nid: 3\n\ndata: x\n\n") == [SSEEvent(data="x", id="3")]

    def test_unterminated_event_at_eof_is_dropped(self):
        assert sse(b"data: a\n\ndata: b\n") == [SSEEvent(data="a")]
        assert sse(b"data: a\n\ndata: b") == [SSEEvent(data="a")]

    def test_megabyte_line_in_small_chunks_is_linear(self):
        raw = b"data: " + b"x" * 1_000_000 + b"\n\n"
        chunks = [raw[i:i + 64] for i in range(0, len(raw), 64)]
        start = time.perf_counter()
        events = list(iter_sse(chunks))
        assert time.perf_counter() - start < 2.0
        assert len(events) == 1 and len(events[0].data) == 1_000_000

    def test_is_lazy(self):
        def chunks():
            yield b"data: a\n\n"
            raise AssertionError("read past the first event")

        assert next(iter_sse(chunks())) == SSEEvent(data="a")


class TestNDJSON:
    def test_split_lines(self):
        raw = b'{"a": 1}\n{"b": [2, 3]}\n'
        for cut in range(len(raw) + 1):
            assert list(iter_ndjson([raw[:cut], raw[cut:]])) == [{"a": 1}, {"b": [2, 3]}], cut

    def test_blank_lines_are_skipped(self):
        assert list(iter_ndjson([b"\n1\n\n  \r\n2\r\n\n"])) == [1, 2]

    def test_final_line_without_newline(self):
        assert list(iter_ndjson([b'{"a": 1}\n{"done": tr', b"ue}"])) == [
            {"a": 1}, {"done": True},
        ]

    def test_invalid_json_raises_protocol_without_line(self):
        events = iter_ndjson([b'{"a": 1}\n{"secret": sk-live-abc\n'])
        assert next(events) == {"a": 1}
        with pytest.raises(ProviderError) as info:
            next(events)
        err = info.value
        assert err.kind is ErrorKind.PROTOCOL
        assert "invalid JSON line" in str(err)
        assert "sk-live-abc" not in str(err)
        assert err.__cause__ is None and err.__context__ is None

    def test_invalid_utf8_raises_protocol(self):
        with pytest.raises(ProviderError) as info:
            list(iter_ndjson([b'"\xff"\n']))
        assert info.value.kind is ErrorKind.PROTOCOL
