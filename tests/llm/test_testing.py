"""Test helpers: cassette replay, the redacting recorder, find_secrets and ScriptedModel."""

import json
import os
from pathlib import Path
import subprocess
import sys
from urllib.parse import parse_qsl, urlsplit
import zlib

import httpx2
import pytest

import yian.llm
from yian.llm.auth import BearerAuth, HeaderKeyAuth
from yian.llm.events import (
    EventStream,
    Finish,
    PartDone,
    ReasoningDelta,
    TextDelta,
    ToolCallDelta,
    UsageReport,
    collect,
)
from yian.llm.executor import Executor
from yian.llm.messages import (
    Message,
    ReasoningPart,
    Request,
    Response,
    TextPart,
    ToolCallPart,
    Usage,
)
from yian.llm.route import Route
from yian.llm.testing import (
    REDACTED,
    CassetteTransport,
    RecordingTransport,
    ScriptedModel,
    find_secrets,
    load_cassette,
)
from yian.llm.transport import build_client

CASSETTE = Path(__file__).parent / "cassettes" / "example_sse.json"
# Fake keys shaped like real ones, so find_secrets and the redaction both see them.
KEY = "sk-test-0123456789abcdefghijKLMN"
OTHER_KEY = "sk-other-9876543210zyxwvutsrqPONM"
BASE = "https://api.example.test/v1"


def example_route(auth=None, **kwargs):
    kwargs.setdefault("query", {"api-version": "2026-01-01"})
    return Route("example", BASE, "responses", auth or BearerAuth(KEY), **kwargs)


def example_body(text):
    return {"model": "m", "input": [{"role": "user", "content": text}]}


def one_interaction(request, response=None):
    """Build a cassette dict holding one interaction."""
    return {
        "format": "yian.cassette.v1",
        "source": "synthetic",
        "description": "test",
        "interactions": [
            {"request": request, "response": response or {"status": 200, "json": {"ok": True}}}
        ],
    }


class Body(httpx2.SyncByteStream):
    """A response body that yields byte chunks and records whether it was closed."""

    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.closed = False

    def __iter__(self):
        yield from self.chunks

    def close(self):
        self.closed = True


def gzip_chunks(*texts):
    """Gzip ``texts`` as one stream, flushing after each so each text is one raw chunk."""
    compressor = zlib.compressobj(wbits=31)
    chunks = [
        compressor.compress(text.encode()) + compressor.flush(zlib.Z_SYNC_FLUSH) for text in texts
    ]
    chunks[-1] += compressor.flush()
    return chunks


def query_of(url):
    return dict(parse_qsl(urlsplit(url).query, keep_blank_values=True))


# --- replay -----------------------------------------------------------------------------


def test_cassette_replays_interactions_in_order_through_the_executor():
    transport = CassetteTransport(CASSETTE)
    executor = Executor(build_client(transport=transport))
    replies = []
    for text in ("first", "second"):
        with executor.open(example_route(), example_body(text), framing="sse") as frames:
            replies.append([frame.data for frame in frames])

    assert replies == [['{"text": "one"}', "[DONE]"], ['{"text": "two"}', "[DONE]"]]
    sent = [json.loads(request.content)["input"][0]["content"] for request in transport.requests]
    assert sent == ["first", "second"]
    transport.assert_done()


def test_assert_done_fails_naming_the_unused_count():
    transport = CassetteTransport(CASSETTE)
    executor = Executor(build_client(transport=transport))
    executor.open(example_route(), example_body("first"), framing="sse").close()

    with pytest.raises(AssertionError, match="1 of 2"):
        transport.assert_done()


def test_an_extra_request_fails():
    transport = CassetteTransport(one_interaction({"method": "GET", "url": "https://h.test/a"}))
    client = httpx2.Client(transport=transport)
    client.get("https://h.test/a")

    with pytest.raises(AssertionError, match="unexpected request 2"):
        client.get("https://h.test/a")


def test_load_cassette_names_the_problem(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"format": "other"}))
    with pytest.raises(ValueError, match="format"):
        load_cassette(path)

    bad_response = one_interaction({"method": "GET", "url": "https://h.test/"}, {"status": 200})
    path.write_text(json.dumps(bad_response))
    with pytest.raises(ValueError, match=r"interactions\[0\]\.response"):
        load_cassette(path)

    assert load_cassette(CASSETTE)["source"] == "synthetic"


def test_replay_serves_json_and_text_responses():
    for response, expected in (
        ({"status": 201, "json": {"a": 1}}, b'{"a": 1}'),
        ({"status": 500, "headers": {"x-y": "z"}, "text": "oops"}, b"oops"),
    ):
        request = {"method": "GET", "url": "https://h.test/"}
        transport = CassetteTransport(one_interaction(request, response))
        reply = httpx2.Client(transport=transport).get("https://h.test/")
        assert reply.status_code == response["status"]
        assert reply.content == expected


# --- mismatches -------------------------------------------------------------------------


def test_a_body_mismatch_names_the_json_path():
    transport = CassetteTransport(
        one_interaction(
            {"method": "POST", "url": "https://h.test/p", "json": example_body("first")}
        )
    )
    client = httpx2.Client(transport=transport)

    with pytest.raises(AssertionError, match=r"input\[0\]\.content") as caught:
        client.post("https://h.test/p", json=example_body("changed"))
    assert "changed" not in str(caught.value)


def test_redacted_values_match_any_value_but_only_when_present():
    recorded = {
        "method": "POST",
        "url": "https://h.test/p?key=<redacted>&v=1",
        "headers": {"x-api-key": REDACTED},
        "json": {"token": REDACTED, "n": 1},
    }

    def send(url, headers, body):
        transport = CassetteTransport(one_interaction(recorded))
        httpx2.Client(transport=transport).post(url, headers=headers, json=body)

    send("https://h.test/p?key=anything&v=1", {"x-api-key": "whatever"}, {"token": "abc", "n": 1})
    with pytest.raises(AssertionError, match="x-api-key"):
        send("https://h.test/p?key=anything&v=1", {}, {"token": "abc", "n": 1})
    with pytest.raises(AssertionError, match="query parameter 'key'"):
        send("https://h.test/p?v=1", {"x-api-key": "whatever"}, {"token": "abc", "n": 1})
    with pytest.raises(AssertionError, match="token"):
        send("https://h.test/p?key=anything&v=1", {"x-api-key": "whatever"}, {"n": 1})


def test_query_parameters_match_as_a_multimap():
    recorded = {"method": "GET", "url": "https://h.test/p?a=1&a=2&b=3"}

    def get(url):
        transport = CassetteTransport(one_interaction(recorded))
        httpx2.Client(transport=transport).get(url)

    get("https://h.test/p?b=3&a=2&a=1")
    with pytest.raises(AssertionError, match="'a'"):
        get("https://h.test/p?a=1&b=3")


def test_ignored_headers_may_differ():
    recorded = {
        "method": "GET",
        "url": "https://h.test/p",
        "headers": {"user-agent": "someone-else/1.0", "content-length": "999", "accept": "*/*"},
    }
    transport = CassetteTransport(one_interaction(recorded))
    httpx2.Client(transport=transport).get("https://h.test/p")
    transport.assert_done()


def test_a_mismatch_never_shows_header_or_query_values():
    recorded = {
        "method": "POST",
        "url": "https://h.test/p?key=recorded-query-value",
        "headers": {"x-api-key": "recorded-header-value"},
        "json": {},
    }
    transport = CassetteTransport(one_interaction(recorded))
    client = httpx2.Client(transport=transport)

    with pytest.raises(AssertionError) as caught:
        client.post(f"https://h.test/p?key={KEY}", headers={"x-api-key": KEY}, json={})
    message = str(caught.value)
    assert "x-api-key" in message and "'key'" in message
    for value in (KEY, "recorded-query-value", "recorded-header-value"):
        assert value not in message


def test_a_method_or_url_mismatch_fails():
    recorded = {"method": "POST", "url": "https://h.test/p"}
    for call in (
        lambda client: client.get("https://h.test/p"),
        lambda client: client.post("https://other.test/p"),
        lambda client: client.post("https://h.test/q"),
    ):
        with pytest.raises(AssertionError):
            call(httpx2.Client(transport=CassetteTransport(one_interaction(recorded))))


# --- recording --------------------------------------------------------------------------


def record_gzip_exchange():
    """Record one gzip SSE exchange whose key is in the header, query, body and reply."""
    reply_texts = ('data: {"echo": "' + KEY[:12], KEY[12:] + '"}\n\ndata: [DONE]\n\n')

    def handler(request):
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream", "content-encoding": "gzip"},
            stream=Body(gzip_chunks(*reply_texts)),
        )

    recorder = RecordingTransport(httpx2.MockTransport(handler), secrets=[KEY])
    executor = Executor(build_client(transport=recorder))
    route = example_route(HeaderKeyAuth("x-api-key", KEY), query={"key": KEY})
    with executor.open(route, {"model": "m", "note": KEY}, framing="sse") as frames:
        assert [frame.data for frame in frames] == [f'{{"echo": "{KEY}"}}', "[DONE]"]
    return recorder.cassette(description="a recorded exchange")


def test_recording_redacts_the_key_everywhere_and_decodes_the_body():
    cassette = record_gzip_exchange()
    assert cassette["format"] == "yian.cassette.v1"
    assert cassette["source"] == "recorded"
    assert cassette["description"] == "a recorded exchange"
    [interaction] = cassette["interactions"]
    request, response = interaction["request"], interaction["response"]

    assert request["method"] == "POST"
    assert request["headers"]["x-api-key"] == REDACTED
    assert query_of(request["url"])["key"] == REDACTED
    assert request["json"] == {"model": "m", "note": REDACTED}
    assert "user-agent" not in request["headers"]
    assert response["chunks"] == ['data: {"echo": "<redacted>"}\n\n', "data: [DONE]\n\n"]
    assert "content-encoding" not in response["headers"]
    assert response["headers"]["content-type"] == "text/event-stream"
    assert KEY not in json.dumps(cassette)
    assert find_secrets(json.dumps(cassette)) == []
    CassetteTransport(json.loads(json.dumps(cassette)))


def test_a_recorded_cassette_replays_with_a_different_key():
    cassette = record_gzip_exchange()
    transport = CassetteTransport(cassette)
    executor = Executor(build_client(transport=transport))
    route = example_route(HeaderKeyAuth("x-api-key", OTHER_KEY), query={"key": OTHER_KEY})

    with executor.open(route, {"model": "m", "note": OTHER_KEY}, framing="sse") as frames:
        assert [frame.data for frame in frames] == ['{"echo": "<redacted>"}', "[DONE]"]
    transport.assert_done()


def test_closing_a_stream_early_records_what_was_read():
    body = Body([b"data: one\n\n", b"data: two\n\n"])
    inner = httpx2.MockTransport(lambda request: httpx2.Response(200, stream=body))
    recorder = RecordingTransport(inner)
    executor = Executor(build_client(transport=recorder))

    frames = executor.open(example_route(), example_body("first"), framing="sse")
    assert next(frames).data == "one"
    frames.close()

    assert body.closed
    [interaction] = recorder.cassette(description="half")["interactions"]
    assert interaction["response"]["chunks"] == ["data: one\n\n"]
    assert interaction["request"]["headers"]["authorization"] == REDACTED


# --- find_secrets -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "key sk-proj-abcdefghijklmnop1234",
        "key sk-ant-api03-abcdefghijklmnop",
        "AIzaSyA1234567890abcdefghijklmnopqrstu",
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N",
        "Authorization: Bearer abcdefghijklmnopqrstuvwxyz",
    ],
)
def test_find_secrets_flags_each_pattern(text):
    assert find_secrets(text)


def test_find_secrets_names_kinds_not_values():
    found = find_secrets(
        "Bearer abcdefghijklmnopqrstuvwxyz and AIzaSyA1234567890abcdefghijklmnopqrstu"
    )
    assert found
    assert all("abcdefghijklmnop" not in kind and "AIza" not in kind for kind in found)


@pytest.mark.parametrize(
    "text",
    [
        "Authorization: Bearer <redacted>",
        "task-0123456789abcdefgh",
        "risk-0123456789abcdefghijk",
        "no secrets here",
    ],
)
def test_find_secrets_ignores_redacted_and_embedded_text(text):
    assert find_secrets(text) == []


# --- ScriptedModel ----------------------------------------------------------------------


REQUEST = Request((Message("user", [TextPart("hi")]),))


def full_response():
    message = Message(
        "assistant",
        [
            ReasoningPart("thinking", provider_payload={"signature": "s"}),
            TextPart("hello"),
            ToolCallPart("c1", "search", {"q": "x"}),
            ToolCallPart("c2", "broken", None, raw_arguments="{not json"),
        ],
        protocol="openai_responses",
        provider_payload={"id": "msg_1"},
    )
    return Response(message, usage=Usage(3, 4, 7), finish_reason="tool_calls", response_id="r1")


def test_scripted_model_replays_a_response_as_events():
    reply = full_response()
    model = ScriptedModel([reply])

    with model.stream(REQUEST) as stream:
        events = list(stream)

    assert isinstance(stream, EventStream)
    assert events[:2] == [ReasoningDelta(0, "thinking"), PartDone(0, reply.message.parts[0])]
    assert events[2:4] == [TextDelta(1, "hello"), PartDone(1, reply.message.parts[1])]
    assert events[4] == ToolCallDelta(2, "c1", "search", '{"q": "x"}')
    assert events[6] == ToolCallDelta(3, "c2", "broken", "{not json")
    assert events[-2:] == [
        UsageReport(Usage(3, 4, 7)),
        Finish("tool_calls", "r1", protocol="openai_responses", provider_payload={"id": "msg_1"}),
    ]
    assert collect(events) == reply


def test_scripted_model_generate_collects_the_same_events():
    reply = full_response()
    model = ScriptedModel([reply])

    assert model.generate(REQUEST) == reply
    assert model.calls == ["generate"]
    assert model.requests == [REQUEST]


def test_scripted_model_replays_a_raw_event_list_as_given():
    events = [TextDelta(0, "a"), TextDelta(0, "b"), Finish("stop")]
    model = ScriptedModel([events])

    assert list(model.stream(REQUEST)) == events


class Cut(Exception):
    """A scripted mid-stream failure."""


def test_scripted_model_raises_a_bare_exception_reply():
    model = ScriptedModel([Cut("down")])
    with pytest.raises(Cut):
        model.stream(REQUEST)
    assert model.calls == ["stream"]


def test_scripted_model_raises_an_exception_mid_stream():
    model = ScriptedModel([[TextDelta(0, "par"), Cut("cut")]])
    stream = model.stream(REQUEST)

    assert next(stream) == TextDelta(0, "par")
    with pytest.raises(Cut):
        next(stream)
    assert model.closed == 1


def test_scripted_model_records_requests_calls_and_closed_streams():
    second = Request((Message("user", [TextPart("again")]),))
    replies = [full_response(), full_response(), full_response()]
    model = ScriptedModel(replies, provider="p", model="m")
    assert (model.provider, model.model) == ("p", "m")

    stream = model.stream(REQUEST)
    assert model.closed == 0
    stream.close()
    stream.close()
    model.generate(second)
    list(model.stream(REQUEST))

    assert model.requests == [REQUEST, second, REQUEST]
    assert model.calls == ["stream", "generate", "stream"]
    assert model.closed == 2


def test_scripted_model_fails_when_replies_run_out():
    model = ScriptedModel([])
    with pytest.raises(AssertionError, match="no reply"):
        model.stream(REQUEST)
    with pytest.raises(AssertionError, match="no reply"):
        model.generate(REQUEST)


# --- exports ----------------------------------------------------------------------------


def test_llm_exports():
    assert yian.llm.__all__
    for name in yian.llm.__all__:
        assert hasattr(yian.llm, name), name
    expected = ("Message", "collect", "ProviderError", "Executor", "Route", "iter_sse")
    for name in expected:
        assert name in yian.llm.__all__
    assert "testing" not in yian.llm.__all__

    code = "import sys, yian.llm; print('yian.llm.testing' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True,
        env={**os.environ, "PYTHONPATH": str(Path(yian.llm.__file__).parents[2])},
    )
    assert result.stdout.strip() == "False"
