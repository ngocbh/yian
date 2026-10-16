"""Routes and the executor: request shape, retries before the first frame, error mapping."""

import itertools
import json

import httpx2
import pytest

from yian.llm.auth import BearerAuth, HeaderKeyAuth, NoAuth
from yian.llm.errors import ErrorKind, ProviderError
from yian.llm.executor import Executor
from yian.llm.framing import SSEEvent
from yian.llm.route import Route
from yian.llm.transport import USER_AGENT, build_client

JITTER = 0.4  # fixed random(): each backoff is multiplied by 1 - 0.25 * 0.4 = 0.9


class Body(httpx2.SyncByteStream):
    """A response body that yields byte chunks, raises exceptions in place, and records close."""

    def __init__(self, items):
        self.items = list(items)
        self.closed = False
        self.yielded = 0

    def __iter__(self):
        for item in self.items:
            if isinstance(item, BaseException):
                raise item
            self.yielded += 1
            yield item

    def close(self):
        self.closed = True


class Harness:
    """A mock transport answering with scripted responses, plus a log of requests and waits."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests = []
        self.log = []
        self.stops = 0

    def handler(self, request):
        self.requests.append(request)
        self.log.append("request")
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply() if callable(reply) else reply

    def sleep(self, seconds):
        self.log.append(seconds)

    def check_stopped(self):
        self.stops += 1

    def waits(self):
        """Return the total of each wait between requests (sleep slices summed)."""
        groups, current = [], None
        for entry in self.log:
            if entry == "request":
                if current is not None:
                    groups.append(current)
                current = None
            else:
                current = (current or 0.0) + entry
        if current is not None:
            groups.append(current)
        return groups

    def executor(self, *, check_stopped=None):
        client = build_client(transport=httpx2.MockTransport(self.handler))
        return Executor(
            client,
            check_stopped=check_stopped or self.check_stopped,
            sleep=self.sleep,
            random=lambda: JITTER,
        )


def route(**overrides):
    values = dict(
        provider="openai",
        base_url="https://api.example/v1",
        path="responses",
        auth=BearerAuth("sk-test-key-0123456789"),
    )
    values.update(overrides)
    return Route(**values)


def sse(*chunks, status=200, headers=None):
    return lambda: httpx2.Response(
        status,
        headers={"content-type": "text/event-stream", **(headers or {})},
        stream=Body([chunk.encode() if isinstance(chunk, str) else chunk for chunk in chunks]),
    )


def status(code, body="", headers=None):
    return lambda: httpx2.Response(code, headers=headers or {}, content=body.encode())


# Route.url()


@pytest.mark.parametrize("base", ["https://api.example/v1", "https://api.example/v1/"])
@pytest.mark.parametrize("path", ["chat/completions", "/chat/completions"])
def test_route_url_joins_with_one_slash(base, path):
    assert route(base_url=base, path=path).url() == "https://api.example/v1/chat/completions"


def test_route_copies_query_and_headers_read_only():
    query = {"api-version": "1"}
    headers = {"x-extra": "a"}
    r = route(query=query, headers=headers)
    query["api-version"] = "2"
    headers["x-extra"] = "b"
    assert dict(r.query) == {"api-version": "1"}
    assert dict(r.headers) == {"x-extra": "a"}
    with pytest.raises(TypeError):
        r.query["api-version"] = "3"


def test_route_query_is_encoded():
    h = Harness(sse("data: 1\n\n"))
    h.executor().open(route(query={"api-version": "a b&c=d"}), {}, framing="sse").close()
    (request,) = h.requests
    assert request.url.params["api-version"] == "a b&c=d"
    assert "a b" not in str(request.url) and "&c=d" not in str(request.url)
    assert request.url.path == "/v1/responses"


# Request shape


def test_request_shape():
    h = Harness(sse("data: 1\n\n"))
    body = {"model": "m", "input": "héllo ✓", "n": 1.5}
    h.executor().open(route(headers={"x-extra": "a"}), body, framing="sse").close()
    (request,) = h.requests
    assert request.method == "POST"
    assert request.content == json.dumps(body, ensure_ascii=False, allow_nan=False).encode()
    assert request.headers["content-type"] == "application/json"
    assert request.headers["accept"] == "text/event-stream"
    assert request.headers["authorization"] == "Bearer sk-test-key-0123456789"
    assert request.headers["x-extra"] == "a"
    assert request.headers["user-agent"] == USER_AGENT


def test_auth_beats_route_header_of_any_case():
    h = Harness(sse("data: 1\n\n"))
    h.executor().open(route(headers={"authorization": "x"}), {}, framing="sse").close()
    (request,) = h.requests
    assert request.headers.get_list("authorization") == ["Bearer sk-test-key-0123456789"]
    assert [name for name, _ in request.headers.raw if name.lower() == b"authorization"] == [
        b"Authorization"
    ]


def test_framing_headers_beat_route_headers():
    h = Harness(lambda: httpx2.Response(200, json={"ok": True}))
    r = route(headers={"accept": "text/plain", "CONTENT-TYPE": "text/plain"}, auth=NoAuth())
    h.executor().open(r, {}, framing="json").close()
    (request,) = h.requests
    assert request.headers.get_list("accept") == ["application/json"]
    assert request.headers.get_list("content-type") == ["application/json"]
    assert "authorization" not in request.headers


@pytest.mark.parametrize(
    "framing, accept",
    [("sse", "text/event-stream"), ("ndjson", "application/json"), ("json", "application/json")],
)
def test_accept_per_framing(framing, accept):
    h = Harness(status(200, "{}\n"))
    h.executor().open(route(), {}, framing=framing).close()
    assert h.requests[0].headers["accept"] == accept


def test_body_rejects_nan():
    h = Harness()
    with pytest.raises(ValueError):
        h.executor().open(route(), {"x": float("nan")}, framing="json")
    assert h.requests == []


@pytest.mark.parametrize(
    "body, reason",
    [
        ({"text": "a\ud800b"}, "request body has an unpaired surrogate at position "),
        ({"x": {1, 2}}, "request body is not JSON: "),
        ({"x": float("inf")}, "request body is not JSON: "),
    ],
)
def test_unencodable_body_makes_no_request(body, reason):
    h = Harness()
    with pytest.raises(ValueError) as info:
        h.executor().open(route(), body, framing="json")
    assert str(info.value).startswith(reason)
    # Unchained, so the serialized body held by the encoder's error is not kept.
    assert info.value.__cause__ is None
    assert info.value.__context__ is None
    assert h.requests == []


def test_route_timeout_is_passed_and_client_default_otherwise():
    h = Harness(status(200, "{}"), status(200, "{}"))
    ex = h.executor()
    ex.open(route(timeout=7.0), {}, framing="json").close()
    ex.open(route(), {}, framing="json").close()
    assert h.requests[0].extensions["timeout"]["read"] == 7.0
    assert h.requests[1].extensions["timeout"]["read"] == 600.0


def test_executor_refuses_redirect_following_client():
    with pytest.raises(ValueError):
        Executor(httpx2.Client(follow_redirects=True))


# Retries


def test_two_429s_then_200():
    h = Harness(status(429), status(429), sse("data: ok\n\n"))
    with h.executor().open(route(), {}, framing="sse") as frames:
        assert list(frames) == [SSEEvent(data="ok")]
    assert len(h.requests) == 3
    assert h.waits() == pytest.approx([0.5 * 0.9, 1.0 * 0.9])


def test_waits_are_sliced():
    h = Harness(status(429), sse("data: ok\n\n"))
    h.executor().open(route(), {}, framing="sse").close()
    slices = [entry for entry in h.log if entry != "request"]
    assert all(0 < s <= 0.25 for s in slices)
    assert sum(slices) == pytest.approx(0.45)


def test_retry_after_ms_is_honored():
    h = Harness(status(429, headers={"retry-after-ms": "1500"}), sse("data: ok\n\n"))
    h.executor().open(route(), {}, framing="sse").close()
    assert h.waits() == pytest.approx([1.5])


def test_long_retry_after_is_not_retried():
    h = Harness(status(429, headers={"retry-after": "300"}))
    with pytest.raises(ProviderError) as info:
        h.executor().open(route(), {}, framing="sse")
    assert info.value.kind is ErrorKind.RATE_LIMIT
    assert len(h.requests) == 1
    assert h.waits() == []


def test_long_retry_after_waits_when_unbounded():
    h = Harness(status(429, headers={"retry-after": "300"}), sse("data: ok\n\n"))
    h.executor().open(route(max_retry_after=None), {}, framing="sse").close()
    assert h.waits() == pytest.approx([300.0])


def test_x_should_retry_false_is_not_retried():
    h = Harness(status(500, headers={"x-should-retry": "false"}))
    with pytest.raises(ProviderError) as info:
        h.executor().open(route(), {}, framing="sse")
    assert info.value.kind is ErrorKind.SERVER
    assert len(h.requests) == 1


def test_max_retries_zero_makes_one_request():
    h = Harness(status(500))
    with pytest.raises(ProviderError):
        h.executor().open(route(max_retries=0), {}, framing="sse")
    assert len(h.requests) == 1


def test_retries_end_with_the_last_error():
    h = Harness(status(500, "first"), status(502, "second"), status(503, "third"))
    with pytest.raises(ProviderError) as info:
        h.executor().open(route(), {}, framing="sse")
    assert len(h.requests) == 3
    assert info.value.status == 503
    assert "third" in str(info.value)


def test_waits_cap_at_eight_seconds():
    h = Harness(*[status(500)] * 7)
    ex = Executor(
        build_client(transport=httpx2.MockTransport(h.handler)),
        sleep=h.sleep,
        random=lambda: 0.0,
    )
    with pytest.raises(ProviderError):
        ex.open(route(max_retries=6), {}, framing="sse")
    assert h.waits() == pytest.approx([0.5, 1.0, 2.0, 4.0, 8.0, 8.0])


def test_backoff_stays_finite_after_many_retries():
    h = Harness(*[status(500)] * 1101)
    ex = Executor(
        build_client(transport=httpx2.MockTransport(h.handler)),
        sleep=h.sleep,
        random=lambda: 0.0,
    )
    with pytest.raises(ProviderError):
        ex.open(route(max_retries=1100), {}, framing="sse")
    waits = h.waits()
    assert len(waits) == 1100
    assert waits[-1] == pytest.approx(8.0)


def test_non_retryable_status_raises_at_once():
    h = Harness(status(400, '{"error": {"message": "bad field"}}'))
    with pytest.raises(ProviderError) as info:
        h.executor().open(route(), {}, framing="sse")
    assert info.value.kind is ErrorKind.BAD_REQUEST
    assert "bad field" in str(info.value)
    assert len(h.requests) == 1


def test_error_body_read_is_bounded():
    body = Body([b"x" * 16384] * 100)
    h = Harness(lambda: httpx2.Response(400, stream=body))
    with pytest.raises(ProviderError):
        h.executor().open(route(), {}, framing="sse")
    assert body.yielded <= 4
    assert body.closed


# Connection and timeout errors


@pytest.mark.parametrize(
    "exc, kind",
    [
        (httpx2.ConnectError("refused"), ErrorKind.CONNECTION),
        (httpx2.ReadTimeout("slow"), ErrorKind.TIMEOUT),
    ],
)
def test_transport_errors_are_retried_then_raised(exc, kind):
    h = Harness(exc, exc, exc)
    with pytest.raises(ProviderError) as info:
        h.executor().open(route(), {}, framing="sse")
    assert len(h.requests) == 3
    assert info.value.kind is kind
    assert info.value.__cause__ is exc
    assert str(info.value) == f"openai request failed ({kind.value}): {type(exc).__name__}: {exc}"


def test_transport_error_then_success():
    h = Harness(httpx2.ConnectError("refused"), sse("data: ok\n\n"))
    with h.executor().open(route(), {}, framing="sse") as frames:
        assert list(frames) == [SSEEvent(data="ok")]
    assert len(h.requests) == 2


# Redirect


def test_redirect_is_not_followed_or_retried():
    body = Body([b"moved"])
    h = Harness(
        lambda: httpx2.Response(
            307, headers={"Location": "https://evil.example/steal?k=1"}, stream=body
        )
    )
    with pytest.raises(ProviderError) as info:
        h.executor().open(route(), {}, framing="sse")
    assert info.value.kind is ErrorKind.REDIRECT
    assert "https://evil.example" in str(info.value)
    assert "/steal" not in str(info.value) and "k=1" not in str(info.value)
    assert len(h.requests) == 1
    assert body.closed


# No retry once streaming


@pytest.mark.parametrize(
    "exc, kind",
    [(httpx2.ReadError("reset"), ErrorKind.CONNECTION), (httpx2.ReadTimeout("slow"), ErrorKind.TIMEOUT)],
)
def test_no_retry_after_first_frame(exc, kind):
    h = Harness(sse("data: one\n\n", exc), sse("data: never\n\n"))
    frames = h.executor().open(route(), {}, framing="sse")
    assert next(frames) == SSEEvent(data="one")
    with pytest.raises(ProviderError) as info:
        next(frames)
    assert info.value.kind is kind
    assert info.value.__cause__ is exc
    assert len(h.requests) == 1


def test_retry_before_first_frame():
    body = Body([b": still connecting\n\n", httpx2.ReadError("reset")])
    h = Harness(lambda: httpx2.Response(200, stream=body), sse("data: one\n\n", "data: two\n\n"))
    frames = h.executor().open(route(), {}, framing="sse")
    # open() returned only after the second attempt's first frame was parsed.
    assert len(h.requests) == 2
    assert body.closed
    assert list(frames) == [SSEEvent(data="one"), SSEEvent(data="two")]


def test_partial_event_then_error_is_retried():
    h = Harness(sse("data: half", httpx2.ReadError("reset")), sse("data: whole\n\n"))
    with h.executor().open(route(), {}, framing="sse") as frames:
        assert list(frames) == [SSEEvent(data="whole")]
    assert len(h.requests) == 2


# Framing


def test_ndjson_framing():
    h = Harness(sse('{"a": 1}\n', '\n{"b"', ": 2}\n{\"c\": 3}"))
    with h.executor().open(route(), {}, framing="ndjson") as frames:
        assert list(frames) == [{"a": 1}, {"b": 2}, {"c": 3}]


def test_json_framing():
    h = Harness(sse('{"id": ', '"r1", "x": [1, 2]}'))
    with h.executor().open(route(), {}, framing="json") as frames:
        assert list(frames) == [{"id": "r1", "x": [1, 2]}]


@pytest.mark.parametrize("framing, body", [("json", '{"id": '), ("ndjson", '{"a": 1\n')])
def test_invalid_json_is_protocol_without_retry(framing, body):
    h = Harness(status(200, body), status(200, "{}"))
    with pytest.raises(ProviderError) as info:
        h.executor().open(route(), {}, framing=framing)
    assert info.value.kind is ErrorKind.PROTOCOL
    assert info.value.provider == "openai"
    assert len(h.requests) == 1


def test_decoding_error_is_protocol_without_retry():
    h = Harness(
        lambda: httpx2.Response(
            200, headers={"content-encoding": "gzip"}, stream=Body([b"not gzip at all"])
        ),
        sse("data: ok\n\n"),
    )
    with pytest.raises(ProviderError) as info:
        h.executor().open(route(), {}, framing="sse")
    assert info.value.kind is ErrorKind.PROTOCOL
    assert isinstance(info.value.__cause__, httpx2.DecodingError)
    assert len(h.requests) == 1


HTTPX_ERRORS = [
    httpx2.ConnectError,
    httpx2.ReadError,
    httpx2.WriteError,
    httpx2.RemoteProtocolError,
    httpx2.ConnectTimeout,
    httpx2.ReadTimeout,
    httpx2.PoolTimeout,
    httpx2.DecodingError,
    httpx2.TooManyRedirects,
]


@pytest.mark.parametrize("error_type", HTTPX_ERRORS)
def test_no_raw_httpx_error_from_send(error_type):
    h = Harness(*[error_type("boom")] * 3)
    with pytest.raises(ProviderError) as info:
        h.executor().open(route(), {}, framing="sse")
    assert isinstance(info.value.__cause__, error_type)


@pytest.mark.parametrize("error_type", HTTPX_ERRORS)
def test_no_raw_httpx_error_before_first_frame(error_type):
    h = Harness(*[sse(error_type("boom")) for _ in range(3)])
    with pytest.raises(ProviderError) as info:
        h.executor().open(route(), {}, framing="sse")
    assert isinstance(info.value.__cause__, error_type)
    # Transport errors and timeouts are retried (max_retries + 1 requests); decoding and
    # other httpx2 errors are not.
    retried = issubclass(error_type, (httpx2.TransportError, httpx2.TimeoutException))
    assert len(h.requests) == (3 if retried else 1)


@pytest.mark.parametrize("error_type", HTTPX_ERRORS)
def test_no_raw_httpx_error_after_first_frame(error_type):
    h = Harness(sse("data: one\n\n", error_type("boom")))
    frames = h.executor().open(route(), {}, framing="sse")
    assert next(frames) == SSEEvent(data="one")
    with pytest.raises(ProviderError) as info:
        next(frames)
    assert isinstance(info.value.__cause__, error_type)
    assert info.value.retryable is False
    assert len(h.requests) == 1


def test_comment_only_sse_is_empty():
    body = Body([b": keep-alive\n\n", b": again\n\n"])
    h = Harness(lambda: httpx2.Response(200, stream=body))
    frames = h.executor().open(route(), {}, framing="sse")
    assert list(frames) == []
    assert body.closed
    assert len(h.requests) == 1


def test_empty_json_body_is_empty():
    h = Harness(status(200, ""))
    assert list(h.executor().open(route(), {}, framing="json")) == []


def test_unknown_framing_is_refused():
    h = Harness()
    with pytest.raises(ValueError):
        h.executor().open(route(), {}, framing="xml")
    assert h.requests == []


# Stop checks


class Stop(Exception):
    pass


def test_stop_during_backoff_makes_no_more_requests():
    h = Harness(status(500), sse("data: ok\n\n"))
    stopping = []

    def sleep(seconds):
        h.log.append(seconds)
        stopping.append(True)

    def check_stopped():
        if stopping:
            raise Stop

    ex = Executor(
        build_client(transport=httpx2.MockTransport(h.handler)),
        check_stopped=check_stopped,
        sleep=sleep,
        random=lambda: JITTER,
    )
    with pytest.raises(Stop):
        ex.open(route(), {}, framing="sse")
    assert len(h.requests) == 1
    assert [entry for entry in h.log if entry != "request"] == [0.25]


def test_stop_check_runs_on_every_chunk():
    h = Harness(sse(*[": ping\n\n"] * 50))
    assert list(h.executor().open(route(), {}, framing="sse")) == []
    assert h.stops >= 50


def test_stop_check_runs_before_each_frame():
    h = Harness(sse("data: 1\n\ndata: 2\n\ndata: 3\n\n"))
    frames = h.executor().open(route(), {}, framing="sse")
    seen = []
    for _ in range(3):
        before = h.stops
        next(frames)
        seen.append(h.stops - before)
    assert all(count >= 1 for count in seen)


def test_stop_check_raising_mid_stream_closes():
    body = Body([b"data: 1\n\n", b"data: 2\n\n"])
    h = Harness(lambda: httpx2.Response(200, stream=body))
    calls = itertools.count()

    def check_stopped():
        # The attempt check and the first chunk pass; the check before the first frame stops.
        if next(calls) >= 2:
            raise Stop

    frames = h.executor(check_stopped=check_stopped).open(route(), {}, framing="sse")
    with pytest.raises(Stop):
        next(frames)
    assert body.closed


def test_close_half_read_stream_closes_response():
    body = Body([b"data: 1\n\n", b"data: 2\n\n", b"data: 3\n\n"])
    h = Harness(lambda: httpx2.Response(200, stream=body))
    frames = h.executor().open(route(), {}, framing="sse")
    assert next(frames) == SSEEvent(data="1")
    assert not body.closed
    frames.close()
    assert body.closed
    frames.close()
    assert list(frames) == []


def test_exhaustion_closes_response():
    body = Body([b"data: 1\n\n"])
    h = Harness(lambda: httpx2.Response(200, stream=body))
    frames = h.executor().open(route(), {}, framing="sse")
    assert list(frames) == [SSEEvent(data="1")]
    assert body.closed


def test_context_manager_closes():
    body = Body([b"data: 1\n\n", b"data: 2\n\n"])
    h = Harness(lambda: httpx2.Response(200, stream=body))
    with h.executor().open(route(), {}, framing="sse") as frames:
        next(frames)
    assert body.closed


# Credentials


def test_auth_is_resolved_per_attempt():
    keys = iter(["key-one", "key-two"])
    auth = BearerAuth(lambda: next(keys))
    h = Harness(status(500), sse("data: ok\n\n"))
    h.executor().open(route(auth=auth), {}, framing="sse").close()
    assert [r.headers["authorization"] for r in h.requests] == ["Bearer key-one", "Bearer key-two"]


def test_status_error_scrubs_the_key():
    key = "sk-live-secret-key-abcdef"
    body = json.dumps({"error": {"message": f"Incorrect API key provided: {key}."}})
    h = Harness(status(401, body))
    with pytest.raises(ProviderError) as info:
        h.executor().open(route(auth=HeaderKeyAuth("x-api-key", key)), {}, framing="sse")
    assert info.value.kind is ErrorKind.AUTH
    assert "[REDACTED]" in str(info.value)
    assert key not in str(info.value)
    assert key not in repr(info.value.args)


def test_status_error_scrubs_a_json_escaped_key():
    key = "sk-live/secret/key-abcdef"
    body = json.dumps({"error": {"message": f"Incorrect API key provided: {key}."}})
    # JSON may escape "/" as "\/", which the body scrub does not match but the parse undoes.
    body = body.replace("/", "\\/")
    assert key not in body
    h = Harness(status(401, body))
    with pytest.raises(ProviderError) as info:
        h.executor().open(route(auth=BearerAuth(key)), {}, framing="sse")
    assert "Incorrect API key provided: [REDACTED]." in str(info.value)
    assert key not in str(info.value)
    assert key not in info.value.message
    assert key not in repr(info.value.args)


def test_scrub_replaces_longest_credential_first():
    keys = iter(["abcdefgh", "abcdefgh-longer"])
    auth = BearerAuth(lambda: next(keys))
    h = Harness(status(500), status(401, "saw abcdefgh-longer"))
    with pytest.raises(ProviderError) as info:
        h.executor().open(route(auth=auth), {}, framing="sse")
    assert str(info.value).endswith("saw [REDACTED]")


def test_transport_error_scrubs_the_key():
    key = "sk-live-secret-key-abcdef"
    h = Harness(*[httpx2.ConnectError(f"proxy rejected {key}")] * 3)
    with pytest.raises(ProviderError) as info:
        h.executor().open(route(auth=BearerAuth(key)), {}, framing="sse")
    assert key not in str(info.value)
    assert "proxy rejected [REDACTED]" in str(info.value)
