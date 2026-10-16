"""Provider errors: status mapping, retry policy, retry-after parsing, messages."""

import copy
import email.utils
import json
import pickle

import pytest

from yian.llm.errors import ErrorKind, ProviderError, error_from_response, parse_retry_after
from yian.llm_clients.redact import scrub_keys


@pytest.mark.parametrize(
    "status, body, kind",
    [
        (401, "", ErrorKind.AUTH),
        (403, "", ErrorKind.PERMISSION),
        (404, "", ErrorKind.NOT_FOUND),
        (408, "", ErrorKind.TIMEOUT),
        (429, "", ErrorKind.RATE_LIMIT),
        (529, "", ErrorKind.OVERLOADED),
        (503, '{"error": {"type": "Overloaded_error"}}', ErrorKind.OVERLOADED),
        (503, "busy", ErrorKind.SERVER),
        (500, "", ErrorKind.SERVER),
        (502, "", ErrorKind.SERVER),
        (599, "", ErrorKind.SERVER),
        (301, "", ErrorKind.REDIRECT),
        (307, "", ErrorKind.REDIRECT),
        (400, "", ErrorKind.BAD_REQUEST),
        (409, "", ErrorKind.BAD_REQUEST),
        (413, "", ErrorKind.BAD_REQUEST),
        (422, "", ErrorKind.BAD_REQUEST),
    ],
)
def test_status_maps_to_kind(status, body, kind):
    err = error_from_response(status, {}, body, provider="openai")
    assert err.kind is kind
    assert err.status == status
    assert err.provider == "openai"


@pytest.mark.parametrize("status", [400, 413, 422])
@pytest.mark.parametrize(
    "phrase", ["context_length_exceeded", "Prompt is too long", "MAXIMUM CONTEXT LENGTH"]
)
def test_context_overflow_body(status, phrase):
    body = json.dumps({"error": {"message": f"oops: {phrase} here"}})
    assert error_from_response(status, {}, body, provider="p").kind is ErrorKind.CONTEXT_LENGTH


def test_context_phrase_on_other_status_is_not_context_length():
    err = error_from_response(500, {}, "maximum context length", provider="p")
    assert err.kind is ErrorKind.SERVER


@pytest.mark.parametrize(
    "status, retryable", [(400, False), (401, False), (404, False), (408, True), (409, True),
                          (429, True), (500, True), (503, True), (529, True), (301, False)]
)
def test_retryable_by_status(status, retryable):
    assert error_from_response(status, {}, "", provider="p").retryable is retryable


def test_should_retry_header_beats_status():
    assert error_from_response(400, {"x-should-retry": "true"}, "", provider="p").retryable
    assert not error_from_response(500, {"x-should-retry": "false"}, "", provider="p").retryable
    assert not error_from_response(429, {"X-Should-Retry": "false"}, "", provider="p").retryable


def test_long_retry_after_disables_retry():
    err = error_from_response(429, {"retry-after": "121"}, "", provider="p")
    assert not err.retryable
    assert err.retry_after == 121.0
    # The retry-after refusal comes before x-should-retry, as in openai.
    headers = {"retry-after": "500", "x-should-retry": "true"}
    assert not error_from_response(429, headers, "", provider="p").retryable
    assert error_from_response(429, {"retry-after": "120"}, "", provider="p").retryable


def test_no_max_retry_after_keeps_retry():
    err = error_from_response(429, {"retry-after": "500"}, "", provider="p", max_retry_after=None)
    assert err.retryable
    assert err.retry_after == 500.0


@pytest.mark.parametrize("value", ["inf", "nan"])
def test_literal_infinite_retry_after_falls_through(value):
    assert error_from_response(429, {"retry-after": value}, "", provider="p").retryable
    assert not error_from_response(400, {"retry-after": value}, "", provider="p").retryable


def test_pickle_and_copy_keep_every_attribute():
    err = ProviderError(ErrorKind.RATE_LIMIT, "slow down", status=429, retry_after=1.5,
                        retryable=True, provider="openai")
    for clone in (pickle.loads(pickle.dumps(err)), copy.copy(err), copy.deepcopy(err)):
        assert type(clone) is ProviderError
        assert clone.kind is ErrorKind.RATE_LIMIT
        assert clone.message == "slow down" == str(clone) == clone.args[0]
        assert clone.status == 429
        assert clone.retry_after == 1.5
        assert clone.retryable is True
        assert clone.provider == "openai"


def test_error_text_is_message_and_args():
    err = ProviderError(ErrorKind.PROTOCOL, "bad frame")
    assert str(err) == err.message == err.args[0] == "bad frame"
    assert err.args == ("bad frame",)
    assert err.status is None and err.retry_after is None and err.provider is None
    assert err.retryable is False


class TestParseRetryAfter:
    def test_milliseconds_preferred(self):
        assert parse_retry_after({"retry-after-ms": "1500", "retry-after": "9"}) == 1.5

    def test_seconds(self):
        assert parse_retry_after({"retry-after": "7"}) == 7.0

    def test_float_seconds(self):
        assert parse_retry_after({"Retry-After": "0.25"}) == 0.25

    def test_http_date(self):
        now = 1_800_000_000.0
        date = email.utils.formatdate(now + 30, usegmt=True)
        assert parse_retry_after({"retry-after": date}, now=now) == 30.0

    def test_garbage_ms_falls_back_to_seconds(self):
        assert parse_retry_after({"retry-after-ms": "soon", "retry-after": "3"}) == 3.0

    @pytest.mark.parametrize("headers", [{}, {"retry-after": "soon"}, {"retry-after": ""}])
    def test_absent_or_garbage(self, headers):
        assert parse_retry_after(headers) is None

    @pytest.mark.parametrize("name", ["retry-after", "retry-after-ms"])
    def test_overflow_is_sentinel(self, name):
        assert parse_retry_after({name: "1e400"}) == 121.0

    def test_overflow_refuses_retry_by_default(self):
        err = error_from_response(503, {"retry-after": "9" * 400}, "", provider="p")
        assert err.retry_after == 121.0
        assert not err.retryable


class TestMessage:
    def test_json_error_message(self):
        body = json.dumps({"error": {"message": "Invalid model", "type": "x"}})
        err = error_from_response(400, {}, body, provider="openai")
        assert str(err) == "openai returned HTTP 400 (bad_request): Invalid model"
        assert err.message == str(err)

    def test_json_top_level_message_and_detail(self):
        msg = error_from_response(404, {}, '{"message": "no such"}', provider="p")
        assert msg.message.endswith(": no such")
        detail = error_from_response(422, {}, '{"detail": "bad field"}', provider="p")
        assert detail.message.endswith(": bad field")

    def test_plain_text_body_collapses_whitespace(self):
        err = error_from_response(502, {}, "  Bad\n\n gateway\t here ", provider="azure")
        assert str(err) == "azure returned HTTP 502 (server): Bad gateway here"

    def test_json_without_known_fields_uses_text(self):
        err = error_from_response(500, {}, '{"code": 7}', provider="p")
        assert err.message.endswith(': {"code": 7}')

    def test_empty_body(self):
        err = error_from_response(500, {}, "  \n", provider="p")
        assert str(err) == "p returned HTTP 500 (server): no body"

    def test_detail_cut_at_2000(self):
        err = error_from_response(500, {}, "x" * 5000, provider="p")
        detail = err.message.split(": ", 1)[1]
        assert detail == "x" * 2000 + "…"

    def test_redirect_names_only_scheme_and_host(self):
        headers = {"Location": "https://user:pw@evil.example.com:8443/steal/path?token=abc#f"}
        err = error_from_response(302, headers, "see /steal/path?token=abc", provider="p")
        assert err.kind is ErrorKind.REDIRECT
        assert "https://evil.example.com:8443" in err.message
        for leak in ("steal", "token", "abc", "user", "pw"):
            assert leak not in err.message

    def test_redirect_without_location(self):
        err = error_from_response(301, {}, "", provider="p")
        assert err.message.startswith("p returned HTTP 301 (redirect): ")


def test_caller_scrub_removes_echoed_key():
    body = json.dumps({"error": {"message": "Incorrect API key provided: sk-test-123456."}})
    err = error_from_response(401, {}, body, provider="openai")
    assert "sk-test-123456" in str(err)
    scrub_keys(err, ["sk-test-123456"])
    assert "sk-test-123456" not in str(err)
    assert "sk-test-123456" not in err.message
    assert "[REDACTED]" in err.message
