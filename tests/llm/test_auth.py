"""Auth objects: the headers each sends and the credentials each lists."""

from types import SimpleNamespace

import pytest

from yian.llm.auth import BearerAuth, HeaderKeyAuth, NoAuth, SubscriptionAuth


def counting_source(*values):
    """Return a source yielding ``values`` in turn, and a list of call counts."""
    calls = []
    pending = list(values)

    def source():
        calls.append(1)
        return pending.pop(0)

    return source, calls


def access(token, account, residency=None):
    return SimpleNamespace(access_token=token, account_id=account, residency=residency)


class TestHeaders:
    def test_no_auth(self):
        auth = NoAuth()
        assert auth.headers() == {}
        assert auth.credentials() == ()

    def test_bearer(self):
        auth = BearerAuth("sk-static")
        assert auth.headers() == {"Authorization": "Bearer sk-static"}
        assert auth.credentials() == ("sk-static",)

    @pytest.mark.parametrize("header", ["x-api-key", "api-key", "x-goog-api-key"])
    def test_header_key(self, header):
        auth = HeaderKeyAuth(header, "k-static")
        assert auth.headers() == {header: "k-static"}
        assert auth.credentials() == ("k-static",)

    def test_subscription_without_residency(self):
        auth = SubscriptionAuth(lambda: access("tok", "acct"))
        assert auth.headers() == {
            "Authorization": "Bearer tok",
            "ChatGPT-Account-Id": "acct",
            "originator": "yian",
        }

    def test_subscription_with_residency(self):
        auth = SubscriptionAuth(lambda: access("tok", "acct", "us"))
        assert auth.headers() == {
            "Authorization": "Bearer tok",
            "ChatGPT-Account-Id": "acct",
            "originator": "yian",
            "x-openai-internal-codex-residency": "us",
        }


class TestCallableKeys:
    @pytest.mark.parametrize(
        "make, render",
        [
            (BearerAuth, lambda key: {"Authorization": f"Bearer {key}"}),
            (lambda key: HeaderKeyAuth("x-api-key", key), lambda key: {"x-api-key": key}),
        ],
    )
    def test_resolved_on_every_call(self, make, render):
        source, calls = counting_source("k1", "k2", "k1")
        auth = make(source)
        assert calls == []
        assert auth.credentials() == ()
        assert auth.headers() == render("k1")
        assert auth.headers() == render("k2")
        assert auth.headers() == render("k1")
        assert len(calls) == 3
        # In first-seen order with no duplicates, and listing never calls the source.
        assert auth.credentials() == ("k1", "k2")
        assert len(calls) == 3

    def test_subscription_resolved_on_every_call(self):
        source, calls = counting_source(access("t1", "a1"), access("t2", "a1"))
        auth = SubscriptionAuth(source)
        assert auth.credentials() == ()
        assert auth.headers()["Authorization"] == "Bearer t1"
        assert auth.headers()["Authorization"] == "Bearer t2"
        assert auth.credentials() == ("t1", "a1", "t2")
        assert len(calls) == 2

    @pytest.mark.parametrize("bad", [None, 42, b"bytes", ""])
    def test_callable_returning_non_string_or_empty_raises(self, bad):
        with pytest.raises(ValueError):
            BearerAuth(lambda: bad).headers()
        with pytest.raises(ValueError):
            HeaderKeyAuth("x-api-key", lambda: bad).headers()


BAD_VALUES = ["", "sk-abc\r\ninjected: 1", "sk-abc\nx", "sk-abc\rx", "sk-\x00abc", "sk-\tabc",
              "sk-\x7fabc", "sk-é"]


class TestValueChecks:
    @pytest.mark.parametrize("value", BAD_VALUES)
    def test_bad_static_key(self, value):
        for auth in (BearerAuth(value), HeaderKeyAuth("x-api-key", value)):
            with pytest.raises(ValueError) as info:
                auth.headers()
            if value:
                assert value not in str(info.value)

    @pytest.mark.parametrize("value", BAD_VALUES)
    def test_bad_callable_key(self, value):
        with pytest.raises(ValueError) as info:
            BearerAuth(lambda: value).headers()
        if value:
            assert value not in str(info.value)

    @pytest.mark.parametrize("field", ["access_token", "account_id", "residency"])
    @pytest.mark.parametrize("value", BAD_VALUES)
    def test_bad_subscription_value(self, field, value):
        values = {"access_token": "tok", "account_id": "acct", "residency": "us"}
        values[field] = value
        auth = SubscriptionAuth(lambda: SimpleNamespace(**values))
        with pytest.raises(ValueError) as info:
            auth.headers()
        if value:
            assert value not in str(info.value)

    def test_inner_space_allowed(self):
        assert BearerAuth("a b").headers() == {"Authorization": "Bearer a b"}
