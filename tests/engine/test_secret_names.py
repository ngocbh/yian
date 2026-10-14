"""The short list of names whose literal value is a credential."""

from __future__ import annotations

import pytest

from yian.secret_names import SECRET_NAMES, SECRET_SUFFIXES, is_secret_name, normalize_name


def test_the_list_is_the_documented_short_list() -> None:
    assert SECRET_NAMES == {
        "password",
        "passwd",
        "secret",
        "client_secret",
        "private_key",
        "access_token",
        "refresh_token",
        "api_key",
    }
    assert SECRET_SUFFIXES == ("_api_key", "_password", "_secret")


@pytest.mark.parametrize(
    "name",
    [
        "password",
        "passwd",
        "secret",
        "client_secret",
        "private_key",
        "access_token",
        "refresh_token",
        "api_key",
        "openai_api_key",
        "db_password",
        "aws_secret",
        "PASSWORD",
        "clientSecret",
        "Client-Secret",
        "OPENAI_API_KEY",
        "dbPassword",
        "privateKey",
        "refresh-token",
        "HTTPProxyPassword",
        "_api_key",
    ],
)
def test_secret_names_match_exact_suffixed_and_respelled(name: str) -> None:
    assert is_secret_name(name) is True


@pytest.mark.parametrize(
    "name",
    [
        "key",
        "user",
        "auth",
        "token",
        "sig",
        "signature",
        "page_token",
        "secret_key",
        "apikey",
        "password_hint",
        "max_password_length",
        "passwords",
        "tokens",
        "url",
        "mode",
        "db.password",
        "db password",
        "",
    ],
)
def test_ordinary_names_are_not_secret(name: str) -> None:
    assert is_secret_name(name) is False


@pytest.mark.parametrize(
    ("name", "normalized"),
    [
        ("clientSecret", "client_secret"),
        ("HTTPProxyPassword", "http_proxy_password"),
        ("Client-Secret", "client_secret"),
        ("db.password", "db.password"),
    ],
)
def test_normalize_name_splits_case_and_dashes_only(name: str, normalized: str) -> None:
    assert normalize_name(name) == normalized
