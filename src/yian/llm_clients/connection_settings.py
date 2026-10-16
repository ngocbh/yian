"""Checks for connection settings: an OpenAI organization or project id and extra headers.

The settings travel with a provider key, so every caller that accepts them (the
``llm_config`` carrier, the auth store, ``yian auth``, resuming a run) checks them here.
This module imports nothing from ``config.py`` or ``auth/``, so each of those can use it.
Every check raises `ValueError` with a message that never contains a header value or an
id, since header values are secrets.
"""

from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any, Optional

from pydantic import SecretStr

from yian.defaults import current

_CREDENTIAL = "yian sends the credential itself"
_TRANSPORT = "yian's client sets it"
_OPENAI_IDS = "use llm_config.organization / project"

# Header names refused for every provider, lowercase -> why. Credential headers would
# replace or duplicate the key yian sends; framing headers would corrupt the request.
REFUSED_HEADER_NAMES: dict[str, str] = {
    "authorization": _CREDENTIAL,
    "proxy-authorization": _CREDENTIAL,
    "x-api-key": _CREDENTIAL,
    "api-key": _CREDENTIAL,
    "x-goog-api-key": _CREDENTIAL,
    "host": _TRANSPORT,
    "content-length": _TRANSPORT,
    "transfer-encoding": _TRANSPORT,
    "connection": _TRANSPORT,
    "content-type": _TRANSPORT,
}
# Header names refused only for ``provider: openai``, which has its own fields for them.
# Other providers (an OpenAI-compatible gateway, say) may send them as ordinary headers.
OPENAI_REFUSED_HEADER_NAMES: dict[str, str] = {
    "openai-organization": _OPENAI_IDS,
    "openai-project": _OPENAI_IDS,
}

# RFC 9110 token characters.
_TOKEN = re.compile(r"[!#$%&'*+\-.^_`|~0-9A-Za-z]+")
# RFC 9110 field-value restricted to visible ASCII: no surrounding whitespace, controls or
# non-ASCII, which h11/httpx would otherwise reject at request time with the value in
# the error.
_FIELD_VALUE = re.compile(r"[\x21-\x7e]+(?:[ \t]+[\x21-\x7e]+)*")
_VISIBLE_ASCII = re.compile(r"[\x21-\x7e]+")


def validate_header_name(name: Any, provider: Optional[str]) -> str:
    """Return ``name`` unchanged if it may be sent as a connection header.

    Args:
        name (`Any`): The candidate header name.
        provider (`str | None`): The provider the header goes to; ``"openai"`` also refuses
            `OPENAI_REFUSED_HEADER_NAMES`. `None` checks only the names refused for every
            provider.

    Returns:
        `str`: ``name`` itself.

    Raises:
        `ValueError`: ``name`` is not an HTTP token or is a refused name.
    """
    if not isinstance(name, str) or not _TOKEN.fullmatch(name):
        raise ValueError(
            "header names must be HTTP tokens: letters, digits and !#$%&'*+-.^_`|~"
        )
    reason = REFUSED_HEADER_NAMES.get(name.lower())
    if reason is None and provider == "openai":
        reason = OPENAI_REFUSED_HEADER_NAMES.get(name.lower())
    if reason is not None:
        raise ValueError(f"header {name} is refused: {reason}")
    return name


def validate_header_value(name: str, value: Any) -> str:
    """Return the plain text of a header value if it may be sent.

    Args:
        name (`str`): The header's name, already checked; error messages name it.
        value (`Any`): The candidate value, a `str` or `SecretStr`.

    Returns:
        `str`: The value as plain text.

    Raises:
        `ValueError`: The value is not a string, is longer than
            ``YIAN_MAX_HEADER_VALUE_CHARS`` characters (default 8192), or is not visible
            ASCII with single spaces or tabs between words.
    """
    if isinstance(value, SecretStr):
        value = value.get_secret_value()
    if not isinstance(value, str):
        raise ValueError(f"header {name} value must be a string")
    limit = current().max_header_value_chars
    if len(value) > limit:
        raise ValueError(f"header {name} value must be at most {limit} characters")
    if not _FIELD_VALUE.fullmatch(value):
        raise ValueError(
            f"header {name} value must be non-empty visible ASCII, with spaces or tabs "
            "only between characters"
        )
    return value


def validate_headers(headers: Any, provider: Optional[str]) -> dict[str, str]:
    """Return connection headers as plain text if every name and value may be sent.

    Names compare case-insensitively, so two spellings of one name are refused.

    Args:
        headers (`Any`): The candidate mapping of name to `str` or `SecretStr` value.
        provider (`str | None`): The provider the headers go to (see
            `validate_header_name`).

    Returns:
        `dict[str, str]`: The headers in their given order, values as plain text.

    Raises:
        `ValueError`: ``headers`` is not a mapping, has more than
            ``YIAN_MAX_CONNECTION_HEADERS`` entries (default 32), repeats a name, or has
            a name or value refused by `validate_header_name` or `validate_header_value`.
    """
    if not isinstance(headers, Mapping):
        raise ValueError("headers must be a mapping of header name to value")
    limit = current().max_connection_headers
    if len(headers) > limit:
        raise ValueError(f"headers must have at most {limit} headers")
    checked: dict[str, str] = {}
    seen: set[str] = set()
    for name, value in headers.items():
        validate_header_name(name, provider)
        if name.lower() in seen:
            raise ValueError(f"header {name} is given more than once (names ignore case)")
        seen.add(name.lower())
        checked[name] = validate_header_value(name, value)
    return checked


def validate_organization(value: Any, field: str) -> str:
    """Return ``value`` unchanged if it is an allowed OpenAI organization or project id.

    Args:
        value (`Any`): The candidate id.
        field (`str`): ``"organization"`` or ``"project"``; error messages name it.

    Returns:
        `str`: ``value`` itself.

    Raises:
        `ValueError`: ``value`` is not a string, is longer than
            ``YIAN_MAX_ORGANIZATION_CHARS`` characters (default 256), or is not non-empty
            visible ASCII.
    """
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    limit = current().max_organization_chars
    if len(value) > limit:
        raise ValueError(f"{field} must be at most {limit} characters")
    if not _VISIBLE_ASCII.fullmatch(value):
        raise ValueError(f"{field} must be non-empty visible ASCII, without spaces")
    return value
