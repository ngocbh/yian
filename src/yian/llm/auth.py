"""Auth objects: the credential headers for one request and the secrets they sent.

Each object resolves its headers per request, so a key source that rotates or refreshes
is read every time. `credentials()` lists every secret the object holds or has resolved,
so the executor can replace those exact values in error text without calling a source
again.
"""

from __future__ import annotations

from collections.abc import Callable
import threading
from typing import Any, Protocol

Secret = str | Callable[[], str]


class Auth(Protocol):
    """Credential headers for a model call."""

    def headers(self) -> dict[str, str]:
        """Resolve the credential headers for one request.

        Returns:
            `dict[str, str]`: Header names mapped to their values.

        Raises:
            `ValueError`: If a resolved value is not valid header text.
        """
        ...

    def credentials(self) -> tuple[str, ...]:
        """List the static secrets and every value resolved so far; never calls a source.

        Returns:
            `tuple[str, ...]`: The secrets, in first-seen order, without duplicates.
        """
        ...


def _check_header_value(name: str, value: Any) -> str:
    """Return ``value`` if it is valid header text, never echoing it on failure.

    This is the subscription-access rule widened to allow inner spaces, so a value such
    as ``Bearer x`` passes while CR, LF, other control characters, non-ASCII and
    leading or trailing spaces fail. HTTP strips the padding from a header value, so a
    padded key would send something other than the secret it was given.

    Args:
        name (`str`): What the value is, for the error message.
        value (`Any`): The value to check.

    Returns:
        `str`: ``value`` unchanged.

    Raises:
        `ValueError`: If ``value`` is not a non-empty string of printable ASCII and
            inner spaces.
    """
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip(" ")
        or any((ord(char) < 0x21 and char != " ") or ord(char) >= 0x7F for char in value)
    ):
        raise ValueError(f"{name} must be non-empty HTTP header text")
    return value


class _Seen:
    """An ordered, thread-safe set of secrets an auth object has resolved."""

    def __init__(self) -> None:
        self._values: dict[str, None] = {}
        self._lock = threading.Lock()

    def add(self, *values: Any) -> None:
        """Record each non-empty string in ``values``.

        Args:
            *values (`Any`): Resolved values; anything else is ignored since it was
                never sent.
        """
        with self._lock:
            for value in values:
                if isinstance(value, str) and value:
                    self._values.setdefault(value, None)

    def snapshot(self) -> tuple[str, ...]:
        """Return the recorded values in first-seen order.

        Returns:
            `tuple[str, ...]`: The values.
        """
        with self._lock:
            return tuple(self._values)


class _KeyAuth:
    """Shared resolution for a key that is a string or a callable returning one.

    Args:
        key (`Secret`): The key, or a callable that returns it on each request.
    """

    def __init__(self, key: Secret) -> None:
        self._key = key
        self._seen = _Seen()
        if isinstance(key, str):
            self._seen.add(key)

    def _resolve(self) -> str:
        """Resolve the key for one request and record it for the scrub.

        Returns:
            `str`: The key, checked as header text.

        Raises:
            `ValueError`: If the key is not a non-empty string of valid header text.
        """
        key = self._key() if callable(self._key) else self._key
        # Record before checking: a value that fails the check may still reach error
        # text elsewhere, and listing it only widens the scrub.
        self._seen.add(key)
        return _check_header_value("API key", key)

    def credentials(self) -> tuple[str, ...]:
        """List the static key and every key resolved so far; never calls a source.

        Returns:
            `tuple[str, ...]`: The keys, in first-seen order, without duplicates.
        """
        return self._seen.snapshot()


class NoAuth:
    """Send no credentials, for local servers such as Ollama."""

    def headers(self) -> dict[str, str]:
        """Return no headers.

        Returns:
            `dict[str, str]`: An empty dict.
        """
        return {}

    def credentials(self) -> tuple[str, ...]:
        """Return no secrets.

        Returns:
            `tuple[str, ...]`: An empty tuple.
        """
        return ()


class BearerAuth(_KeyAuth):
    """Send the key as ``Authorization: Bearer <key>``.

    Args:
        key (`Secret`): The key, or a callable that returns it on each request.
    """

    def headers(self) -> dict[str, str]:
        """Resolve the key and build the ``Authorization`` header.

        Returns:
            `dict[str, str]`: ``{"Authorization": "Bearer <key>"}``.

        Raises:
            `ValueError`: If the key is not a non-empty string of valid header text.
        """
        return {"Authorization": f"Bearer {self._resolve()}"}


class HeaderKeyAuth(_KeyAuth):
    """Send the key as the value of a named header.

    Args:
        header (`str`): The header name, for example ``"x-api-key"``, ``"api-key"`` or
            ``"x-goog-api-key"``.
        key (`Secret`): The key, or a callable that returns it on each request.
    """

    def __init__(self, header: str, key: Secret) -> None:
        super().__init__(key)
        self.header = header

    def headers(self) -> dict[str, str]:
        """Resolve the key and build the header.

        Returns:
            `dict[str, str]`: ``{header: key}``.

        Raises:
            `ValueError`: If the key is not a non-empty string of valid header text.
        """
        return {self.header: self._resolve()}


class SubscriptionAuth:
    """Send a ChatGPT subscription login's access token and account id.

    Args:
        access_source (`Callable[[], Any]`): Called once per request; returns an object
            with ``access_token`` (`str`), ``account_id`` (`str`) and ``residency``
            (`str | None`) attributes, refreshed as the source sees fit.
    """

    def __init__(self, access_source: Callable[[], Any]) -> None:
        self._source = access_source
        self._seen = _Seen()

    def headers(self) -> dict[str, str]:
        """Resolve the current access and build the subscription headers.

        Returns:
            `dict[str, str]`: ``Authorization``, ``ChatGPT-Account-Id`` and
            ``originator``, plus ``x-openai-internal-codex-residency`` when the access
            has a residency.

        Raises:
            `ValueError`: If a resolved value is not a non-empty string of valid header
                text.
        """
        access = self._source()
        token = getattr(access, "access_token", None)
        account_id = getattr(access, "account_id", None)
        residency = getattr(access, "residency", None)
        self._seen.add(token, account_id)
        headers = {
            "Authorization": f"Bearer {_check_header_value('Subscription access_token', token)}",
            "ChatGPT-Account-Id": _check_header_value("Subscription account_id", account_id),
            "originator": "yian",
        }
        if residency is not None:
            headers["x-openai-internal-codex-residency"] = _check_header_value(
                "Subscription residency", residency
            )
        return headers

    def credentials(self) -> tuple[str, ...]:
        """List every access token and account id resolved so far; never calls the source.

        Returns:
            `tuple[str, ...]`: The values, in first-seen order, without duplicates.
        """
        return self._seen.snapshot()
