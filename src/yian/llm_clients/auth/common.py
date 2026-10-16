"""Immutable credential and selection values shared by authentication services."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
import sys
from typing import ClassVar, Literal

from yian.defaults import current
from yian.llm_clients.connection_settings import (
    validate_header_name,
    validate_headers,
    validate_organization,
)


class AuthError(RuntimeError):
    """An authentication or credential-storage failure safe to show to a user."""


def _text(value: object) -> bool:
    return (
        isinstance(value, str) and bool(value) and all(32 < ord(c) < 127 for c in value)
    )


@dataclass(frozen=True)
class ApiKeyCredentials:
    """An API key staged for validation or read from the private store.

    ``headers`` are extra request headers sent beside the key, such as a gateway
    token. They are secrets like the key. Given as a mapping or as name/value
    pairs, they are kept as pairs sorted by name.
    """

    api_key: str = field(repr=False)
    headers: tuple[tuple[str, str], ...] = field(default=(), repr=False)
    method: ClassVar[Literal["api_key"]] = "api_key"

    def __post_init__(self) -> None:
        if not _text(self.api_key):
            raise AuthError("Invalid API key credential")
        headers = self.headers
        if isinstance(headers, Mapping):
            headers = tuple(headers.items())
        elif isinstance(headers, Iterable) and not isinstance(headers, (str, bytes)):
            headers = tuple(headers)
        else:
            raise AuthError("Invalid connection headers: give a mapping of name to value")
        if not all(type(pair) is tuple and len(pair) == 2 for pair in headers):
            raise AuthError("Invalid connection headers: give a mapping of name to value")
        try:
            # A dict keeps only one of two pairs with the same name, so count exact
            # repeats before building it.
            if len({name for name, _ in headers}) != len(headers):
                raise ValueError("a header name is given more than once")
            checked = validate_headers(dict(headers), None)
        except (ValueError, TypeError) as error:
            raise AuthError(f"Invalid connection headers: {error}") from None
        object.__setattr__(self, "headers", tuple(sorted(checked.items())))


@dataclass(frozen=True)
class OAuthCredentials:
    """Subscription tokens and routing metadata returned by authentication."""

    access_token: str = field(repr=False)
    refresh_token: str = field(repr=False)
    expires_at: float
    account_id: str
    residency: str | None = None
    method: ClassVar[Literal["subscription"]] = "subscription"

    def __post_init__(self) -> None:
        if (
            not all(
                _text(value)
                for value in (self.access_token, self.refresh_token, self.account_id)
            )
            or (self.residency is not None and not _text(self.residency))
            or type(self.expires_at) not in (int, float)
            or not 0 < self.expires_at <= sys.float_info.max
        ):
            raise AuthError("Invalid subscription credentials")


Credentials = ApiKeyCredentials | OAuthCredentials


@dataclass(frozen=True)
class ProviderSelection:
    """The nonsecret provider choice that a run may pin to one connection.

    ``method`` is ``"none"`` for a provider saved without a credential, such as
    a local server. ``endpoint`` is the provider's base URL when it needs one,
    such as an Azure resource, and ``None`` otherwise. ``organization`` and
    ``project`` are the OpenAI ids an ``openai`` API-key login sends.
    ``header_names`` names, sorted, the extra headers its API-key credential
    sends; their values stay with the credential.
    """

    provider: str
    method: Literal["api_key", "subscription", "none"]
    model: str
    credential_id: str
    endpoint: str | None = None
    organization: str | None = None
    project: str | None = None
    header_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            not all(
                _text(value)
                for value in (self.provider, self.model, self.credential_id)
            )
            or self.method not in ("api_key", "subscription", "none")
            or (self.endpoint is not None and not _text(self.endpoint))
            or type(self.header_names) is not tuple
        ):
            raise AuthError("Invalid provider selection")
        openai_key = self.provider == "openai" and self.method == "api_key"
        if not openai_key and (self.organization, self.project) != (None, None):
            raise AuthError("organization and project are only for an openai API-key login")
        if self.method != "api_key" and self.header_names:
            raise AuthError("Connection headers are only for an API-key login")
        try:
            for name, value in (("organization", self.organization), ("project", self.project)):
                if value is not None:
                    validate_organization(value, name)
            for name in self.header_names:
                validate_header_name(name, self.provider)
            if len({name.lower() for name in self.header_names}) != len(self.header_names):
                raise ValueError("a header name is given more than once (names ignore case)")
            limit = current().max_connection_headers
            if len(self.header_names) > limit:
                raise ValueError(f"headers must have at most {limit} headers")
        except ValueError as error:
            raise AuthError(f"Invalid provider selection: {error}") from None
        object.__setattr__(self, "header_names", tuple(sorted(self.header_names)))


@dataclass(frozen=True)
class AuthState:
    """A detached, nonsecret snapshot of saved provider choices."""

    default_provider: str | None = None
    selections: tuple[ProviderSelection, ...] = ()


@dataclass(frozen=True)
class HttpResponse:
    """A decoded HTTP result; the body may contain credentials and is redacted."""

    status_code: int
    data: object = field(repr=False)
