"""Immutable credential and selection values shared by authentication services."""

from __future__ import annotations

from dataclasses import dataclass, field
import sys
from typing import ClassVar, Literal


class AuthError(RuntimeError):
    """An authentication or credential-storage failure safe to show to a user."""


def _text(value: object) -> bool:
    return (
        isinstance(value, str) and bool(value) and all(32 < ord(c) < 127 for c in value)
    )


@dataclass(frozen=True)
class ApiKeyCredentials:
    """An API key staged for validation or read from the private store."""

    api_key: str = field(repr=False)
    method: ClassVar[Literal["api_key"]] = "api_key"

    def __post_init__(self) -> None:
        if not _text(self.api_key):
            raise AuthError("Invalid API key credential")


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
    such as an Azure resource, and ``None`` otherwise.
    """

    provider: str
    method: Literal["api_key", "subscription", "none"]
    model: str
    credential_id: str
    endpoint: str | None = None

    def __post_init__(self) -> None:
        if (
            not all(
                _text(value)
                for value in (self.provider, self.model, self.credential_id)
            )
            or self.method not in ("api_key", "subscription", "none")
            or (self.endpoint is not None and not _text(self.endpoint))
        ):
            raise AuthError("Invalid provider selection")


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
