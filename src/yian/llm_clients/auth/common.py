"""Immutable credential and selection values shared by authentication services."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
import sys
from typing import ClassVar, Literal

from yian.defaults import current
from yian.llm_clients.config import validate_base_url
from yian.llm_clients.connection_settings import (
    validate_header_name,
    validate_headers,
    validate_organization,
)


class AuthError(RuntimeError):
    """An authentication or credential-storage failure safe to show to a user."""


def is_auth_failure(exc: BaseException) -> bool:
    """Whether a model call's exception means its credentials were refused or missing.

    Missing means yian couldn't supply them, which it reports as an `AuthError`.
    Walks ``exc`` and its ``__cause__`` chain, since LangChain re-raises a
    provider error ``from`` the SDK's; the walk stops at a repeated link or
    after 16 links, so a cyclic or runaway chain ends. A link counts when it is
    an `AuthError`, or carries an ``int`` ``status_code`` or ``code`` of 401 or
    403, or of 400 with ``api_key_invalid`` in its text: Gemini reports an
    invalid key as a 400 whose only distinguishing mark is that reason, so the
    substring match is deliberate. Callers run this inside an ``except``
    handler, so it never raises: a link whose attribute or text can't be read
    doesn't count.
    """
    seen: set[int] = set()
    link: BaseException | None = exc
    while link is not None and id(link) not in seen and len(seen) < 16:
        seen.add(id(link))
        if isinstance(link, AuthError):
            return True
        for name in ("status_code", "code"):
            try:
                status = getattr(link, name, None)
            except Exception:
                status = None
            if type(status) is not int:
                continue
            if status in (401, 403):
                return True
            if status == 400:
                try:
                    text = str(link).lower()
                except Exception:
                    text = ""
                if "api_key_invalid" in text:
                    return True
        link = link.__cause__
    return False


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
        if self.provider == "ollama" and self.header_names:
            raise AuthError("Connection headers are not sent to provider ollama")
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


# Providers whose saved login may hold no credential: Ollama sends none, and vLLM
# runs keyless unless its server was started with a key.
_KEYLESS_LOGINS = frozenset({"ollama", "vllm"})


def is_usable_login(selection: ProviderSelection) -> bool:
    """Whether a supported provider's login is a kind it can use.

    Args:
        selection (`ProviderSelection`): A login of a supported provider.

    Returns:
        `bool`: False if a provider other than Ollama or vLLM has no credential;
            a provider other than OpenAI has a subscription; a provider other
            than Azure has an endpoint; Azure has no valid endpoint; or Ollama
            has a key. True otherwise.
    """
    provider, method, endpoint = selection.provider, selection.method, selection.endpoint
    if (
        (method == "none" and provider not in _KEYLESS_LOGINS)
        or (provider != "openai" and method == "subscription")
        or (provider != "azure" and endpoint is not None)
        or (provider == "azure" and endpoint is None)
        or (provider == "ollama" and method != "none")
    ):
        return False
    if provider != "azure":
        return True
    try:
        validate_base_url(endpoint)
    except ValueError:
        return False
    return True


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
