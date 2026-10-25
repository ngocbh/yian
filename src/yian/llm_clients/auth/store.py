"""Private atomic credentials with process-safe refresh and connection identity."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import secrets
import stat
import sys
import time

from yian.defaults import current
from yian.llm_clients.config import SUPPORTED_LLM_PROVIDERS

from .common import (
    ApiKeyCredentials,
    AuthError,
    AuthState,
    Credentials,
    OAuthCredentials,
    ProviderSelection,
    is_usable_login,
)


# Entry keys written only when set, so a login without them is stored as before.
_OPTIONAL_ENTRY_KEYS = frozenset({"endpoint", "organization", "project"})


def _header_names(credentials: Credentials | None) -> tuple[str, ...]:
    if isinstance(credentials, ApiKeyCredentials):
        return tuple(name for name, _ in credentials.headers)
    return ()


def _dump_credentials(credentials: Credentials | None) -> dict | None:
    # Written field by field so a login without headers keeps its stored bytes.
    if credentials is None:
        return None
    if isinstance(credentials, ApiKeyCredentials):
        value: dict = {"api_key": credentials.api_key}
        if credentials.headers:
            value["headers"] = dict(credentials.headers)
        return value
    return {
        "access_token": credentials.access_token,
        "refresh_token": credentials.refresh_token,
        "expires_at": credentials.expires_at,
        "account_id": credentials.account_id,
        "residency": credentials.residency,
    }


class AuthStore:
    """Persist one saved login per provider, with an optional default, privately.

    A login holds an API key, a subscription, or no credential; an Azure login
    also holds its endpoint. An API-key login may hold connection settings: extra
    request headers and, for ``openai``, an organization and project id. Reading
    a missing store never creates a directory or lock file. Selection and
    credentials are committed together by ``save`` after host validation;
    ``remove``, ``set_default`` and ``delete`` change a saved store and never
    create one.
    Refresh reloads under the same lock and cannot replace a newer connection.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        clock: Callable[[], float] = time.time,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        check_stopped: Callable[[], None] = lambda: None,
        lock_timeout: float = 30,
        refresh_margin: float = 60,
    ) -> None:
        if path is None:
            path = os.environ.get("YIAN_AUTH_FILE")
        if path is None:
            config = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
            path = config / "yian" / "auth.json"
        self.path = Path(path).expanduser().absolute()
        self._clock = clock
        self._monotonic = monotonic
        self._sleep = sleep
        self._check_stopped = check_stopped
        self._lock_timeout = lock_timeout
        self._refresh_margin = refresh_margin
        if (
            type(lock_timeout) not in (int, float)
            or not 0 < lock_timeout <= sys.float_info.max
            or type(refresh_margin) not in (int, float)
            or not 0 <= refresh_margin <= sys.float_info.max
        ):
            raise ValueError("Invalid credential-store timing")

    @staticmethod
    def _private(fd: int, *, directory: bool = False) -> None:
        info = os.fstat(fd)
        kind = stat.S_ISDIR if directory else stat.S_ISREG
        # An atomic replacement can unlink a snapshot already opened by a reader.
        # Its descriptor remains valid; multiple links would expose the secrets.
        if not kind(info.st_mode) or (not directory and info.st_nlink > 1):
            raise AuthError("Credential storage must use a regular private path")
        if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
            raise AuthError("Credential storage must be private to its owner")

    @contextmanager
    def _directory(self, *, create: bool = False) -> Iterator[int | None]:
        fd = None
        try:
            if create:
                self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            try:
                fd = os.open(
                    self.path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
                )
            except FileNotFoundError:
                if create:
                    raise
                yield None
                return
            self._private(fd, directory=True)
            yield fd
        except OSError:
            raise AuthError("Could not access private credential storage") from None
        finally:
            if fd is not None:
                os.close(fd)

    @contextmanager
    def _locked(self, directory: int) -> Iterator[None]:
        fd = os.open(
            self.path.name + ".lock",
            os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory,
        )
        try:
            self._private(fd)
            deadline = self._monotonic() + self._lock_timeout
            while True:
                self._check_stopped()
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    remaining = deadline - self._monotonic()
                    if remaining <= 0:
                        raise AuthError("Timed out waiting for credential storage")
                    self._sleep(min(0.1, remaining))
            yield
        finally:
            os.close(fd)

    def _load(self, directory: int | None) -> tuple[str | None, dict]:
        if directory is None:
            return None, {}
        try:
            fd = os.open(
                self.path.name,
                os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                dir_fd=directory,
            )
        except FileNotFoundError:
            return None, {}
        with os.fdopen(fd, "rb") as file:
            self._private(file.fileno())
            limit = current().max_auth_store_bytes
            raw = file.read(limit + 1)
        if len(raw) > limit:
            raise AuthError("Credential storage exceeds its size limit")
        try:
            value = json.loads(raw)
            if type(value) is not dict or set(value) != {
                "version",
                "default_provider",
                "providers",
            }:
                raise ValueError
            if (
                type(value["version"]) is not int
                or value["version"] != 1
                or type(value["providers"]) is not dict
            ):
                raise ValueError
            providers = {}
            for provider, entry in value["providers"].items():
                if type(entry) is not dict or set(entry) - _OPTIONAL_ENTRY_KEYS != {
                    "method",
                    "model",
                    "credential_id",
                    "credentials",
                }:
                    raise ValueError
                # The writer omits unset ids, so a stored null is not canonical. A null
                # endpoint stays readable, as stores before these ids allowed it.
                if any(entry.get(key, "") is None for key in ("organization", "project")):
                    raise ValueError
                credentials = self._load_credentials(entry["method"], entry["credentials"])
                selection = ProviderSelection(
                    provider,
                    entry["method"],
                    entry["model"],
                    entry["credential_id"],
                    entry.get("endpoint"),
                    entry.get("organization"),
                    entry.get("project"),
                    _header_names(credentials),
                )
                providers[provider] = (selection, credentials)
            default = value["default_provider"]
            if default is not None and (
                not isinstance(default, str) or default not in providers
            ):
                raise ValueError
            return default, providers
        except (ValueError, TypeError, KeyError, AuthError):
            raise AuthError(
                "Credential storage is malformed; reconnect after repairing it"
            ) from None

    @staticmethod
    def _load_credentials(method: object, value: object) -> Credentials | None:
        if method == "none":
            if value is not None:
                raise ValueError
            return None
        if type(value) is not dict:
            raise ValueError
        if method != "api_key":
            return OAuthCredentials(**value)
        if set(value) - {"headers"} != {"api_key"}:
            raise ValueError
        headers = value.get("headers", {})
        # The writer omits empty headers, so a stored empty mapping is not canonical.
        if "headers" in value and (type(headers) is not dict or not headers):
            raise ValueError
        return ApiKeyCredentials(value["api_key"], tuple(headers.items()))

    def _write(self, directory: int, default: str | None, providers: dict) -> None:
        value = {
            "version": 1,
            "default_provider": default,
            "providers": {
                provider: {
                    "method": selection.method,
                    "model": selection.model,
                    "credential_id": selection.credential_id,
                    "credentials": _dump_credentials(credentials),
                    **{
                        key: getattr(selection, key)
                        for key in sorted(_OPTIONAL_ENTRY_KEYS)
                        if getattr(selection, key) is not None
                    },
                }
                for provider, (selection, credentials) in providers.items()
            },
        }
        raw = json.dumps(value, sort_keys=True).encode() + b"\n"
        if len(raw) > current().max_auth_store_bytes:
            raise AuthError("Credential storage exceeds its size limit")
        temporary = ".auth-" + secrets.token_hex(16)
        fd = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory,
        )
        try:
            with os.fdopen(fd, "wb") as file:
                file.write(raw)
                file.flush()
                os.fsync(file.fileno())
            self._check_stopped()
            os.replace(
                temporary, self.path.name, src_dir_fd=directory, dst_dir_fd=directory
            )
            os.fsync(directory)
        finally:
            try:
                os.unlink(temporary, dir_fd=directory)
            except FileNotFoundError:
                pass

    def read(self) -> AuthState:
        """Read a nonsecret snapshot without creating or changing storage."""
        self._check_stopped()
        with self._directory() as directory:
            default, providers = self._load(directory)
        return AuthState(default, tuple(value[0] for value in providers.values()))

    def selection(self, provider: str | None = None) -> ProviderSelection | None:
        """Find the explicit provider's selection, or the saved default."""
        state = self.read()
        provider = provider if provider is not None else state.default_provider
        return next(
            (item for item in state.selections if item.provider == provider), None
        )

    def saved_secrets(self, provider: str) -> tuple[str, ...]:
        """Return a provider's saved secrets as stored, without refresh or writes.

        Reads without the lock and returns expired subscription tokens as they
        are, so callers can scrub every secret the login may send.

        Args:
            provider (`str`): Saved provider name.

        Returns:
            `tuple[str, ...]`: ``(api_key, *header_values)`` for an API-key
                login, ``(access_token, refresh_token)`` for a subscription, and
                ``()`` for a login without a credential or no login.

        Raises:
            `AuthError`: The store is malformed or cannot be read.
        """
        self._check_stopped()
        with self._directory() as directory:
            _, providers = self._load(directory)
        credentials = providers.get(provider, (None, None))[1]
        if isinstance(credentials, ApiKeyCredentials):
            return (credentials.api_key, *(value for _, value in credentials.headers))
        if isinstance(credentials, OAuthCredentials):
            return (credentials.access_token, credentials.refresh_token)
        return ()

    def save(
        self,
        provider: str,
        credentials: Credentials | None,
        model: str,
        *,
        endpoint: str | None = None,
        organization: str | None = None,
        project: str | None = None,
        make_default: bool = True,
    ) -> ProviderSelection:
        """Atomically reconnect a provider and optionally make it the default.

        The new login replaces the old one with its settings: settings not given
        here are not kept.

        Args:
            provider (str): Provider name to save the login under.
            credentials (Credentials | None): The validated credential, or
                ``None`` for a provider that needs none. An API key's headers
                are saved with it.
            model (str): Model to select for the provider.
            endpoint (str | None): Provider base URL, such as an Azure resource.
            organization (str | None): OpenAI organization id; ``openai`` API
                key only.
            project (str | None): OpenAI project id; ``openai`` API key only.
            make_default (bool): Whether the provider becomes the default.

        Returns:
            ProviderSelection: The saved selection with a new credential id.

        Raises:
            AuthError: If the values are invalid or storage cannot be written.
        """
        if credentials is not None and type(credentials) not in (
            ApiKeyCredentials,
            OAuthCredentials,
        ):
            raise AuthError("Unsupported credential type")
        method = "none" if credentials is None else credentials.method
        selection = ProviderSelection(
            provider,
            method,
            model,
            secrets.token_hex(16),
            endpoint,
            organization,
            project,
            _header_names(credentials),
        )
        # The snapshot refuses such a login, so saving it would make every run fail.
        # A provider this version doesn't run is never read, so its login isn't checked.
        if provider in SUPPORTED_LLM_PROVIDERS and not is_usable_login(selection):
            raise AuthError(f"A {provider} login can't be saved with this credential or endpoint")
        self._check_stopped()
        with self._directory(create=True) as directory:
            with self._locked(directory):
                default, providers = self._load(directory)
                providers[provider] = (selection, credentials)
                self._write(directory, provider if make_default else default, providers)
        return selection

    def _update(self, provider: str, remove: bool) -> None:
        missing = AuthError(
            f"There is no saved {provider} login; run 'yian auth' to list them"
        )
        self._check_stopped()
        with self._directory() as directory:
            if directory is None:
                raise missing
            with self._locked(directory):
                default, providers = self._load(directory)
                if provider not in providers:
                    raise missing
                if remove:
                    del providers[provider]
                    if default == provider:
                        default = None
                else:
                    default = provider
                self._write(directory, default, providers)

    def remove(self, provider: str) -> None:
        """Delete a provider's saved login, clearing the default if it names it.

        Args:
            provider (str): Saved provider name, supported or not.

        Raises:
            AuthError: If the provider has no saved login or storage fails.
        """
        self._update(provider, remove=True)

    def set_default(self, provider: str) -> None:
        """Make a saved provider the default.

        Args:
            provider (str): Saved provider name.

        Raises:
            AuthError: If the provider has no saved login or storage fails.
        """
        self._update(provider, remove=False)

    def delete(self) -> None:
        """Delete the whole saved store under its lock, keeping the lock file.

        Creates nothing when neither the store nor its lock file exists. A
        symbolic link at the store's path is removed, never followed.

        Raises:
            AuthError: If the directory is not private, the lock wait times out,
                or storage fails.
        """
        self._check_stopped()
        with self._directory() as directory:
            if directory is None:
                return

            def exists(name):
                try:
                    os.stat(name, dir_fd=directory, follow_symlinks=False)
                except FileNotFoundError:
                    return False
                return True

            if not (exists(self.path.name) or exists(self.path.name + ".lock")):
                return
            # Unlinking the lock file would let a waiter and a newcomer lock
            # different files, so only the store goes.
            with self._locked(directory):
                try:
                    os.unlink(self.path.name, dir_fd=directory)
                except FileNotFoundError:
                    pass
                else:
                    os.fsync(directory)

    def credentials(
        self,
        provider: str,
        expected_credential_id: str,
        *,
        refresh: Callable[[OAuthCredentials], OAuthCredentials] | None = None,
    ) -> Credentials:
        """Resolve a pinned connection, refreshing expiring tokens under lock."""

        def resolve(providers):
            pair = providers.get(provider)
            if pair is None or pair[0].credential_id != expected_credential_id:
                raise AuthError("The saved connection changed")
            return pair

        def fresh(credentials):
            return (
                isinstance(credentials, ApiKeyCredentials)
                or credentials.expires_at > self._clock() + self._refresh_margin
            )

        self._check_stopped()
        with self._directory() as directory:
            _, providers = self._load(directory)
            _, credentials = resolve(providers)
            if credentials is None:
                raise AuthError(f"{provider} has no saved credential")
            if fresh(credentials):
                return credentials
            with self._locked(directory):
                default, providers = self._load(directory)
                selection, credentials = resolve(providers)
                if fresh(credentials):
                    return credentials
                if refresh is None:
                    raise AuthError("Subscription credentials expired; sign in again")
                self._check_stopped()
                updated = refresh(credentials)
                if (
                    type(updated) is not OAuthCredentials
                    or updated.account_id != credentials.account_id
                ):
                    raise AuthError("Subscription refresh changed account identity")
                if not fresh(updated):
                    raise AuthError("Subscription refresh returned expired credentials")
                providers[provider] = (selection, updated)
                self._write(directory, default, providers)
                return updated
