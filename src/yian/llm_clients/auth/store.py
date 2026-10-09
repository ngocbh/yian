"""Private atomic credentials with process-safe refresh and connection identity."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import secrets
import stat
import sys
import time

from .common import (
    ApiKeyCredentials,
    AuthError,
    AuthState,
    Credentials,
    OAuthCredentials,
    ProviderSelection,
)

_LIMIT = 1024 * 1024


class AuthStore:
    """Persist one active connection per provider in an owner-only POSIX file.

    Reading a missing store never creates a directory or lock file. Selection
    and credentials are committed together by ``save`` after host validation.
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
            raw = file.read(_LIMIT + 1)
        if len(raw) > _LIMIT:
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
                if type(entry) is not dict or set(entry) != {
                    "method",
                    "model",
                    "credential_id",
                    "credentials",
                }:
                    raise ValueError
                selection = ProviderSelection(
                    provider, entry["method"], entry["model"], entry["credential_id"]
                )
                kind = (
                    ApiKeyCredentials
                    if selection.method == "api_key"
                    else OAuthCredentials
                )
                credentials = kind(**entry["credentials"])
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

    def _write(self, directory: int, default: str | None, providers: dict) -> None:
        value = {
            "version": 1,
            "default_provider": default,
            "providers": {
                provider: {
                    "method": selection.method,
                    "model": selection.model,
                    "credential_id": selection.credential_id,
                    "credentials": asdict(credentials),
                }
                for provider, (selection, credentials) in providers.items()
            },
        }
        raw = json.dumps(value, sort_keys=True).encode() + b"\n"
        if len(raw) > _LIMIT:
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

    def save(
        self,
        provider: str,
        credentials: Credentials,
        model: str,
        *,
        make_default: bool = True,
    ) -> ProviderSelection:
        """Atomically reconnect a provider and optionally make it the default."""
        if type(credentials) not in (ApiKeyCredentials, OAuthCredentials):
            raise AuthError("Unsupported credential type")
        selection = ProviderSelection(
            provider, credentials.method, model, secrets.token_hex(16)
        )
        self._check_stopped()
        with self._directory(create=True) as directory:
            with self._locked(directory):
                default, providers = self._load(directory)
                providers[provider] = (selection, credentials)
                self._write(directory, provider if make_default else default, providers)
        return selection

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
                raise AuthError(
                    "The saved connection changed; start a new run or reconnect"
                )
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
