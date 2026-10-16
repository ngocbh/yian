"""OpenAI browser/device OAuth, refresh, and authenticated model discovery."""

from __future__ import annotations

import base64
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import partial
import hashlib
import json
import math
import secrets
import socket
import sys
import time
from urllib.parse import parse_qs, urlencode, urlsplit

from yian.defaults import current
from yian.llm_clients.connection_settings import validate_headers, validate_organization

from .common import ApiKeyCredentials, AuthError, HttpResponse, OAuthCredentials, _text
from .http import _RequestCancelled, _request

_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
_ISSUER = "https://auth.openai.com"
_REDIRECT_URI = "http://localhost:1455/auth/callback"


def _claims(token: object) -> dict:
    """Read routing metadata only from a token just received from OpenAI."""
    if not isinstance(token, str) or len(token.split(".")) != 3:
        return {}
    try:
        encoded = token.split(".")[1]
        raw = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
        value = json.loads(raw)
    except (ValueError, UnicodeError):
        return {}
    return value if type(value) is dict else {}


@contextmanager
def _callback_server(monotonic: Callable[[], float]) -> Iterator[Callable]:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        try:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind(("127.0.0.1", 1455))
            server.listen(5)
        except OSError:
            raise AuthError(
                "Could not open the local login callback; use device login"
            ) from None

        def wait(redirect_uri, timeout, check_stopped):
            deadline = monotonic() + timeout
            limit = current().max_oauth_callback_bytes

            def remaining():
                check_stopped()
                seconds = deadline - monotonic()
                if seconds <= 0:
                    raise AuthError("OpenAI browser login timed out")
                return min(0.2, seconds)

            while True:
                server.settimeout(remaining())
                try:
                    connection, _ = server.accept()
                except socket.timeout:
                    continue
                with connection:
                    headers = bytearray()
                    # Socket inactivity timeouts alone let slowly arriving bytes
                    # bypass cancellation and the overall authorization deadline.
                    while b"\r\n\r\n" not in headers and len(headers) <= limit:
                        connection.settimeout(remaining())
                        try:
                            chunk = connection.recv(4096)
                        except socket.timeout:
                            continue
                        except OSError:
                            break
                        if not chunk:
                            break
                        headers.extend(chunk)
                    remaining()
                    target = None
                    status, body = "400 Bad Request", b"Invalid callback request."
                    try:
                        method, path, version = (
                            bytes(headers)
                            .split(b"\r\n", 1)[0]
                            .decode("ascii")
                            .split(" ")
                        )
                        if method != "GET" or version not in ("HTTP/1.0", "HTTP/1.1"):
                            raise ValueError
                        if b"\r\n\r\n" not in headers or len(headers) > limit:
                            raise ValueError
                        parsed = urlsplit(path)
                        if (
                            parsed.scheme
                            or parsed.netloc
                            or parsed.path != "/auth/callback"
                        ):
                            status, body = "404 Not Found", b"Unknown callback path."
                        else:
                            target = "http://localhost:1455" + path
                            status, body = (
                                "200 OK",
                                b"Authorization received. Return to Yian.",
                            )
                    except (UnicodeError, ValueError):
                        pass
                    response = (
                        f"HTTP/1.1 {status}\r\nContent-Type: text/plain; charset=utf-8\r\n"
                        f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n"
                    ).encode() + body
                    connection.settimeout(remaining())
                    try:
                        connection.sendall(response)
                    except OSError:
                        pass
                    remaining()
                    if target is not None:
                        return target

        yield wait


class OpenAIAuth:
    """Stage validated credentials without changing saved provider selection.

    ``request`` receives method, URL, headers, timeout, and optional ``json_body``
    or ``form`` keyword arguments and returns ``HttpResponse``. Host callbacks
    own presentation; ``check_stopped`` may raise to cancel any waiting phase.
    """

    def __init__(
        self,
        *,
        request: Callable[..., HttpResponse] | None = None,
        clock: Callable[[], float] = time.time,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        check_stopped: Callable[[], None] = lambda: None,
        request_timeout: float = 30,
    ) -> None:
        if not math.isfinite(request_timeout) or request_timeout <= 0:
            raise ValueError("Invalid authentication request timeout")
        self._request = (
            request
            if request is not None
            else partial(_request, check_stopped=check_stopped, monotonic=monotonic)
        )
        self._clock = clock
        self._monotonic = monotonic
        self._sleep = sleep
        self._check_stopped = check_stopped
        self._request_timeout = request_timeout

    def _remaining(self, deadline: float) -> float:
        self._check_stopped()
        remaining = deadline - self._monotonic()
        if remaining <= 0:
            raise AuthError("OpenAI login timed out")
        return remaining

    def _deadline(self, timeout: float) -> float:
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Invalid authentication timeout")
        return self._monotonic() + timeout

    def _send(
        self, method, url, *, deadline=None, headers=None, json_body=None, form=None
    ):
        self._check_stopped()
        timeout = self._request_timeout
        if deadline is not None:
            timeout = min(timeout, self._remaining(deadline))
        headers = {"User-Agent": "yian", **(headers or {})}
        arguments = {"headers": headers, "timeout": timeout}
        if json_body is not None:
            headers["Content-Type"] = "application/json"
            arguments["json_body"] = json_body
        if form is not None:
            headers["Content-Type"] = "application/x-www-form-urlencoded"
            arguments["form"] = form
        try:
            response = self._request(method, url, **arguments)
        except _RequestCancelled as cancelled:
            raise cancelled.original from None
        except Exception:
            raise AuthError("OpenAI authentication request failed") from None
        self._check_stopped()
        if deadline is not None:
            self._remaining(deadline)
        if (
            type(response) is not HttpResponse
            or type(response.status_code) is not int
            or not 100 <= response.status_code <= 599
        ):
            raise AuthError("OpenAI authentication returned an invalid response")
        return response

    @staticmethod
    def _data(response: HttpResponse) -> dict:
        if not 200 <= response.status_code < 300:
            raise AuthError(
                f"OpenAI authentication request failed (HTTP {response.status_code})"
            )
        if type(response.data) is not dict:
            raise AuthError("OpenAI authentication returned malformed data")
        return response.data

    def _tokens(
        self, data: dict, previous: OAuthCredentials | None = None
    ) -> OAuthCredentials:
        account_ids = set()
        residency = previous.residency if previous else None
        for token in (data.get("id_token"), data.get("access_token")):
            claims = _claims(token)
            auth = claims.get("https://api.openai.com/auth", {})
            if type(auth) is not dict:
                auth = {}
            account = claims.get("chatgpt_account_id") or auth.get("chatgpt_account_id")
            organizations = claims.get("organizations")
            if (
                account is None
                and isinstance(organizations, list)
                and organizations
                and isinstance(organizations[0], dict)
            ):
                account = organizations[0].get("id")
            if account is not None:
                if not _text(account):
                    raise AuthError("OpenAI returned invalid account metadata")
                account_ids.add(account)
            if (
                "chatgpt_compute_residency" in auth
                or "chatgpt_compute_residency" in claims
            ):
                residency = auth.get(
                    "chatgpt_compute_residency", claims.get("chatgpt_compute_residency")
                )
                if residency == "no_constraint":
                    residency = None
        if len(account_ids) > 1 or (
            previous and account_ids and account_ids != {previous.account_id}
        ):
            raise AuthError("OpenAI refresh changed account identity")
        account_id = next(iter(account_ids), previous.account_id if previous else None)
        expires_in = data.get("expires_in", 3600)
        if (
            type(expires_in) not in (int, float)
            or not 0 < expires_in <= sys.float_info.max
        ):
            raise AuthError("OpenAI returned invalid credential expiry")
        refresh = data.get(
            "refresh_token", previous.refresh_token if previous else None
        )
        return OAuthCredentials(
            data.get("access_token"),
            refresh,
            self._clock() + expires_in,
            account_id,
            residency,
        )

    def _exchange(
        self, code: str, verifier: str, redirect_uri: str, deadline: float
    ) -> OAuthCredentials:
        if not _text(code) or not _text(verifier):
            raise AuthError("OpenAI returned invalid authorization data")
        response = self._send(
            "POST",
            _ISSUER + "/oauth/token",
            deadline=deadline,
            form={
                "grant_type": "authorization_code",
                "code": code,
                "code_verifier": verifier,
                "client_id": _CLIENT_ID,
                "redirect_uri": redirect_uri,
            },
        )
        return self._tokens(self._data(response))

    def browser_login(
        self,
        open_url: Callable[[str], None],
        *,
        await_callback: Callable[[str, float, Callable[[], None]], str] | None = None,
        timeout: float = 600,
    ) -> OAuthCredentials:
        """Authorize with PKCE and a bounded local callback, returning staged tokens.

        An injected callback receives the redirect URI, remaining timeout, and
        cancellation check; it returns the complete callback URL. Without one,
        this method owns a localhost listener for the entire authorization.
        """
        deadline = self._deadline(timeout)
        verifier = secrets.token_urlsafe(32)
        state = secrets.token_urlsafe(32)
        challenge = (
            base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
            .decode()
            .rstrip("=")
        )
        url = (
            _ISSUER
            + "/oauth/authorize?"
            + urlencode(
                {
                    "response_type": "code",
                    "client_id": _CLIENT_ID,
                    "redirect_uri": _REDIRECT_URI,
                    "scope": "openid profile email offline_access",
                    "code_challenge": challenge,
                    "code_challenge_method": "S256",
                    "id_token_add_organizations": "true",
                    "codex_cli_simplified_flow": "true",
                    "state": state,
                    "originator": "yian",
                }
            )
        )

        def authorize(callback):
            self._remaining(deadline)
            open_url(url)
            result = callback(
                _REDIRECT_URI, self._remaining(deadline), self._check_stopped
            )
            self._remaining(deadline)
            try:
                parsed = urlsplit(result)
                expected = urlsplit(_REDIRECT_URI)
                query = parse_qs(parsed.query, keep_blank_values=True)
                if (parsed.scheme, parsed.netloc, parsed.path) != (
                    expected.scheme,
                    expected.netloc,
                    expected.path,
                ):
                    raise AuthError("OpenAI returned an invalid callback location")
                states = query.get("state", [])
                if len(states) != 1 or not secrets.compare_digest(states[0], state):
                    raise AuthError("OpenAI login state did not match")
                if "error" in query:
                    raise AuthError("OpenAI browser authorization was denied")
                codes = query.get("code", [])
                if len(codes) != 1 or not codes[0]:
                    raise AuthError(
                        "OpenAI callback did not include an authorization code"
                    )
            except (ValueError, TypeError):
                raise AuthError("OpenAI returned an invalid browser callback") from None
            return self._exchange(codes[0], verifier, _REDIRECT_URI, deadline)

        if await_callback is not None:
            return authorize(await_callback)
        with _callback_server(self._monotonic) as callback:
            return authorize(callback)

    def device_login(
        self, display_code: Callable[[str, str], None], *, timeout: float = 900
    ) -> OAuthCredentials:
        """Authorize headlessly using OpenAI's device-auth polling protocol."""
        deadline = self._deadline(timeout)
        data = self._data(
            self._send(
                "POST",
                _ISSUER + "/api/accounts/deviceauth/usercode",
                deadline=deadline,
                json_body={"client_id": _CLIENT_ID},
            )
        )
        device_id = data.get("device_auth_id")
        user_code = data.get("user_code", data.get("usercode"))
        if not _text(device_id) or not _text(user_code):
            raise AuthError("OpenAI returned invalid device authorization data")
        try:
            raw_interval = data.get("interval", 5)
            interval = float(raw_interval)
            if isinstance(raw_interval, bool) or not math.isfinite(interval):
                raise ValueError
            interval = max(1.0, interval)
        except (ValueError, TypeError, OverflowError):
            raise AuthError(
                "OpenAI returned an invalid device polling interval"
            ) from None
        display_code(_ISSUER + "/codex/device", user_code)
        while True:
            response = self._send(
                "POST",
                _ISSUER + "/api/accounts/deviceauth/token",
                deadline=deadline,
                json_body={"device_auth_id": device_id, "user_code": user_code},
            )
            if response.status_code not in (403, 404):
                data = self._data(response)
                return self._exchange(
                    data.get("authorization_code"),
                    data.get("code_verifier"),
                    _ISSUER + "/deviceauth/callback",
                    deadline,
                )
            next_poll = self._monotonic() + interval
            while self._monotonic() < next_poll:
                remaining = self._remaining(deadline)
                self._sleep(min(0.2, remaining, next_poll - self._monotonic()))

    def refresh(self, credentials: OAuthCredentials) -> OAuthCredentials:
        """Refresh staged tokens while preserving the authorized account."""
        response = self._send(
            "POST",
            _ISSUER + "/oauth/token",
            form={
                "grant_type": "refresh_token",
                "refresh_token": credentials.refresh_token,
                "client_id": _CLIENT_ID,
            },
        )
        return self._tokens(self._data(response), credentials)

    def api_models(
        self,
        credentials: ApiKeyCredentials,
        *,
        organization: str | None = None,
        project: str | None = None,
    ) -> tuple[str, ...]:
        """Validate an API key and return the model identifiers it can list.

        The request carries the credential's connection headers and, when given, the
        organization and project ids as ``OpenAI-Organization`` and ``OpenAI-Project``.

        Args:
            credentials (`ApiKeyCredentials`): The key and its connection headers.
            organization (`str | None`): OpenAI organization id to send.
            project (`str | None`): OpenAI project id to send.

        Returns:
            `tuple[str, ...]`: The model ids, sorted and without duplicates.

        Raises:
            `AuthError`: A setting is invalid (checked before any request, and the
                message holds no header value or id), the request fails (a mismatched
                organization or project is an HTTP status), or the list is invalid or
                empty.
        """
        try:
            headers = validate_headers(dict(credentials.headers), "openai")
            for header, field, value in (
                ("OpenAI-Organization", "organization", organization),
                ("OpenAI-Project", "project", project),
            ):
                if value is not None:
                    headers[header] = validate_organization(value, field)
        except ValueError as error:
            raise AuthError(f"Invalid OpenAI connection settings: {error}") from None
        headers["Authorization"] = "Bearer " + credentials.api_key
        response = self._send("GET", "https://api.openai.com/v1/models", headers=headers)
        data = self._data(response).get("data")
        if (
            not isinstance(data, list)
            or not data
            or any(
                not isinstance(item, dict) or not _text(item.get("id")) for item in data
            )
        ):
            raise AuthError("OpenAI returned an invalid or empty model list")
        return tuple(sorted({item["id"] for item in data}))

    def subscription_models(
        self, credentials: OAuthCredentials, *, client_version: str = "0.155.1"
    ) -> tuple[str, ...]:
        """List visible subscription models using the backend catalog protocol."""
        if not _text(client_version):
            raise ValueError("Invalid model catalog client version")
        headers = {
            "Authorization": "Bearer " + credentials.access_token,
            "ChatGPT-Account-Id": credentials.account_id,
            "originator": "yian",
        }
        if credentials.residency is not None:
            headers["x-openai-internal-codex-residency"] = credentials.residency
        response = self._send(
            "GET",
            "https://chatgpt.com/backend-api/codex/models?"
            + urlencode({"client_version": client_version}),
            headers=headers,
        )
        models = self._data(response).get("models")
        if not isinstance(models, list) or any(
            not isinstance(item, dict)
            or not _text(item.get("slug"))
            or item.get("visibility") not in ("list", "hide", "none")
            or type(item.get("priority")) is not int
            for item in models
        ):
            raise AuthError("OpenAI returned an invalid subscription model list")
        visible = sorted(
            (item for item in models if item["visibility"] == "list"),
            key=lambda item: item["priority"],
        )
        if not visible:
            raise AuthError("OpenAI returned no available subscription models")
        return tuple(dict.fromkeys(item["slug"] for item in visible))
