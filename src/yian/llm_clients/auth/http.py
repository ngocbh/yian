"""Owned stdlib HTTP requests with guarded response reads and secret-safe results.

Header/body reads share one deadline and cancellation check. TCP connect and TLS
handshake use stdlib timeouts for their remaining budget; system DNS resolution
is synchronous and is not interrupted by those socket timeouts.

Failures stay distinguishable for callers that report them: an exhausted deadline
raises `TimeoutError`, a body over the limit raises `ValueError`, and connection
failures raise `URLError` or another `OSError`.
"""

from __future__ import annotations

from http.client import HTTPConnection, HTTPSConnection, HTTPResponse
import io
import json
import socket
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import (
    HTTPHandler,
    HTTPRedirectHandler,
    HTTPSHandler,
    Request,
    build_opener,
)

from yian.defaults import current

from .common import HttpResponse


class _RequestCancelled(Exception):
    """Carry a host cancellation through network-error sanitization."""

    def __init__(self, original: BaseException):
        super().__init__("Authentication request cancelled")
        self.original = original


class _Deadline:
    def __init__(self, timeout, monotonic, check_stopped):
        self.end = monotonic() + timeout
        self.monotonic = monotonic
        self.check_stopped = check_stopped

    def remaining(self):
        try:
            self.check_stopped()
        except BaseException as error:
            raise _RequestCancelled(error) from None
        remaining = self.end - self.monotonic()
        if remaining <= 0:
            raise TimeoutError("HTTP request timed out")
        return remaining


class _ResponseReader(io.RawIOBase):
    """Check the request deadline below buffered HTTP header/body parsing."""

    def __init__(self, connection, deadline):
        self.connection = connection
        self.deadline = deadline
        # urllib closes its connection once the response owns the socket. A raw
        # file reference preserves that standard lifetime without using its reads:
        # SocketIO permanently poisons reads after the first polling timeout.
        self.reference = connection.makefile("rb", buffering=0)

    def readable(self):
        return True

    def readinto(self, buffer):
        while True:
            self.connection.settimeout(min(0.2, self.deadline.remaining()))
            try:
                count = self.connection.recv_into(buffer)
            except socket.timeout:
                continue
            self.deadline.remaining()
            return count

    def close(self):
        try:
            self.reference.close()
        finally:
            super().close()


class _ResponseSocket:
    def __init__(self, connection, deadline):
        self.connection = connection
        self.deadline = deadline

    def makefile(self, mode):
        return io.BufferedReader(_ResponseReader(self.connection, self.deadline))


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, file, code, message, headers, url):
        return None


def _request(
    method,
    url,
    *,
    headers,
    timeout,
    json_body=None,
    form=None,
    check_stopped=lambda: None,
    monotonic=time.monotonic,
    body_limit: int | None = None,
):
    """Read one response without leaking credentials through redirects or logs.

    ``body_limit`` caps the bytes read from the body; `None` means
    `current().max_auth_response_bytes`.
    """
    if body_limit is None:
        body_limit = current().max_auth_response_bytes
    deadline = _Deadline(timeout, monotonic, check_stopped)

    class Response(HTTPResponse):
        def __init__(self, connection, *args, **kwargs):
            super().__init__(_ResponseSocket(connection, deadline), *args, **kwargs)

    class Connection:
        response_class = Response

        def send(self, data):
            if self.sock is None:
                self.connect()
            self.sock.settimeout(deadline.remaining())
            super().send(data)
            deadline.remaining()

    class HTTP(Connection, HTTPConnection):
        def connect(self):
            self.timeout = deadline.remaining()
            super().connect()
            deadline.remaining()

    class HTTPS(Connection, HTTPSConnection):
        def connect(self):
            self.timeout = deadline.remaining()
            HTTPConnection.connect(self)
            # A proxy CONNECT response may have used the short polling timeout.
            self.sock.settimeout(deadline.remaining())
            self.sock = self._context.wrap_socket(
                self.sock, server_hostname=self._tunnel_host or self.host
            )
            deadline.remaining()

    class HttpHandler(HTTPHandler):
        def http_open(self, request):
            return self.do_open(HTTP, request)

    class HttpsHandler(HTTPSHandler):
        def https_open(self, request):
            return self.do_open(HTTPS, request)

    data = None
    if json_body is not None:
        data = json.dumps(json_body).encode()
    elif form is not None:
        data = urlencode(form).encode()
    request = Request(url, data=data, headers=headers, method=method)
    try:
        response = build_opener(_NoRedirect(), HttpHandler(), HttpsHandler()).open(
            request, timeout=deadline.remaining()
        )
    except HTTPError as error:
        response = error
    except URLError as error:
        # urllib wraps connect and send failures, including the deadline's own.
        if isinstance(error.reason, TimeoutError):
            raise TimeoutError("HTTP request timed out") from None
        raise
    with response:
        status = response.code
        body = response.read(body_limit + 1)
        deadline.remaining()
    if len(body) > body_limit:
        raise ValueError("HTTP response is too large")
    try:
        data = json.loads(body)
    except (ValueError, UnicodeError):
        data = None
    return HttpResponse(status, data)
