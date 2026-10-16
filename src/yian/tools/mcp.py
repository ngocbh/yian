"""MCP servers whose tools a flow calls as host tools.

``mcp_server`` describes a Model Context Protocol server: a command started as a
subprocess that speaks over stdio, or a streamable HTTP endpoint. ``McpServer.open``
starts the server's client on a background thread with its own asyncio loop, lists
the server's tools once, and exposes each as a ``FunctionTool``; ``close`` stops the
client and the subprocess. A call returns the server's structured content, or its
content rendered as text. A failed call raises an ``McpError`` whose classes tell an
AGENT's tool loop whether its model reads the error and a TOOL node whether
``retries:`` apply; a stop of the calling run or ``close()`` ends a waiting call.

Knows about: the ``mcp`` SDK and ``httpx2`` (imported when a server opens),
``yian.defaults``, ``tools.context``, ``tools.errors``, ``tools.function_tools``,
and the Python standard library.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import concurrent.futures
import contextlib
import copy
import math
import numbers
import os
import tempfile
import threading
import time
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from typing import Any

from yian.defaults import current
from yian.tools.context import ToolContext, _current_stop
from yian.tools.errors import PermanentToolError, ReportedToolError, TransientToolError
from yian.tools.function_tools import FunctionTool

# A server that keeps returning cursors would otherwise keep `open()` listing forever.
_MAX_PAGES = 100
# The SDK's stdio shutdown waits up to about 6 s for the subprocess to exit.
_CLOSE_JOIN_SECONDS = 10
_STDERR_LINES = 20


class McpError(Exception):
    """Base class of the errors yian raises for an MCP server."""


class McpConnectError(McpError, RuntimeError):
    """An MCP server didn't start, didn't list its tools in time, or listed bad tools.

    The message names the server's label. For a stdio server it ends with the last
    lines the server wrote to stderr; for an HTTP server it names only the error type
    and HTTP status, never the URL, which may hold a token.
    """


class McpToolError(McpError, ReportedToolError):
    """An MCP server returned an error result for a tool call, or the call failed locally.

    For an error result the message is the result's content rendered as text, as a
    successful call's text result is: text blocks, then one-line markers for media.
    For any other failure while the server is up (for example structured content
    that fails the tool's output schema) it names only the server, the tool and the
    exception type. An AGENT's model reads it as the call's result.
    """


class McpRequestError(McpError, ReportedToolError):
    """An MCP server answered a tool call with a JSON-RPC error.

    The message names the server and the tool and holds the server's error message
    and code. An AGENT's model reads it as the call's result.

    Args:
        message (`str`):
            The error message.
        code (`int`):
            The JSON-RPC error code.
    """

    def __init__(self, message: str, code: int) -> None:
        super().__init__(message)
        self.code = code

    def __reduce__(self) -> tuple[Any, ...]:
        return (type(self), (str(self), self.code))


class McpInvalidRequestError(McpRequestError, PermanentToolError):
    """An MCP server rejected a tool call as malformed or unknown; retrying can't help.

    Raised for the JSON-RPC codes -32700, -32600, -32601 and -32602.
    """


class McpTimeoutError(McpError, ReportedToolError, TransientToolError):
    """A tool call took longer than the server's call timeout."""


class McpTransportError(McpError, ReportedToolError, TransientToolError):
    """A tool call failed because the HTTP connection to the server failed.

    The message names only the server and the error type, never the URL or the error
    text, which may hold a token.
    """


class McpServerClosed(McpError, PermanentToolError):
    """The MCP server is gone: yian closed it, or its process or connection ended.

    Later calls to the same server raise it at once; the host still closes it.
    """


class McpCallCancelled(McpError, PermanentToolError):
    """The calling run stopped while the tool call waited, so the call was cancelled."""


# JSON-RPC codes for a request the server can't parse, route or accept.
_INVALID_REQUEST_CODES = frozenset({-32700, -32600, -32601, -32602})
_CONNECTION_CLOSED = -32000
_REQUEST_TIMEOUT = -32001
_INTERNAL_ERROR = -32603
# How long the liveness probe after a -32000 answer may take.
_PING_SECONDS = 2
# The slice a waiting call sleeps between checks of the run's stop event.
_WAIT_SLICE_SECONDS = 0.1


def mcp_server(
    *,
    command: str | None = None,
    args: Sequence[str] = (),
    env: Mapping[str, str] | None = None,
    cwd: str | os.PathLike[str] | None = None,
    url: str | None = None,
    headers: Mapping[str, str] | None = None,
    timeout: float | None = None,
    connect_timeout: float | None = None,
    label: str | None = None,
) -> McpServer:
    """Describe an MCP server, started as a command over stdio or reached by URL.

    Nothing starts until the server is opened, with `with server:` or `open()`.

    Args:
        command (`str`, *optional*, defaults to `None`):
            The executable that starts a stdio server. Pass exactly one of `command`
            and `url`.
        args (`Sequence[str]`, *optional*, defaults to `()`):
            The command's arguments. Only with `command`.
        env (`Mapping[str, str]`, *optional*, defaults to `None`):
            Variables merged over the MCP SDK's safe default environment (`PATH`,
            `HOME` and a few others), so the host's secrets reach the server only when
            passed here. Only with `command`.
        cwd (`str | os.PathLike[str]`, *optional*, defaults to `None`):
            The command's working directory. `None` uses the host's. Only with
            `command`.
        url (`str`, *optional*, defaults to `None`):
            The `http` or `https` endpoint of a streamable HTTP server.
        headers (`Mapping[str, str]`, *optional*, defaults to `None`):
            HTTP headers sent with every request, for example an `Authorization`
            header. Only with `url`.
        timeout (`float`, *optional*, defaults to `None`):
            Longest wait in seconds for one tool call. `None` reads
            `YIAN_MCP_CALL_TIMEOUT_SECONDS` (300) when the server opens.
        connect_timeout (`float`, *optional*, defaults to `None`):
            Longest wait in seconds for the server to start and list its tools.
            `None` reads `YIAN_MCP_CONNECT_TIMEOUT_SECONDS` (30) when the server
            opens.
        label (`str`, *optional*, defaults to `None`):
            The name used in errors and the thread name. `None` uses the command's
            basename, or the URL's host.

    Returns:
        `McpServer`:
            The server, not yet open.

    Raises:
        `TypeError`:
            Raised when an argument has the wrong type: `args` a string or holding a
            non-string, `env` or `headers` not a mapping of strings to strings, a
            timeout not a number.
        `ValueError`:
            Raised when neither or both of `command` and `url` are given, `url` is not
            an `http`/`https` URL with a host, an option is given for the other kind
            of server, a timeout is not positive and finite, or `command` or `label`
            is empty.

    Example:
        ```python
        files = mcp_server(command="npx", args=["-y", "@modelcontextprotocol/server-filesystem", "."])
        with files:
            print([tool.name for tool in files.tools])
        ```
    """
    if (command is None) == (url is None):
        raise ValueError("mcp_server needs exactly one of command or url")
    if command is not None:
        if not isinstance(command, str):
            raise TypeError(f"mcp_server command must be a string, got {type(command).__name__}")
        if not command:
            raise ValueError("mcp_server command must not be empty")
        if headers is not None:
            raise ValueError("mcp_server headers needs url; a command server takes env instead")
    else:
        if not isinstance(url, str):
            raise TypeError(f"mcp_server url must be a string, got {type(url).__name__}")
        parts = urllib.parse.urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ValueError("mcp_server url must be an http or https URL with a host")
        for name, given in (("args", bool(args)), ("env", env is not None), ("cwd", cwd is not None)):
            if given:
                raise ValueError(f"mcp_server {name} needs command; a url server takes headers instead")
    if isinstance(args, (str, bytes)) or not isinstance(args, Sequence):
        raise TypeError("mcp_server args must be a sequence of strings, not a single string")
    if not all(isinstance(arg, str) for arg in args):
        raise TypeError("mcp_server args must hold only strings")
    _check_str_mapping("env", env)
    _check_str_mapping("headers", headers)
    if cwd is not None and not isinstance(cwd, (str, os.PathLike)):
        raise TypeError(f"mcp_server cwd must be a str or path, got {type(cwd).__name__}")
    _check_seconds("timeout", timeout)
    _check_seconds("connect_timeout", connect_timeout)
    if label is not None:
        if not isinstance(label, str):
            raise TypeError(f"mcp_server label must be a string, got {type(label).__name__}")
        if not label:
            raise ValueError("mcp_server label must not be empty")
    return McpServer(
        command=command,
        args=tuple(args),
        env=dict(env) if env is not None else None,
        cwd=os.fspath(cwd) if cwd is not None else None,
        url=url,
        headers=dict(headers) if headers is not None else None,
        timeout=timeout,
        connect_timeout=connect_timeout,
        label=label,
    )


def _check_str_mapping(name: str, value: Mapping[str, str] | None) -> None:
    """Raise `TypeError` unless `value` is `None` or a mapping of strings to strings."""
    if value is None:
        return
    if not isinstance(value, Mapping):
        raise TypeError(f"mcp_server {name} must be a mapping of strings to strings")
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, str):
            raise TypeError(f"mcp_server {name} must map strings to strings; {key!r} doesn't")


def _check_seconds(name: str, value: float | None) -> None:
    """Raise unless `value` is `None` or a positive, finite number of seconds."""
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"mcp_server {name} must be a number of seconds, got {type(value).__name__}")
    if not (0 < value < math.inf):
        raise ValueError(f"mcp_server {name} must be positive and finite, got {value!r}")


class McpServer:
    """An MCP server whose tools a flow calls as host tools.

    Build one with `mcp_server`. While open, a daemon thread named
    `yian-mcp-<label>` runs the MCP SDK client on its own asyncio loop; `tools` holds
    the tools the server listed when it opened. A later change to the server's tool
    list is ignored. A server opened with `with server:` or `open()` belongs to the
    host, which closes it.

    Example:
        ```python
        server = mcp_server(url="https://tools.example.com/mcp", headers={"Authorization": token})
        with server:
            names = [tool.name for tool in server.tools]
        ```
    """

    def __init__(
        self,
        *,
        command: str | None,
        args: tuple[str, ...],
        env: dict[str, str] | None,
        cwd: str | None,
        url: str | None,
        headers: dict[str, str] | None,
        timeout: float | None,
        connect_timeout: float | None,
        label: str | None,
    ) -> None:
        self._command = command
        self._args = args
        self._env = env
        self._cwd = cwd
        self._url = url
        self._headers = headers
        self._timeout = timeout
        self._connect_timeout = connect_timeout
        if label is None:
            if command is not None:
                label = os.path.basename(command) or command
            else:
                label = urllib.parse.urlsplit(url).hostname or "mcp"
        self._label = label
        # The SDK's connect mode; tests set "legacy" to use the initialize handshake.
        self._mode = "auto"
        # None while closed, "host" for a server the host opened, or the token of the
        # run that opened it.
        self._owner: str | None = None
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._close_requested = False
        self._error: BaseException | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._closing: asyncio.Event | None = None
        self._scope: Any = None
        self._client: Any = None
        self._tools: tuple[FunctionTool, ...] | None = None
        self._stderr: Any = None
        self._transport_failures = 0
        self._failures_lock = threading.Lock()
        self._call_timeout: float | None = None
        # Set once the server is known gone; calls then fail at once until reopened.
        self._dead = False
        # Futures of the calls in flight, which close() cancels; guarded by `_lock`.
        self._calls: set[concurrent.futures.Future[Any]] = set()

    @property
    def label(self) -> str:
        """`str`: The name used in errors: `label`, else the command's basename, else the URL host."""
        return self._label

    @property
    def is_open(self) -> bool:
        """`bool`: Whether the server is open and its tools are listed."""
        return self._owner is not None

    @property
    def tools(self) -> tuple[FunctionTool, ...]:
        """`tuple[FunctionTool, ...]`: The server's tools, in the order it listed them.

        Raises:
            `RuntimeError`:
                Raised when the server is not open.
        """
        if self._owner is None or self._tools is None:
            raise RuntimeError(f"MCP server '{self._label}' is not open")
        return self._tools

    def __enter__(self) -> McpServer:
        return self.open()

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def open(self) -> McpServer:
        """Start the server, wait until it lists its tools, and keep it running.

        The server then belongs to the host, which must `close()` it; `with server:`
        does both.

        Returns:
            `McpServer`:
                This server.

        Raises:
            `RuntimeError`:
                Raised when the server is already open.
            `ImportError`:
                Raised when the MCP SDK is not installed (`pip install 'yian[mcp]'`).
            `McpConnectError`:
                Raised when the server doesn't start, doesn't list its tools within
                the connect timeout, or lists a tool twice or a tool whose input
                schema is not an object.
        """
        return self._open("host")

    def _open_for_run(self, token: str) -> McpServer:
        """Open the server on behalf of the run `token`, which closes it when it ends.

        Args:
            token (`str`):
                The opening run's ownership token.

        Returns:
            `McpServer`:
                This server.

        Raises:
            `RuntimeError`, `ImportError`, `McpConnectError`:
                Raised as `open()` raises them.
        """
        return self._open(token)

    def _open(self, owner: str) -> McpServer:
        """Start the server and record `owner` once its tools are listed."""
        try:
            import mcp  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "MCP servers need the MCP SDK; install it with pip install 'yian[mcp]'"
            ) from exc
        settings = current()
        connect_timeout = self._connect_timeout or settings.mcp_connect_timeout_seconds
        call_timeout = self._timeout or settings.mcp_call_timeout_seconds

        # Under the lock, so two open() calls, or a close() from another thread, can't
        # interleave with the check and the reset.
        with self._lock:
            if self._owner is not None or self._thread is not None:
                raise RuntimeError(f"MCP server '{self._label}' is already open")
            ready = self._ready = threading.Event()
            self._close_requested = False
            self._error = None
            self._loop = self._closing = self._scope = None
            self._client = self._tools = None
            self._transport_failures = 0
            self._call_timeout = call_timeout
            self._dead = False
            if self._url is None:
                # A real file, not a pipe yian must drain: the subprocess writes it
                # directly, and its stderr never reaches the host's.
                self._stderr = tempfile.TemporaryFile("w+")
            thread = self._thread = threading.Thread(
                target=self._run,
                args=(connect_timeout, call_timeout),
                name=f"yian-mcp-{self._label}",
                daemon=True,
            )
        try:
            thread.start()
            deadline = time.monotonic() + connect_timeout + 10
            # Short waits, so a SIGINT in the main thread raises promptly.
            while not ready.wait(0.1):
                if not thread.is_alive():
                    break
                if time.monotonic() > deadline:
                    raise McpConnectError(
                        f"MCP server '{self._label}' didn't start within {connect_timeout:g} s"
                    )
            with self._lock:
                if self._close_requested:
                    raise McpConnectError(f"MCP server '{self._label}' was closed while starting")
                if not ready.is_set():
                    raise self._error or McpConnectError(
                        f"MCP server '{self._label}' stopped while starting"
                    )
                self._owner = owner
        except BaseException:
            self.close()
            raise
        return self

    def close(self) -> None:
        """Stop the server and its thread. Never raises; closing twice is fine.

        Waits up to 10 s for the server to shut down. Safe to call from any thread and
        before `open()`.
        """
        try:
            with self._lock:
                self._close_requested = True
                loop, thread = self._loop, self._thread
                calls, self._calls = self._calls, set()
            # Waiting calls end now, not after the SDK's shutdown, and raise McpServerClosed.
            for future in calls:
                future.cancel()
            if loop is not None and thread is not None and thread.is_alive():
                with contextlib.suppress(RuntimeError):
                    loop.call_soon_threadsafe(self._stop)
            if thread is not None and thread is not threading.current_thread():
                thread.join(_CLOSE_JOIN_SECONDS)
        except Exception:
            pass
        finally:
            self._owner = None
            self._client = None
            self._tools = None
            self._thread = None
            self._loop = None
            stderr, self._stderr = self._stderr, None
            if stderr is not None:
                with contextlib.suppress(Exception):
                    stderr.close()

    def _stop(self) -> None:
        """In the loop thread: end `_main`, cancelling a connect still in progress."""
        if self._closing is not None:
            self._closing.set()
        if self._scope is not None and not self._ready.is_set():
            self._scope.cancel()

    def _run(self, connect_timeout: float, call_timeout: float) -> None:
        """The thread's target: run `_main` on a fresh asyncio loop."""
        ready = self._ready
        try:
            asyncio.run(self._main(connect_timeout, call_timeout))
        except BaseException as exc:
            if not self._ready.is_set() and self._error is None:
                self._error = self._connect_error(exc)
        finally:
            # Once the loop has ended no call can reach the server, whether close()
            # ended it or the server went away. A later open() replaces `_ready`, so a
            # thread that outlived close() leaves the new connection alone.
            if ready.is_set() and self._ready is ready:
                self._dead = True

    async def _main(self, connect_timeout: float, call_timeout: float) -> None:
        """Connect, list the tools, then hold the client open until `close()`."""
        import anyio
        from mcp import Client

        with self._lock:
            if self._close_requested:
                return
            self._loop = asyncio.get_running_loop()
            self._closing = asyncio.Event()
        try:
            # The deadline covers only the connect and the listing, but the scope must
            # enclose the client: anyio scopes exit in LIFO order, so a timeout scope
            # closed while the client stays open breaks the client.
            with anyio.CancelScope(deadline=anyio.current_time() + connect_timeout) as scope:
                self._scope = scope
                async with contextlib.AsyncExitStack() as stack:
                    transport = await self._enter_transport(stack, call_timeout)
                    # No client-wide read timeout: it would bound the handshake too,
                    # which the connect deadline covers; each call passes its own.
                    client = await stack.enter_async_context(Client(transport, mode=self._mode))
                    tools = await self._list_all(client)
                    scope.deadline = math.inf
                    self._client = client
                    self._tools = tools
                    self._ready.set()
                    await self._closing.wait()
                    # Bounds the shutdown: an HTTP close sends a DELETE that a silent
                    # host could hold past close()'s join. The SDK shields stdio's.
                    scope.deadline = anyio.current_time() + 5
        except BaseException as exc:
            if not self._ready.is_set():
                self._error = self._connect_error(exc)
            return
        # A close() during the connect cancels the scope too; open() reports that itself.
        if scope.cancelled_caught and not self._ready.is_set() and not self._close_requested:
            self._error = McpConnectError(
                f"MCP server '{self._label}' didn't start within {connect_timeout:g} s"
                + self._stderr_tail()
            )

    async def _enter_transport(self, stack: contextlib.AsyncExitStack, call_timeout: float) -> Any:
        """Return the SDK transport for this server, entering any HTTP client on `stack`."""
        if self._url is None:
            from mcp import StdioServerParameters
            from mcp.client.stdio import stdio_client

            params = StdioServerParameters(
                command=self._command, args=list(self._args), env=self._env, cwd=self._cwd
            )
            return stdio_client(params, errlog=self._stderr)

        import httpx2
        from mcp.client.streamable_http import streamable_http_client

        http = await stack.enter_async_context(
            httpx2.AsyncClient(
                headers=dict(self._headers or {}),
                # The read timeout outlasts a call's own timeout, so the call's timeout
                # is the one that fires.
                timeout=httpx2.Timeout(30, read=max(call_timeout + 30, 300)),
                transport=_soft_fail_transport(httpx2, _proxy_for(self._url), self._count_failure),
            )
        )
        return streamable_http_client(self._url, http_client=http)

    def _count_failure(self) -> None:
        """Count one HTTP request that failed at the transport level."""
        with self._failures_lock:
            self._transport_failures += 1

    async def _list_all(self, client: Any) -> tuple[FunctionTool, ...]:
        """List every page of the server's tools and wrap each in a `FunctionTool`."""
        tools: list[FunctionTool] = []
        seen: set[str] = set()
        cursor: str | None = None
        for _ in range(_MAX_PAGES):
            page = await client.list_tools(cursor=cursor)
            for tool in page.tools:
                if tool.name in seen:
                    raise McpConnectError(
                        f"MCP server '{self._label}' lists the tool {tool.name!r} twice"
                    )
                if tool.input_schema.get("type") != "object":
                    raise McpConnectError(
                        f"MCP server '{self._label}' tool {tool.name!r} has an input schema "
                        "whose type is not 'object'"
                    )
                seen.add(tool.name)
                tools.append(self._function_tool(tool))
            cursor = page.next_cursor
            if cursor is None:
                return tuple(tools)
        raise McpConnectError(
            f"MCP server '{self._label}' lists more than {_MAX_PAGES} pages of tools"
        )

    def _function_tool(self, tool: Any) -> FunctionTool:
        """Wrap a listed MCP tool in a `FunctionTool` that calls it on this server."""
        name = tool.name

        def on_invoke(ctx: ToolContext[Any], args: dict[str, Any]) -> Any:
            return self._call(name, args, {"yian/call_id": ctx.call_id, "yian/attempt": ctx.attempt})

        return FunctionTool(
            name=name,
            description=tool.description or tool.title or name,
            params_json_schema=copy.deepcopy(tool.input_schema),
            on_invoke=on_invoke,
        )

    def _call(self, tool_name: str, args: dict[str, Any], meta: dict[str, Any]) -> Any:
        """Call `tool_name` on the server and return its result as a yian value.

        Waits in short slices, so a SIGINT, a stop of the calling run, or `close()`
        ends the wait promptly.

        Args:
            tool_name (`str`):
                The server's name for the tool.
            args (`dict[str, Any]`):
                The checked arguments.
            meta (`dict[str, Any]`):
                The request `_meta`: the yian call id and attempt.

        Returns:
            `Any`:
                The result's structured content, or its content rendered as text.

        Raises:
            `RuntimeError`:
                Raised when the server was never opened.
            `McpToolError`:
                Raised when the server returns an error result, or the call fails in
                a way no other error here covers.
            `McpRequestError`:
                Raised when the server answers with a JSON-RPC error;
                `McpInvalidRequestError` for a malformed or unknown request.
            `McpTimeoutError`:
                Raised when the call outlasts the call timeout.
            `McpTransportError`:
                Raised when the HTTP connection to the server fails.
            `McpServerClosed`:
                Raised when the server is closed or gone.
            `McpCallCancelled`:
                Raised when the calling run's stop event is set while the call waits.
        """
        stop = _current_stop()
        with self._lock:
            # The connection this call goes to; a later re-open replaces all three.
            client, loop, thread, ready = self._client, self._loop, self._thread, self._ready
            if self._close_requested or self._dead:
                raise self._closed()
            if client is None or loop is None:
                raise RuntimeError(f"MCP server '{self._label}' is not open")
            if stop is not None and stop.is_set():
                raise self._cancelled(tool_name)
            with self._failures_lock:
                failures_before = self._transport_failures
            started = time.monotonic()
            coro = client.call_tool(tool_name, args, read_timeout_seconds=self._call_timeout, meta=meta)
            try:
                future = asyncio.run_coroutine_threadsafe(coro, loop)
            except RuntimeError as exc:
                # The loop has closed: the server went away.
                coro.close()
                raise self._closed(ready) from exc
            self._calls.add(future)
        try:
            loop_gone = False
            while not concurrent.futures.wait([future], timeout=_WAIT_SLICE_SECONDS).done:
                if stop is not None and stop.is_set():
                    # Cancels the request task; the SDK tells the server from a shielded scope.
                    future.cancel()
                    raise self._cancelled(tool_name)
                if thread is None or not thread.is_alive():
                    # The loop resolves its calls as it ends; give it one more slice.
                    if loop_gone:
                        raise self._closed(ready)
                    loop_gone = True
            try:
                result = future.result()
            except Exception as exc:
                raise self._call_error(tool_name, exc, started, failures_before, ready) from exc
        finally:
            if not future.done():
                future.cancel()
            with self._lock:
                self._calls.discard(future)
        return _result_value(result)

    def _call_error(
        self,
        tool_name: str,
        exc: Exception,
        started: float,
        failures_before: int,
        ready: threading.Event,
    ) -> McpError:
        """Return the error a failed tool call raises.

        Only the server's own JSON-RPC message reaches the returned error's text; for
        everything else the text is fixed, since an exception's text may hold the URL.

        Args:
            tool_name (`str`):
                The server's name for the tool.
            exc (`Exception`):
                What the call raised.
            started (`float`):
                The `time.monotonic()` at which the call was sent.
            failures_before (`int`):
                The HTTP transport failure count when the call was sent.
            ready (`threading.Event`):
                The connection generation (`_ready`) the call was sent on.

        Returns:
            `McpError`:
                The error to raise.
        """
        from mcp import MCPError

        if self._close_requested or not self._loop_alive():
            return self._closed(ready)
        if isinstance(exc, concurrent.futures.CancelledError):
            # Not a stop of the calling run: that raised McpCallCancelled already.
            return self._closed(ready)
        if isinstance(exc, MCPError):
            code = exc.code
            elapsed = time.monotonic() - started
            if code == _REQUEST_TIMEOUT and elapsed >= (self._call_timeout or math.inf):
                return self._timed_out(tool_name)
            with self._failures_lock:
                failed_since = self._transport_failures > failures_before
            if code == _INTERNAL_ERROR and failed_since:
                # The soft-fail transport turned a connection error into this answer.
                return McpTransportError(f"MCP server '{self._label}' transport error (connection failed)")
            # A server also answers -32000 when merely busy, and an HTTP stream that
            # ends without a response arrives as -32000 too; only a failed probe means
            # the connection is gone.
            if code == _CONNECTION_CLOSED and not self._probe_alive():
                return self._closed(ready)
            kind = McpInvalidRequestError if code in _INVALID_REQUEST_CODES else McpRequestError
            return kind(f"MCP server '{self._label}' tool '{tool_name}': {exc.message} (code {code})", code)
        if isinstance(exc, TimeoutError):
            return self._timed_out(tool_name)
        if _is_transport_error(exc):
            return McpTransportError(f"MCP server '{self._label}' transport error ({type(exc).__name__})")
        return McpToolError(f"MCP server '{self._label}' tool '{tool_name}' failed ({type(exc).__name__})")

    def _loop_alive(self) -> bool:
        """Return whether the client's loop thread still runs and the server isn't known gone."""
        thread = self._thread
        return not self._dead and thread is not None and thread.is_alive()

    def _probe_alive(self) -> bool:
        """Ping the server and return whether it answered at all within 2 s.

        Any answer, even an error other than a closed connection, or no answer in
        time, counts as alive; a closed connection or an ended loop does not.
        """
        from mcp import MCPError

        client, loop = self._client, self._loop
        if client is None or loop is None:
            return False

        async def ping() -> None:
            await asyncio.wait_for(client.session.send_ping(), _PING_SECONDS)

        coro = ping()
        try:
            future = asyncio.run_coroutine_threadsafe(coro, loop)
        except RuntimeError:
            coro.close()
            return False
        try:
            future.result(timeout=_PING_SECONDS + 1)
        except concurrent.futures.CancelledError:
            return False
        except MCPError as exc:
            return exc.code != _CONNECTION_CLOSED
        except TimeoutError:
            future.cancel()
            return self._loop_alive()
        except Exception:
            return True
        return True

    def _closed(self, ready: threading.Event | None = None) -> McpServerClosed:
        """Mark the server gone and return the `McpServerClosed` a call raises.

        Args:
            ready (`threading.Event`, *optional*, defaults to `None`):
                The connection generation (`_ready`) the failed call was sent on. The
                server is marked gone only while that generation is current, so a call
                that resolves after a re-open leaves the new connection alone. `None`
                means the current one.
        """
        if ready is None or ready is self._ready:
            self._dead = True
        return McpServerClosed(f"MCP server '{self._label}' is closed")

    def _timed_out(self, tool_name: str) -> McpTimeoutError:
        """Return the `McpTimeoutError` for a call to `tool_name`."""
        return McpTimeoutError(
            f"MCP server '{self._label}' tool '{tool_name}' timed out after {self._call_timeout:g} s"
        )

    @staticmethod
    def _cancelled(tool_name: str) -> McpCallCancelled:
        """Return the `McpCallCancelled` for a call to `tool_name`."""
        return McpCallCancelled(f"MCP call to '{tool_name}' cancelled by a run stop")

    def _connect_error(self, exc: BaseException) -> McpConnectError:
        """Return the `McpConnectError` for a failure while the server was starting."""
        leaf = _leaf(exc)
        if isinstance(leaf, McpConnectError):
            return leaf
        if self._url is None:
            message = (
                f"MCP server '{self._label}' didn't start: {type(leaf).__name__}: {leaf}"
                + self._stderr_tail()
            )
        else:
            # Only the type and status: the URL and the error text may hold a token.
            response = getattr(leaf, "response", None)
            status = getattr(response, "status_code", None)
            message = f"MCP server '{self._label}' didn't start: {type(leaf).__name__}"
            if isinstance(status, int):
                message += f" (HTTP {status})"
        error = McpConnectError(message)
        error.__cause__ = leaf
        return error

    def _stderr_tail(self) -> str:
        """Return the last lines the stdio server wrote to stderr, as a message suffix."""
        if self._stderr is None:
            return ""
        try:
            self._stderr.flush()
            self._stderr.seek(0)
            lines = self._stderr.read().splitlines()[-_STDERR_LINES:]
        except Exception:
            return ""
        if not lines:
            return ""
        return "\nserver stderr:\n" + "\n".join(lines)


def _result_value(result: Any) -> Any:
    """Return a tool call's result as a yian value.

    Args:
        result (`mcp_types.CallToolResult`):
            The server's result.

    Returns:
        `Any`:
            The structured content when the server sent it, else the content
            rendered as one string (see `_render_content`).

    Raises:
        `McpToolError`:
            Raised when the result is an error result; its message is the rendered
            content.
    """
    if result.is_error:
        raise McpToolError(_render_content(result.content))
    if result.structured_content is not None:
        return result.structured_content
    return _render_content(result.content)


def _render_content(content: Sequence[Any]) -> str:
    """Render MCP content blocks as text, joined by blank lines.

    Text blocks and embedded text resources give their text. Images, audio, embedded
    blob resources and resource links give a one-line marker with the mime type,
    when known, and the decoded size; their bytes are dropped.

    Args:
        content (`Sequence[mcp_types.ContentBlock]`):
            The result's content blocks.

    Returns:
        `str`:
            The rendered text; `""` for no content.
    """
    parts: list[str] = []
    for block in content:
        kind = getattr(block, "type", None)
        if kind == "text":
            parts.append(block.text)
        elif kind in ("image", "audio"):
            parts.append(f"[{kind} {block.mime_type}, {_size(block.data)}]")
        elif kind == "resource":
            resource = block.resource
            text = getattr(resource, "text", None)
            if text is not None:
                parts.append(text)
            else:
                head = " ".join(str(part) for part in (resource.uri, resource.mime_type) if part)
                parts.append(f"[resource {head}, {_size(resource.blob)}]")
        elif kind == "resource_link":
            head = " ".join(str(part) for part in (block.uri, block.mime_type) if part)
            parts.append(f"[resource {head}]")
        else:
            parts.append(f"[{kind}]")
    return "\n\n".join(parts)


def _size(data: str) -> str:
    """Return the size of base64 `data` once decoded: `<n> B`, `<x.y> KB` or `<x.y> MB`."""
    try:
        size = len(base64.b64decode(data))
    except (binascii.Error, ValueError):
        # Estimate from the encoded length when the server sent malformed base64.
        size = len(data) * 3 // 4
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 * 1024):.1f} MB"


def _leaf(exc: BaseException) -> BaseException:
    """Return the first leaf of nested exception groups, or `exc` itself."""
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    return exc


def _is_transport_error(exc: BaseException) -> bool:
    """Return whether `exc` is a network-level failure: an `OSError` or an `httpx2` transport error."""
    if isinstance(exc, OSError):
        return True
    try:
        import httpx2
    except ImportError:
        return False
    return isinstance(exc, httpx2.TransportError)


def _proxy_for(url: str) -> str | None:
    """Return the proxy the environment sets for `url`, or `None`."""
    parts = urllib.parse.urlsplit(url)
    if parts.hostname and urllib.request.proxy_bypass(parts.hostname):
        return None
    return urllib.request.getproxies().get(parts.scheme)


def _soft_fail_transport(httpx2: Any, proxy: str | None, on_failure: Any) -> Any:
    """Return an `httpx2` transport that turns connection errors into 502 responses.

    A connection error raised inside the SDK's streamable HTTP transport escapes its
    task group and ends the whole client; a 502 fails only that request, so the
    client survives a server that is briefly down.

    Args:
        httpx2 (`module`):
            The `httpx2` module.
        proxy (`str`, *optional*):
            The proxy URL, or `None` for a direct connection.
        on_failure (`Callable[[], None]`):
            Called once for each request that failed at the transport level.

    Returns:
        `httpx2.AsyncBaseTransport`:
            The transport.
    """

    class _SoftFailTransport(httpx2.AsyncBaseTransport):
        def __init__(self) -> None:
            self._inner = httpx2.AsyncHTTPTransport(proxy=proxy)

        async def handle_async_request(self, request: Any) -> Any:
            try:
                return await self._inner.handle_async_request(request)
            except httpx2.TransportError:
                on_failure()
                return httpx2.Response(502, request=request)

        async def aclose(self) -> None:
            await self._inner.aclose()

    return _SoftFailTransport()
