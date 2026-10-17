"""Helpers for tests that run the fake MCP servers in `tests/fixtures/mcp/`."""

import contextlib
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

_FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "mcp"
FAKE = _FIXTURES / "fake_server.py"
BAD_OUTPUT = _FIXTURES / "bad_output_server.py"
PAGED = _FIXTURES / "paged_server.py"


def child_env():
    """Return the environment a stdio server child needs to import `mcp`.

    The SDK starts children with a minimal environment that drops `PYTHONPATH`.

    Returns:
        `dict[str, str]`:
            The child's extra environment.
    """
    return {"PYTHONPATH": os.pathsep.join(sys.path)}


def stdio_params(*flags, script=FAKE):
    """Return the keyword arguments that start a fake server over stdio.

    Args:
        *flags (`str`):
            Command-line flags for the server script.
        script (`Path`, *optional*, defaults to `FAKE`):
            The server script.

    Returns:
        `dict[str, Any]`:
            `command`, `args`, and `env`, ready for `StdioServerParameters`.
    """
    return {"command": sys.executable, "args": [str(script), *flags], "env": child_env()}


class _RecordHeaders:
    """ASGI middleware that records each HTTP request's headers.

    Args:
        app (`Callable`):
            The wrapped ASGI app.
        recorded (`list[dict[str, str]]`):
            The list each request's headers are appended to.
    """

    def __init__(self, app, recorded):
        self.app = app
        self.recorded = recorded

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            self.recorded.append({key.decode("latin-1"): value.decode("latin-1") for key, value in scope["headers"]})
        await self.app(scope, receive, send)


class HttpServer:
    """The fake server over streamable HTTP on 127.0.0.1, in a daemon thread.

    Each `start` serves a fresh server, so a test can stop it and start it again on the
    same port.

    Args:
        only (`set[str]`, *optional*, defaults to `None`):
            Tool names to keep. `None` keeps every tool.
    """

    def __init__(self, only=None):
        self.only = only
        self.port = None
        self.recorded_headers = []
        self._server = None
        self._thread = None

    @property
    def url(self):
        """`str`: The server's MCP endpoint."""
        return f"http://localhost:{self.port}/mcp"

    def start(self, port=0):
        """Start serving and wait until the server listens.

        Args:
            port (`int`, *optional*, defaults to `0`):
                The port to listen on. 0 picks a free one.

        Raises:
            `RuntimeError`:
                Raised when the server is not listening within 10 seconds.
        """
        import uvicorn

        from tests.fixtures.mcp.fake_server import build_server

        app = _RecordHeaders(build_server(only=self.only).streamable_http_app(), self.recorded_headers)
        config = uvicorn.Config(app, host="127.0.0.1", port=port, lifespan="on", log_level="warning")
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        deadline = time.monotonic() + 10
        while not self._server.started:
            if not self._thread.is_alive() or time.monotonic() > deadline:
                raise RuntimeError(f"fake MCP HTTP server did not start on port {port}")
            time.sleep(0.02)
        self.port = self._server.servers[0].sockets[0].getsockname()[1]

    def stop(self):
        """Stop serving and wait for the thread to end."""
        if self._server is None:
            return
        self._server.should_exit = True
        self._thread.join(10)
        self._server = None
        self._thread = None


@contextlib.contextmanager
def run_http_server(only=None):
    """Serve the fake server over streamable HTTP for the duration of a `with` block.

    Args:
        only (`set[str]`, *optional*, defaults to `None`):
            Tool names to keep. `None` keeps every tool.

    Yields:
        `tuple[str, list[dict[str, str]]]`:
            The MCP endpoint URL and the headers of every HTTP request served so far.
    """
    server = HttpServer(only)
    server.start()
    try:
        yield server.url, server.recorded_headers
    finally:
        server.stop()


def child_alive(pid):
    """Return whether process `pid` exists and is not a zombie.

    Args:
        pid (`int`):
            The process id.

    Returns:
        `bool`:
            Whether the process is alive.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # A zombie still answers signal 0; only its parent can reap it.
    stat = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return bool(stat) and not stat.startswith("Z")


def mcp_threads():
    """Return the names of live threads whose name starts with `yian-mcp-`.

    Returns:
        `list[str]`:
            The thread names.
    """
    return [thread.name for thread in threading.enumerate() if thread.name.startswith("yian-mcp-")]


def free_port():
    """Return a TCP port on 127.0.0.1 that was free a moment ago.

    Returns:
        `int`:
            The port.
    """
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]
