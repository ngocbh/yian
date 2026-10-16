"""Read `mcpServers` JSON files into unopened MCP servers for `yian run`.

The CLI reads the user's `~/.yian/mcp.json` when it exists, plus each file named
with `--mcp-config`; nothing is discovered from the working directory. Each file
holds the common `{"mcpServers": {name: entry}}` shape. `${VAR}` and
`${VAR:-default}` in string values expand from the given environment at load.
"""

from __future__ import annotations

import json
import re
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from yian.tools.mcp import McpServer, mcp_server

_KNOWN_KEYS = frozenset({"command", "args", "env", "cwd", "url", "headers", "type", "disabled"})
# A tuple, not a set, so an unhashable `type` value compares instead of raising.
_TYPES = ("stdio", "http", "streamable-http")
# `$$` first so `$${X}` is a literal `${X}`; any other `${` that doesn't match the
# reference form falls to the last branch and is an error.
_REFERENCE = re.compile(r"\$\$|\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}|\$\{")


class McpConfigError(ValueError):
    """An MCP config file can't be read; the message names the file and server (and field)."""


def user_mcp_config() -> Path:
    """Return the path of the user's MCP config file.

    Returns:
        `Path`:
            `~/.yian/mcp.json`, whether or not it exists.
    """
    return Path.home() / ".yian" / "mcp.json"


def load_mcp_config(paths: Sequence[Path], environ: Mapping[str, str]) -> dict[str, McpServer]:
    """Read `mcpServers` files, in order, into unopened MCP servers keyed by name.

    A later file's entry replaces an earlier entry of the same name whole; an entry
    with `"disabled": true` drops the server, so `{"disabled": true}` alone turns off
    a server an earlier file defines. Entry keys read are `command`, `args`, `env`,
    `cwd`, `url`, `headers`, `type` (`"stdio"`, `"http"` or `"streamable-http"`) and
    `disabled`; any other key is ignored with a `UserWarning`. In `command`, each
    `args` item, `cwd`, `url`, and the values of `env` and `headers`, `${VAR}` expands
    to `environ[VAR]`, `${VAR:-default}` to `default` when `VAR` is unset or empty,
    and `$$` to `$`; any other `$` stays as written.

    Args:
        paths (`Sequence[Path]`):
            The config files to read, each of which must exist. Later files win.
        environ (`Mapping[str, str]`):
            The variables `${VAR}` references expand from, usually `os.environ`.

    Returns:
        `dict[str, McpServer]`:
            The enabled servers by name, each labelled with its name and not yet open.

    Raises:
        `McpConfigError`:
            Raised when a file can't be read or isn't JSON, has no `mcpServers`
            object, or an entry isn't an object, has both or neither of `command` and
            `url`, has a value of the wrong type, a `type` that is unknown, `"sse"`,
            or disagrees with `command`/`url`, references an unset variable without a
            default, or fails `mcp_server`'s checks.

    Example:
        ```python
        paths = [user_mcp_config()] if user_mcp_config().exists() else []
        servers = load_mcp_config(paths, os.environ)
        ```
    """
    servers: dict[str, McpServer] = {}
    for path in paths:
        for name, entry in _read_entries(path).items():
            server = _build(path, name, entry, environ)
            if server is None:
                servers.pop(name, None)
            else:
                servers[name] = server
    return servers


def _read_entries(path: Path) -> dict[str, Any]:
    """Return the `mcpServers` object of the file at `path`."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as error:
        raise McpConfigError(f"{path}: can't read the MCP config: {error.strerror or error}") from error
    except UnicodeDecodeError as error:
        raise McpConfigError(f"{path}: the MCP config is not valid JSON: {error}") from error
    try:
        document = json.loads(text)
    except json.JSONDecodeError as error:
        raise McpConfigError(f"{path}: the MCP config is not valid JSON: {error}") from error
    if not isinstance(document, dict) or not isinstance(document.get("mcpServers"), dict):
        raise McpConfigError(f'{path}: the MCP config needs an "mcpServers" object mapping names to servers')
    return document["mcpServers"]


def _build(path: Path, name: str, entry: Any, environ: Mapping[str, str]) -> McpServer | None:
    """Return the server one entry describes, or `None` when it is disabled."""
    where = f"{path}: server {name!r}"
    if not isinstance(entry, dict):
        raise McpConfigError(f"{where}: the entry must be an object, got {_kind(entry)}")
    for key in entry:
        if key not in _KNOWN_KEYS:
            warnings.warn(f"{path}: server {name!r}: ignoring unknown key {key!r}", UserWarning, stacklevel=3)
    disabled = entry.get("disabled", False)
    if not isinstance(disabled, bool):
        raise McpConfigError(f"{where}: 'disabled' must be true or false, got {_kind(disabled)}")
    if disabled:
        return None

    command = _optional_str(where, entry, "command")
    url = _optional_str(where, entry, "url")
    if (command is None) == (url is None):
        raise McpConfigError(
            f"{where}: give exactly one of 'command' (a stdio server) and 'url' (an HTTP server)"
        )
    kind = entry.get("type")
    if kind is not None:
        if kind == "sse":
            raise McpConfigError(f"{where}: 'type': the SSE transport isn't supported; use streamable HTTP")
        if kind not in _TYPES:
            raise McpConfigError(f"{where}: 'type' must be one of {', '.join(_TYPES)}; got {kind!r}")
        needed, other = ("command", "url") if kind == "stdio" else ("url", "command")
        if entry.get(needed) is None:
            raise McpConfigError(f"{where}: 'type' {kind!r} needs {needed!r}, not {other!r}")

    args = entry.get("args", [])
    if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
        raise McpConfigError(f"{where}: 'args' must be a list of strings")
    env = _optional_str_object(where, entry, "env")
    headers = _optional_str_object(where, entry, "headers")
    cwd = _optional_str(where, entry, "cwd")

    def expand(field: str, text: str | None) -> str | None:
        return None if text is None else _expand(where, field, text, environ)

    def expand_object(field: str, value: dict[str, str] | None) -> dict[str, str] | None:
        if value is None:
            return None
        return {key: _expand(where, f"{field}[{key!r}]", item, environ) for key, item in value.items()}

    fields = {
        "command": expand("command", command),
        "args": [expand(f"args[{index}]", arg) for index, arg in enumerate(args)],
        "env": expand_object("env", env),
        "cwd": expand("cwd", cwd),
        "url": expand("url", url),
        "headers": expand_object("headers", headers),
    }
    try:
        return mcp_server(**fields, label=name)
    except (TypeError, ValueError) as error:
        raise McpConfigError(f"{where}: {error}") from error


def _optional_str(where: str, entry: dict[str, Any], key: str) -> str | None:
    """Return `entry[key]` when it is a string, `None` when absent, else raise."""
    value = entry.get(key)
    if value is not None and not isinstance(value, str):
        raise McpConfigError(f"{where}: {key!r} must be a string, got {_kind(value)}")
    return value


def _optional_str_object(where: str, entry: dict[str, Any], key: str) -> dict[str, str] | None:
    """Return `entry[key]` when it is an object of strings, `None` when absent, else raise."""
    value = entry.get(key)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise McpConfigError(f"{where}: {key!r} must be an object of strings, got {_kind(value)}")
    for item_key, item in value.items():
        if not isinstance(item, str):
            raise McpConfigError(f"{where}: {key}[{item_key!r}] must be a string, got {_kind(item)}")
    return value


def _expand(where: str, field: str, text: str, environ: Mapping[str, str]) -> str:
    """Return `text` with `$$`, `${VAR}` and `${VAR:-default}` expanded from `environ`."""
    if "$" not in text:
        return text

    def replace(match: re.Match[str]) -> str:
        whole, variable, default = match.group(0), match.group(1), match.group(2)
        if whole == "$$":
            return "$"
        if variable is None:
            raise McpConfigError(
                f"{where}: {field}: '${{' must start a ${{VAR}} or ${{VAR:-default}} reference;"
                " write '$$' for a literal '$'"
            )
        value = environ.get(variable)
        if default is not None:
            return value if value else default
        if value is None:
            raise McpConfigError(
                f"{where}: {field}: ${{{variable}}} is not set; set it or write ${{{variable}:-default}}"
            )
        return value

    return _REFERENCE.sub(replace, text)


def _kind(value: Any) -> str:
    """Return the JSON name of `value`'s type, for error messages."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "a boolean"
    if isinstance(value, (int, float)):
        return "a number"
    if isinstance(value, str):
        return "a string"
    if isinstance(value, list):
        return "an array"
    return "an object"
