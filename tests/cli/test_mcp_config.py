"""Tests for reading `mcpServers` files into unopened MCP servers."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest

from yian.cli.mcp_config import McpConfigError, load_mcp_config, user_mcp_config
from yian.tools import McpServer


def _write(path: Path, servers: object) -> Path:
    path.write_text(json.dumps({"mcpServers": servers}), encoding="utf-8")
    return path


def test_user_config_is_under_the_home_yian_directory(isolated_cli_home: Path) -> None:
    assert user_mcp_config() == isolated_cli_home / ".yian" / "mcp.json"


def test_stdio_entry_expands_env_and_keeps_args(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "mcp.json",
        {
            "files": {
                "type": "stdio",
                "command": "npx",
                "args": ["-y", "server-files", "."],
                "env": {"T": "${TOK}"},
                "cwd": "/srv",
            }
        },
    )

    servers = load_mcp_config([path], {"TOK": "abc"})

    assert list(servers) == ["files"]
    server = servers["files"]
    assert isinstance(server, McpServer)
    assert server._command == "npx"
    assert server._args == ("-y", "server-files", ".")
    assert server._env == {"T": "abc"}
    assert server._cwd == "/srv"
    assert server._url is None
    assert server._label == "files"
    assert server._owner is None


@pytest.mark.parametrize("kind", ["http", "streamable-http"])
def test_http_entry_expands_headers(tmp_path: Path, kind: str) -> None:
    path = _write(
        tmp_path / "mcp.json",
        {
            "remote": {
                "type": kind,
                "url": "https://tools.example.com/mcp",
                "headers": {"Authorization": "Bearer ${TOKEN}"},
            }
        },
    )

    server = load_mcp_config([path], {"TOKEN": "s3cret"})["remote"]

    assert server._url == "https://tools.example.com/mcp"
    assert server._headers == {"Authorization": "Bearer s3cret"}
    assert server._command is None
    assert server._label == "remote"


def test_sse_type_is_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path / "mcp.json", {"old": {"type": "sse", "url": "https://x.example.com/sse"}})

    message = "the SSE transport isn't supported; use streamable HTTP"
    with pytest.raises(McpConfigError, match=message) as info:
        load_mcp_config([path], {})

    assert str(path) in str(info.value)
    assert "'old'" in str(info.value)


@pytest.mark.parametrize(
    "entry",
    [
        {"type": "websocket", "url": "https://x.example.com"},
        {"type": "stdio", "url": "https://x.example.com"},
        {"type": "http", "command": "srv"},
        {"type": 3, "command": "srv"},
        {"type": ["stdio"], "command": "srv"},
    ],
)
def test_type_must_be_known_and_agree_with_command_or_url(tmp_path: Path, entry: dict) -> None:
    path = _write(tmp_path / "mcp.json", {"s": entry})

    with pytest.raises(McpConfigError, match="type") as info:
        load_mcp_config([path], {})

    assert str(path) in str(info.value)
    assert "'s'" in str(info.value)


def test_missing_variable_names_file_server_and_field(tmp_path: Path) -> None:
    path = _write(tmp_path / "mcp.json", {"files": {"command": "srv", "args": ["--key", "${MISSING}"]}})

    with pytest.raises(McpConfigError) as info:
        load_mcp_config([path], {})

    message = str(info.value)
    assert str(path) in message
    assert "'files'" in message
    assert "args[1]" in message
    assert "MISSING" in message


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("${MISSING:-d}", "d"),
        ("${SET:-d}", "v"),
        ("x${SET}y", "xvy"),
        ("a$$b", "a$b"),
        ("$${SET}", "${SET}"),
        ("cost $5", "cost $5"),
        ("plain", "plain"),
    ],
)
def test_expansion_forms(tmp_path: Path, text: str, expected: str) -> None:
    path = _write(tmp_path / "mcp.json", {"s": {"command": "srv", "args": [text]}})

    server = load_mcp_config([path], {"SET": "v"})["s"]

    assert server._args == (expected,)


def test_expansion_covers_command_cwd_url_and_header_values(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "mcp.json",
        {
            "local": {"command": "${BIN}/srv", "cwd": "${HOME_DIR}"},
            "remote": {"url": "https://${HOST}/mcp", "headers": {"X-Key": "${KEY}"}},
        },
    )
    environ = {"BIN": "/opt/bin", "HOME_DIR": "/home/me", "HOST": "tools.example.com", "KEY": "k"}

    servers = load_mcp_config([path], environ)

    assert servers["local"]._command == "/opt/bin/srv"
    assert servers["local"]._cwd == "/home/me"
    assert servers["remote"]._url == "https://tools.example.com/mcp"
    assert servers["remote"]._headers == {"X-Key": "k"}


def test_unterminated_reference_is_an_error(tmp_path: Path) -> None:
    path = _write(tmp_path / "mcp.json", {"s": {"command": "srv", "env": {"A": "${OPEN"}}})

    with pytest.raises(McpConfigError, match=r"env\['A'\]"):
        load_mcp_config([path], {"OPEN": "x"})


def test_a_reference_inside_a_default_is_an_error(tmp_path: Path) -> None:
    path = _write(tmp_path / "mcp.json", {"s": {"command": "srv", "args": ["${A:-${B}}"]}})

    with pytest.raises(McpConfigError, match=r"args\[0\]: '\$\{' must start"):
        load_mcp_config([path], {"B": "b"})


def test_a_reserved_server_name_is_an_error_naming_the_file(tmp_path: Path) -> None:
    first = _write(tmp_path / "first.json", {"ok": {"command": "srv"}})
    second = _write(tmp_path / "second.json", {"ask_user": {"command": "srv"}})

    with pytest.raises(McpConfigError) as caught:
        load_mcp_config([first, second], {}, reserved={"ask_user", "load_skill"})

    assert str(caught.value) == (
        f"{second}: server 'ask_user': the name is reserved (a registered tool, ask_user"
        " or load_skill); rename the server"
    )


def test_a_reserved_name_a_later_file_disables_is_no_error(tmp_path: Path) -> None:
    first = _write(tmp_path / "first.json", {"ask_user": {"command": "srv"}})
    second = _write(tmp_path / "second.json", {"ask_user": {"disabled": True}})

    assert load_mcp_config([first, second], {}, reserved={"ask_user"}) == {}


def test_later_file_replaces_same_named_server(tmp_path: Path) -> None:
    first = _write(tmp_path / "a.json", {"s": {"command": "old", "args": ["1"]}, "keep": {"command": "k"}})
    second = _write(tmp_path / "b.json", {"s": {"url": "https://new.example.com/mcp"}})

    servers = load_mcp_config([first, second], {})

    assert set(servers) == {"s", "keep"}
    assert servers["s"]._url == "https://new.example.com/mcp"
    assert servers["s"]._command is None
    assert servers["s"]._args == ()


def test_later_disabled_entry_removes_server(tmp_path: Path) -> None:
    first = _write(tmp_path / "a.json", {"s": {"command": "srv"}, "keep": {"command": "k"}})
    second = _write(tmp_path / "b.json", {"s": {"disabled": True}})

    servers = load_mcp_config([first, second], {})

    assert list(servers) == ["keep"]


def test_disabled_entry_with_command_is_skipped(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "mcp.json",
        {"off": {"command": "srv", "disabled": True}, "on": {"command": "srv", "disabled": False}},
    )

    servers = load_mcp_config([path], {})

    assert list(servers) == ["on"]


def test_unknown_keys_warn_once_each_and_server_loads(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "mcp.json",
        {"s": {"command": "srv", "timeout": 30, "autoApprove": [], "description": "x"}},
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        servers = load_mcp_config([path], {})

    messages = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert messages == [
        f"{path}: server 's': ignoring unknown key 'timeout'",
        f"{path}: server 's': ignoring unknown key 'autoApprove'",
        f"{path}: server 's': ignoring unknown key 'description'",
    ]
    assert servers["s"]._command == "srv"


def test_unknown_key_warning_is_a_user_warning(tmp_path: Path) -> None:
    path = _write(tmp_path / "mcp.json", {"s": {"command": "srv", "alwaysAllow": []}})

    with pytest.warns(UserWarning, match="ignoring unknown key 'alwaysAllow'"):
        load_mcp_config([path], {})


def test_not_json_names_the_file(tmp_path: Path) -> None:
    path = tmp_path / "mcp.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(McpConfigError, match="not valid JSON") as info:
        load_mcp_config([path], {})

    assert str(path) in str(info.value)


@pytest.mark.parametrize("document", [{}, {"servers": {}}, [], {"mcpServers": []}, {"mcpServers": "x"}])
def test_missing_or_wrong_mcp_servers_names_the_file(tmp_path: Path, document: object) -> None:
    path = tmp_path / "mcp.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(McpConfigError, match="mcpServers") as info:
        load_mcp_config([path], {})

    assert str(path) in str(info.value)


@pytest.mark.parametrize(
    ("entry", "field"),
    [
        ("srv", None),
        (["srv"], None),
        ({"command": "srv", "url": "https://x.example.com/mcp"}, "command"),
        ({}, "command"),
        ({"args": ["a"]}, "command"),
        ({"command": "srv", "args": "a b"}, "args"),
        ({"command": "srv", "args": ["a", 1]}, "args"),
        ({"command": "srv", "env": {"A": 1}}, "env"),
        ({"command": "srv", "env": ["A=1"]}, "env"),
        ({"url": "https://x.example.com/mcp", "headers": {"A": True}}, "headers"),
        ({"command": 5}, "command"),
        ({"command": "srv", "cwd": 1}, "cwd"),
        ({"url": 5}, "url"),
        ({"command": "srv", "disabled": "yes"}, "disabled"),
    ],
)
def test_bad_entries_name_file_server_and_field(tmp_path: Path, entry: object, field: str | None) -> None:
    path = _write(tmp_path / "mcp.json", {"s": entry})

    with pytest.raises(McpConfigError) as info:
        load_mcp_config([path], {})

    message = str(info.value)
    assert str(path) in message
    assert "'s'" in message
    if field is not None:
        assert field in message


def test_mcp_server_validation_errors_are_reraised(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "mcp.json",
        {"s": {"url": "ftp://x.example.com/mcp"}, "t": {"command": ""}},
    )

    with pytest.raises(McpConfigError, match="http or https") as info:
        load_mcp_config([path], {})

    assert str(path) in str(info.value)
    assert "'s'" in str(info.value)


def test_missing_file_names_it(tmp_path: Path) -> None:
    path = tmp_path / "absent.json"

    with pytest.raises(McpConfigError) as info:
        load_mcp_config([path], {})

    assert str(path) in str(info.value)


def test_no_files_gives_no_servers() -> None:
    assert load_mcp_config([], {}) == {}
