"""`yian run` starts the MCP servers a flow needs from `~/.yian/mcp.json` and `--mcp-config`."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp")

from langchain_core.messages import AIMessage
from typer.testing import CliRunner

import yian.cli.run as run_cli
import yian.llm_clients as llm_clients
from yian.cli import app
from yian.compose import load_flow
from yian.run import run_flow
from yian.tools.registry import TOOL_REGISTRY

from tests.engine.mcp_helpers import FAKE, child_alive, child_env, mcp_threads, run_http_server

runner = CliRunner()

_WHOAMI_FLOW = """\
id: mcp_resume
name: mcp_resume
nodes:
  call:
    kind: tool
    tool_id: whoami
output: ${call.output}
"""

_AGENT_FLOW = """\
id: mcp_agent
name: mcp_agent
llm_config: {provider: openai, model: test-model}
nodes:
  worker:
    kind: agent
    prompt: go
    tools: [fake]
output: ${worker.output}
"""

_PLAIN_FLOW = """\
id: plain
name: plain
nodes:
  pick:
    kind: code
    code: "return 'plain'"
    output: str
output: ${pick.output}
"""


class _Model:
    """Answer at once and record the tools bound."""

    def __init__(self) -> None:
        self.bound: list = []

    def bind_tools(self, tools):
        self.bound = list(tools)
        return self

    def invoke(self, messages):
        return AIMessage(content="done")


class _SimulatedCrash(BaseException):
    """Process-loss marker that bypasses node-failure conversion."""


def _tool_flow(tmp_path: Path, tool_id: str = "echo") -> Path:
    """Write a flow whose one TOOL node calls `tool_id` with `text: "hi"`."""
    flow = tmp_path / "flow.yaml"
    flow.write_text(
        f"id: mcp_tool\nname: mcp_tool\nnodes:\n  call:\n    kind: tool\n    tool_id: {tool_id}\n"
        '    input: {text: "hi"}\noutput: ${call.output}\n',
        encoding="utf-8",
    )
    return flow


def _stdio(*flags: str) -> dict:
    """Return a config entry that starts the fake server over stdio with `flags`."""
    return {"command": sys.executable, "args": [str(FAKE), *flags], "env": child_env()}


def _config(path: Path, servers: dict) -> Path:
    """Write `servers` as an `mcpServers` file at `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"mcpServers": servers}), encoding="utf-8")
    return path


def _pids(marker: Path) -> list[int]:
    """Return the pids the fake server wrote to `marker`, one per start."""
    return [int(line) for line in marker.read_text().split()] if marker.exists() else []


def _assert_no_children(*markers: Path) -> None:
    """Assert no MCP thread runs and every server child recorded in `markers` is gone."""
    assert mcp_threads() == []
    for marker in markers:
        for pid in _pids(marker):
            assert not child_alive(pid)


def _run(*args: str):
    return runner.invoke(app, ["run", *args, "--quiet"])


def test_a_tool_node_calls_a_server_from_mcp_config(tmp_path):
    marker = tmp_path / "pid"
    cfg = _config(tmp_path / "cfg.json", {"fake": _stdio("--marker", str(marker))})

    result = _run(str(_tool_flow(tmp_path)), "--mcp-config", str(cfg))

    assert result.exit_code == 0, result.output
    assert "hi" in result.output
    assert len(_pids(marker)) == 1
    _assert_no_children(marker)


def test_a_tool_node_calls_an_http_server(tmp_path):
    with run_http_server() as (url, _headers):
        cfg = _config(tmp_path / "cfg.json", {"web": {"url": url}})
        result = _run(str(_tool_flow(tmp_path)), "--mcp-config", str(cfg))

    assert result.exit_code == 0, result.output
    assert "hi" in result.output
    _assert_no_children()


def test_the_user_config_is_read_and_a_flag_file_replaces_its_server(tmp_path, isolated_cli_home):
    flow = _tool_flow(tmp_path)
    marker = tmp_path / "pid"
    _config(isolated_cli_home / ".yian" / "mcp.json", {"fake": _stdio("--marker", str(marker))})

    from_user = _run(str(flow))

    assert from_user.exit_code == 0, from_user.output
    assert "hi" in from_user.output
    assert len(_pids(marker)) == 1

    _config(isolated_cli_home / ".yian" / "mcp.json", {"fake": {"command": str(tmp_path / "missing")}})
    cfg = _config(tmp_path / "cfg.json", {"fake": _stdio("--marker", str(marker))})

    replaced = _run(str(flow), "--mcp-config", str(cfg))

    assert replaced.exit_code == 0, replaced.output
    assert "hi" in replaced.output
    assert len(_pids(marker)) == 2
    _assert_no_children(marker)


def test_a_flow_without_tools_starts_no_server(tmp_path):
    flow = tmp_path / "flow.yaml"
    flow.write_text(_PLAIN_FLOW, encoding="utf-8")
    marker = tmp_path / "pid"
    cfg = _config(tmp_path / "cfg.json", {"fake": _stdio("--marker", str(marker))})

    result = _run(str(flow), "--mcp-config", str(cfg))

    assert result.exit_code == 0, result.output
    assert "plain" in result.output
    assert not marker.exists()


def test_unused_servers_with_colliding_tools_dont_fail_the_run(tmp_path):
    markers = {name: tmp_path / f"{name}.pid" for name in "abc"}
    cfg = _config(
        tmp_path / "cfg.json",
        {
            "a": _stdio("--only", "add", "--marker", str(markers["a"])),
            "b": _stdio("--only", "add", "--marker", str(markers["b"])),
            "c": _stdio("--only", "echo", "--marker", str(markers["c"])),
        },
    )

    result = _run(str(_tool_flow(tmp_path)), "--mcp-config", str(cfg))

    assert result.exit_code == 0, result.output
    assert "hi" in result.output
    # No server is named `echo`, so every server opened to find it; a and b then closed.
    assert [len(_pids(marker)) for marker in markers.values()] == [1, 1, 1]
    _assert_no_children(*markers.values())


def test_an_agent_grant_starts_only_the_granted_server(monkeypatch, tmp_path):
    flow = tmp_path / "flow.yaml"
    flow.write_text(_AGENT_FLOW, encoding="utf-8")
    model = _Model()
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _config: model)
    monkeypatch.setattr(run_cli, "_ensure_provider_keys", lambda *args: None)
    fake_marker, other_marker = tmp_path / "fake.pid", tmp_path / "other.pid"
    cfg = _config(
        tmp_path / "cfg.json",
        {
            "fake": _stdio("--marker", str(fake_marker)),
            "other": _stdio("--marker", str(other_marker)),
        },
    )

    result = _run(str(flow), "--mcp-config", str(cfg))

    assert result.exit_code == 0, result.output
    assert "echo" in {tool.name for tool in model.bound}
    assert len(_pids(fake_marker)) == 1
    assert not other_marker.exists()
    _assert_no_children(fake_marker)


def test_resume_re_supplies_the_servers(tmp_path):
    flow = tmp_path / "flow.yaml"
    flow.write_text(_WHOAMI_FLOW, encoding="utf-8")
    log = tmp_path / "run.jsonl"

    def whoami() -> str:
        """Crash the process on the call."""
        raise _SimulatedCrash()

    with pytest.raises(_SimulatedCrash):
        run_flow(load_flow(_WHOAMI_FLOW), {}, log=log, tools={"whoami": whoami}, run_id="run-5c2e")
    marker = tmp_path / "pid"
    cfg = _config(tmp_path / "cfg.json", {"fake": _stdio("--marker", str(marker))})

    result = runner.invoke(
        app, ["run", str(flow), "--resume", str(log), "--mcp-config", str(cfg), "--quiet"]
    )

    assert result.exit_code == 0, result.output
    assert "run-5c2e/call" in result.output
    assert len(_pids(marker)) == 1
    _assert_no_children(marker)


def test_a_bad_config_exits_with_its_message(tmp_path):
    cfg = tmp_path / "cfg.json"
    cfg.write_text("{not json", encoding="utf-8")

    result = _run(str(_tool_flow(tmp_path)), "--mcp-config", str(cfg))

    assert result.exit_code == 1
    assert "the MCP config is not valid JSON" in result.output


def test_a_needed_server_that_fails_to_open_exits_naming_it(tmp_path):
    cfg = _config(tmp_path / "cfg.json", {"broken": {"command": str(tmp_path / "missing")}})

    result = _run(str(_tool_flow(tmp_path)), "--mcp-config", str(cfg))

    assert result.exit_code == 1
    assert "MCP server 'broken'" in result.output
    _assert_no_children()


def test_a_server_named_like_a_registered_tool_exits(tmp_path, monkeypatch):
    import yian.cli  # noqa: F401 - registers the built-in flow tools

    monkeypatch.setitem(TOOL_REGISTRY, "shadowed", TOOL_REGISTRY["read_flow"])
    marker = tmp_path / "pid"
    cfg = _config(tmp_path / "cfg.json", {"shadowed": _stdio("--marker", str(marker))})

    result = _run(str(_tool_flow(tmp_path)), "--mcp-config", str(cfg))

    assert result.exit_code == 1
    assert "'shadowed'" in result.output and "registered tool" in result.output
    assert not marker.exists()


def test_a_server_named_ask_user_exits(tmp_path):
    cfg = _config(tmp_path / "cfg.json", {"ask_user": _stdio()})

    result = _run(str(_tool_flow(tmp_path)), "--mcp-config", str(cfg))

    assert result.exit_code == 1
    assert "'ask_user'" in result.output


def test_a_later_file_disables_a_user_server(tmp_path, isolated_cli_home):
    marker = tmp_path / "pid"
    _config(isolated_cli_home / ".yian" / "mcp.json", {"fake": _stdio("--marker", str(marker))})
    cfg = _config(tmp_path / "cfg.json", {"fake": {"disabled": True}})

    result = _run(str(_tool_flow(tmp_path)), "--mcp-config", str(cfg))

    assert result.exit_code == 1
    assert "unknown tool id 'echo'" in result.output
    assert not marker.exists()
