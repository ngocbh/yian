"""Tests for the `ac chat` CLI — exercised through typer's `CliRunner`.

Covers: the subcommand is wired and renders its options; a scripted multi-turn drive; the
cwd-local fold-module import path; and turn-failure resilience (a failed turn is surfaced
and the session survives).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from typer.testing import CliRunner

import agent_composer.llm_clients as llm_clients_mod
from agent_composer.cli import app

runner = CliRunner()

# The bundled example chat flow, resolved from THIS file's location — not cwd. The `chat`
# command's `flow` arg is `exists=True`, so a cwd-relative path only works when pytest runs
# from the repo root; anchoring to `__file__` keeps these tests cwd-independent.
EXAMPLE_CHAT = Path(__file__).resolve().parents[2] / "examples" / "chat.yaml"

# Matches an ANSI SGR (color/style) escape sequence, e.g. "\x1b[1;36m".
_ANSI_SGR = re.compile(r"\x1b\[[0-9;]*m")


class _FakeModel:
    """A stand-in chat model: plain agent mode calls `model.invoke(msgs).content`."""

    def bind_tools(self, tools):
        return self

    def invoke(self, msgs):
        class R:
            content = "hello back"

        return R()


class _Msg:
    """A minimal AI message: `.content` for a final answer, `.tool_calls` for tool requests."""

    def __init__(self, content="", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


class _LoopyModel:
    """Never gives a final answer — always requests a tool, so the agent hits its
    tool-iteration cap and the turn fails (the failure the resilience path must survive)."""

    def bind_tools(self, tools):
        return self

    def invoke(self, msgs):
        return _Msg(tool_calls=[{"name": "list_flows", "args": {}, "id": "c"}])


class _GoodModel:
    """Answers immediately with text (no tool calls)."""

    def bind_tools(self, tools):
        return self

    def invoke(self, msgs):
        return _Msg(content="recovered reply")


class _ToolThenAnswer:
    """First `invoke` requests one tool; the second returns a final answer. Proves the
    `tool_calling` loop wires model -> tool -> model within a single turn. A fresh instance
    per turn (the `model_from_config` factory) resets the call counter, so each turn is
    "one tool call, then answer."""

    def __init__(self):
        self._n = 0

    def bind_tools(self, tools):
        return self

    def invoke(self, msgs):
        self._n += 1
        if self._n == 1:
            return _Msg(tool_calls=[{"name": "list_flows", "args": {}, "id": "c1"}])
        return _Msg(content="here are your flows")



def test_chat_help():
    # typer renders --help as a rich options table with ANSI color. Newer rich (what a
    # fresh dependency resolve pulls, e.g. on CI) styles each leading dash of an option
    # as its own color run, so the captured bytes split the flag apart:
    # "\x1b[1;36m-\x1b[0m\x1b[1;36m-workspace\x1b[0m". The token renders fine on screen
    # but "--workspace" is no longer a contiguous substring, so a raw assertion misses
    # (green on an older local rich, red on CI). Strip the color codes before asserting;
    # a wide width additionally keeps the option column from wrapping.
    r = runner.invoke(app, ["chat", "--help"], env={"COLUMNS": "200"})
    assert r.exit_code == 0
    assert "--workspace" in _ANSI_SGR.sub("", r.output)


def test_chat_two_turns(tmp_path, monkeypatch):
    # The AGENT node resolves its model via `agent_composer.llm_clients.model_from_config`
    # (the engine's `_default_llm` re-imports it lazily) — patch it to the fake.
    monkeypatch.setattr(llm_clients_mod, "model_from_config", lambda cfg: _FakeModel())
    # `--provider openai` makes `_ensure_provider_keys` pick openai for the flow's agent
    # (which sets no provider); satisfy its key check with a dummy env var (no product change).
    monkeypatch.setenv("OPENAI_API_KEY", "x")

    # Use the plain example flow (mode: plain) so no tools/keys are needed for the drive test.
    # It grows the transcript in its `output:` template bindings (no code node), so it needs
    # nothing on sys.path beyond the package.
    flow = EXAMPLE_CHAT
    # two user turns then EOF (CliRunner closes stdin -> loop breaks)
    r = runner.invoke(app, ["chat", str(flow), "--provider", "openai"], input="hi\nagain\n")
    assert r.exit_code == 0
    # The assistant reply is rendered as Markdown each turn; two turns -> it appears.
    assert "hello back" in r.output
    assert "session ended" in r.output


def test_chat_eof_exit_is_newline_terminated(tmp_path, monkeypatch):
    """Ctrl+D (EOF) at the `You:` prompt echoes no newline, so the exit notice must not be
    glued to the prompt line (`You: session ended`) — the command emits its own newline first.
    Regression for "the screen doesn't refresh / no trailing newline on exit"."""
    monkeypatch.setattr(llm_clients_mod, "model_from_config", lambda cfg: _FakeModel())
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    flow = EXAMPLE_CHAT
    r = runner.invoke(app, ["chat", str(flow), "--provider", "openai"], input="hi\n")
    assert r.exit_code == 0
    assert "session ended" in r.output
    assert "You: session ended" not in r.output   # the prompt line was terminated first


def test_chat_resolves_cwd_local_fold_module(tmp_path, monkeypatch):
    """`ac chat` must put the cwd on `sys.path` so a chat flow's `code: module:fn` fold
    ref resolves without the caller exporting PYTHONPATH — mirroring `ac run`.

    The bundled/example chat flows no longer use a CODE fold (they grow the transcript in
    `output:` template bindings), so this test uses a self-contained legacy-style chat flow
    whose turn folds via a cwd-local module. Without `_ensure_cwd_importable()`, the run
    fails to import the fold, the turn loop never yields a reply, and the user sees only
    "session ended".
    """
    monkeypatch.setattr(llm_clients_mod, "model_from_config", lambda cfg: _FakeModel())
    monkeypatch.setenv("OPENAI_API_KEY", "x")

    # A cwd-local fold module + a chat flow that references it by bare module name.
    (tmp_path / "localfold.py").write_text(
        "def fold_turn(inputs):\n"
        "    grown = inputs['transcript'] + '\\n' + inputs['message'] + '\\n' + inputs['reply']\n"
        "    return {'transcript': grown, 'exited': bool(inputs['exited'])}\n"
    )
    (tmp_path / "mychat.yaml").write_text(
        "id: mychat\n"
        "name: mychat\n"
        "input:\n"
        "  opening: str\n"
        "defs:\n"
        "  turn:\n"
        "    input:\n"
        "      transcript: str\n"
        "      exited: bool\n"
        "    nodes:\n"
        "      ask:\n"
        "        kind: human_input\n"
        '        prompt: "You:"\n'
        "        output: str\n"
        "      reply:\n"
        "        kind: agent\n"
        "        mode: plain\n"
        "        input:\n"
        "          transcript: ${input.transcript}\n"
        "          message:    ${ask.output}\n"
        "        output: str\n"
        "        prompt: hi ${transcript} ${message}\n"
        "      fold:\n"
        "        kind: code\n"
        "        code: localfold:fold_turn\n"
        "        input:\n"
        "          transcript: ${input.transcript}\n"
        "          message:    ${ask.output}\n"
        "          reply:      ${reply.output}\n"
        "          exited:     ${input.exited}\n"
        "        output:\n"
        "          transcript: str\n"
        "          exited: bool\n"
        "    output: ${fold.output}\n"
        "nodes:\n"
        "  chat:\n"
        "    kind: loop\n"
        "    call: turn\n"
        "    input:\n"
        "      transcript: ${input.opening}\n"
        "      exited: false\n"
        '    while: "not ${exited}"\n'
        "    max: 1000\n"
        "output: ${chat.output.transcript}\n"
    )

    # Run from tmp_path; it is NOT on sys.path until the command adds it.
    monkeypatch.chdir(tmp_path)
    assert str(tmp_path) not in sys.path
    try:
        r = runner.invoke(app, ["chat", "mychat.yaml", "--provider", "openai"], input="hi\n")
        assert r.exit_code == 0
        assert "hello back" in r.output  # the fold imported and the turn produced a reply
    finally:
        # The command prepends cwd to sys.path (idempotent); undo so it doesn't leak.
        if str(tmp_path) in sys.path:
            sys.path.remove(str(tmp_path))
        sys.modules.pop("localfold", None)


def test_chat_survives_a_failed_turn(tmp_path, monkeypatch):
    """A single bad turn must NOT end the session. When the reply agent fails (here: it hits
    its tool-iteration cap), the run terminates as `failed`; the loop must surface the error,
    say it is continuing, and RESTART the flow so the next turn works — not print a silent
    "session ended". Regression for the reported "session ended right after I run a flow" bug.
    """
    # First model resolution (turn 1) loops forever -> tool cap -> the turn fails. Every
    # later resolution (the restarted run's turn) answers cleanly. `model_from_config` is
    # called once per agent-node run, i.e. once per turn.
    calls = {"n": 0}

    def _factory(cfg):
        calls["n"] += 1
        return _LoopyModel() if calls["n"] == 1 else _GoodModel()

    monkeypatch.setattr(llm_clients_mod, "model_from_config", _factory)
    monkeypatch.setenv("OPENAI_API_KEY", "x")

    # The bundled composer chat (tool_calling reply). A workspace with one flow keeps
    # list_flows fast and confined.
    (tmp_path / "f.yaml").write_text("id: f\nname: f\ninput: {}\nkind: agent\noutput: str\nprompt: hi\n")

    # turn 1 fails (loopy), then turn 2 recovers, then EOF ends the session.
    r = runner.invoke(
        app, ["chat", "--workspace", str(tmp_path), "--provider", "openai"],
        input="run something\ngood turn\n",
    )
    assert r.exit_code == 0
    assert "turn failed" in r.output          # the failure was surfaced, not swallowed
    assert "tool-iteration cap" in r.output   # with the real reason
    assert "continuing" in r.output           # and the session was kept alive
    assert "recovered reply" in r.output      # the restarted run produced a working turn
    assert "session ended" in r.output


def test_chat_missing_key_no_tty_exits_clean(monkeypatch):
    """A key-requiring provider with no key and no TTY must exit cleanly (red on stderr, exit 1),
    NOT dump a raw traceback — parity with `ac run`. Both the buggy and fixed code exit 1 under
    CliRunner, so the guard is the exception TYPE: a clean `typer.Exit` (surfaces as `SystemExit`),
    never a bare `RuntimeError`. Regression for #16."""
    # `--provider openai` makes `_ensure_provider_keys` pick openai for the flow's agent (which sets
    # none); delete its key so the preflight raises regardless of the ambient environment. No model
    # is invoked — the raise is at the preflight, before any model call, so no key/network is needed.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    flow = EXAMPLE_CHAT
    r = runner.invoke(app, ["chat", str(flow), "--provider", "openai"], input="")
    assert r.exit_code == 1
    assert isinstance(r.exception, SystemExit)          # a clean typer.Exit, not a crash
    assert not isinstance(r.exception, RuntimeError)     # the real regression guard
    assert "OPENAI_API_KEY" in _ANSI_SGR.sub("", r.stderr)  # the red key message, on stderr


def test_chat_malformed_flow_exits_clean(tmp_path):
    """A malformed chat flow must exit cleanly (located `.yaml` error on stderr, exit 1), NOT dump a
    raw traceback — the SAME parity with `ac run` as the missing-key path. `load_flow` raises
    `LoadError` (a `ValueError`, not `RuntimeError`), so the missing-key guard alone would let it
    escape; the guard here is the exception TYPE — a clean `typer.Exit` (`SystemExit`), never a bare
    `ValueError`/`LoadError`. Regression for #16."""
    bad = tmp_path / "bad.yaml"
    bad.write_text("id: f\nname: f\ninput: {}\noutput: int\nnodes:\n  a:\n    kind: nonesuch\n")
    r = runner.invoke(app, ["chat", str(bad)], input="")
    assert r.exit_code == 1
    assert isinstance(r.exception, SystemExit)      # a clean typer.Exit, not a crash
    assert not isinstance(r.exception, ValueError)  # the real regression guard (LoadError is a ValueError)
    assert "bad.yaml" in _ANSI_SGR.sub("", r.stderr)  # the located source-frame error, on stderr


def test_chat_tool_call_through_the_loop(tmp_path, monkeypatch):
    """The reply agent (tool_calling) must invoke a flow-op tool THROUGH the loop and then answer —
    model -> tool -> model -> reply — not merely echo text. The five flow-op tools are unit-tested
    directly in test_chat_tools.py; this covers the loop wiring, which otherwise only the failure/
    cap path exercises."""
    monkeypatch.setattr(llm_clients_mod, "model_from_config", lambda cfg: _ToolThenAnswer())
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    # A workspace with one flow so `list_flows` (the tool the model calls) has something to return.
    (tmp_path / "a.yaml").write_text(
        "id: a\nname: a\ninput: {}\noutput: int\nkind: code\ncode: tests.seeds.fns:const_one\n"
    )
    # The bundled composer chat (tool_calling reply). One turn, then EOF ends the session.
    r = runner.invoke(
        app, ["chat", "--workspace", str(tmp_path), "--provider", "openai"],
        input="list my flows\n",
    )
    assert r.exit_code == 0
    assert "here are your flows" in r.output   # the loop reached a final answer after the tool call
    assert "turn failed" not in r.output       # the tool call was handled, not looped to the cap

