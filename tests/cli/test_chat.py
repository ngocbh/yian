"""Tests for the broker-driven ``ac chat`` CLI through Typer's ``CliRunner``.

Covers flow-owned exits, broker EOF normalization, multi-turn ordering, a user flow
that treats exit text as ordinary input, cwd-local imports, and failed-turn recovery.
"""

from __future__ import annotations

import os
import re
import sys
import threading
import time
from concurrent.futures import CancelledError
from pathlib import Path

import pytest
from typer.testing import CliRunner

import agent_composer.llm_clients as llm_clients_mod
import agent_composer.tools as tools_module
from agent_composer.cli import app
import agent_composer.cli.chat.command as chat_command
from agent_composer.cli.chat import tools as chat_tools
import agent_composer.cli.run as run_cli
from agent_composer.cli.terminal import PromptRequest
from agent_composer.nodes.base import _HostStop

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

    def __init__(self, content="recovered reply", prompts=None):
        self.content = content
        self.prompts = prompts

    def bind_tools(self, tools):
        return self

    def invoke(self, msgs):
        if self.prompts is not None:
            self.prompts.append(msgs[-1].content)
        return _Msg(content=self.content)


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
    # from its selected mode at call time — patch it to the fake.
    monkeypatch.setattr(llm_clients_mod, "model_from_config", lambda cfg: _FakeModel())
    # `--provider openai` makes `_ensure_provider_keys` pick openai for the flow's agent
    # (which sets no provider); satisfy its key check with a dummy env var (no product change).
    monkeypatch.setenv("OPENAI_API_KEY", "x")

    # Use the plain example flow (mode: plain) so no tools/keys are needed for the drive test.
    # Its typed CODE branch records keep transcript growth deterministic.
    flow = EXAMPLE_CHAT
    # Two user turns, then the authored CASE routes the textual command to its exit record.
    r = runner.invoke(
        app,
        ["chat", str(flow), "--provider", "openai"],
        input="hi\nagain\n/exit\n",
    )
    assert r.exit_code == 0
    assert r.output.count("hello back") == 2
    assert "session ended" in r.output
    # Each reply event is queued before the following HUMAN_INPUT request, so the first
    # answer must render before the next prompt becomes visible.
    first_reply = r.output.index("hello back")
    assert r.output.find("You:", first_reply + len("hello back")) > first_reply


def test_chat_eof_is_normalized_to_flow_owned_exit(monkeypatch):
    """EOF becomes ``/exit`` at the broker and reaches the flow without a model call."""
    calls = []

    def factory(config):
        calls.append(config)
        return _FakeModel()

    monkeypatch.setattr(llm_clients_mod, "model_from_config", factory)
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    r = runner.invoke(
        app,
        ["chat", str(EXAMPLE_CHAT), "--provider", "openai"],
        input="",
    )
    assert r.exit_code == 0
    assert calls == []
    assert "session ended" in r.output


def test_chat_bundled_exit_commands_skip_reply_model(monkeypatch):
    """The authored CASE handles both exit spellings before the reply AGENT."""
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    calls = []
    monkeypatch.setattr(
        llm_clients_mod,
        "model_from_config",
        lambda config: calls.append(config) or _FakeModel(),
    )

    for command in ("/exit", "/quit"):
        result = runner.invoke(
            app,
            ["chat", str(EXAMPLE_CHAT), "--provider", "openai"],
            input=f"{command}\n",
        )
        assert result.exit_code == 0

    assert calls == []


def test_descriptor_chat_prompt_does_not_prefetch_the_next_line(monkeypatch):
    """Two already-written lines remain independently readable by two prompts."""
    read_fd, write_fd = os.pipe()
    reader = os.fdopen(read_fd, "r", encoding="utf-8")
    second_prompted = threading.Event()
    prompt_count = 0
    answers = []
    outcome = {}
    second = PromptRequest(question="Second:")

    def render_prompt(*args, **kwargs):
        nonlocal prompt_count
        prompt_count += 1
        if prompt_count == 2:
            second_prompted.set()

    def read_both():
        try:
            answers.append(chat_command._prompt_chat_request(PromptRequest(question="First:")))
            answers.append(chat_command._prompt_chat_request(second))
        except BaseException as exception:
            outcome["exception"] = exception

    monkeypatch.setattr(chat_command.sys, "stdin", reader)
    monkeypatch.setattr(chat_command.console, "print", render_prompt)
    os.write(write_fd, b"one\ntwo\n")
    thread = threading.Thread(target=read_both)
    thread.start()
    try:
        assert second_prompted.wait(2)
        thread.join(0.5)
        if thread.is_alive():
            callback = second._cancel()
            if callback is not None:
                callback()
            thread.join(2)
    finally:
        os.close(write_fd)
        thread.join(2)
        reader.close()

    assert not thread.is_alive()
    assert "exception" not in outcome
    assert answers == ["one", "two"]


def test_descriptor_chat_prompt_preserves_a_blank_submitted_line(monkeypatch):
    """A newline is an empty answer, distinct from descriptor EOF."""
    read_fd, write_fd = os.pipe()
    reader = os.fdopen(read_fd, "r", encoding="utf-8")
    monkeypatch.setattr(chat_command.sys, "stdin", reader)
    monkeypatch.setattr(chat_command.console, "print", lambda *args, **kwargs: None)
    try:
        os.write(write_fd, b"\n")
        answer = chat_command._prompt_chat_request(PromptRequest(question="You:"))
    finally:
        os.close(write_fd)
        reader.close()

    assert answer == ""


def test_descriptor_chat_prompt_raises_on_actual_eof(monkeypatch):
    """A closed descriptor with no bytes remains distinct from a blank line."""
    read_fd, write_fd = os.pipe()
    reader = os.fdopen(read_fd, "r", encoding="utf-8")
    monkeypatch.setattr(chat_command.sys, "stdin", reader)
    monkeypatch.setattr(chat_command.console, "print", lambda *args, **kwargs: None)
    os.close(write_fd)
    try:
        with pytest.raises(EOFError):
            chat_command._prompt_chat_request(PromptRequest(question="You:"))
    finally:
        reader.close()


def test_tty_ctrl_d_returns_the_partial_line(monkeypatch):
    """Canonical VEOF terminates existing text instead of stranding the prompt."""
    pty = pytest.importorskip("pty")
    master_fd, slave_fd = pty.openpty()
    reader = os.fdopen(slave_fd, "r", encoding="utf-8")
    request = PromptRequest(question="You:")
    outcome = {}

    def prompt():
        try:
            outcome["answer"] = chat_command._prompt_chat_request(request)
        except BaseException as exception:
            outcome["exception"] = exception

    monkeypatch.setattr(chat_command.sys, "stdin", reader)
    monkeypatch.setattr(chat_command.console, "print", lambda *args, **kwargs: None)
    thread = threading.Thread(target=prompt)
    thread.start()
    try:
        os.write(master_fd, b"partial\x04")
        thread.join(2)
    finally:
        if thread.is_alive():
            callback = request._cancel()
            if callback is not None:
                callback()
        os.close(master_fd)
        thread.join(2)
        reader.close()

    assert not thread.is_alive()
    assert "exception" not in outcome
    assert outcome.get("answer") == "partial"


def test_tty_ctrl_d_on_an_empty_line_is_eof(monkeypatch):
    """Canonical VEOF with no pending text remains an actual EOF signal."""
    pty = pytest.importorskip("pty")
    master_fd, slave_fd = pty.openpty()
    reader = os.fdopen(slave_fd, "r", encoding="utf-8")
    request = PromptRequest(question="You:")
    outcome = {}

    def prompt():
        try:
            outcome["answer"] = chat_command._prompt_chat_request(request)
        except BaseException as exception:
            outcome["exception"] = exception

    monkeypatch.setattr(chat_command.sys, "stdin", reader)
    monkeypatch.setattr(chat_command.console, "print", lambda *args, **kwargs: None)
    thread = threading.Thread(target=prompt)
    thread.start()
    try:
        os.write(master_fd, b"\x04")
        thread.join(2)
    finally:
        if thread.is_alive():
            callback = request._cancel()
            if callback is not None:
                callback()
        os.close(master_fd)
        thread.join(2)
        reader.close()

    assert not thread.is_alive()
    assert isinstance(outcome.get("exception"), EOFError)


def test_descriptor_chat_prompt_cancellation_interrupts_a_partial_line(monkeypatch):
    """Cancellation remains observable after bytes arrive but before newline."""
    read_fd, write_fd = os.pipe()
    reader = os.fdopen(read_fd, "r", encoding="utf-8")
    request = PromptRequest(question="You:")
    partial_consumed = threading.Event()
    outcome = {}
    real_read = os.read
    seen = bytearray()

    def recording_read(fd, size):
        chunk = real_read(fd, size)
        if fd == read_fd and chunk:
            seen.extend(chunk)
            if seen == b"partial":
                partial_consumed.set()
        return chunk

    def prompt():
        try:
            outcome["answer"] = chat_command._prompt_chat_request(request)
        except BaseException as exception:
            outcome["exception"] = exception

    monkeypatch.setattr(chat_command.sys, "stdin", reader)
    monkeypatch.setattr(chat_command.console, "print", lambda *args, **kwargs: None)
    monkeypatch.setattr(chat_command.os, "read", recording_read)
    os.write(write_fd, b"partial")
    thread = threading.Thread(target=prompt)
    thread.start()
    consumed = partial_consumed.wait(0.5)
    callback = request._cancel()
    if callback is not None:
        callback()
    thread.join(0.5)
    cancelled_prompt_woke = not thread.is_alive()
    if thread.is_alive():
        # Regression cleanup: EOF releases a buffered readline so the test cannot leak.
        os.close(write_fd)
        write_fd = -1
        thread.join(2)
    if write_fd >= 0:
        os.close(write_fd)
    reader.close()

    assert consumed
    assert cancelled_prompt_woke
    assert isinstance(outcome.get("exception"), CancelledError)


def test_delayed_prompt_cancellation_callback_cannot_write_after_cleanup(monkeypatch):
    """An already-extracted callback becomes inert before its pipe fd is closed."""
    read_fd, write_fd = os.pipe()
    reader = os.fdopen(read_fd, "r", encoding="utf-8")
    request = PromptRequest(question="You:")
    outcome = {}

    def prompt():
        try:
            outcome["answer"] = chat_command._prompt_chat_request(request)
        except BaseException as exception:
            outcome["exception"] = exception

    monkeypatch.setattr(chat_command.sys, "stdin", reader)
    monkeypatch.setattr(chat_command.console, "print", lambda *args, **kwargs: None)
    thread = threading.Thread(target=prompt)
    thread.start()
    callback = None
    writes = []
    cleaned_up = False
    real_write = os.write

    def record_late_write(fd, payload):
        writes.append((fd, payload))
        raise OSError("late write")

    try:
        for _ in range(200):
            with request._lock:
                registered = request._cancel_callback is not None
            if registered:
                callback = request._cancel()
                break
            threading.Event().wait(0.01)

        if callback is not None:
            # Wake select through stdin but deliberately delay the extracted
            # cancellation callback until prompt cleanup closes its self-pipe.
            real_write(write_fd, b"\n")
            thread.join(2)
            cleaned_up = not thread.is_alive()
            if cleaned_up:
                monkeypatch.setattr(chat_command.os, "write", record_late_write)
                callback()
    finally:
        if thread.is_alive():
            wake = callback if callback is not None else request._cancel()
            if wake is not None:
                wake()
            try:
                real_write(write_fd, b"\n")
            except OSError:
                pass
            thread.join(2)
        os.close(write_fd)
        reader.close()

    assert callback is not None
    assert cleaned_up
    assert not thread.is_alive()
    assert isinstance(outcome.get("exception"), CancelledError)
    assert writes == []


def test_active_chat_prompt_is_woken_by_controller_failure(monkeypatch):
    """A controller stop cancels a real descriptor-backed terminal read."""
    read_fd, write_fd = os.pipe()
    reader = os.fdopen(read_fd, "r")
    prompt_entered = threading.Event()
    prompt_outcome = {}
    failure = RuntimeError("sibling failed")

    monkeypatch.setattr(chat_command.sys, "stdin", reader)
    monkeypatch.setattr(
        chat_command.console,
        "print",
        lambda *args, **kwargs: prompt_entered.set(),
    )

    def controller(stop_event, on_event, ask_user, render):
        def block_on_prompt():
            try:
                ask_user(question="You:")
            except BaseException as exception:
                prompt_outcome["exception"] = exception

        prompt_thread = threading.Thread(target=block_on_prompt)
        prompt_thread.start()
        assert prompt_entered.wait(2)
        stop_event.set()
        prompt_thread.join(2)
        assert not prompt_thread.is_alive()
        raise failure

    try:
        with pytest.raises(RuntimeError) as caught:
            run_cli._run_with_terminal_controller(
                controller,
                prompt_handler=chat_command._prompt_chat_request,
                reporter=None,
                eof_value="/exit",
            )
    finally:
        reader.close()
        os.close(write_fd)

    assert caught.value is failure
    assert isinstance(prompt_outcome["exception"], _HostStop)


def test_chat_host_does_not_intercept_exit_text_for_user_flow(tmp_path, monkeypatch):
    """Without an authored exit CASE, ``/exit`` is ordinary HUMAN_INPUT data."""
    seen = []

    class RecordingModel(_FakeModel):
        def invoke(self, messages):
            seen.append(messages[-1].content)
            return super().invoke(messages)

    monkeypatch.setattr(llm_clients_mod, "model_from_config", lambda config: RecordingModel())
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    flow = tmp_path / "one-turn.yaml"
    flow.write_text(
        "id: one-turn\n"
        "name: one_turn\n"
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
        "        input: {message: '${ask.output}'}\n"
        "        output: str\n"
        '        prompt: "Message: ${message}"\n'
        "    output:\n"
        "      transcript: '${reply.output}'\n"
        "      exited: ${input.exited}\n"
        "nodes:\n"
        "  chat:\n"
        "    kind: loop\n"
        "    call: turn\n"
        "    input:\n"
        "      transcript: ${input.opening}\n"
        "      exited: false\n"
        "    times: 1\n"
        "output: ${chat.output.transcript}\n"
    )

    result = runner.invoke(
        app,
        ["chat", str(flow), "--provider", "openai"],
        input="/exit\n",
    )

    assert result.exit_code == 0
    assert seen and "/exit" in seen[0]


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
        "    times: 1\n"
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
    """A bad turn restarts from the last completed transcript, not partial text.

    The reply agent fails on turn two by hitting its tool-iteration cap. The host
    surfaces the error, starts a fresh run, and carries turn one's completed
    transcript into turn three without preserving turn two's in-progress message.
    """
    # `model_from_config` is called once per agent-node run, i.e. once per turn.
    calls = {"n": 0}
    recovered_prompts = []

    def _factory(cfg):
        calls["n"] += 1
        if calls["n"] == 1:
            return _GoodModel(content="first reply")
        if calls["n"] == 2:
            return _LoopyModel()
        return _GoodModel(prompts=recovered_prompts)

    monkeypatch.setattr(llm_clients_mod, "model_from_config", _factory)
    monkeypatch.setenv("OPENAI_API_KEY", "x")

    # The bundled composer chat (tool_calling reply). A workspace with one flow keeps
    # list_flows fast and confined.
    (tmp_path / "f.yaml").write_text("id: f\nname: f\ninput: {}\nkind: agent\noutput: str\nprompt: hi\n")

    # Turn 1 completes, turn 2 fails, turn 3 recovers, then the authored exit branch ends.
    r = runner.invoke(
        app, ["chat", "--workspace", str(tmp_path), "--provider", "openai"],
        input="first turn\nfailed turn\nrecovered turn\n/exit\n",
    )
    assert r.exit_code == 0
    assert "turn failed" in r.output          # the failure was surfaced, not swallowed
    assert "tool-iteration cap" in r.output   # with the real reason
    assert "continuing" in r.output           # and the session was kept alive
    assert "recovered reply" in r.output      # the restarted run produced a working turn
    assert "session ended" in r.output
    assert calls["n"] == 3
    assert len(recovered_prompts) == 1
    recovered_prompt = recovered_prompts[0]
    assert "first turn" in recovered_prompt
    assert "first reply" in recovered_prompt
    assert "recovered turn" in recovered_prompt
    assert "failed turn" not in recovered_prompt
    # Recovery installed a fresh process-owned handler, which proves the failed
    # lifecycle released its owner; the final lifecycle releases its owner too.
    assert "ask_user" not in tools_module.TOOL_REGISTRY


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
    # The bundled composer chat (tool_calling reply). One turn, then flow-owned exit.
    r = runner.invoke(
        app, ["chat", "--workspace", str(tmp_path), "--provider", "openai"],
        input="list my flows\n/exit\n",
    )
    assert r.exit_code == 0
    assert "here are your flows" in r.output   # the loop reached a final answer after the tool call
    assert "turn failed" not in r.output       # the tool call was handled, not looped to the cap


def test_nested_chat_wait_observes_outer_controller_stop(tmp_path, monkeypatch):
    """Outer cancellation wakes a WAIT running inside the chat ``run_flow`` tool."""
    (tmp_path / "wait.yaml").write_text(
        "id: nested-wait\n"
        "name: nested_wait\n"
        "nodes:\n"
        "  wait:\n"
        "    kind: wait\n"
        '    until: "2999-01-01T00:00:00+00:00"\n'
        "  done:\n"
        "    kind: code\n"
        "    runs_after: [wait]\n"
        "    output: str\n"
        "    code: |\n"
        '      return "done"\n'
        "output: ${done.output}\n"
    )
    monkeypatch.setattr(chat_tools, "_WORKSPACE", tmp_path.resolve())

    def controller(stop_event, on_event, ask_user, render):
        timer = threading.Timer(0.05, stop_event.set)
        timer.start()
        try:
            with chat_tools._bind_controller_stop(stop_event):
                return chat_tools.run_flow("wait.yaml")
        finally:
            timer.cancel()
            timer.join()

    started = time.monotonic()
    with pytest.raises(RuntimeError, match="terminal controller stopped"):
        run_cli._run_with_terminal_controller(
            controller,
            prompt_handler=lambda request: None,
            reporter=None,
        )

    assert time.monotonic() - started < 0.8


def test_nested_chat_failure_does_not_set_outer_controller_stop(tmp_path, monkeypatch):
    """A child engine failure remains a tool result instead of cancelling chat."""
    (tmp_path / "fail.yaml").write_text(
        "id: nested-failure\n"
        "name: nested_failure\n"
        "nodes:\n"
        "  fail:\n"
        "    kind: code\n"
        "    output: str\n"
        "    code: |\n"
        '      raise RuntimeError("nested boom")\n'
        '      return "unreachable"\n'
        "output: ${fail.output}\n"
    )
    monkeypatch.setattr(chat_tools, "_WORKSPACE", tmp_path.resolve())
    parent_states = []

    def controller(stop_event, on_event, ask_user, render):
        with chat_tools._bind_controller_stop(stop_event):
            result = chat_tools.run_flow("fail.yaml")
        parent_states.append(stop_event.is_set())
        return result

    result = run_cli._run_with_terminal_controller(
        controller,
        prompt_handler=lambda request: None,
        reporter=None,
    )

    assert result == "FAILED: nested boom"
    assert parent_states == [False]
