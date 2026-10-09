"""yian run memory flags, registry lifetime, and run-boundary integration."""

from __future__ import annotations

import re
import subprocess
import sys
import threading

import pytest
from typer.testing import CliRunner

from yian.cli import app
import yian.cli.run as run_cli
from yian.cli.memory import (
    MemoryOptionError,
    build_cli_memories,
    parse_memory_storage,
)
from yian.compose import LoadError, load_flow
from yian.events import NodeStarted, RunStarted
from yian.memory.contracts import MemoryOperationError, MemorySetupError
from yian.memory_host import build_memories
from yian.run import run_flow
from yian.runtime.log import RunStartedRecord, decode_record

_RUNNER = CliRunner()

_CODE_FLOW = """IDLINE
name: cli_memory
long_term_memory: MEMORY
nodes:
  work:
    kind: code
    code: return 7
    output: int
output: ${work.output}
"""

_SESSION_FLOW = """id: cli-memory
name: cli_memory
nodes:
  work:
    kind: agent
    mode: plain
    long_term_memory: {chat: session}
    prompt: hello
    output: str
output: ${work.output}
"""


@pytest.fixture(autouse=True)
def _offline_model(monkeypatch):
    """Answer every AGENT call locally so session flows run without a provider."""
    from langchain_core.messages import AIMessage
    import yian.llm_clients as llm_clients

    class _Model:
        def invoke(self, messages):
            return AIMessage(content="7")

    monkeypatch.setattr(llm_clients, "model_from_config", lambda config: _Model())
    monkeypatch.setattr(run_cli, "_ensure_provider_keys", lambda *args: None)


def _flow_text(memory="{notes: episodic}", flow_id="cli-memory"):
    """Return the CODE flow; ``memory=None`` drops the declaration line."""
    text = _CODE_FLOW.replace("IDLINE\n", f"id: {flow_id}\n" if flow_id else "")
    if memory is None:
        return text.replace("long_term_memory: MEMORY\n", "")
    return text.replace("MEMORY", memory)


def _flow_file(tmp_path, memory="{notes: episodic}", flow_id="cli-memory"):
    path = tmp_path / "flow.yaml"
    path.write_text(_flow_text(memory, flow_id))
    return path


def _session_flow_file(tmp_path):
    path = tmp_path / "flow.yaml"
    path.write_text(_SESSION_FLOW)
    return path


def _configured(policy, config):
    """Return a flow that declares memory ``policy`` of type ``policy`` with a YAML config."""
    if policy == "session":
        return _SESSION_FLOW.replace(
            "{chat: session}", f"{{session: {{type: session, config: {config}}}}}"
        )
    return _flow_text(f"{{{policy}: {{type: {policy}, config: {config}}}}}")


def _plain(text):
    """Strip ANSI styles and collapse whitespace, so wrapped messages match."""
    # cli.run's err_console fixes its width when the module is imported, so a
    # test cannot widen it; long messages wrap at 80 columns.
    return " ".join(re.sub(r"\x1b\[[0-9;]*m", "", text).split())


def _records(path):
    return [decode_record(line).record for line in path.read_bytes().splitlines()]


def _run(tmp_path, *extra, flow=None):
    flow = flow or _flow_file(tmp_path)
    log = tmp_path / "run.jsonl"
    result = _RUNNER.invoke(app, ["run", str(flow), "--log", str(log), "--quiet", *extra])
    return result, log


def _no_inputs_or_provider(monkeypatch):
    monkeypatch.setattr(run_cli, "_gather_missing_inputs", lambda *a, **k: pytest.fail("inputs"))
    monkeypatch.setattr(run_cli, "_ensure_provider_keys", lambda *a, **k: pytest.fail("provider"))


def test_memory_root_persists_every_builtin_under_the_flow_id(tmp_path):
    root = tmp_path / "mem"
    flow = _flow_file(tmp_path, "{notes: episodic, lessons: reflective, about: profile}")
    result, log = _run(tmp_path, "--memory-root", str(root), flow=flow)
    assert result.exit_code == 0, result.output
    assert (root / "cli-memory" / "notes.sqlite").is_file()
    assert (root / "cli-memory" / "lessons.sqlite").is_file()
    assert (root / "cli-memory" / ".about.md.lock").exists()
    assert isinstance(_records(log)[0], RunStartedRecord)


def test_memory_storage_is_keyed_by_memory_key(tmp_path):
    path = tmp_path / "notes.sqlite"
    result, _ = _run(tmp_path, "--memory-storage", f"notes={path}")
    assert result.exit_code == 0, result.output
    assert path.is_file()


def test_no_flags_keep_builtins_in_ram(tmp_path):
    flow = _flow_file(tmp_path, "{notes: episodic, lessons: reflective, about: profile}")
    result, _ = _run(tmp_path, flow=flow)
    assert result.exit_code == 0, result.output
    # tests/cli/conftest.py creates tmp_path/home for every CLI test.
    stored = [p for p in tmp_path.rglob("*") if p.suffix in {".sqlite", ".md", ".lock"}]
    assert stored == []


@pytest.mark.parametrize(
    "options",
    [
        ["notes"],
        ["notes="],
        ["=x.sqlite"],
        ["missing=a.sqlite"],
        ["notes=a.sqlite", "notes=b.sqlite"],
    ],
)
def test_bad_memory_storage_is_a_bad_parameter_before_inputs_provider_and_log(
    tmp_path, options, monkeypatch
):
    _no_inputs_or_provider(monkeypatch)
    args = []
    for option in options:
        args += ["--memory-storage", option]
    result, log = _run(tmp_path, *args)
    assert result.exit_code == 2, result.output
    # Typer prefixes the flag name; the message itself does not repeat it.
    assert _plain(result.output).count("--memory-storage") == 1
    assert not log.exists()


def test_memory_storage_key_is_stripped_and_path_kept() -> None:
    assert parse_memory_storage([" notes =./a b.sqlite"]) == {"notes": "./a b.sqlite"}
    with pytest.raises(MemoryOptionError, match="duplicate key 'notes'"):
        parse_memory_storage(["notes=a", " notes=b"])


def test_bad_storage_path_shape_fails_before_inputs_provider_and_log(tmp_path, monkeypatch):
    _no_inputs_or_provider(monkeypatch)
    result, log = _run(tmp_path, "--memory-storage", "notes=:memory:")
    assert result.exit_code == 1, result.output
    assert "memory 'notes': storage ':memory:' is not a local file path" in _plain(result.output)
    assert "Traceback" not in result.output
    assert not log.exists()


def test_unsafe_flow_id_with_memory_root_names_the_memory_root(tmp_path):
    root = tmp_path / "mem"
    flow = _flow_file(tmp_path, flow_id='"a/b"')
    result, log = _run(tmp_path, "--memory-root", str(root), flow=flow)
    assert result.exit_code == 1, result.output
    text = _plain(result.output)
    assert "flow id 'a/b' cannot name a directory under the memory root" in text
    assert "root=" not in text
    assert not root.exists() and not log.exists()


def test_memory_storage_for_a_custom_key_is_a_bad_parameter(tmp_path):
    flow = _flow_file(tmp_path, "{vec: vector}")
    result, _ = _run(tmp_path, "--memory-storage", "vec=v.sqlite", flow=flow)
    assert result.exit_code == 2
    assert "custom" in _plain(result.output)


def test_memory_root_without_flow_id_is_a_bad_parameter(tmp_path):
    # YAML requires id:, so only a flow compiled in Python can lack one.
    flow = load_flow(_flow_text())
    flow.compiled.flow_id = None
    with pytest.raises(MemoryOptionError, match="--memory-root") as caught:
        build_cli_memories(flow, root=tmp_path / "mem", storage_options=())
    assert caught.value.param_hint == "--memory-root"


@pytest.mark.parametrize("resume", [False, True], ids=["fresh", "resume"])
def test_custom_type_points_to_the_python_api(tmp_path, monkeypatch, resume):
    if resume:
        finished, log = _run(tmp_path, flow=_flow_file(tmp_path, memory=None))
        assert finished.exit_code == 0, finished.output
        before = log.read_bytes()
    _no_inputs_or_provider(monkeypatch)
    flow = _flow_file(tmp_path, "{vec: vector}")
    if resume:
        result = _RUNNER.invoke(app, ["run", str(flow), "--resume", str(log), "--quiet"])
    else:
        result, log = _run(tmp_path, flow=flow)
    assert result.exit_code == 1
    text = _plain(result.output)
    assert text.startswith("resume failed:" if resume else "run failed:")
    assert "Python API" in text and "build_memories" in text
    assert "Traceback" not in result.output
    if resume:
        assert log.read_bytes() == before
    else:
        assert not log.exists()


def test_flags_are_accepted_with_resume(tmp_path):
    root = tmp_path / "mem"
    result, log = _run(tmp_path, "--memory-root", str(root))
    assert result.exit_code == 0, result.output
    before = log.read_bytes()
    resumed = _RUNNER.invoke(
        app, ["run", str(_flow_file(tmp_path)), "--resume", str(log), "--quiet",
              "--memory-root", str(root)]
    )
    assert resumed.exit_code == 0, resumed.output
    assert log.read_bytes() == before


@pytest.mark.parametrize("policy, config, detail", [
    ("episodic", "{bogus: 1}", r"episodic memory config has unknown key\(s\) \['bogus'\]"),
    ("reflective", "{llm_config: {model: x}}", r"unknown key\(s\) \['llm_config'\]"),
    ("session", "{storage: x.sqlite}", "config key 'storage' is not allowed"),
])
def test_yaml_config_outside_the_allowlist_fails_at_load(policy, config, detail):
    with pytest.raises(LoadError, match=detail):
        load_flow(_configured(policy, config))


def _wrap_builtins(monkeypatch, wrap):
    """Pass every built-in backend that build_memories constructs to ``wrap(backend, type)``."""
    import yian.memory_host as host

    real = host._builtin

    def builtin(type_name, *args, **kwargs):
        backend = real(type_name, *args, **kwargs)
        wrap(backend, type_name)
        return backend

    monkeypatch.setattr(host, "_builtin", builtin)


class _Backend:
    def __init__(self, calls, error=None):
        self.calls = calls
        self.error = error

    def open_run(self, *, snapshot=None):
        self.calls.append(("open", snapshot))
        return snapshot

    def reset(self):
        self.calls.append("reset")

    def close(self):
        self.calls.append("close")
        if self.error is not None:
            raise self.error


def test_no_declarations_build_an_empty_registry(monkeypatch):
    import yian.memory_host as host

    monkeypatch.setattr(host, "_builtin", lambda *a, **k: pytest.fail("built"))
    flow = load_flow(_flow_text(None))
    with build_cli_memories(flow, root=None, storage_options=()) as registry:
        assert len(registry) == 0


@pytest.mark.parametrize("memory_type", ["episodic", "reflective", "session"])
@pytest.mark.parametrize("durable", [False, True])
def test_cli_activates_declared_memory_without_learning(memory_type, durable, tmp_path):
    if memory_type == "session":
        key, flow = "chat", _session_flow_file(tmp_path)
    else:
        key, flow = "notes", _flow_file(tmp_path, f"{{notes: {memory_type}}}")
    path = tmp_path / "memory.sqlite"
    options = [f"{key}={path}"] if durable else []
    extra = [arg for option in options for arg in ("--memory-storage", option)]
    result, log = _run(tmp_path, *extra, flow=flow)
    assert result.exit_code == 0, result.output
    records = _records(log)
    started = records[0]
    assert isinstance(started, RunStartedRecord)
    assert [spec.key for spec in started.memories] == [key]
    assert len(started.memory_snapshots) == 1
    assert not started.learnable
    assert all(not type(record).__name__.startswith("Backward") for record in records)
    assert path.exists() == durable
    if durable:
        # The CLI must release the file before returning.
        loaded = load_flow(flow.read_text())
        with build_cli_memories(loaded, root=None, storage_options=options) as registry:
            with registry[key].open_run() as opened:
                assert str(path) not in repr(opened.snapshot)


@pytest.mark.parametrize("terminal", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_resume_opens_memory_only_when_the_record_needs_it(
    terminal, enabled, tmp_path, monkeypatch
):
    from yian.memory.session import SessionMemory

    flow = _session_flow_file(tmp_path)
    log = tmp_path / "run.jsonl"
    memory_path = tmp_path / "memory.sqlite"
    loaded = load_flow(flow.read_text())
    primary = KeyboardInterrupt("stop before work")

    def stop(event):
        if not terminal and isinstance(event, NodeStarted) and event.node_id == "work":
            raise primary

    with SessionMemory() as backend:
        kwargs = {"memory": {"chat": backend}} if enabled else {}
        if terminal:
            run_flow(loaded, {}, log=log, on_event=stop, **kwargs)
        else:
            with pytest.raises(KeyboardInterrupt) as caught:
                run_flow(loaded, {}, log=log, on_event=stop, **kwargs)
            assert caught.value is primary
    before = log.read_bytes()
    opened = []

    def count_opens(backend, type_name):
        open_run = backend.open_run

        def counting(**kwargs):
            opened.append(type_name)
            return open_run(**kwargs)

        backend.open_run = counting

    _wrap_builtins(monkeypatch, count_opens)
    args = ["run", str(flow), "--resume", str(log), "--quiet",
            "--memory-storage", f"chat={memory_path}"]
    if terminal or not enabled:
        # An occupied store must not matter to terminal or historically dormant
        # recovery. Entering the holder opens and locks the file.
        with SessionMemory(storage=memory_path):
            result = _RUNNER.invoke(app, args)
        assert opened == []
    else:
        result = _RUNNER.invoke(app, args)
        assert opened == ["session"]
    assert result.exit_code == 0, result.output
    assert log.read_bytes().startswith(before)
    if terminal:
        assert log.read_bytes() == before


def test_cli_cleanup_failure_has_concise_host_error(tmp_path, monkeypatch):
    error = MemoryOperationError("close", "test.close", "memory close failed", False)

    def fail_close(backend, _type):
        close = backend.close

        def wrapped():
            close()
            raise error

        backend.close = wrapped

    _wrap_builtins(monkeypatch, fail_close)
    result, _ = _run(tmp_path)
    assert result.exit_code == 1
    assert "run failed: memory close failed" in _plain(result.output)
    assert "Traceback" not in result.output


def test_first_open_and_cleanup_run_in_controller_after_worker_drain(tmp_path, monkeypatch):
    events = []
    primary = KeyboardInterrupt("cancel")

    def watch(backend, _type):
        open_run, close = backend.open_run, backend.close

        def open_backend(**kwargs):
            events.append(("open", threading.current_thread().name))
            return open_run(**kwargs)

        def close_backend():
            assert events[-1][0] == "drained"
            events.append(("close", threading.current_thread().name))
            close()

        backend.open_run, backend.close = open_backend, close_backend

    def run(*args, memory, **kwargs):
        with memory["chat"].open_run():
            events.append(("drained", threading.current_thread().name))
        raise primary

    _wrap_builtins(monkeypatch, watch)
    monkeypatch.setattr(run_cli, "_run_flow_controlled", run)
    result = _RUNNER.invoke(app, ["run", str(_session_flow_file(tmp_path)), "--quiet"])
    assert result.exit_code == 130, result.output
    assert events == [(name, "ac-controller") for name in ["open", "drained", "close"]]


def test_no_memory_flow_imports_no_concrete_backend():
    script = """
import sys
from yian.cli.memory import build_cli_memories
from yian.compose import load_flow
flow = load_flow('id: no-memory\\nname: no_memory\\nnodes:\\n  work: {kind: code, code: return 7}\\noutput: ${work.output}\\n')
with build_cli_memories(flow, root=None, storage_options=()) as registry:
    assert len(registry) == 0
assert not any('yian.memory.' + name in sys.modules
               for name in ('episodic', 'reflective', 'session', 'profile', 'storage'))
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=15
    )
    assert result.returncode == 0, result.stderr


def test_open_failure_closes_every_backend_and_creates_no_directory(tmp_path):
    from yian.memory.episodic import EpisodicMemory

    flow = tmp_path / "flow.yaml"
    flow.write_text(
        _SESSION_FLOW.replace(
            "name: cli_memory\n", "name: cli_memory\nlong_term_memory: {archive: episodic}\n"
        )
    )
    # Run setup opens memories in sorted key order: archive opens (and creates
    # its file), then chat fails because its parent directory is missing.
    archive_path = tmp_path / "archive.sqlite"
    absent = tmp_path / "absent"
    result, _ = _run(
        tmp_path,
        "--memory-storage", f"archive={archive_path}",
        "--memory-storage", f"chat={absent / 'chat.sqlite'}",
        flow=flow,
    )
    assert result.exit_code == 1
    assert "run failed:" in _plain(result.output)
    assert "Traceback" not in result.output
    assert not absent.exists()
    assert archive_path.is_file()  # archive really opened before chat failed
    with EpisodicMemory(storage=archive_path):  # entering opens; busy if still held
        pass


def test_cli_preserves_primary_when_cleanup_also_fails(tmp_path, monkeypatch):
    import yian.memory_host as host

    primary = ValueError("initiating failure")
    calls = []
    monkeypatch.setattr(
        host, "_builtin", lambda *a, **k: _Backend(calls, OSError("cleanup failure"))
    )

    def fail_run(*args, memory, **kwargs):
        memory["chat"].open_run()
        raise primary

    monkeypatch.setattr(run_cli, "_run_flow_controlled", fail_run)
    result = _RUNNER.invoke(app, ["run", str(_session_flow_file(tmp_path)), "--quiet"])
    assert result.exit_code == 1
    assert "run failed: initiating failure" in _plain(result.output)
    assert "cleanup failure" not in _plain(result.output)
    assert calls[-1] == "close"


@pytest.mark.parametrize(
    "policy, config, expected",
    [
        ("episodic", "{max_episodes: 4, recall_limit: 2}", {"max_episodes": 4, "recall_limit": 2}),
        ("reflective", "{max_items: 5, recall_limit: 3}", {"max_items": 5, "recall_limit": 3}),
    ],
)
def test_yaml_config_reaches_the_built_in_constructor(policy, config, expected):
    flow = load_flow(_configured(policy, config))
    with build_cli_memories(flow, root=None, storage_options=()) as registry:
        with registry[policy].open_run() as session:
            assert session.snapshot["config"] == expected


@pytest.mark.parametrize(
    "policy, config, detail, cause",
    [
        ("reflective", "{max_items: 0}", "max_items", ValueError),
        ("episodic", "{recall_limit: two}", "recall_limit", TypeError),
        ("session", "{recent_turns: -1}", "recent_turns", ValueError),
    ],
)
def test_rejected_yaml_config_is_a_setup_error_naming_the_key(policy, config, detail, cause):
    flow = load_flow(_configured(policy, config))
    with pytest.raises(MemorySetupError, match=rf"memory {policy!r}.*{detail}") as caught:
        build_cli_memories(flow, root=None, storage_options=())
    assert type(caught.value.__cause__) is cause


def test_real_pooled_cancellation_keeps_storage_until_model_callback_drains(
    tmp_path, monkeypatch
):
    from langchain_core.messages import AIMessage
    import yian.llm_clients as llm_clients
    from yian.memory.session import SessionMemory

    flow = _session_flow_file(tmp_path)
    path = tmp_path / "memory.sqlite"
    started = threading.Event()
    drained = threading.Event()
    close_threads = []
    owned_errors = []
    broker_ref = []

    class Model:
        def invoke(self, messages):
            started.set()
            assert broker_ref[0]._stop_event.wait(5)
            try:
                with SessionMemory(storage=path):
                    pytest.fail("storage released before callback drained")
            except MemoryOperationError as exc:
                owned_errors.append(exc.code)
            drained.set()
            return AIMessage(content="finished")

    def wrap_close(backend, _type):
        close = backend.close

        def close_backend():
            assert drained.is_set()
            close_threads.append(threading.current_thread().name)
            close()

        backend.close = close_backend

    def interrupt(broker):
        broker_ref.append(broker)
        assert started.wait(5)
        raise KeyboardInterrupt("cancel")

    monkeypatch.setattr(llm_clients, "model_from_config", lambda _: Model())
    monkeypatch.setattr(run_cli, "_ensure_provider_keys", lambda *a: None)
    monkeypatch.setattr(run_cli.TerminalBroker, "serve", interrupt)
    _wrap_builtins(monkeypatch, wrap_close)
    result = _RUNNER.invoke(
        app,
        [
            "run",
            str(flow),
            "--quiet",
            "--num-workers",
            "1",
            "--memory-storage",
            f"chat={path}",
        ],
    )
    assert result.exit_code == 130, result.output
    assert drained.is_set()
    assert owned_errors == ["storage.busy"]
    assert close_threads == ["ac-controller"]
    assert not any(
        t.name.startswith(("ac-worker-", "ac-controller"))
        for t in threading.enumerate()
    )
    with SessionMemory(storage=path):
        pass


def test_run_help_describes_memory_flags():
    result = _RUNNER.invoke(app, ["run", "--help"], terminal_width=200)
    text = _plain(result.output)
    assert "--memory-root" in text
    assert "KEY=PATH" in text
    # Rich wraps help inside a box; drop its borders before matching a sentence.
    unboxed = _plain(result.output.replace("\u2502", " "))
    assert "Without either flag, memory lives in process RAM." in unboxed
    assert "POLICY" not in text


def test_cli_module_docstrings_name_the_new_flags():
    import yian.cli as cli_package

    for doc in (cli_package.__doc__, run_cli.__doc__):
        assert "--memory-root" in doc
        assert "POLICY=PATH" not in doc


def test_resume_accepts_one_def_called_from_two_nodes(tmp_path):
    flow = tmp_path / "flow.yaml"
    flow.write_text(
        "id: two-calls\nname: two_calls\nlong_term_memory: {notes: episodic}\n"
        "defs:\n  child:\n    nodes:\n      work: {kind: code, code: return 1, output: int}\n"
        "    output: ${work.output}\n"
        "nodes:\n  one: {kind: call, call: child}\n  two: {kind: call, call: child}\n"
        "output: ${one.output + two.output}\n"
    )
    log = tmp_path / "run.jsonl"

    def stop_after_start(event):
        if isinstance(event, RunStarted):
            raise RuntimeError("host stop")

    composed = load_flow(flow.read_text())
    with build_memories(composed) as registry, pytest.raises(RuntimeError, match="host stop"):
        run_flow(composed, {}, log=log, memory=registry, on_event=stop_after_start)
    resumed = _RUNNER.invoke(app, ["run", str(flow), "--resume", str(log), "--quiet"])
    assert resumed.exit_code == 0, resumed.output
