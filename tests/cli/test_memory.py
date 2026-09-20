"""CLI-owned memory selection, lazy lifetime, and run-boundary integration."""

import importlib
import re
import subprocess
import sys
import threading

import pytest
from typer.testing import CliRunner

from agent_composer.cli import app
import agent_composer.cli.run as run_cli
from agent_composer.compose import load_flow
from agent_composer.events import NodeStarted
from agent_composer.memory.contracts import (
    Memory,
    MemoryOperationError,
    MemorySetupError,
)
from agent_composer.run import run_flow
from agent_composer.runtime.log import RunStartedRecord, decode_record

_CODE_FLOW = """id: cli-memory
name: cli_memory
long_term_memory: POLICY
nodes:
  work:
    kind: code
    code: return 7
    output: int
output: ${work.output}
"""


def _host(policy="session", storage=()):
    module = importlib.import_module("agent_composer.cli.memory")
    source = _CODE_FLOW.replace("POLICY", policy)
    return module.MemoryHost(load_flow(source).compiled, storage)


def test_no_declarations_leave_memory_dormant(monkeypatch):
    module = importlib.import_module("agent_composer.cli.memory")
    flow = load_flow(_CODE_FLOW.replace("long_term_memory: POLICY\n", ""))
    monkeypatch.setattr(module, "_create_backend", lambda *args: pytest.fail("opened"))
    with module.MemoryHost(flow.compiled, ()) as host:
        assert host.registry is None


@pytest.mark.parametrize("policy", ["episodic", "reflective", "session"])
def test_backend_is_lazy_and_reused_per_policy(policy, monkeypatch, tmp_path):
    module = importlib.import_module("agent_composer.cli.memory")
    calls = []
    backend = _Backend(calls)
    monkeypatch.setattr(
        module,
        "_create_backend",
        lambda name, path: (calls.append((name, path)), backend)[1],
    )
    path = tmp_path / "memory=state.sqlite"
    with _host(policy, [f"{policy}={path}"]) as host:
        lazy = host.registry[policy]
        assert isinstance(lazy, Memory)
        assert calls == []
        assert lazy.open_run(snapshot={"pin": 1}) == {"pin": 1}
        assert lazy.open_run() is None
        assert calls == [(policy, str(path)), ("open", {"pin": 1}), ("open", None)]
    assert calls[-1] == "close"


@pytest.mark.parametrize(
    "options, message",
    [
        (["session"], "POLICY=PATH"),
        (["=memory.sqlite"], "POLICY=PATH"),
        (["session="], "POLICY=PATH"),
        (["session=   "], "POLICY=PATH"),
        ([" =memory.sqlite"], "POLICY=PATH"),
        (["session=a", "session=b"], "duplicate"),
        (["unknown=a"], "built-in"),
        (["episodic=a"], "not declared"),
    ],
)
def test_storage_options_reject_invalid_assignments(options, message):
    with pytest.raises(ValueError, match=message):
        _host(storage=options)


def test_nested_declarations_deduplicate_without_mutating_layout():
    module = importlib.import_module("agent_composer.cli.memory")
    flow = load_flow("""id: nested-memory
name: nested_memory
long_term_memory: episodic
defs:
  leaf:
    long_term_memory: session
    nodes:
      reply:
        kind: agent
        long_term_memory: reflective
        prompt: hello
        output: str
    output: ${reply.output}
nodes:
  first: {kind: call, call: leaf}
  second: {kind: call, call: leaf}
output: ${second.output}
""")
    assert flow.compiled.nodes["first"].memory_scope is None
    with module.MemoryHost(flow.compiled, ()) as host:
        assert list(host.registry) == ["episodic", "session", "reflective"]
    assert flow.compiled.nodes["first"].memory_scope is None


def test_custom_policy_fails_only_when_opened():
    with _host("custom_policy") as host:
        with pytest.raises(MemorySetupError, match="Python.*memory="):
            host.registry["custom_policy"].open_run()


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


def test_unused_wrapper_cannot_construct_after_host_close(monkeypatch):
    module = importlib.import_module("agent_composer.cli.memory")
    monkeypatch.setattr(module, "_create_backend", lambda *args: pytest.fail("opened"))
    with _host() as host:
        lazy = host.registry["session"]
    with pytest.raises(MemorySetupError, match="closed"):
        lazy.open_run()
    with pytest.raises(MemorySetupError, match="closed"):
        lazy.reset()
    host.close()


@pytest.mark.parametrize(
    "primary", [ValueError("primary"), KeyboardInterrupt("primary")]
)
def test_cleanup_preserves_primary_exception(primary, monkeypatch):
    module = importlib.import_module("agent_composer.cli.memory")
    calls = []
    monkeypatch.setattr(
        module, "_create_backend", lambda *args: _Backend(calls, OSError("close"))
    )
    with pytest.raises(type(primary)) as caught:
        with _host() as host:
            host.registry["session"].open_run()
            raise primary
    assert caught.value is primary
    assert calls[-1] == "close"


def test_all_backends_close_before_first_cleanup_failure_is_raised(monkeypatch):
    module = importlib.import_module("agent_composer.cli.memory")
    flow = load_flow("""id: two-memory
name: two_memory
long_term_memory: episodic
nodes:
  work:
    kind: agent
    long_term_memory: session
    prompt: hello
    output: str
output: ${work.output}
""")
    calls = {"episodic": [], "session": []}
    error = OSError("session close")
    monkeypatch.setattr(
        module,
        "_create_backend",
        lambda name, path: _Backend(calls[name], error if name == "session" else None),
    )
    with pytest.raises(OSError) as caught:
        with module.MemoryHost(flow.compiled, ()) as host:
            for lazy in host.registry.values():
                lazy.open_run()
    assert caught.value is error
    assert all(values[-1] == "close" for values in calls.values())


_RUNNER = CliRunner()


def _flow_file(tmp_path, policy="session"):
    path = tmp_path / "flow.yaml"
    path.write_text(_CODE_FLOW.replace("POLICY", policy))
    return path


def _records(path):
    return [decode_record(line).record for line in path.read_bytes().splitlines()]


def _plain(text):
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


@pytest.mark.parametrize("policy", ["episodic", "reflective", "session"])
@pytest.mark.parametrize("durable", [False, True])
def test_cli_activates_declared_policy_without_learning(policy, durable, tmp_path):
    flow = _flow_file(tmp_path, policy)
    log = tmp_path / "run.jsonl"
    path = tmp_path / "memory.sqlite"
    args = ["run", str(flow), "--log", str(log), "--quiet"]
    if durable:
        args += ["--memory-storage", f"{policy}={path}"]
    result = _RUNNER.invoke(app, args)
    assert result.exit_code == 0, result.output
    records = _records(log)
    started = records[0]
    assert isinstance(started, RunStartedRecord)
    assert started.memory_layout is not None
    assert len(started.memory_snapshots) == 1
    assert not started.learnable
    assert all(not type(record).__name__.startswith("Backward") for record in records)
    assert path.exists() == durable
    if durable:
        # The CLI owner must release the lifetime lock before returning.
        module = importlib.import_module("agent_composer.cli.memory")
        with module._create_backend(policy, str(path)) as reopened:
            with reopened.open_run() as opened:
                assert str(path) not in repr(opened.snapshot)


@pytest.mark.parametrize(
    "options",
    [
        ["session"],
        ["session="],
        ["missing=a"],
        ["episodic=a"],
        ["session=a", "session=b"],
    ],
)
def test_invalid_cli_storage_fails_before_inputs_provider_and_log(
    options, tmp_path, monkeypatch
):
    flow = _flow_file(tmp_path)
    log = tmp_path / "run.jsonl"
    monkeypatch.setattr(
        run_cli, "_gather_missing_inputs", lambda *a, **k: pytest.fail("inputs")
    )
    monkeypatch.setattr(
        run_cli, "_ensure_provider_keys", lambda *a, **k: pytest.fail("provider")
    )
    args = ["run", str(flow), "--log", str(log)]
    for option in options:
        args += ["--memory-storage", option]
    result = _RUNNER.invoke(app, args)
    assert result.exit_code == 2, result.output
    assert "--memory-storage" in _plain(result.output)
    assert not log.exists()


@pytest.mark.parametrize("terminal", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_resume_laziness_follows_recorded_memory_authority(
    terminal, enabled, tmp_path, monkeypatch
):
    module = importlib.import_module("agent_composer.cli.memory")
    from agent_composer.memory.session import SessionMemory

    flow = _flow_file(tmp_path)
    log = tmp_path / "run.jsonl"
    memory_path = tmp_path / "memory.sqlite"
    loaded = load_flow(flow.read_text())
    primary = KeyboardInterrupt("stop before work")

    def stop(event):
        if not terminal and isinstance(event, NodeStarted) and event.node_id == "work":
            raise primary

    with SessionMemory() as backend:
        kwargs = {"memory": {"session": backend}} if enabled else {}
        if terminal:
            run_flow(loaded, {}, log=log, on_event=stop, **kwargs)
        else:
            with pytest.raises(KeyboardInterrupt) as caught:
                run_flow(loaded, {}, log=log, on_event=stop, **kwargs)
            assert caught.value is primary
    before = log.read_bytes()
    opened = []
    real_create = module._create_backend

    def create(policy, path):
        opened.append(policy)
        return real_create(policy, path)

    monkeypatch.setattr(module, "_create_backend", create)
    if terminal or not enabled:
        # An occupied store must not matter to terminal or historically dormant recovery.
        with SessionMemory(storage=memory_path):
            result = _RUNNER.invoke(
                app,
                [
                    "run",
                    str(flow),
                    "--resume",
                    str(log),
                    "--quiet",
                    "--memory-storage",
                    f"session={memory_path}",
                ],
            )
        assert opened == []
    else:
        result = _RUNNER.invoke(
            app,
            [
                "run",
                str(flow),
                "--resume",
                str(log),
                "--quiet",
                "--memory-storage",
                f"session={memory_path}",
            ],
        )
        assert opened == ["session"]
    assert result.exit_code == 0, result.output
    assert log.read_bytes().startswith(before)
    if terminal:
        assert log.read_bytes() == before


def test_cli_custom_policy_fails_with_python_host_guidance(tmp_path):
    flow = _flow_file(tmp_path, "custom_policy")
    result = _RUNNER.invoke(app, ["run", str(flow), "--quiet"])
    assert result.exit_code == 1
    assert "Python API" in _plain(result.output)
    assert "memory=" in _plain(result.output)
    assert "Traceback" not in result.output


def test_cli_cleanup_failure_has_concise_host_error(tmp_path, monkeypatch):
    module = importlib.import_module("agent_composer.cli.memory")
    real_create = module._create_backend
    error = MemoryOperationError("close", "test.close", "memory close failed", False)

    def create(policy, path):
        backend = real_create(policy, path)
        close = backend.close

        def fail_close():
            close()
            raise error

        backend.close = fail_close
        return backend

    monkeypatch.setattr(module, "_create_backend", create)
    result = _RUNNER.invoke(app, ["run", str(_flow_file(tmp_path)), "--quiet"])
    assert result.exit_code == 1
    assert "run failed: memory close failed" in _plain(result.output)
    assert "Traceback" not in result.output


def test_backend_creation_and_cleanup_run_in_controller_after_worker_drain(
    tmp_path, monkeypatch
):
    module = importlib.import_module("agent_composer.cli.memory")
    real_create = module._create_backend
    events = []
    primary = KeyboardInterrupt("cancel")

    def create(policy, path):
        events.append(("create", threading.current_thread().name))
        backend = real_create(policy, path)
        close = backend.close

        def close_backend():
            assert events[-1][0] == "drained"
            events.append(("close", threading.current_thread().name))
            close()

        backend.close = close_backend
        return backend

    def run(*args, memory, **kwargs):
        with memory["session"].open_run():
            events.append(("drained", threading.current_thread().name))
        raise primary

    monkeypatch.setattr(module, "_create_backend", create)
    monkeypatch.setattr(run_cli, "_run_flow_controlled", run)
    result = _RUNNER.invoke(app, ["run", str(_flow_file(tmp_path)), "--quiet"])
    assert result.exit_code == 130, result.output
    assert events == [
        (name, "ac-controller") for name in ["create", "drained", "close"]
    ]


def test_no_memory_flow_imports_no_concrete_backend(tmp_path):
    script = """
import sys
from agent_composer.cli.memory import MemoryHost
from agent_composer.compose import load_flow
flow = load_flow('id: no-memory\\nname: no_memory\\nnodes:\\n  work: {kind: code, code: return 7}\\noutput: ${work.output}\\n')
with MemoryHost(flow.compiled, ()) as host:
    assert host.registry is None
assert not any('agent_composer.memory.' + name in sys.modules
               for name in ('episodic', 'reflective', 'session', 'storage'))
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=15
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "kind, extra",
    [("map", "    over: [1, 2]\n"), ("loop", "    while: false\n    max: 1\n")],
)
def test_discovery_includes_map_and_loop_children(kind, extra):
    module = importlib.import_module("agent_composer.cli.memory")
    loaded = load_flow("""id: child-memory
name: child_memory
defs:
  leaf:
    long_term_memory: session
    nodes:
      work: {kind: code, code: 'return {}', output: {}}
    output: ${work.output}
nodes:
  child:
    kind: KIND
    call: leaf
EXTRAoutput: ${child.output}
""".replace("KIND", kind).replace("EXTRA", extra))
    with module.MemoryHost(loaded.compiled, ()) as host:
        assert list(host.registry) == ["session"]


def test_repeatable_storage_and_later_constructor_failure_close_prior_backend(
    tmp_path, monkeypatch
):
    from agent_composer.memory.episodic import EpisodicMemory

    flow = _flow_file(tmp_path, "episodic")
    flow.write_text(
        flow.read_text().replace(
            "    kind: code\n    code: return 7\n    output: int",
            "    kind: agent\n    long_term_memory: session\n    prompt: hello\n    output: str",
        )
    )
    monkeypatch.setattr(run_cli, "_ensure_provider_keys", lambda *a: None)
    episodic_path = tmp_path / "episodic.sqlite"
    absent_parent = tmp_path / "absent"
    session_path = absent_parent / "session.sqlite"
    result = _RUNNER.invoke(
        app,
        [
            "run",
            str(flow),
            "--quiet",
            "--memory-storage",
            f"episodic={episodic_path}",
            "--memory-storage",
            f"session={session_path}",
        ],
    )
    assert result.exit_code == 1
    assert "run failed:" in _plain(result.output)
    assert "Traceback" not in result.output
    assert not absent_parent.exists()
    with EpisodicMemory(storage=episodic_path):
        pass


def test_cli_preserves_primary_when_cleanup_also_fails(tmp_path, monkeypatch):
    module = importlib.import_module("agent_composer.cli.memory")
    primary = ValueError("initiating failure")
    calls = []
    monkeypatch.setattr(
        module,
        "_create_backend",
        lambda *args: _Backend(calls, OSError("cleanup failure")),
    )

    def fail_run(*args, memory, **kwargs):
        memory["session"].open_run()
        raise primary

    monkeypatch.setattr(run_cli, "_run_flow_controlled", fail_run)
    result = _RUNNER.invoke(app, ["run", str(_flow_file(tmp_path)), "--quiet"])
    assert result.exit_code == 1
    assert "run failed: initiating failure" in _plain(result.output)
    assert "cleanup failure" not in _plain(result.output)
    assert calls[-1] == "close"


def test_real_pooled_cancellation_keeps_storage_until_model_callback_drains(
    tmp_path, monkeypatch
):
    from langchain_core.messages import AIMessage
    import agent_composer.llm_clients as llm_clients
    from agent_composer.memory.session import SessionMemory

    module = importlib.import_module("agent_composer.cli.memory")
    flow = _flow_file(tmp_path)
    flow.write_text(
        flow.read_text().replace(
            "    kind: code\n    code: return 7\n    output: int",
            "    kind: agent\n    mode: plain\n    prompt: hello\n    output: str",
        )
    )
    path = tmp_path / "memory.sqlite"
    started = threading.Event()
    drained = threading.Event()
    close_threads = []
    owned_errors = []
    broker_ref = []
    real_create = module._create_backend

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

    def create(policy, storage):
        backend = real_create(policy, storage)
        close = backend.close

        def close_backend():
            assert drained.is_set()
            close_threads.append(threading.current_thread().name)
            close()

        backend.close = close_backend
        return backend

    def interrupt(broker):
        broker_ref.append(broker)
        assert started.wait(5)
        raise KeyboardInterrupt("cancel")

    monkeypatch.setattr(llm_clients, "model_from_config", lambda _: Model())
    monkeypatch.setattr(run_cli, "_ensure_provider_keys", lambda *a: None)
    monkeypatch.setattr(run_cli.TerminalBroker, "serve", interrupt)
    monkeypatch.setattr(module, "_create_backend", create)
    result = _RUNNER.invoke(
        app,
        [
            "run",
            str(flow),
            "--quiet",
            "--num-workers",
            "1",
            "--memory-storage",
            f"session={path}",
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
