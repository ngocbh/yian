"""Offline protocol and supervised official-SDK lifecycle regressions."""

import math
import importlib.metadata
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from agent_composer.llm_clients.codex.protocol import (
    CodexWorkerError,
    Decoder,
    MAX_FRAME_BYTES,
    encode,
)


def test_protocol_roundtrip_detaches_finite_json_and_handles_partial_frames():
    payload = {"nested": [None, True, 1, 2.5, {"text": "日本語"}]}
    wire = encode("command", 1, payload)
    payload["nested"].append("changed")
    decoder = Decoder()
    assert list(decoder.feed(wire[:3])) == []
    assert list(decoder.feed(wire[3:8])) == []
    assert list(decoder.feed(wire[8:])) == [
        {
            "v": 1,
            "kind": "command",
            "id": 1,
            "payload": {"nested": [None, True, 1, 2.5, {"text": "日本語"}]},
        }
    ]


@pytest.mark.parametrize(
    "value",
    [
        [],
        {"x": math.nan},
        {"x": math.inf},
        {"x": ()},
        {1: "bad"},
        {"x": object()},
        {"x": "\ud800"},
    ],
)
def test_protocol_rejects_non_json(value):
    with pytest.raises(CodexWorkerError):
        encode("command", 1, value)


def test_protocol_rejects_cycles_unknown_envelope_and_oversized_prefix():
    cycle = {}
    cycle["self"] = cycle
    with pytest.raises(CodexWorkerError):
        encode("command", 1, cycle)
    with pytest.raises(CodexWorkerError):
        list(Decoder().feed((MAX_FRAME_BYTES + 1).to_bytes(4, "big")))
    raw = b'{"v":1,"kind":"command","id":1,"payload":{},"extra":1}'
    with pytest.raises(CodexWorkerError):
        list(Decoder().feed(len(raw).to_bytes(4, "big") + raw))


@pytest.mark.parametrize(
    "raw",
    [
        b'{"v":1,"v":1,"kind":"command","id":1,"payload":{}}',
        b'{"v":true,"kind":"command","id":1,"payload":{}}',
        b'{"v":1,"kind":"command","id":true,"payload":{}}',
        b'{"v":1,"kind":"command","id":1,"payload":{"x":NaN}}',
    ],
)
def test_protocol_rejects_ambiguous_wire(raw):
    with pytest.raises(CodexWorkerError):
        list(Decoder().feed(len(raw).to_bytes(4, "big") + raw))


@pytest.mark.parametrize(
    "trailing,code",
    [
        (b"\x00\x00\x00\x01{", "protocol"),
        ((MAX_FRAME_BYTES + 1).to_bytes(4, "big"), "capacity"),
    ],
)
def test_endpoint_delivers_valid_prefix_before_bad_frame_in_same_pipe_read(
    trailing, code
):
    from agent_composer.llm_clients.codex.protocol import Endpoint

    read_fd, peer = os.pipe()
    sink, write_fd = os.pipe()
    endpoint = None
    event = {"method": "item/completed", "params": {"actual": 42}}
    result = {"value": {"thread": {"id": "native-existing"}}}
    wire = encode("event", 1, event) + encode("result", 1, result) + trailing
    try:
        # Fill the pipe before starting its only reader: both valid frames and
        # the malformed suffix are present in the same bounded OS read.
        assert os.write(peer, wire) == len(wire)
        endpoint = Endpoint(read_fd, write_fd)
        eventually(lambda: endpoint.fault is not None)
        assert endpoint.receive()["payload"] == event
        assert endpoint.receive()["payload"] == result
        with pytest.raises(CodexWorkerError) as caught:
            endpoint.receive()
        assert caught.value is endpoint.fault
        assert caught.value.code == code
    finally:
        if endpoint is not None:
            endpoint.stop()
        for descriptor in (read_fd, peer, sink, write_fd):
            os.close(descriptor)


try:
    HAS_SDK = importlib.metadata.version("openai-codex") == "0.155.1"
except importlib.metadata.PackageNotFoundError:
    HAS_SDK = False
real_sdk = pytest.mark.skipif(not HAS_SDK, reason="requires pinned optional SDK")
SERVER = Path(__file__).parent / "fixtures" / "codex_app_server.py"


def client(scenario="normal", **kwargs):
    from agent_composer.llm_clients.codex.client import CodexWorkerClient

    return CodexWorkerClient(
        _launch_args_override=(sys.executable, "-u", str(SERVER), scenario), **kwargs
    )


@real_sdk
@pytest.mark.parametrize("resume", [False, True])
@pytest.mark.parametrize(
    "scenario,status", [("normal", "completed"), ("failed", "failed")]
)
def test_real_sdk_preserves_native_events_and_terminal_disposition(
    resume, scenario, status
):
    events = []
    lane = threading.get_ident()

    def event(method, payload):
        assert threading.get_ident() == lane
        events.append((method, payload))

    with client(scenario, on_event=event) as worker:
        assert worker.initialize()["serverInfo"]["version"] == "0.155.1"
        response = worker.resume_thread("retained") if resume else worker.start_thread()
        thread_id = response["thread"]["id"]
        terminal = worker.run_turn(thread_id, "hello")
    assert terminal["threadId"] == thread_id
    assert terminal["turn"]["status"] == status
    assert any(method == "item/completed" for method, _ in events)
    assert events[-1][0] == "turn/completed"


@real_sdk
def test_real_sdk_callback_runs_on_callers_lane_and_roundtrips():
    lane = threading.get_ident()
    callbacks = []

    def callback(method, params):
        assert threading.get_ident() == lane
        callbacks.append((method, params))
        return {
            "contentItems": [{"type": "inputText", "text": "actual result"}],
            "success": True,
        }

    with client("callback", callback=callback) as worker:
        thread = worker.start_thread()["thread"]["id"]
        assert worker.run_turn(thread, "hello")["turn"]["status"] == "completed"
    assert callbacks[0][0] == "item/tool/call"
    assert callbacks[0][1]["arguments"] == {"value": 1}


@real_sdk
def test_real_sdk_callback_exception_retains_identity():
    primary = ValueError("private callback error")

    def callback(method, params):
        raise primary

    with client("callback", callback=callback) as worker:
        thread = worker.start_thread()["thread"]["id"]
        with pytest.raises(ValueError) as caught:
            worker.run_turn(thread, "hello")
    assert caught.value is primary


@real_sdk
def test_absent_callback_handler_never_implicitly_approves():
    with client("callback") as worker:
        thread = worker.start_thread()["thread"]["id"]
        with pytest.raises(CodexWorkerError) as caught:
            worker.run_turn(thread, "hello")
    assert caught.value.code == "callback"


def test_worker_becomes_idle_before_publishing_result():
    from agent_composer.llm_clients.codex.worker import _Worker

    records = []
    endpoint = SimpleNamespace(
        send=lambda kind, identity, payload: records.append((kind, worker.operation))
    )
    worker = _Worker(endpoint)
    worker.operation = worker.last_command = 1
    response = SimpleNamespace(model_dump=lambda **kwargs: {"thread": {"id": "thread"}})
    worker.client = SimpleNamespace(thread_start=lambda params: response)
    worker.run(1, {"op": "start_thread", "params": {}})
    assert records == [("result", 0)]


@pytest.mark.parametrize("already_stopping", [False, True])
def test_transport_failure_forbids_later_interrupt_rpc(already_stopping):
    from agent_composer.llm_clients.codex.worker import _Worker

    calls = []
    worker = _Worker(SimpleNamespace(send=lambda *args: None))
    worker.client = SimpleNamespace(turn_interrupt=lambda *args: calls.append(args))
    worker.thread_id, worker.turn_id = "thread", "turn"
    if already_stopping:
        worker.stopping.set()
    worker.fail(RuntimeError("reader died"))
    worker.interrupt()
    threading.Event().wait(0.05)
    assert calls == []


def test_close_and_operation_admission_are_atomic():
    from agent_composer.llm_clients.codex.client import CodexWorkerClient

    entering, release = threading.Event(), threading.Event()
    errors = []

    class Gate:
        def acquire(self, **kwargs):
            entering.set()
            assert release.wait(3)
            return True

        def release(self):
            pass

    worker = CodexWorkerClient(
        _worker_command=(sys.executable, "-u", str(SERVER), "fake_worker")
    )
    worker._operation_lock = Gate()

    def call():
        try:
            worker.initialize()
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=call, daemon=True)
    thread.start()
    try:
        assert entering.wait(3)
        worker.close()
        release.set()
        thread.join(5)
        assert not thread.is_alive()
        assert worker._process is None
        assert len(errors) == 1 and errors[0].code == "closed"
    finally:
        release.set()
        thread.join(5)
        if worker._process is not None:
            worker._terminate_group()


@pytest.mark.parametrize(
    "scenario", ["fake_worker", "fake_worker_duplicate", "fake_worker_late_result"]
)
def test_actual_result_is_observed_after_monitor_latched_stop(scenario):
    from agent_composer.llm_clients.codex.client import CodexWorkerClient

    stopped = threading.Event()
    primary = KeyboardInterrupt("stop after surfaced response")
    results = []

    def check():
        if stopped.is_set():
            raise primary

    def event(method, payload):
        stopped.set()
        deadline = __import__("time").monotonic() + 3
        while worker._primary is None:
            assert __import__("time").monotonic() < deadline
            threading.Event().wait(0.01)

    with CodexWorkerClient(
        on_event=event,
        on_result=lambda op, value: results.append((op, value)),
        check_stopped=check,
        _worker_command=(sys.executable, "-u", str(SERVER), scenario),
    ) as worker:
        worker.initialize()
        with pytest.raises(KeyboardInterrupt) as caught:
            worker.start_thread()
    assert caught.value is primary
    assert results == [
        ("initialize", {}),
        ("start_thread", {"thread": {"id": "actual-thread"}}),
    ]


@real_sdk
def test_worker_uses_loaded_checkout_after_parent_cwd_changes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTHONPATH", "src")
    with client() as worker:
        assert worker.start_thread()["thread"]["id"] == "thread-1"


def eventually(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "bounded offline process wait expired"
        time.sleep(0.01)


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    if sys.platform != "linux":
        return True
    try:
        # A reparented zombie is dead, though kill(pid, 0) still succeeds.
        stat = Path(f"/proc/{pid}/stat").read_text()
    except FileNotFoundError:
        return False
    return stat.rsplit(")", 1)[1].split()[0] != "Z"


@pytest.mark.parametrize("scenario", ["malformed", "wrong_id", "partial", "exit"])
def test_supervisor_rejects_broken_workers_without_optional_sdk(scenario):
    from agent_composer.llm_clients.codex.client import CodexWorkerClient

    with CodexWorkerClient(
        _worker_command=(sys.executable, "-u", str(SERVER), f"fake_worker_{scenario}")
    ) as worker:
        with pytest.raises(CodexWorkerError):
            worker.initialize()
        assert worker._process.poll() is not None


def test_invalid_result_identity_is_rejected_before_observation():
    from agent_composer.llm_clients.codex.client import CodexWorkerClient

    results = []
    with CodexWorkerClient(
        on_result=lambda op, value: results.append(op),
        _worker_command=(
            sys.executable,
            "-u",
            str(SERVER),
            "fake_worker_invalid_thread",
        ),
    ) as worker:
        with pytest.raises(CodexWorkerError):
            worker.start_thread()
    assert results == ["initialize"]


@pytest.mark.parametrize(
    "installed,code", [(None, "sdk_missing"), ("0.155.2", "sdk_version")]
)
def test_dependency_pin_is_lazy_and_checked_before_provider_effects(
    monkeypatch, installed, code
):
    from agent_composer.llm_clients.codex import worker as module
    from agent_composer.llm_clients.codex.client import CodexWorkerClient

    def version(name):
        assert name == "openai-codex"
        if installed is None:
            raise importlib.metadata.PackageNotFoundError(name)
        return installed

    monkeypatch.setattr(module, "version", version)
    with CodexWorkerClient() as client:
        assert client._process is None
    with pytest.raises(CodexWorkerError) as caught:
        module._sdk()
    assert caught.value.code == code


@pytest.mark.parametrize("params", [{"threadId": "other"}, {"thread_id": "other"}])
def test_resume_identity_cannot_be_overridden_before_launch(params):
    from agent_composer.llm_clients.codex.client import CodexWorkerClient

    with CodexWorkerClient() as worker:
        with pytest.raises(CodexWorkerError):
            worker.resume_thread("retained", params)
        assert worker._process is None


@real_sdk
def test_quiet_healthy_inference_is_not_timed_out_or_retried(tmp_path):
    audit = tmp_path / "audit.jsonl"
    from agent_composer.llm_clients.codex.client import CodexWorkerClient

    with CodexWorkerClient(
        _launch_args_override=(sys.executable, "-u", str(SERVER), "slow", str(audit))
    ) as worker:
        thread = worker.start_thread()["thread"]["id"]
        started = time.monotonic()
        assert worker.run_turn(thread, "quiet")["turn"]["status"] == "completed"
        assert time.monotonic() - started >= 0.5
    assert (
        sum(
            json.loads(line).get("method") == "turn/start"
            for line in audit.read_text().splitlines()
        )
        == 1
    )


@real_sdk
def test_reader_failure_poison_prevents_late_rpc_and_sanitizes_native_errors(tmp_path):
    from agent_composer.llm_clients.codex.client import CodexWorkerClient

    audit = tmp_path / "audit.jsonl"
    with CodexWorkerClient(
        _launch_args_override=(
            sys.executable,
            "-u",
            str(SERVER),
            "reader_death",
            str(audit),
        )
    ) as worker:
        thread = worker.start_thread()["thread"]["id"]
        with pytest.raises(CodexWorkerError) as first:
            worker.run_turn(thread, "hello")
        with pytest.raises(CodexWorkerError) as later:
            worker.start_thread()
        assert first.value is later.value
    assert "secret-marker" not in str(first.value)
    methods = [
        json.loads(line).get("method") for line in audit.read_text().splitlines()
    ]
    assert methods.count("thread/start") == 1
    assert "turn/interrupt" not in methods


@real_sdk
def test_partial_initialization_failure_is_sanitized_and_reaped():
    with client("init_failure") as worker:
        with pytest.raises(CodexWorkerError) as caught:
            worker.initialize()
        assert worker._process.poll() is not None
    assert caught.value.code == "sdk"
    assert "secret-marker" not in str(caught.value)


@real_sdk
@pytest.mark.parametrize("scenario", ["descendants", "stuck_shutdown"])
def test_normal_and_stuck_shutdown_kill_owned_descendants(scenario):
    seen = []

    def event(method, payload):
        if method == "offline/started":
            seen.append(payload)

    with client(scenario, on_event=event) as worker:
        worker.start_thread()
        assert seen
        assert alive(seen[0]["server_pid"]) and alive(seen[0]["child_pid"])
        process = worker._process
    assert process.poll() is not None
    eventually(
        lambda: not alive(seen[0]["server_pid"]) and not alive(seen[0]["child_pid"])
    )


@real_sdk
def test_parent_eof_closes_app_server_and_owned_descendants():
    from agent_composer.llm_clients.codex.protocol import Endpoint

    process = subprocess.Popen(
        [sys.executable, "-u", "-m", "agent_composer.llm_clients.codex.worker"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=0,
        start_new_session=True,
    )
    endpoint = Endpoint(process.stdout.fileno(), process.stdin.fileno())
    seen = None
    try:
        endpoint.send(
            "command",
            1,
            {
                "op": "initialize",
                "config": {},
                "launch_args": [sys.executable, "-u", str(SERVER), "descendants"],
            },
        )
        deadline = time.monotonic() + 5
        while seen is None:
            assert time.monotonic() < deadline
            try:
                frame = endpoint.receive()
            except queue.Empty:
                continue
            if (
                frame["kind"] == "event"
                and frame["payload"]["method"] == "offline/started"
            ):
                seen = frame["payload"]["params"]
        process.stdin.close()
        process.wait(timeout=6)
        eventually(
            lambda: not alive(seen["server_pid"]) and not alive(seen["child_pid"])
        )
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=2)
        endpoint.stop()
        process.stdin.close()
        process.stdout.close()


def test_pipe_backpressure_is_bounded_and_stop_joins_pumps():
    from agent_composer.llm_clients.codex.protocol import Endpoint

    incoming, peer_writer = os.pipe()
    peer_reader, outgoing = os.pipe()
    endpoint = Endpoint(incoming, outgoing)
    try:
        endpoint.send("event", 0, {"large": "x" * 1_000_000})
        eventually(lambda: endpoint.fault is not None, timeout=4)
        assert endpoint.fault.code == "transport"
        endpoint.stop()
        assert all(not thread.is_alive() for thread in endpoint._threads)
    finally:
        endpoint.stop()
        for descriptor in (incoming, peer_writer, peer_reader, outgoing):
            os.close(descriptor)


@pytest.mark.parametrize(
    "params",
    [
        {"threadId": "different", "turnId": "turn"},
        {"threadId": "thread", "turnId": "different"},
    ],
)
def test_wrong_native_event_identity_is_rejected_before_publication(params):
    from agent_composer.llm_clients.codex.worker import _Worker

    published = []
    worker = _Worker(
        SimpleNamespace(send=lambda kind, identity, payload: published.append(kind))
    )
    worker.thread_id = "thread"
    notification = SimpleNamespace(
        method="item/completed", payload=SimpleNamespace(params=params)
    )
    worker.client = SimpleNamespace(
        turn_start=lambda *args: SimpleNamespace(turn=SimpleNamespace(id="turn")),
        next_turn_notification=lambda turn: notification,
        unregister_turn_notifications=lambda turn: None,
    )
    with pytest.raises(CodexWorkerError):
        worker.run_turn({"input": "hello", "params": {}})
    assert published == ["turn"]


def test_monitor_start_failure_cannot_leave_a_waitable_unstarted_monitor(monkeypatch):
    from agent_composer.llm_clients.codex.client import CodexWorkerClient

    original = threading.Thread.start
    primary = RuntimeError("thread resource unavailable")

    def start(thread):
        if thread._target.__name__ == "_monitor":
            raise primary
        original(thread)

    monkeypatch.setattr(threading.Thread, "start", start)
    worker = CodexWorkerClient(
        _worker_command=(sys.executable, "-u", str(SERVER), "fake_worker")
    )
    try:
        with pytest.raises(RuntimeError) as caught:
            worker._start()
        assert caught.value is primary
        assert worker._process.poll() is not None
        assert worker._monitor_thread is None
    finally:
        worker._terminate_group()


def test_deferred_interrupt_rechecks_reader_failure_before_rpc(monkeypatch):
    from agent_composer.llm_clients.codex.worker import _Worker

    deferred, calls = [], []

    class Deferred:
        def __init__(self, target, **kwargs):
            self.target = target

        def start(self):
            deferred.append(self.target)

    monkeypatch.setattr(threading, "Thread", Deferred)
    worker = _Worker(SimpleNamespace(send=lambda *args: None))
    worker.client = SimpleNamespace(turn_interrupt=lambda *args: calls.append(args))
    worker.thread_id, worker.turn_id = "thread", "turn"
    worker.interrupt()
    worker.fail(RuntimeError("reader transport died before interrupt admission"))
    deferred[0]()
    assert calls == []


def test_startup_reader_failure_cannot_issue_initialize_after_poison(monkeypatch):
    from agent_composer.llm_clients.codex import worker as module

    calls = []
    worker = module._Worker(SimpleNamespace(send=lambda *args: None))
    fake = SimpleNamespace(
        start=lambda: worker.fail(RuntimeError("reader transport died")),
        initialize=lambda: calls.append("initialize"),
        next_notification=lambda: (_ for _ in ()).throw(
            RuntimeError("reader transport died")
        ),
    )
    monkeypatch.setattr(
        module, "_sdk", lambda: (lambda *args, **kwargs: fake, lambda **kwargs: None)
    )
    worker.operation = 1
    worker.run(1, {"op": "initialize", "config": {}, "launch_args": None})
    assert calls == []


def test_early_native_callback_waits_for_authoritative_turn_without_blocking_events():
    from agent_composer.llm_clients.codex.client import CodexWorkerClient

    order = []

    def event(method, params):
        if method == "offline/before_turn":
            order.append("event")

    def callback(method, params):
        assert worker._turn_id == params["turnId"]
        order.append("tool")
        return {"success": True}

    with CodexWorkerClient(
        callback=callback,
        on_event=event,
        _worker_command=(sys.executable, "-u", str(SERVER), "fake_worker_tool_early"),
    ) as worker:
        thread = worker.start_thread()["thread"]["id"]
        for turn in ("turn-1", "turn-2"):
            assert worker.run_turn(thread, "hello")["turn"]["id"] == turn
    assert order == ["event", "tool", "event", "tool"]


@pytest.mark.parametrize(
    "scenario,expected_calls",
    [
        ("wrong_thread", 0),
        ("wrong_turn", 0),
        ("missing_call", 0),
        ("empty_call", 0),
        ("duplicate", 1),
        ("start_failed", 0),
        ("start_closed", 0),
        ("missing_thread", 0),
        ("missing_turn", 0),
        ("after_complete", 0),
        ("two_pending", 0),
    ],
)
def test_native_tool_identity_and_startup_failures_prevent_host_effects(
    scenario, expected_calls
):
    from agent_composer.llm_clients.codex.client import CodexWorkerClient

    calls = []

    def callback(method, params):
        calls.append(params)
        return {"success": True}

    with CodexWorkerClient(
        callback=callback,
        _worker_command=(
            sys.executable,
            "-u",
            str(SERVER),
            f"fake_worker_tool_{scenario}",
        ),
    ) as worker:
        thread = worker.start_thread()["thread"]["id"]
        with pytest.raises(CodexWorkerError):
            worker.run_turn(thread, "hello")
    assert len(calls) == expected_calls


@pytest.mark.parametrize("parent_error", [False, True])
def test_known_ipc_failure_closes_callback_admission_but_drains_observed_events(
    parent_error,
):
    from agent_composer.llm_clients.codex.client import CodexWorkerClient

    calls, observed = [], []
    primary = KeyboardInterrupt("initiating publication stop")

    def event(method, params):
        observed.append(method)
        if method == "offline/before_fault":
            eventually(lambda: worker._endpoint.fault is not None)
            if parent_error:
                raise primary

    with CodexWorkerClient(
        callback=lambda *args: calls.append(args) or {},
        on_event=event,
        _worker_command=(
            sys.executable,
            "-u",
            str(SERVER),
            "fake_worker_tool_broken_ipc",
        ),
    ) as worker:
        thread = worker.start_thread()["thread"]["id"]
        with pytest.raises(
            KeyboardInterrupt if parent_error else CodexWorkerError
        ) as caught:
            worker.run_turn(thread, "hello")
    assert "offline/before_fault" in observed
    assert "offline/observed_after_callback" in observed
    if parent_error:
        assert caught.value is primary
    assert calls == []


@pytest.mark.parametrize(
    "failure,host_error_at",
    [
        ("broken", None),
        ("broken", "before"),
        ("broken", "after"),
        ("broken", "result"),
        ("exit", None),
        ("close", None),
    ],
)
def test_known_terminal_fault_drains_result_and_following_evidence(
    failure, host_error_at
):
    from agent_composer.llm_clients.codex.client import CodexWorkerClient

    observed, closers = [], []
    primary = KeyboardInterrupt("initiating caller publication failed")

    def event(method, params):
        stage = "before" if method == "offline/before_result_fault" else "after"
        observed.append(stage)
        if stage == "before":
            if failure == "broken":
                eventually(lambda: worker._endpoint.fault is not None)
            elif failure == "exit":
                eventually(lambda: worker._process.poll() is not None)
            else:
                closer = threading.Thread(target=worker.close, daemon=True)
                closers.append(closer)
                closer.start()
                assert worker._close_requested.wait(3)
        if host_error_at == stage:
            raise primary

    def result(op, value):
        assert op == "initialize" and value == {}
        observed.append("result")
        if host_error_at == "result":
            raise primary

    with CodexWorkerClient(
        on_event=event,
        on_result=result,
        _worker_command=(
            sys.executable,
            "-u",
            str(SERVER),
            f"fake_worker_result_{failure}",
        ),
    ) as worker:
        with pytest.raises(
            KeyboardInterrupt if host_error_at else CodexWorkerError
        ) as caught:
            worker.initialize()
    for closer in closers:
        closer.join(5)
        assert not closer.is_alive()
    assert observed == ["before", "result", "after"]
    if host_error_at:
        assert caught.value is primary
    elif failure == "broken":
        assert caught.value.code == "protocol"


@pytest.mark.parametrize("capability", ["killpg", "setsid", "getpgrp"])
def test_unsupported_process_groups_fail_before_launch(monkeypatch, capability):
    from agent_composer.llm_clients.codex.client import CodexWorkerClient

    launches = []

    def launch(*args, **kwargs):
        launches.append(args)
        raise AssertionError("unsupported process must never launch")

    monkeypatch.delattr(os, capability)
    monkeypatch.setattr(subprocess, "Popen", launch)
    with CodexWorkerClient() as worker:
        with pytest.raises(CodexWorkerError) as caught:
            worker.initialize()
    assert caught.value.code == "platform"
    assert launches == []


def test_process_liveness_helper_does_not_require_linux_proc(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")

    def no_proc(*args, **kwargs):
        raise AssertionError("non-Linux liveness must not read /proc")

    monkeypatch.setattr(Path, "read_text", no_proc)
    assert alive(os.getpid())
