"""The native worker drains caller-owned effects before a memory lease exits."""

from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path
import signal
import subprocess
import sys
from threading import Event, Thread, get_ident
import time

import pytest

import agent_composer.memory.contracts as c
from agent_composer.memory.session import SessionMemory
from tests.engine.test_layer_direction import _import_targets

ROOT = Path(__file__).parents[1]
SERVER = ROOT / "tests" / "fixtures" / "codex_app_server.py"
REF = c.MemoryFrameRef("root", ("root",), c.MemoryOwner.FLOW)


def _client_type():
    try:
        installed = version("openai-codex")
    except PackageNotFoundError:
        pytest.skip("offline SDK integration requires the codex extra")
    if installed != "0.155.1":
        pytest.skip("offline SDK integration requires openai-codex==0.155.1")
    from agent_composer.llm_clients.codex.client import CodexWorkerClient

    return CodexWorkerClient


def _request(run_id):
    return c.ConversationRequest(
        ("root", "agent"),
        c.ConversationInvocation(run_id, 1),
        {"format": "offline-native-boundary"},
    )


def _running(pid):
    """A reaped-or-zombie fixture no longer owns a live external operation."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    if sys.platform != "linux":
        return True
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except FileNotFoundError:
        return False
    return stat.rsplit(")", 1)[1].split()[0] != "Z"


def _wait_until(predicate, seconds=8):
    deadline = time.monotonic() + seconds
    while not predicate():
        assert time.monotonic() < deadline, "supervised cleanup did not finish"
        time.sleep(0.02)


def test_codex_boundary_remains_a_provider_leaf():
    package = ROOT / "src" / "agent_composer" / "llm_clients" / "codex"
    assert package.is_dir()
    violations = []
    for path in sorted(package.glob("*.py")):
        for line, target, _ in _import_targets(path):
            if target == "agent_composer" or (
                target.startswith("agent_composer.")
                and not target.startswith("agent_composer.llm_clients.codex")
            ):
                violations.append(f"{path.name}:{line}: {target}")
    assert not violations, "\n".join(violations)


def test_importing_client_does_not_load_optional_sdk():
    code = """
import sys
from importlib.abc import MetaPathFinder

class NoNativeSDK(MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == 'openai_codex' or fullname.startswith('openai_codex.'):
            raise AssertionError('optional SDK was imported in caller process')

sys.meta_path.insert(0, NoNativeSDK())
from agent_composer.llm_clients.codex.client import CodexWorkerClient
assert not any(name.startswith('openai_codex') for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=10,
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("native_status", ["completed", "failed"])
def test_native_events_are_retained_before_terminal_disposition(native_status):
    client_type = _client_type()
    memory = SessionMemory()
    lanes, events = [], []
    scenario = "normal" if native_status == "completed" else "failed"
    with memory.open_run() as session:
        frame = session.open_frame(c.MemoryFrameSpec(REF, "session", {}))
        with frame.open_turn(_request("native"), check_stopped=lambda: None) as turn:
            effect = turn.request("model", {"input": "offline request"})

            def observe(method, payload):
                lanes.append(get_ident())
                events.append(method)
                turn.observe(
                    effect,
                    "pending",
                    items=(
                        c.ConversationItem(
                            f"native-{len(events)}",
                            "offline-native-event",
                            {"method": method, "payload": payload},
                        ),
                    ),
                )

            with client_type(
                on_event=observe,
                _launch_args_override=(sys.executable, "-u", str(SERVER), scenario),
            ) as client:
                client.initialize()
                native = client.start_thread()
                assert native["thread"]["id"] == "thread-1"
                assert native["thread"]["sessionId"] != native["thread"]["id"]
                result = client.run_turn(native["thread"]["id"], "offline request")
                assert result["turn"]["status"] == native_status
                assert "item/completed" in events
                assert "turn/completed" in events
                assert events.index("item/completed") < events.index("turn/completed")
            # A failed native disposition is data, not a lost/raised SDK result.
            assert result["turn"]["items"][0]["text"] == "offline answer"
            assert lanes and set(lanes) == {get_ident()}
            turn.observe(effect, "completed")
    with memory.open_run() as reopened:
        retained = reopened.snapshot["conversations"][0]
        methods = [item["payload"]["method"] for item in retained["items"]]
        assert "item/completed" in methods and "turn/completed" in methods


@pytest.mark.parametrize("operation", ["start_thread", "resume_thread"])
def test_actual_native_reference_can_be_saved_before_cancellation_escapes(operation):
    client_type = _client_type()
    memory = SessionMemory()
    stop = Event()
    primary = KeyboardInterrupt("stop after native session reply")
    observed = []
    thread_id = "thread-1" if operation == "start_thread" else "thread-existing"

    def check_stopped():
        if stop.is_set():
            raise primary

    with memory.open_run() as session:
        frame = session.open_frame(c.MemoryFrameSpec(REF, "session", {}))
        with pytest.raises(KeyboardInterrupt) as caught:
            with frame.open_turn(
                _request("reference"), check_stopped=check_stopped
            ) as turn:
                ordinal = turn.request("native_session", {"operation": operation})
                owner_lane = get_ident()

                def publish_result(actual_operation, result):
                    if actual_operation != operation:
                        return
                    observed.append(result["thread"]["id"])
                    assert get_ident() == owner_lane
                    stop.set()
                    _wait_until(lambda: client._primary is primary)
                    turn.observe(
                        ordinal,
                        "completed",
                        reference=c.ConversationReference(
                            "codex",
                            result["thread"]["id"],
                        ),
                    )

                with client_type(
                    on_result=publish_result,
                    check_stopped=check_stopped,
                    _launch_args_override=(sys.executable, "-u", str(SERVER), "normal"),
                ) as client:
                    if operation == "start_thread":
                        client.start_thread()
                    else:
                        client.resume_thread(thread_id)
        assert caught.value is primary
    assert observed == [thread_id]
    with memory.open_run() as reopened:
        retained = reopened.snapshot["conversations"][0]
        assert retained["reference"] == {"provider": "codex", "session_id": thread_id}
        assert retained["effects"][0]["status"] == "completed"
        assert retained["closed_invocations"][0]["status"] == "interrupted"


@pytest.mark.parametrize("close_run_handle", [False, True])
@pytest.mark.parametrize("stop_source", ["cancellation", "worker_exit"])
def test_stopped_worker_cannot_release_lease_before_host_callback_finishes(
    close_run_handle,
    stop_source,
):
    client_type = _client_type()
    from agent_composer.llm_clients.codex.protocol import CodexWorkerError

    memory = SessionMemory()
    owner_run = memory.open_run()
    follower_run = memory.open_run()
    owner_frame = owner_run.open_frame(c.MemoryFrameSpec(REF, "session", {}))
    follower_frame = follower_run.open_frame(c.MemoryFrameSpec(REF, "session", {}))
    callback_started, release_callback = Event(), Event()
    stopped, follower_entered, callback_finished = Event(), Event(), Event()
    primary = KeyboardInterrupt("host cancellation")
    failures, owner_errors, lanes, server_pids, clients = [], [], [], [], []

    def check_stopped():
        if stopped.is_set():
            raise primary

    def owner():
        try:
            with owner_frame.open_turn(
                _request("owner"), check_stopped=check_stopped
            ) as turn:
                owner_lane = get_ident()

                def observe(method, payload):
                    lanes.append(get_ident())
                    assert get_ident() == owner_lane
                    if method == "offline/started":
                        server_pids.append(payload["server_pid"])

                def callback(method, payload):
                    lanes.append(get_ident())
                    assert get_ident() == owner_lane
                    assert method == "item/tool/call"
                    assert payload["tool"] == "lookup"
                    ordinal = turn.request("tool", {"name": "lookup"})
                    callback_started.set()
                    assert release_callback.wait(15), "test did not release host tool"
                    callback_finished.set()
                    turn.observe(
                        ordinal,
                        "completed",
                        items=(
                            c.ConversationItem(
                                "actual-tool-result",
                                "offline-host-result",
                                {"value": 42},
                            ),
                        ),
                    )
                    return {
                        "contentItems": [{"type": "inputText", "text": "42"}],
                        "success": True,
                    }

                with client_type(
                    callback=callback,
                    on_event=observe,
                    check_stopped=check_stopped,
                    _launch_args_override=(
                        sys.executable,
                        "-u",
                        str(SERVER),
                        "callback",
                    ),
                ) as client:
                    clients.append(client)
                    client.initialize()
                    native = client.start_thread()
                    client.run_turn(native["thread"]["id"], "offline request")
        except BaseException as error:
            owner_errors.append(error)

    def follower():
        try:
            with follower_frame.open_turn(
                _request("follower"), check_stopped=lambda: None
            ) as turn:
                follower_entered.set()
                assert callback_finished.is_set()
                assert turn.view.items[-1].payload == {"value": 42}
        except BaseException as error:
            failures.append(error)

    first = Thread(target=owner, daemon=True)
    second = Thread(target=follower, daemon=True)
    first.start()
    try:
        assert callback_started.wait(10), owner_errors
        assert server_pids
        second.start()
        with memory._condition:
            assert memory._condition.wait_for(
                lambda: sum(map(len, memory._queues.values())) == 1,
                timeout=5,
            )
        if close_run_handle:
            owner_run.close()
        if stop_source == "cancellation":
            stopped.set()
        else:
            os.kill(clients[0]._process.pid, signal.SIGKILL)
        _wait_until(lambda: not _running(server_pids[0]))
        assert not callback_finished.is_set()
        assert not follower_entered.is_set()
        with pytest.raises(c.MemoryOperationError) as busy:
            memory.reset()
        assert busy.value.code == "session.busy"
        release_callback.set()
        first.join(10)
        second.join(10)
        assert not first.is_alive() and not second.is_alive()
        if stop_source == "cancellation":
            assert owner_errors == [primary]
        else:
            assert len(owner_errors) == 1
            assert isinstance(owner_errors[0], CodexWorkerError)
        assert not failures
        assert follower_entered.is_set()
        assert len(set(lanes)) == 1
    finally:
        stopped.set()
        release_callback.set()
        first.join(10)
        if second.ident is not None:
            second.join(10)
        owner_run.close()
        follower_run.close()
    with memory.open_run() as reopened:
        retained = reopened.snapshot["conversations"][0]
        assert retained["items"][-1]["payload"] == {"value": 42}
        assert retained["effects"][-1]["status"] == "completed"
        expected = (
            "interrupted"
            if close_run_handle or stop_source == "cancellation"
            else "failed"
        )
        assert retained["closed_invocations"][0]["status"] == expected
