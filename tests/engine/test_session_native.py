"""Memory-owned native session selection and response publication."""

import os
import signal
from threading import Event, Thread

import pytest

import agent_composer.memory.contracts as c
from agent_composer.memory.session import SessionMemory
from tests.engine._session_fakes import REF, harness, request, snapshot
from tests.engine.test_session_storage import _bounded


def _unexpected(*args):
    raise AssertionError("wrong native session operation")


def _native(turn, session_id="thread", *, start=None, resume=None, provider="codex"):
    return turn.native_session(
        provider=provider,
        start=(lambda accept: accept(session_id)) if start is None else start,
        resume=(
            (lambda existing, accept: accept(existing)) if resume is None else resume
        ),
    )


def _seed(memory, name="first", session_id="thread"):
    with harness(memory) as h:
        with h.frame.open_turn(request(name), check_stopped=lambda: None) as turn:
            return _native(turn, session_id, resume=_unexpected)


def _view(memory):
    return snapshot(memory)["conversations"][0]


def test_fresh_start_then_resume_uses_memory_reference_and_effects():
    memory = SessionMemory()
    assert _seed(memory) == c.ConversationReference("codex", "thread")
    with harness(memory) as h:
        with h.frame.open_turn(request("second"), check_stopped=lambda: None) as turn:
            assert _native(turn, start=_unexpected) == turn.view.reference
    view = _view(memory)
    assert [effect["request"] for effect in view["effects"]] == [
        {"provider": "codex", "operation": "start"},
        {"provider": "codex", "operation": "resume", "session_id": "thread"},
    ]
    assert [effect["kind"] for effect in view["effects"]] == ["native_session"] * 2
    assert [effect["status"] for effect in view["effects"]] == ["completed"] * 2


def test_reset_starts_fresh_without_deleting_external_sessions():
    memory = SessionMemory()
    _seed(memory)
    memory.reset()
    assert _seed(memory, "after-reset", "new") == c.ConversationReference(
        "codex", "new"
    )


def test_missing_thread_keeps_reference_and_does_not_fallback_to_start():
    memory = SessionMemory()
    _seed(memory)
    primary = RuntimeError("native thread no longer exists")

    def resume(session_id, accept):
        assert session_id == "thread"
        raise primary

    with harness(memory) as h:
        with pytest.raises(RuntimeError) as caught:
            with h.frame.open_turn(
                request("second"), check_stopped=lambda: None
            ) as turn:
                _native(turn, start=_unexpected, resume=resume)
    assert caught.value is primary
    assert _view(memory)["reference"] == {"provider": "codex", "session_id": "thread"}
    assert _view(memory)["effects"][-1]["status"] == "pending"


def test_provider_mismatch_rejects_before_request_or_callback():
    memory = SessionMemory()
    _seed(memory)
    with harness(memory) as h:
        with pytest.raises(c.MemoryOperationError, match="provider"):
            with h.frame.open_turn(
                request("second"), check_stopped=lambda: None
            ) as turn:
                _native(turn, provider="other", start=_unexpected, resume=_unexpected)
    assert len(_view(memory)["effects"]) == 1


def test_resume_must_accept_the_requested_identity():
    memory = SessionMemory()
    _seed(memory)
    with harness(memory) as h:
        with pytest.raises(c.MemoryOperationError, match="session ID"):
            with h.frame.open_turn(
                request("second"), check_stopped=lambda: None
            ) as turn:
                _native(
                    turn, start=_unexpected, resume=lambda _, accept: accept("other")
                )
    assert _view(memory)["reference"]["session_id"] == "thread"
    assert _view(memory)["effects"][-1]["status"] == "pending"


def test_callback_return_is_not_an_accepted_reference():
    memory = SessionMemory()
    with harness(memory) as h:
        with pytest.raises(c.MemoryOperationError, match="accept"):
            with h.frame.open_turn(request(), check_stopped=lambda: None) as turn:
                _native(turn, start=lambda accept: "not-observed")
    assert _view(memory)["reference"] is None
    assert _view(memory)["effects"][0]["status"] == "pending"


@pytest.mark.parametrize("provider", [None, 42, "", "bad\ud800"])
def test_invalid_provider_fails_before_request(provider):
    with harness() as h:
        with h.frame.open_turn(request(), check_stopped=lambda: None) as turn:
            with pytest.raises((TypeError, ValueError)):
                _native(turn, provider=provider, start=_unexpected, resume=_unexpected)
            assert turn.view.effects == ()


@pytest.mark.parametrize("callback", ["start", "resume"])
def test_invalid_callback_fails_before_request(callback):
    with harness() as h:
        with h.frame.open_turn(request(), check_stopped=lambda: None) as turn:
            arguments = {"start": _unexpected, "resume": _unexpected, callback: 42}
            with pytest.raises(TypeError, match="callable"):
                turn.native_session(provider="codex", **arguments)
            assert turn.view.effects == ()


@pytest.mark.parametrize("session_id", [None, 42, "", "bad\ud800"])
def test_invalid_response_never_publishes_reference(session_id):
    memory = SessionMemory()
    with harness(memory) as h:
        with pytest.raises((TypeError, ValueError)):
            with h.frame.open_turn(request(), check_stopped=lambda: None) as turn:
                _native(turn, session_id)
    assert _view(memory)["reference"] is None
    assert _view(memory)["effects"][0]["status"] == "pending"


def test_duplicate_acceptance_preserves_first_response():
    memory = SessionMemory()

    def start(accept):
        accept("first")
        accept("second")

    with harness(memory) as h:
        with pytest.raises(c.MemoryOperationError, match="repeated"):
            with h.frame.open_turn(request(), check_stopped=lambda: None) as turn:
                _native(turn, start=start)
    view = _view(memory)
    assert view["reference"]["session_id"] == "first"
    assert view["effects"][0]["status"] == "completed"
    assert len(view["effects"][0]["observation_ordinals"]) == 1


@pytest.mark.parametrize("accepted", [False, True])
def test_acceptance_closes_with_callback_even_if_turn_is_still_live(accepted):
    retained = []

    def start(accept):
        retained.append(accept)
        if accepted:
            accept("thread")
        return "ignored"

    with harness() as h:
        with h.frame.open_turn(request(), check_stopped=lambda: None) as turn:
            if accepted:
                assert _native(turn, start=start).session_id == "thread"
            else:
                with pytest.raises(c.MemoryOperationError, match="without accepting"):
                    _native(turn, start=start)
            before = turn.view
            with pytest.raises(c.MemoryOperationError, match="late"):
                retained[0]("late")
            assert turn.view == before
            if not accepted:
                # An unknown request cannot be closed as successful.
                turn.observe(turn.view.effects[0].effect_ordinal, "interrupted")
        with pytest.raises(c.MemoryOperationError, match="lease"):
            retained[0]("after-release")


@pytest.mark.parametrize("after_accept", [False, True])
def test_callback_exception_is_primary_and_disables_retained_accept(after_accept):
    retained = []
    primary = KeyboardInterrupt("stop callback")
    memory = SessionMemory()

    def start(accept):
        retained.append(accept)
        if after_accept:
            accept("thread")
        raise primary

    with harness(memory) as h:
        with pytest.raises(KeyboardInterrupt) as caught:
            with h.frame.open_turn(request(), check_stopped=lambda: None) as turn:
                try:
                    _native(turn, start=start)
                finally:
                    with pytest.raises(c.MemoryOperationError, match="late"):
                        retained[0]("late")
    assert caught.value is primary
    view = _view(memory)
    assert view["effects"][0]["status"] == ("completed" if after_accept else "pending")
    assert len(view["effects"][0]["observation_ordinals"]) == int(after_accept)
    assert view["closed_invocations"][-1]["status"] == "interrupted"


@pytest.mark.parametrize("resumed", [False, True])
@pytest.mark.parametrize("swallow_stop", [False, True])
def test_response_is_durable_before_stop_and_is_not_observed_twice(
    tmp_path, resumed, swallow_stop
):
    path = tmp_path / "memory.sqlite"
    primary = RuntimeError("cooperative stop")
    stopped = Event()

    def check():
        if stopped.is_set():
            raise primary

    def start(accept):
        stopped.set()
        if swallow_stop:
            with pytest.raises(RuntimeError) as caught:
                accept("thread")
            assert caught.value is primary
        else:
            accept("thread")

    with SessionMemory(storage=path) as memory:
        if resumed:
            _seed(memory)
        with harness(memory) as h:
            with pytest.raises(RuntimeError) as caught:
                with h.frame.open_turn(request("stopped"), check_stopped=check) as turn:
                    _native(
                        turn, start=start, resume=lambda existing, accept: start(accept)
                    )
            assert caught.value is primary
        saved = snapshot(memory)
        effect = saved["conversations"][0]["effects"][-1]
        assert effect["status"] == "completed"
        assert len(effect["observation_ordinals"]) == 1
    stopped.clear()
    with SessionMemory(storage=path) as memory:
        assert snapshot(memory) == saved
        with harness(memory) as h:
            with h.frame.open_turn(request("retry"), check_stopped=check) as turn:
                assert _native(turn, start=_unexpected).session_id == "thread"


def test_closed_run_still_records_native_response_but_no_new_request():
    memory = SessionMemory()
    with harness(memory) as h:

        def start(accept):
            h.session.close()
            with pytest.raises(c.MemoryOperationError) as caught:
                memory.close()
            assert caught.value.code == "session.busy"
            accept("thread")

        with pytest.raises(c.MemoryOperationError, match="closed"):
            with h.frame.open_turn(request(), check_stopped=lambda: None) as turn:
                _native(turn, start=start)
    assert _view(memory)["reference"]["session_id"] == "thread"
    assert _view(memory)["effects"][0]["status"] == "completed"
    assert _view(memory)["closed_invocations"][0]["status"] == "interrupted"
    memory.close()


def test_old_version_one_pin_does_not_rewind_durable_native_reference(tmp_path):
    path = tmp_path / "memory.sqlite"
    with SessionMemory(storage=path) as memory:
        opening = snapshot(memory)
        opening["version"] = 1
        del opening["notes"]
        _seed(memory)
    with SessionMemory(storage=path) as memory:
        with memory.open_run(snapshot=opening) as opened:
            frame = opened.open_frame(c.MemoryFrameSpec(REF, "session", {}))
            with frame.open_turn(
                request("resumed"), check_stopped=lambda: None
            ) as turn:
                assert _native(turn, start=_unexpected).session_id == "thread"
            assert opened.snapshot == opening


def test_unknown_start_may_repeat_and_does_not_invent_a_reference():
    memory = SessionMemory()
    primary = OSError("connection lost")

    def lost(accept):
        raise primary

    with harness(memory) as h:
        with pytest.raises(OSError) as caught:
            with h.frame.open_turn(request(), check_stopped=lambda: None) as turn:
                _native(turn, start=lost)
        assert caught.value is primary
    _seed(memory, "retry")
    assert [effect["status"] for effect in _view(memory)["effects"]] == [
        "pending",
        "completed",
    ]


def test_changed_bridge_descriptor_rejects_before_native_work():
    memory = SessionMemory()
    initial = {"provider": "codex", "bridge": [{"name": "lookup", "schema": {}}]}
    with harness(memory) as h:
        with h.frame.open_turn(
            request(compatibility=initial), check_stopped=lambda: None
        ) as turn:
            _native(turn)
    with harness(memory) as h:
        with pytest.raises(c.MemoryOperationError, match="compatibility"):
            with h.frame.open_turn(
                request("changed", compatibility={"provider": "codex", "bridge": []}),
                check_stopped=lambda: None,
            ) as turn:
                _native(turn, start=_unexpected, resume=_unexpected)
    assert len(_view(memory)["effects"]) == 1


def test_native_callback_runs_unlocked_and_independent_leaf_does_not_wait():
    memory = SessionMemory()
    failures = []

    def independent():
        try:
            with harness(memory) as h:
                with h.frame.open_turn(
                    request("other", consumer=("root", "other")),
                    check_stopped=lambda: None,
                ) as turn:
                    _native(turn, "independent")
        except BaseException as error:
            failures.append(error)

    def start(accept):
        other = Thread(target=independent, daemon=True)
        other.start()
        other.join(timeout=3)
        assert not other.is_alive(), "native callback held the backend condition"
        assert failures == []
        accept("thread")

    with harness(memory) as h:
        with h.frame.open_turn(request(), check_stopped=lambda: None) as turn:
            _native(turn, start=start)
    assert {
        view["reference"]["session_id"] for view in snapshot(memory)["conversations"]
    } == {"thread", "independent"}


def test_queued_native_turn_resumes_only_after_prior_callback_and_lease_drain():
    memory = SessionMemory()
    entered, release, accepted, finish = Event(), Event(), Event(), Event()
    failures, operations = [], []
    first_session = []

    def first():
        try:
            with harness(memory) as h:
                first_session.append(h.session)

                def start(accept):
                    operations.append("start")
                    entered.set()
                    assert release.wait(5)
                    accept("thread")
                    accepted.set()
                    assert finish.wait(5)

                with h.frame.open_turn(
                    request("first"), check_stopped=lambda: None
                ) as turn:
                    _native(turn, start=start)
        except BaseException as error:
            failures.append(error)

    def second():
        try:
            with harness(memory) as h:

                def resume(session_id, accept):
                    operations.append("resume")
                    accept(session_id)

                with h.frame.open_turn(
                    request("second"), check_stopped=lambda: None
                ) as turn:
                    _native(turn, start=_unexpected, resume=resume)
        except BaseException as error:
            failures.append(error)

    first_thread = Thread(target=first, daemon=True)
    second_thread = Thread(target=second, daemon=True)
    first_thread.start()
    try:
        assert entered.wait(3)
        second_thread.start()
        with memory._condition:
            assert memory._condition.wait_for(lambda: bool(memory._queues), timeout=3)
        first_session[0].close()
        with pytest.raises(c.MemoryOperationError) as caught:
            memory.close()
        assert caught.value.code == "session.busy"
        release.set()
        assert accepted.wait(3)
        assert operations == ["start"]
        with pytest.raises(c.MemoryOperationError) as caught:
            memory.reset()
        assert caught.value.code == "session.busy"
    finally:
        release.set()
        finish.set()
        first_thread.join(timeout=5)
        if second_thread.ident is not None:
            second_thread.join(timeout=5)
    assert not first_thread.is_alive() and not second_thread.is_alive()
    assert len(failures) == 1
    assert isinstance(failures[0], c.MemoryOperationError)
    assert failures[0].code == "session.stale"
    assert operations == ["start", "resume"]
    assert [effect["status"] for effect in _view(memory)["effects"]] == [
        "completed"
    ] * 2
    memory.close()


def test_request_persistence_failure_prevents_external_native_work(
    tmp_path, monkeypatch
):
    memory = SessionMemory(storage=tmp_path / "memory.sqlite")
    primary = c.MemoryOperationError("storage", "test.save", "save failed", False)
    with harness(memory) as h:
        with pytest.raises(c.MemoryOperationError) as caught:
            with h.frame.open_turn(request(), check_stopped=lambda: None) as turn:
                before = turn.view

                def fail_save(value):
                    raise primary

                with monkeypatch.context() as patch:
                    patch.setattr(memory._storage, "save", fail_save)
                    try:
                        _native(turn, start=_unexpected, resume=_unexpected)
                    finally:
                        assert turn.view == before
        assert caught.value is primary
    assert _view(memory)["effects"] == []
    memory.close()


def test_response_persistence_failure_remains_unknown_and_preserves_primary(
    tmp_path, monkeypatch
):
    memory = SessionMemory(storage=tmp_path / "memory.sqlite")
    primary = c.MemoryOperationError("storage", "test.save", "save failed", False)

    def start(accept):
        def fail_save(value):
            raise primary

        with monkeypatch.context() as patch:
            patch.setattr(memory._storage, "save", fail_save)
            accept("thread")

    with harness(memory) as h:
        with pytest.raises(c.MemoryOperationError) as caught:
            with h.frame.open_turn(request(), check_stopped=lambda: None) as turn:
                _native(turn, start=start)
        assert caught.value is primary
    saved = snapshot(memory)
    assert saved["conversations"][0]["reference"] is None
    assert saved["conversations"][0]["effects"][0]["status"] == "pending"
    memory.close()
    with SessionMemory(storage=tmp_path / "memory.sqlite") as reopened:
        assert snapshot(reopened) == saved


def test_callback_return_value_is_ignored_after_acceptance():
    def start(accept):
        accept("thread")
        return object()

    with harness() as h:
        with h.frame.open_turn(request(), check_stopped=lambda: None) as turn:
            assert _native(turn, start=start) == c.ConversationReference(
                "codex", "thread"
            )


def test_stop_before_native_request_performs_no_external_work():
    memory = SessionMemory()
    primary = KeyboardInterrupt("cancel")
    stopped = Event()

    def check():
        if stopped.is_set():
            raise primary

    with harness(memory) as h:
        with pytest.raises(KeyboardInterrupt) as caught:
            with h.frame.open_turn(request(), check_stopped=check) as turn:
                stopped.set()
                _native(turn, start=_unexpected, resume=_unexpected)
        assert caught.value is primary
    assert _view(memory)["effects"] == []


def _fork_in_native_callback(path, phase):
    parent_pid = os.getpid()
    primary = KeyboardInterrupt("callback failed after fork")

    try:
        with SessionMemory(storage=path) as memory:
            with harness(memory) as h:

                def start(accept):
                    locked, release = Event(), Event()

                    def hold():
                        with memory._condition:
                            locked.set()
                            assert release.wait(5)

                    holder = Thread(target=hold, daemon=True)
                    holder.start()
                    assert locked.wait(5)
                    child = os.fork()
                    if child == 0:
                        signal.alarm(3)
                        if phase == "failure":
                            raise primary
                        if phase == "accept":
                            accept("child-thread")
                        return
                    release.set()
                    holder.join(5)
                    _, status = os.waitpid(child, 0)
                    assert os.waitstatus_to_exitcode(status) == 0
                    accept("thread")

                with h.frame.open_turn(request(), check_stopped=lambda: None) as turn:
                    _native(turn, start=start)
    except BaseException as error:
        if os.getpid() == parent_pid:
            raise
        valid = (
            error is primary
            if phase == "failure"
            else isinstance(error, c.MemoryOperationError)
            and error.code == "storage.forked"
        )
        os._exit(0 if valid else 4)
    if os.getpid() != parent_pid:
        os._exit(5)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
@pytest.mark.parametrize("phase", ["accept", "return", "failure"])
def test_native_callback_fork_checks_pid_before_locks_or_followup_work(tmp_path, phase):
    _bounded(_fork_in_native_callback, str(tmp_path / "memory.sqlite"), phase)
