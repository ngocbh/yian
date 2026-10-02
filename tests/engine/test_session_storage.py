"""Optional session persistence retains raw history and private derived state."""

from contextlib import closing
from copy import deepcopy
import json
import multiprocessing
import os
import signal
import sqlite3
from threading import Event, Thread
import time

import pytest

import agent_composer.memory.contracts as c
from agent_composer.memory.session import SessionMemory
from agent_composer.memory.storage import SQLiteStorage
from tests.engine._session_fakes import (
    append_reply,
    harness,
    item,
    request,
    snapshot,
)
from tests.engine.test_session_notes import _note
from tests.engine.test_session_summaries import _prepare, _project, _seed


def _complete(memory, name="a"):
    with harness(memory) as h:
        with h.frame.open_turn(request(name), check_stopped=lambda: None) as turn:
            append_reply(turn, name, name)


def _mutation():
    note = _note()
    return c.MemoryMutation(note["note_id"], {"operation": "append_note", "note": note})


def test_completed_history_and_identity_survive_reopen(tmp_path):
    path = tmp_path / "memory.sqlite"
    with SessionMemory(storage=path) as memory:
        opening = snapshot(memory)
        _complete(memory)
        saved = snapshot(memory)
    with SessionMemory(storage=path) as memory:
        assert snapshot(memory) == saved
        with memory.open_run(snapshot=opening) as restored:
            assert restored.snapshot == opening
            frame = restored
            with frame.open_turn(request("b"), check_stopped=lambda: None) as turn:
                assert turn.view.items == (item("a", "a"),)
                append_reply(turn, "b", "b")
        assert len(snapshot(memory)["conversation"]["items"]) == 2


def test_partial_model_observation_and_unknown_request_survive_reopen(tmp_path):
    path = tmp_path / "memory.sqlite"
    with SessionMemory(storage=path) as memory:
        with harness(memory) as h:
            turn = h.frame.open_turn(request(), check_stopped=lambda: None).__enter__()
            ordinal = turn.request("model", {"prompt": "question"})
            turn.observe(ordinal, "pending", items=(item("partial", "working"),))
            h.session._release(turn._lease)
        saved = snapshot(memory)
    with SessionMemory(storage=path) as memory:
        assert snapshot(memory) == saved
        with harness(memory) as h:
            with h.frame.open_turn(
                request("retry"), check_stopped=lambda: None
            ) as turn:
                assert turn.view.items == (item("partial", "working"),)
                assert turn.view.effects[0].status == "pending"
                repeated = turn.request("model", {"prompt": "question"})
                turn.observe(repeated, "completed")
        assert [
            e["status"] for e in snapshot(memory)["conversation"]["effects"]
        ] == ["pending", "completed"]


def test_notes_and_commit_tombstones_survive_restart_and_reset(tmp_path):
    path = tmp_path / "memory.sqlite"
    mutation = _mutation()
    with SessionMemory(storage=path) as memory:
        _complete(memory)
        with harness(memory) as h:
            h.session.commit("learn", (mutation,))
        saved = snapshot(memory)
    with SessionMemory(storage=path) as memory:
        assert snapshot(memory) == saved
        with harness(memory) as h:
            h.session.commit("learn", (mutation,))
            assert (
                h.frame.recall(c.RecallContext("", "", {}, ("root", "agent"))).text
                == "Check units."
            )
        memory.reset()
    with SessionMemory(storage=path) as memory:
        with harness(memory) as h:
            h.session.commit("learn", (mutation,))
            assert h.session.snapshot["notes"] == []
            with pytest.raises(c.MemoryOperationError) as caught:
                h.session.commit("learn", ())
            assert caught.value.code == "session.commit_collision"


@pytest.mark.parametrize("used", [False, True])
def test_pristine_adoption_eligibility_survives_restart(tmp_path, used):
    source = snapshot(SessionMemory())
    path = tmp_path / "memory.sqlite"
    with SessionMemory(storage=path) as memory:
        identity = memory._namespace
        if used:
            snapshot(memory)
    with SessionMemory(storage=path) as memory:
        assert memory._namespace == identity
        if used:
            with pytest.raises(c.MemoryOperationError, match="proven history"):
                memory.open_run(snapshot=source)
        else:
            with memory.open_run(snapshot=source) as adopted:
                assert adopted.snapshot == source


def test_idle_handles_do_not_block_close_and_are_invalidated(tmp_path):
    path = tmp_path / "memory.sqlite"
    memory = SessionMemory(storage=path)
    opened = memory.open_run()
    frame = opened
    memory.close()
    memory.close()
    for operation in (
        lambda: opened.snapshot,
        frame.recall,
        memory.open_run,
        memory.reset,
    ):
        with pytest.raises(c.MemoryOperationError) as caught:
            operation()
        assert caught.value.code == "session.closed"
    opened.close()
    with SessionMemory(storage=path):
        pass


def test_inventory_has_no_host_settings_or_live_handles(tmp_path):
    path = tmp_path / "memory.sqlite"
    with SessionMemory(storage=path, history_budget_chars=17, recent_turns=0) as memory:
        _complete(memory)
    storage = SQLiteStorage(path, "session")
    try:
        encoded = json.dumps(storage.load())
    finally:
        storage.close()
    for forbidden in (
        "history_budget_chars",
        "recent_turns",
        "_condition",
        "_queues",
        "ticket",
        "lease",
    ):
        assert forbidden not in encoded


def test_process_local_backend_never_opens_sqlite(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("process-local memory must not use SQLite")

    monkeypatch.setattr(sqlite3, "connect", forbidden)
    with SessionMemory() as memory:
        _complete(memory)
        memory.reset()


def test_owner_kind_and_corrupt_store_are_rejected(tmp_path):
    path = tmp_path / "memory.sqlite"
    with SessionMemory(storage=path):
        with pytest.raises(c.MemoryOperationError) as caught:
            SessionMemory(storage=path)
        assert caught.value.code == "storage.busy"
    with pytest.raises(c.MemoryOperationError) as caught:
        SQLiteStorage(path, "episodic")
    assert caught.value.code == "storage.kind"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE memory_state SET envelope = ?", ('{"version":1,"version":1}',)
        )
    with pytest.raises(c.MemoryOperationError) as caught:
        SessionMemory(storage=path)
    assert caught.value.code == "storage.corrupt"


@pytest.mark.parametrize(
    "case",
    [
        "extra",
        "version",
        "used_type",
        "unused",
        "generation",
        "commits_type",
        "commit_id",
        "digest",
        "duplicate_view",
        "foreign_view",
        "notes",
        "summaries_type",
        "summary_extra",
        "summary_namespace",
        "summary_generation",
        "summary_version",
        "summary_covered",
        "summary_empty",
        "summary_current",
        "summary_duplicate",
        "summary_digest",
        "summary_text",
    ],
)
def test_invalid_inventory_never_rewrites_store_and_releases_owner(tmp_path, case):
    path = tmp_path / "memory.sqlite"
    with SessionMemory(storage=path, history_budget_chars=1, recent_turns=0) as memory:
        _seed(memory, 1)
        with harness(memory) as h:
            with h.frame.open_turn(
                request("current"), check_stopped=lambda: None
            ) as turn:
                _prepare(turn, [])
    storage = SQLiteStorage(path, "session")
    value = storage.load()
    summary = value["summaries"][0]
    if case == "extra":
        value["extra"] = None
    elif case == "version":
        value["version"] = True
    elif case == "used_type":
        value["used"] = 1
    elif case == "unused":
        value["used"] = False
    elif case == "generation":
        value["snapshot"]["generation"] = -1
    elif case == "commits_type":
        value["applied_commits"] = []
    elif case == "commit_id":
        value["applied_commits"] = {"": "a" * 64}
    elif case == "digest":
        value["applied_commits"] = {"c": "A" * 64}
    elif case == "duplicate_view":
        value["snapshot"]["conversation"] = [value["snapshot"]["conversation"]] * 2
    elif case == "foreign_view":
        value["snapshot"]["conversation"]["namespace"] = "foreign"
    elif case == "notes":
        value["snapshot"]["notes"] = [_note(run_id="unadmitted")]
    elif case == "summaries_type":
        value["summaries"] = {}
    elif case == "summary_extra":
        summary["extra"] = None
    elif case == "summary_namespace":
        summary["namespace"] = "foreign"
    elif case == "summary_generation":
        summary["generation"] = True
    elif case == "summary_version":
        summary["projection_version"] = ""
    elif case == "summary_covered":
        summary["covered"][0]["start_sequence"] = 9
    elif case == "summary_empty":
        summary["covered"] = []
    elif case == "summary_current":
        summary["covered"].append({"run_id": "current", "start_sequence": 1})
    elif case == "summary_duplicate":
        value["summaries"] *= 2
    elif case == "summary_digest":
        summary["digest"] = "a" * 64
    else:
        summary["text"] = "   "
    storage.save(value)
    storage.close()
    with pytest.raises(c.MemoryOperationError) as caught:
        SessionMemory(storage=path)
    assert caught.value.code == "session.invalid_storage"
    storage = SQLiteStorage(path, "session")
    try:
        assert storage.load() == value
    finally:
        storage.close()


def _ram(memory):
    return deepcopy(
        (
            memory._namespace,
            memory._generation,
            memory._view,
            memory._notes,
            memory._used,
            memory._applied_commits,
            memory._summaries,
        )
    )


@pytest.mark.parametrize(
    "operation",
    [
        "open",
        "adopt",
        "reset",
        "commit",
        "admit",
        "request",
        "observe",
        "summary",
        "finish",
    ],
)
def test_failed_save_does_not_publish_candidate_or_advance_ordinal(
    tmp_path, monkeypatch, operation
):
    memory = SessionMemory(
        storage=tmp_path / "memory.sqlite", history_budget_chars=1, recent_turns=0
    )
    opened = turn = None
    if operation not in {"open", "adopt", "reset"}:
        _complete(memory)
        opened = memory.open_run()
        frame = opened
        if operation != "commit":
            turn = frame.open_turn(request("next"), check_stopped=lambda: None)
            if operation != "admit":
                turn.__enter__()
            if operation == "observe":
                ordinal = turn.request("model", {})
    before = _ram(memory)
    before_ordinal = None if turn is None else turn._ordinal
    error = RuntimeError("save rejected")

    def reject(candidate):
        raise error

    with monkeypatch.context() as patch:
        patch.setattr(memory._storage, "save", reject)
        try:
            with pytest.raises(RuntimeError) as caught:
                if operation == "open":
                    memory.open_run()
                elif operation == "adopt":
                    memory.open_run(snapshot=snapshot(SessionMemory()))
                elif operation == "reset":
                    memory.reset()
                elif operation == "commit":
                    opened.commit("learn", (_mutation(),))
                elif operation == "admit":
                    turn.__enter__()
                elif operation == "request":
                    turn.request("model", {})
                elif operation == "observe":
                    turn.observe(ordinal, "completed", items=(item("new", "observed"),))
                elif operation == "summary":
                    _prepare(turn, [])
                else:
                    turn.__exit__(None, None, None)
            assert caught.value is error
            assert _ram(memory) == before
            if turn is not None:
                assert turn._ordinal == before_ordinal
        finally:
            if turn is not None and turn._lease is not None:
                opened._release(turn._lease)
            if opened is not None:
                opened.close()
    memory.close()


@pytest.mark.parametrize("after_commit", [False, True])
@pytest.mark.parametrize("cancel", [False, True])
def test_uncertain_write_poisoning_keeps_ram_and_drains_lease(
    tmp_path, monkeypatch, after_commit, cancel
):
    connect = sqlite3.connect
    error = (
        KeyboardInterrupt("stop")
        if cancel
        else sqlite3.OperationalError("private-path")
    )

    class Connection(sqlite3.Connection):
        armed = False

        def commit(self):
            if self.armed:
                if after_commit:
                    super().commit()
                raise error
            super().commit()

    monkeypatch.setattr(
        sqlite3,
        "connect",
        lambda *args, **kwargs: connect(*args, **kwargs, factory=Connection),
    )
    path = tmp_path / "memory.sqlite"
    memory = SessionMemory(storage=path)
    opened = memory.open_run()
    frame = opened
    turn = frame.open_turn(request(), check_stopped=lambda: None).__enter__()
    before = _ram(memory)
    before_ordinal = turn._ordinal
    memory._storage._connection.armed = True
    with pytest.raises(BaseException) as caught:
        turn.request("model", {})
    primary = caught.value
    if cancel:
        assert primary is error
    else:
        assert primary.code == "storage.io" and "private-path" not in str(primary)
    assert _ram(memory) == before and turn._ordinal == before_ordinal
    for operation in (
        memory.open_run,
        memory.reset,
        lambda: opened.snapshot,
        frame.recall,
        lambda: opened.commit("c", ()),
        lambda: turn.view,
        lambda: turn.request("model", {}),
        lambda: _prepare(turn, []),
    ):
        with pytest.raises(c.MemoryOperationError) as poisoned:
            operation()
        assert poisoned.value.code == "storage.poisoned"
    with pytest.raises(c.MemoryOperationError) as busy:
        memory.close()
    assert busy.value.code == "session.busy"
    with pytest.raises(BaseException) as original:
        turn.__exit__(type(primary), primary, None)
    assert original.value is primary
    assert not memory._active and not memory._queue
    opened.close()
    memory.close()
    with SessionMemory(storage=path) as reopened:
        view = snapshot(reopened)["conversation"]
        assert len(view["effects"]) == int(after_commit)
        if after_commit:
            assert view["effects"][0]["status"] == "pending"


def test_summary_cache_persists_and_extends_without_losing_raw_prefix(tmp_path):
    path = tmp_path / "memory.sqlite"
    with SessionMemory(storage=path, history_budget_chars=1, recent_turns=1) as memory:
        _seed(memory, 3)
        with harness(memory) as h:
            with h.frame.open_turn(
                request("current"), check_stopped=lambda: None
            ) as turn:
                calls = []
                assert (
                    _prepare(turn, calls, result="oversized saved summary")[0]
                    == "oversized saved summary"
                )
                assert _prepare(turn, calls)[0] == "oversized saved summary"
                assert len(calls) == 1
        saved = snapshot(memory)
        cache = deepcopy(memory._summaries)
    with SessionMemory(storage=path, history_budget_chars=1, recent_turns=1) as memory:
        assert memory._summaries == cache
        assert snapshot(memory) == saved
        with harness(memory) as h:
            with h.frame.open_turn(request("next"), check_stopped=lambda: None) as turn:
                calls = []
                assert _prepare(turn, calls, result="extended")[0] == "extended"
                assert len(calls) == 1
                assert "oversized saved summary" in calls[0][0]
                assert '"2"' in calls[0][0] and '"0"' not in calls[0][0]
        assert (
            snapshot(memory)["conversation"]["items"]
            == saved["conversation"]["items"]
        )
        memory.reset()
    with SessionMemory(storage=path) as memory:
        assert memory._summaries == {}


@pytest.mark.parametrize("phase", ["stop", "project", "summary", "effect"])
@pytest.mark.parametrize("close_run", [False, True])
def test_callback_drain_owns_backend_even_after_run_close(tmp_path, phase, close_run):
    path = tmp_path / "memory.sqlite"
    memory = SessionMemory(storage=path, history_budget_chars=1, recent_turns=0)
    _seed(memory, 1)
    opened = memory.open_run()
    frame = opened
    entered, release = Event(), Event()
    errors = []
    once = False

    def callback():
        nonlocal once
        if not once:
            once = True
            entered.set()
            assert release.wait(5)

    def check():
        if phase == "stop":
            callback()

    def project(view, selected):
        if phase == "project":
            callback()
        return _project(view, selected)

    def summarize(text, target):
        if phase == "summary":
            callback()
        return "saved summary"

    def work():
        try:
            with frame.open_turn(request("next"), check_stopped=check) as turn:
                if phase == "effect":
                    ordinal = turn.request("model", {})
                    callback()
                    turn.observe(
                        ordinal, "completed", items=(item("actual", "result"),)
                    )
                elif phase != "stop":
                    turn.prepare_history(
                        projection_version="v1", project=project, summarize=summarize
                    )
        except BaseException as error:
            errors.append(error)

    worker = Thread(target=work, daemon=True)
    worker.start()
    assert entered.wait(5)
    try:
        for operation in (memory.close, memory.reset):
            started = time.monotonic()
            with pytest.raises(c.MemoryOperationError) as busy:
                operation()
            assert busy.value.code == "session.busy"
            assert time.monotonic() - started < 1
        if close_run:
            opened.close()
            # A stop callback before admission only owns a queued ticket; closing
            # its run cancels that ticket, unlike a lease protecting live effects.
            if phase != "stop":
                with pytest.raises(c.MemoryOperationError) as busy:
                    memory.close()
                assert busy.value.code == "session.busy"
        with pytest.raises(c.MemoryOperationError) as owner:
            SessionMemory(storage=path)
        assert owner.value.code == "storage.busy"
    finally:
        release.set()
        worker.join(5)
    assert not worker.is_alive()
    if close_run and phase != "effect":
        assert len(errors) == 1 and errors[0].code == "session.stale"
    else:
        assert errors == []
    if phase == "summary":
        assert next(iter(memory._summaries.values())).text == "saved summary"
    if phase == "effect":
        assert snapshot(memory)["conversation"]["items"][-1]["item_id"] == "actual"
    opened.close()
    memory.close()
    with SessionMemory(storage=path):
        pass


def test_waiting_turn_keeps_fifo_and_cannot_be_discarded_by_close(tmp_path):
    memory = SessionMemory(storage=tmp_path / "memory.sqlite")
    first = memory.open_run()
    later = memory.open_run()
    frame = first
    other = later
    turn = frame.open_turn(request(), check_stopped=lambda: None).__enter__()
    queued = Event()
    finished = []

    def work():
        with other.open_turn(request("b"), check_stopped=queued.set):
            finished.append("b")

    worker = Thread(target=work, daemon=True)
    worker.start()
    assert queued.wait(5)
    first.close()
    for operation in (memory.close, memory.reset):
        with pytest.raises(c.MemoryOperationError) as busy:
            operation()
        assert busy.value.code == "session.busy"
    assert not finished
    turn.__exit__(None, None, None)
    worker.join(5)
    assert not worker.is_alive() and finished == ["b"]
    later.close()
    memory.close()


def _bounded(function, *args):
    process = multiprocessing.get_context("spawn").Process(target=function, args=args)
    process.start()
    process.join(15)
    if process.is_alive():
        process.terminate()
        process.join(5)
        pytest.fail("session lifecycle probe timed out")
    assert process.exitcode == 0


def _fork_while_locked(path):
    with SessionMemory(storage=path) as memory:
        with memory.open_run() as opened:
            frame = opened
            checks = []
            with frame.open_turn(
                request(), check_stopped=lambda: checks.append(os.getpid())
            ) as turn:
                ordinal = turn.request("model", {})
                checks.clear()
                locked, release = Event(), Event()

                def hold():
                    with memory._condition:
                        locked.set()
                        assert release.wait(5)

                worker = Thread(target=hold, daemon=True)
                worker.start()
                assert locked.wait(5)
                pid = os.fork()
                if pid == 0:
                    signal.alarm(3)
                    try:
                        for operation in (
                            memory.close,
                            memory.reset,
                            memory.open_run,
                            lambda: opened.snapshot,
                            opened.close,
                            frame.recall,
                            lambda: opened.commit("c", ()),
                            lambda: turn.view,
                            lambda: turn.request("model", {}),
                            lambda: turn.observe(ordinal, "completed"),
                            lambda: _prepare(turn, []),
                            lambda: turn.__exit__(None, None, None),
                        ):
                            try:
                                operation()
                            except c.MemoryOperationError as error:
                                assert error.code == "storage.forked"
                            else:
                                raise AssertionError("inherited operation succeeded")
                        assert checks == []
                    except BaseException:
                        os._exit(2)
                    os._exit(0)
                release.set()
                worker.join(5)
                _, status = os.waitpid(pid, 0)
                assert os.waitstatus_to_exitcode(status) == 0
                turn.observe(ordinal, "completed")


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
def test_pid_guard_precedes_condition_and_cleanup_locks(tmp_path):
    _bounded(_fork_while_locked, str(tmp_path / "memory.sqlite"))


def _fork_during_callback(path, phase, interrupt):
    parent_pid = os.getpid()
    primary = KeyboardInterrupt("original callback failure")
    forked = False

    def callback(memory):
        nonlocal forked
        if forked:
            return
        forked = True
        locked, release = Event(), Event()

        def hold():
            with memory._condition:
                locked.set()
                assert release.wait(5)

        worker = Thread(target=hold, daemon=True)
        worker.start()
        assert locked.wait(5)
        pid = os.fork()
        if pid == 0:
            signal.alarm(3)
            if interrupt:
                raise primary
            return
        release.set()
        worker.join(5)
        _, status = os.waitpid(pid, 0)
        assert os.waitstatus_to_exitcode(status) == 0

    try:
        with SessionMemory(
            storage=path, history_budget_chars=1, recent_turns=0
        ) as memory:
            _seed(memory, 1)
            with memory.open_run() as opened:
                frame = opened

                def check():
                    if phase == "stop":
                        callback(memory)

                def project(view, selected):
                    if phase == "project":
                        callback(memory)
                    return _project(view, selected)

                def summarize(text, target):
                    if phase == "summary":
                        callback(memory)
                    return "summary"

                with frame.open_turn(request("next"), check_stopped=check) as turn:
                    turn.prepare_history(
                        projection_version="v1", project=project, summarize=summarize
                    )
    except BaseException as error:
        if os.getpid() == parent_pid:
            raise
        valid = (
            error is primary
            if interrupt
            else isinstance(error, c.MemoryOperationError)
            and error.code == "storage.forked"
        )
        os._exit(0 if valid else 4)
    if os.getpid() != parent_pid:
        os._exit(5)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
@pytest.mark.parametrize("phase", ["stop", "project", "summary"])
@pytest.mark.parametrize("interrupt", [False, True])
def test_callback_fork_does_not_deadlock_or_replace_primary(tmp_path, phase, interrupt):
    _bounded(_fork_during_callback, str(tmp_path / "memory.sqlite"), phase, interrupt)


def test_first_finish_failure_survives_secondary_release_failure(tmp_path, monkeypatch):
    memory = SessionMemory(storage=tmp_path / "memory.sqlite")
    opened = memory.open_run()
    frame = opened
    turn = frame.open_turn(request(), check_stopped=lambda: None).__enter__()
    primary = RuntimeError("close publication failed")

    def fail_save(candidate):
        raise primary

    def fail_release(lease):
        raise RuntimeError("lease cleanup also failed")

    with monkeypatch.context() as patch:
        patch.setattr(memory._storage, "save", fail_save)
        patch.setattr(opened, "_release", fail_release)
        with pytest.raises(RuntimeError) as caught:
            turn.__exit__(None, None, None)
        assert caught.value is primary
    opened._release(turn._lease)
    opened.close()
    memory.close()


def _crash_after_observation(path):
    memory = SessionMemory(storage=path)
    opened = memory.open_run()
    frame = opened
    turn = frame.open_turn(request(), check_stopped=lambda: None).__enter__()
    completed = turn.request("model", {}, items=(item("u", "question"),))
    turn.observe(completed, "completed", items=(item("a", "answer"),))
    turn.request("tool", {"name": "repeatable"})
    os._exit(0)


def test_process_crash_keeps_completed_evidence_and_unknown_effect(tmp_path):
    path = str(tmp_path / "memory.sqlite")
    _bounded(_crash_after_observation, path)
    with SessionMemory(storage=path) as memory:
        with harness(memory) as h:
            with h.frame.open_turn(
                request("retry"), check_stopped=lambda: None
            ) as turn:
                assert [entry.payload["content"] for entry in turn.view.items] == [
                    "question",
                    "answer",
                ]
                assert [effect.status for effect in turn.view.effects] == [
                    "completed",
                    "pending",
                ]
                ordinal = turn.request("tool", {"name": "repeatable"})
                turn.observe(ordinal, "completed")


def test_surfaced_summary_is_saved_before_cooperative_stop(tmp_path):
    path = tmp_path / "memory.sqlite"
    stopped = Event()
    primary = KeyboardInterrupt("stop after summary")

    def check():
        if stopped.is_set():
            raise primary

    def summarize(text, target):
        stopped.set()
        return "summary returned before stop"

    with SessionMemory(storage=path, history_budget_chars=1, recent_turns=0) as memory:
        _seed(memory, 1)
        with harness(memory) as h:
            with pytest.raises(KeyboardInterrupt) as caught:
                with h.frame.open_turn(request("current"), check_stopped=check) as turn:
                    turn.prepare_history(
                        projection_version="v1", project=_project, summarize=summarize
                    )
            assert caught.value is primary
            saved = snapshot(memory)
    with SessionMemory(storage=path) as memory:
        assert snapshot(memory) == saved
        assert (
            next(iter(memory._summaries.values())).text
            == "summary returned before stop"
        )


def test_poisoned_storage_still_drains_queued_and_active_owners(tmp_path):
    memory = SessionMemory(storage=tmp_path / "memory.sqlite")
    opened = memory.open_run()
    other = memory.open_run()
    frame = opened
    waiting = other
    turn = frame.open_turn(request(), check_stopped=lambda: None).__enter__()
    queued, proceed = Event(), Event()
    errors = []

    def check():
        queued.set()
        assert proceed.wait(5)

    def work():
        try:
            with waiting.open_turn(request("next"), check_stopped=check):
                raise AssertionError("poisoned turn was admitted")
        except c.MemoryOperationError as error:
            errors.append(error)

    worker = Thread(target=work, daemon=True)
    worker.start()
    assert queued.wait(5)
    memory._storage._poisoned = True
    proceed.set()
    worker.join(5)
    assert not worker.is_alive() and len(errors) == 1
    assert errors[0].code == "storage.poisoned"
    assert not memory._queue and memory._active
    with pytest.raises(c.MemoryOperationError) as caught:
        turn.__exit__(None, None, None)
    assert caught.value.code == "storage.poisoned"
    assert not memory._active
    opened.close()
    other.close()
    memory.close()


def _message_snapshot_fixture():
    from pathlib import Path

    path = Path(__file__).parents[1] / "fixtures" / "session_message_snapshot.json"
    return json.loads(path.read_text())


def test_frame_era_snapshot_is_rejected_without_mutation(tmp_path):
    original = _message_snapshot_fixture()
    expected = deepcopy(original)
    with SessionMemory(storage=tmp_path / "memory.sqlite") as memory:
        before = snapshot(memory)
        with pytest.raises(c.MemoryOperationError) as caught:
            memory.open_run(snapshot=original)
        assert caught.value.code == "session.invalid_snapshot"
        assert snapshot(memory) == before
    # A pristine backend would adopt any valid snapshot, so this isolates the
    # version check from the used-backend lineage check above.
    with pytest.raises(c.MemoryOperationError) as pristine:
        SessionMemory().open_run(snapshot=original)
    assert pristine.value.code == "session.invalid_snapshot"
    assert original == expected


@pytest.mark.parametrize("legacy", ["reference", "native_session"])
def test_legacy_native_snapshot_is_rejected_without_mutation(tmp_path, legacy):
    source = SessionMemory()
    _complete(source)
    original = snapshot(source)
    view = original["conversation"]
    if legacy == "reference":
        view["reference"] = {"provider": "codex", "session_id": "thread"}
    else:
        view["effects"][0]["kind"] = "native_session"
    expected = deepcopy(original)
    path = tmp_path / "memory.sqlite"
    with SessionMemory(storage=path) as memory:
        before = snapshot(memory)
        stored = memory._storage.load()
        with pytest.raises(c.MemoryOperationError, match="unsupported|reset"):
            memory.open_run(snapshot=original)
        assert snapshot(memory) == before
        assert memory._storage.load() == stored
    assert original == expected
    with SessionMemory(storage=path) as memory:
        assert snapshot(memory) == before


@pytest.mark.parametrize("legacy", ["reference", "native_session"])
def test_legacy_native_storage_is_rejected_without_rewriting(tmp_path, legacy):
    path = tmp_path / "memory.sqlite"
    with SessionMemory(storage=path) as memory:
        _complete(memory)
    with closing(SQLiteStorage(path, "session")) as storage:
        value = storage.load()
        view = value["snapshot"]["conversation"]
        if legacy == "reference":
            view["reference"] = {"provider": "codex", "session_id": "thread"}
        else:
            view["effects"][0]["kind"] = "native_session"
        storage.save(value)
    original_bytes = path.read_bytes()
    with pytest.raises(
        c.MemoryOperationError, match="use fresh SessionMemory"
    ) as caught:
        SessionMemory(storage=path)
    assert caught.value.code == "session.unsupported_history"
    assert path.read_bytes() == original_bytes
    with closing(SQLiteStorage(path, "session")) as storage:
        assert storage.load() == value
