"""Optional episodic persistence preserves state and backend ownership."""

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
from agent_composer.memory.episodic import EpisodicMemory
import agent_composer.memory.episodic as episodic
from agent_composer.memory.storage import SQLiteStorage
from tests.engine.test_episodic_memory import _ref, _spec


def _snapshot(memory):
    with memory.open_run() as opened:
        return opened.snapshot


def _context(run_id="run", ref=None):
    ref = _ref() if ref is None else ref
    episode = c.MemoryEpisode(
        run_id,
        c.MemoryOccurrenceRef(ref),
        ref.static_path,
        c.MemoryEpisodeStatus.SUCCEEDED,
        {},
        run_id,
    )
    return c.LearnContext(run_id, ref, (episode,))


def _append(memory, commit_id, *, name="agent"):
    with memory.open_run() as opened:
        frame = opened.open_frame(_spec(name))
        batch = tuple(frame.learn(_context(commit_id, _ref(name))))
        opened.commit(commit_id, batch)
    return batch


def test_constructor_initializes_identity_before_any_run_and_releases_owner(tmp_path):
    path = tmp_path / "memory.sqlite"
    memory = EpisodicMemory(7, 3, storage=path)
    namespace = memory._namespace
    with pytest.raises(c.MemoryOperationError) as caught:
        EpisodicMemory(storage=path)
    assert caught.value.code == "storage.busy"
    memory.close()
    memory.close()
    with EpisodicMemory(storage=path) as reopened:
        assert reopened._namespace == namespace
        assert not reopened._ever_opened
        assert _snapshot(reopened)["version"] == 1


def test_none_storage_remains_resource_free(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("SQLite must stay dormant")

    monkeypatch.setattr(sqlite3, "connect", forbidden)
    with EpisodicMemory(4, 2, storage=None) as memory:
        assert _snapshot(memory)["frames"] == []


@pytest.mark.parametrize("storage", [None, "file"])
def test_idle_handles_do_not_block_close_and_are_invalidated(tmp_path, storage):
    path = tmp_path / "memory.sqlite" if storage else None
    memory = EpisodicMemory(storage=path)
    opened = memory.open_run()
    frame = opened.open_frame(_spec())
    memory.close()
    for operation in (
        memory.open_run,
        memory.reset,
        lambda: opened.snapshot,
        lambda: opened.open_frame(_spec()),
        lambda: opened.commit("c", ()),
        frame.recall,
        lambda: frame.learn(_context()),
    ):
        with pytest.raises(c.MemoryOperationError) as caught:
            operation()
        assert caught.value.code == "episodic.closed"
    opened.close()
    opened.close()


@pytest.mark.parametrize(
    "case",
    [
        "extra",
        "version",
        "counter",
        "flag",
        "ledger",
        "digest",
        "unopened",
        "uncommitted",
        "reset",
    ],
)
def test_invalid_inventory_is_rejected_without_rewriting_or_retaining_owner(
    tmp_path, case
):
    path = tmp_path / "memory.sqlite"
    with EpisodicMemory(storage=path) as memory:
        _snapshot(memory)
    storage = SQLiteStorage(path, "episodic")
    inventory = storage.load()
    if case == "extra":
        inventory["extra"] = None
    elif case == "version":
        inventory["version"] = True
    elif case == "counter":
        inventory["snapshot"]["revision"] = -1
    elif case == "flag":
        inventory["ever_opened"] = 1
    elif case == "ledger":
        inventory["applied_commits"] = []
    elif case == "digest":
        inventory["applied_commits"] = {"c": "not-a-digest"}
    elif case == "unopened":
        inventory["ever_opened"] = False
        inventory["ever_committed"] = True
    elif case == "uncommitted":
        inventory["snapshot"]["revision"] = 1
    elif case == "reset":
        inventory["ever_reset"] = True
    storage.save(inventory)
    storage.close()
    with pytest.raises(c.MemoryOperationError) as caught:
        EpisodicMemory(storage=path)
    assert caught.value.code == "episodic.invalid_storage"
    storage = SQLiteStorage(path, "episodic")
    try:
        assert storage.load() == inventory
    finally:
        storage.close()


def test_history_frame_limits_and_pinned_snapshots_survive_restart(tmp_path):
    path = tmp_path / "memory.sqlite"
    with EpisodicMemory(2, 1, storage=path) as memory:
        old = _snapshot(memory)
        _append(memory, "first")
        expected = _snapshot(memory)
    with EpisodicMemory(9, 7, storage=path) as memory:
        assert _snapshot(memory) == expected
        with memory.open_run() as opened:
            frame = opened.open_frame(_spec())
            assert "first" in frame.recall()
            opened.open_frame(_spec("new"))
            configs = {
                item["ref"]["static_path"][-1]: item["config"]
                for item in opened.snapshot["frames"]
            }
            assert configs == {
                "agent": {"max_episodes": 2, "recall_limit": 1},
                "new": {"max_episodes": 9, "recall_limit": 7},
            }
            assert "storage" not in json_dump(opened.snapshot)
            assert str(path) not in json_dump(opened.snapshot)
        with memory.open_run(snapshot=old) as opened:
            assert opened.snapshot == old
            assert opened.open_frame(_spec()).recall() == ""
        assert _snapshot(memory) == expected


def json_dump(value):
    return json.dumps(value, sort_keys=True)


@pytest.mark.parametrize("prior", ["opened", "reset", "committed"])
def test_nonpristine_flags_survive_restart(prior, tmp_path):
    path = tmp_path / "memory.sqlite"
    foreign = _snapshot(EpisodicMemory())
    with EpisodicMemory(storage=path) as memory:
        if prior == "opened":
            _snapshot(memory)
        elif prior == "reset":
            memory.reset()
        else:
            with memory.open_run() as opened:
                opened.commit("empty", ())
    with EpisodicMemory(storage=path) as memory:
        with pytest.raises(c.MemoryOperationError) as caught:
            memory.open_run(snapshot=foreign)
        assert caught.value.code == "episodic.invalid_snapshot"
        assert getattr(memory, "_ever_" + prior)


def test_pristine_adoption_persists_across_restart(tmp_path):
    path = tmp_path / "memory.sqlite"
    with EpisodicMemory(5, 2) as source:
        _append(source, "adopted")
        saved = _snapshot(source)
    EpisodicMemory(storage=path).close()
    with EpisodicMemory(storage=path) as memory:
        with memory.open_run(snapshot=saved) as opened:
            assert opened.snapshot == saved
    with EpisodicMemory(storage=path) as memory:
        assert _snapshot(memory) == saved
        assert memory._ever_opened and memory._ever_committed


def test_commit_tombstones_survive_eviction_reset_restart_and_stale_replay(tmp_path):
    path = tmp_path / "memory.sqlite"
    with EpisodicMemory(1, 1, storage=path) as memory:
        old = _snapshot(memory)
        batch = _append(memory, "first")
        _append(memory, "second")
        expected = _snapshot(memory)
        assert "first" not in json_dump(expected)
    with EpisodicMemory(storage=path) as memory:
        with memory.open_run(snapshot=old) as stale:
            stale.commit("first", batch)
            with pytest.raises(c.MemoryOperationError) as caught:
                stale.commit("first", ())
            assert caught.value.code == "episodic.commit_collision"
        assert _snapshot(memory) == expected
        memory.reset()
    with EpisodicMemory(storage=path) as memory:
        with memory.open_run() as opened:
            opened.commit("first", batch)
            opened.commit("first", batch)
            with pytest.raises(c.MemoryOperationError) as caught:
                opened.commit("first", ())
            assert caught.value.code == "episodic.commit_collision"
        current = _snapshot(memory)
        assert current["frames"] == [] and current["revision"] == 0
        assert current["epoch"] == expected["epoch"] + 1


@pytest.mark.parametrize("operation", ["open", "adopt", "reset", "commit"])
def test_failed_save_never_publishes_candidate_ram(tmp_path, monkeypatch, operation):
    memory = EpisodicMemory(storage=tmp_path / "memory.sqlite")
    session = frame = None
    if operation == "commit":
        session = memory.open_run()
        frame = session.open_frame(_spec())
    before = deepcopy(
        (
            memory._namespace,
            memory._epoch,
            memory._revision,
            memory._frames,
            memory._applied_commits,
            memory._ever_opened,
            memory._ever_reset,
            memory._ever_committed,
        )
    )
    primary = RuntimeError("save failed")

    def reject(candidate):
        raise primary

    monkeypatch.setattr(memory._storage, "save", reject)
    try:
        with pytest.raises(RuntimeError) as caught:
            if operation == "open":
                memory.open_run()
            elif operation == "adopt":
                memory.open_run(snapshot=_snapshot(EpisodicMemory()))
            elif operation == "reset":
                memory.reset()
            else:
                session.commit("commit", tuple(frame.learn(_context())))
        assert caught.value is primary
        assert (
            memory._namespace,
            memory._epoch,
            memory._revision,
            memory._frames,
            memory._applied_commits,
            memory._ever_opened,
            memory._ever_reset,
            memory._ever_committed,
        ) == before
    finally:
        memory.close()


@pytest.mark.parametrize("after_commit", [False, True])
@pytest.mark.parametrize("cancel", [False, True])
def test_uncertain_save_poisoning_blocks_ram_reads_and_keeps_primary(
    tmp_path, monkeypatch, after_commit, cancel
):
    connect = sqlite3.connect
    primary = (
        KeyboardInterrupt("cancel")
        if cancel
        else sqlite3.OperationalError("private path")
    )

    class Connection(sqlite3.Connection):
        armed = False

        def commit(self):
            if self.armed:
                if after_commit:
                    super().commit()
                raise primary
            return super().commit()

    monkeypatch.setattr(
        sqlite3,
        "connect",
        lambda *args, **kwargs: connect(*args, **kwargs, factory=Connection),
    )
    path = tmp_path / "memory.sqlite"
    memory = EpisodicMemory(storage=path)
    opened = memory.open_run()
    frame = opened.open_frame(_spec())
    before = opened.snapshot
    memory._storage._connection.armed = True
    try:
        with pytest.raises(BaseException) as caught:
            opened.commit("uncertain", tuple(frame.learn(_context())))
        if cancel:
            assert caught.value is primary
        else:
            assert caught.value.code == "storage.io"
            assert "private path" not in str(caught.value)
        assert (
            memory._revision == 0
            and memory._frames == {}
            and memory._applied_commits == {}
        )
        for operation in (
            memory.open_run,
            memory.reset,
            lambda: opened.snapshot,
            frame.recall,
            lambda: frame.learn(_context()),
            lambda: opened.open_frame(_spec()),
            lambda: opened.commit("c", ()),
        ):
            with pytest.raises(c.MemoryOperationError) as failed:
                operation()
            assert failed.value.code == "storage.poisoned"
        with pytest.raises(c.MemoryOperationError) as busy:
            EpisodicMemory(storage=path)
        assert busy.value.code == "storage.busy"
    finally:
        opened.close()
        memory.close()
    with EpisodicMemory(storage=path) as reopened:
        assert _snapshot(reopened)["revision"] == int(after_commit)
        assert reopened._namespace == before["namespace"]


def test_constructor_cleanup_preserves_cancellation_and_releases_ownership(
    tmp_path, monkeypatch
):
    path = tmp_path / "memory.sqlite"
    EpisodicMemory(storage=path).close()
    primary = KeyboardInterrupt("initialization stopped")
    original = episodic._parse_inventory
    close = SQLiteStorage.close

    def reject(value):
        raise primary

    def cleanup(self):
        close(self)
        raise RuntimeError("cleanup problem")

    monkeypatch.setattr(episodic, "_parse_inventory", reject)
    monkeypatch.setattr(SQLiteStorage, "close", cleanup)
    with pytest.raises(KeyboardInterrupt) as caught:
        EpisodicMemory(storage=path)
    assert caught.value is primary
    monkeypatch.setattr(episodic, "_parse_inventory", original)
    monkeypatch.setattr(SQLiteStorage, "close", close)
    with EpisodicMemory(storage=path):
        pass


def _close_during_callback(path, reentrant):
    entered, release = Event(), Event()
    errors = []
    memory = EpisodicMemory(storage=path)
    opened = memory.open_run()
    frame = opened.open_frame(_spec())

    def rewrite(context):
        if reentrant:
            opened.close()
            with pytest.raises(c.MemoryOperationError) as caught:
                memory.close()
            assert caught.value.code == "episodic.busy"
        else:
            entered.set()
            assert release.wait(5)
        return "query"

    def recall():
        try:
            frame.recall(c.RecallContext("", "query", {}, _ref().static_path))
        except BaseException as error:
            errors.append(error)

    memory.rewrite_query = rewrite
    if reentrant:
        recall()
    else:
        worker = Thread(target=recall, daemon=True)
        worker.start()
        assert entered.wait(5)
        try:
            with pytest.raises(c.MemoryOperationError) as caught:
                memory.close()
            assert caught.value.code == "episodic.busy"
            with pytest.raises(c.MemoryOperationError) as busy:
                EpisodicMemory(storage=path)
            assert busy.value.code == "storage.busy"
        finally:
            release.set()
            worker.join(5)
        assert not worker.is_alive()
    assert not errors
    opened.close()
    memory.close()
    with EpisodicMemory(storage=path):
        pass


def _fork_while_locked(path):
    memory = EpisodicMemory(storage=path)
    opened = memory.open_run()
    frame = opened.open_frame(_spec())
    locked, release = Event(), Event()

    def hold():
        with memory._lock:
            locked.set()
            release.wait(10)

    thread = Thread(target=hold, daemon=True)
    thread.start()
    assert locked.wait(5)
    pid = os.fork()
    if pid == 0:
        try:
            for operation in (
                memory.open_run,
                memory.reset,
                memory.close,
                lambda: opened.snapshot,
                opened.close,
                frame.recall,
                lambda: opened.commit("c", ()),
            ):
                try:
                    operation()
                except c.MemoryOperationError as error:
                    assert error.code == "storage.forked"
                else:
                    raise AssertionError("inherited handle accepted")
        except BaseException:
            os._exit(2)
        os._exit(0)
    try:
        deadline = time.monotonic() + 4
        while True:
            finished, status = os.waitpid(pid, os.WNOHANG)
            if finished:
                break
            if time.monotonic() >= deadline:
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
                raise AssertionError("inherited handle waited on a parent lock")
            time.sleep(0.01)
        assert os.waitstatus_to_exitcode(status) == 0
    finally:
        release.set()
        thread.join(5)
        opened.close()
        memory.close()


def _bounded(target, args):
    child = multiprocessing.get_context("spawn").Process(target=target, args=args)
    child.start()
    child.join(12)
    if child.is_alive():
        child.terminate()
        child.join(3)
        pytest.fail("owned lifecycle probe timed out")
    assert child.exitcode == 0


@pytest.mark.parametrize("reentrant", [False, True])
def test_active_callback_makes_close_busy_without_releasing_file(tmp_path, reentrant):
    _bounded(_close_during_callback, (str(tmp_path / "memory.sqlite"), reentrant))


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires fork")
def test_inherited_handle_rejects_before_an_inherited_backend_lock(tmp_path):
    _bounded(_fork_while_locked, (str(tmp_path / "memory.sqlite"),))


def test_invalid_limits_do_not_create_storage(tmp_path):
    path = tmp_path / "memory.sqlite"
    with pytest.raises(ValueError):
        EpisodicMemory(0, storage=path)
    assert not path.exists()


def test_recorded_empty_frame_limits_override_reopened_host_defaults(tmp_path):
    path = tmp_path / "memory.sqlite"
    with EpisodicMemory(4, 2, storage=path) as memory:
        with memory.open_run() as opened:
            opened.open_frame(_spec())
            recorded = opened.snapshot
    with EpisodicMemory(99, 88, storage=path) as memory:
        with memory.open_run(snapshot=recorded) as replay:
            replay.open_frame(_spec())
            assert replay.snapshot == recorded
            replay.commit("declare", ())
    with EpisodicMemory(storage=path) as memory:
        assert _snapshot(memory)["frames"][0]["config"] == {
            "max_episodes": 4,
            "recall_limit": 2,
        }


def test_corrupt_private_state_never_partially_adopts_ram(tmp_path):
    path = tmp_path / "memory.sqlite"
    with EpisodicMemory(storage=path) as memory:
        _append(memory, "saved")
    storage = SQLiteStorage(path, "episodic")
    value = storage.load()
    value["applied_commits"]["bad"] = "invalid"
    storage.save(value)
    storage.close()
    partial = EpisodicMemory.__new__(EpisodicMemory)
    with pytest.raises(c.MemoryOperationError):
        partial.__init__(storage=path)
    assert partial._namespace != value["snapshot"]["namespace"]
    assert partial._frames == {} and partial._applied_commits == {}
    assert partial._epoch == partial._revision == 0


def test_invalid_last_mutation_never_reaches_storage(tmp_path, monkeypatch):
    with EpisodicMemory(storage=tmp_path / "memory.sqlite") as memory:
        with memory.open_run() as opened:
            frame = opened.open_frame(_spec())
            first = frame.learn(_context())[0]
            second = c.MemoryMutation("wrong", _ref(), first.payload)
            calls = []
            monkeypatch.setattr(
                memory._storage, "save", lambda value: calls.append(value)
            )
            with pytest.raises(c.MemoryOperationError) as caught:
                opened.commit("invalid", (first, second))
            assert caught.value.code == "episodic.invalid_batch"
            assert calls == [] and memory._frames == {} and memory._revision == 0
