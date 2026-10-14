"""Atomic file ownership and finite-JSON storage for memory backends."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import multiprocessing
import os
import sqlite3
import stat
from threading import Barrier
from unittest.mock import patch

import pytest

from yian import defaults
from yian.memory.contracts import MemoryOperationError
from yian.memory.storage import SQLiteStorage, check_storage_path


def _assert_error(error: MemoryOperationError, code: str) -> None:
    assert error.stage == "storage"
    assert error.code == f"storage.{code}"
    assert error.retryable is (code == "busy")


def _attempt_open(path, pipe):
    try:
        store = SQLiteStorage(path, "session")
    except MemoryOperationError as error:
        pipe.send((error.stage, error.code, error.retryable))
    else:
        store.close()
        pipe.send("opened")
    finally:
        pipe.close()


def _crash_at_commit(path, cut):
    connect = sqlite3.connect

    class Connection(sqlite3.Connection):
        armed = False

        def commit(self):
            if self.armed and cut == "before":
                os._exit(23)
            result = super().commit()
            if self.armed and cut == "after":
                os._exit(23)
            return result

    with patch(
        "sqlite3.connect", lambda *a, **kw: connect(*a, **kw, factory=Connection)
    ):
        store = SQLiteStorage(path, "session")
        store._connection.armed = True
        store.save({"value": "new", "large": "x" * 100_000})


def _crash_before_connect(path):
    with patch("sqlite3.connect", lambda *a, **kw: os._exit(23)):
        SQLiteStorage(path, "session")


def _hold_until_killed(path, pipe):
    store = SQLiteStorage(path, "session")
    store.save({"saved": True})
    pipe.send("ready")
    pipe.recv()


def _inherited_operations(store, pipe):
    outcomes = []
    for operation in (
        store.ensure_open,
        store.load,
        lambda: store.save({}),
        store.close,
    ):
        try:
            operation()
        except MemoryOperationError as error:
            outcomes.append(error.code)
    pipe.send(outcomes)
    pipe.close()


def _join(process, *, expected=0):
    process.join(10)
    if process.is_alive():
        process.kill()
        process.join(5)
        pytest.fail("storage child exceeded its deadline")
    assert process.exitcode == expected


def _child_result(target, *args, context="spawn"):
    ctx = multiprocessing.get_context(context)
    receive, send = ctx.Pipe(duplex=False)
    process = ctx.Process(target=target, args=(*args, send))
    process.start()
    send.close()
    try:
        assert receive.poll(10), "storage child did not report within its deadline"
        result = receive.recv()
        _join(process)
        return result
    finally:
        receive.close()
        if process.is_alive():
            process.kill()
            process.join(5)


def test_roundtrip_is_detached_and_survives_reopen(tmp_path):
    path = tmp_path / "memory.sqlite"
    store = SQLiteStorage(path, "session")
    try:
        assert store.load() is None
        value = {"nested": [None, True, 1, 1.5, {"text": "日本語"}], "empty": {}}
        expected = json.loads(json.dumps(value))
        store.save(value)
        value["nested"].append("caller mutation")
        loaded = store.load()
        assert loaded == expected
        loaded["nested"].append("reader mutation")
        assert store.load() == expected
        store.ensure_open()
    finally:
        store.close()
    reopened = SQLiteStorage(str(path), "session")
    try:
        assert reopened.load() == expected
    finally:
        reopened.close()


@pytest.mark.parametrize(
    "invalid",
    [
        None,
        [],
        {"x": float("nan")},
        {"x": float("inf")},
        {"x": object()},
        {1: "bad"},
        {"x": (1,)},
        {"x": "\ud800"},
    ],
)
def test_invalid_candidate_is_rejected_without_poisoning(tmp_path, invalid):
    store = SQLiteStorage(tmp_path / "memory.sqlite", "session")
    try:
        store.save({"old": True})
        with pytest.raises(MemoryOperationError) as caught:
            store.save(invalid)
        _assert_error(caught.value, "invalid_state")
        assert store.load() == {"old": True}
        store.save({"valid": True})
    finally:
        store.close()


def test_cycles_are_rejected_before_a_transaction(tmp_path):
    store = SQLiteStorage(tmp_path / "memory.sqlite", "session")
    cycle = {}
    cycle["self"] = cycle
    try:
        with pytest.raises(MemoryOperationError) as caught:
            store.save(cycle)
        _assert_error(caught.value, "invalid_state")
        assert not store._connection.in_transaction
        assert store.load() is None
    finally:
        store.close()


def _nested_object(containers):
    value = {}
    for _ in range(containers - 1):
        value = {"nested": value}
    return value


def test_envelope_depth_rejection_preserves_old_state_and_usable_handle(tmp_path):
    path = tmp_path / "memory.sqlite"
    store = SQLiteStorage(path, "session")
    try:
        store.save({"old": True})
        statements = []
        store._connection.set_trace_callback(statements.append)
        with pytest.raises(MemoryOperationError) as caught:
            store.save(_nested_object(defaults.current().memory_json_nesting))
        _assert_error(caught.value, "invalid_state")
        assert statements == []
        store._connection.set_trace_callback(None)
        assert not store._connection.in_transaction
        store.ensure_open()
        assert store.load() == {"old": True}
        store.save({"later": True})
    finally:
        store.close()
    reopened = SQLiteStorage(path, "session")
    try:
        assert reopened.load() == {"later": True}
    finally:
        reopened.close()


def test_maximum_valid_envelope_depth_roundtrips(tmp_path):
    path = tmp_path / "memory.sqlite"
    value = _nested_object(defaults.current().memory_json_nesting - 1)
    store = SQLiteStorage(path, "session")
    try:
        store.save(value)
        assert store.load() == value
    finally:
        store.close()
    reopened = SQLiteStorage(path, "session")
    try:
        assert reopened.load() == value
    finally:
        reopened.close()


@pytest.mark.parametrize(
    "path",
    ["", ":memory:", "file:memory.sqlite", b"memory.sqlite", None, "bad\x00path"],
)
def test_only_file_paths_are_accepted(path):
    with pytest.raises(MemoryOperationError) as caught:
        SQLiteStorage(path, "session")
    _assert_error(caught.value, "path")


@pytest.mark.parametrize("backend", ["", None, 1, "\ud800"])
def test_backend_kind_requires_a_nonempty_string(tmp_path, backend):
    path = tmp_path / "memory.sqlite"
    with pytest.raises(MemoryOperationError) as caught:
        SQLiteStorage(path, backend)
    _assert_error(caught.value, "backend")
    assert not path.exists()


def test_missing_parent_and_directory_are_rejected_without_path_disclosure(tmp_path):
    path = tmp_path / "private-host-name" / "memory.sqlite"
    with pytest.raises(MemoryOperationError) as caught:
        SQLiteStorage(path, "session")
    assert caught.value.stage == "storage"
    assert "private-host-name" not in str(caught.value)
    assert not path.parent.exists()
    with pytest.raises(MemoryOperationError):
        SQLiteStorage(tmp_path, "session")


def test_new_file_permissions_and_connection_settings(tmp_path):
    path = tmp_path / "memory.sqlite"
    store = SQLiteStorage(path, "session")
    try:
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert store._connection.execute("PRAGMA synchronous").fetchone() == (3,)
        assert store._connection.execute("PRAGMA locking_mode").fetchone() == (
            "exclusive",
        )
        assert store._connection.execute("PRAGMA journal_mode").fetchone() == (
            "delete",
        )
        assert store._connection.execute("PRAGMA busy_timeout").fetchone() == (0,)
    finally:
        store.close()


def test_new_file_permissions_do_not_depend_on_umask(tmp_path):
    path = tmp_path / "memory.sqlite"
    previous = os.umask(0o777)
    try:
        store = SQLiteStorage(path, "session")
    finally:
        os.umask(previous)
    try:
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    finally:
        store.close()


def test_path_io_errors_are_sanitized(tmp_path, monkeypatch):
    def fail_path(*args):
        raise OSError("secret-host-path")

    monkeypatch.setattr(os.path, "realpath", fail_path)
    with pytest.raises(MemoryOperationError) as caught:
        SQLiteStorage(tmp_path / "memory.sqlite", "session")
    _assert_error(caught.value, "io")
    assert "secret-host-path" not in str(caught.value)


def test_zero_byte_file_initializes_without_changing_permissions(tmp_path):
    path = tmp_path / "empty.sqlite"
    path.touch(mode=0o640)
    store = SQLiteStorage(path, "session")
    try:
        assert store.load() is None
        assert stat.S_IMODE(path.stat().st_mode) == 0o640
    finally:
        store.close()


@pytest.mark.parametrize(
    "damage",
    [
        "kind",
        "version",
        "frame_era_version",
        "boolean_version",
        "extra_field",
        "missing_field",
        "duplicate_key",
        "bad_json",
        "nonfinite",
        "not_object",
        "no_row",
        "extra_table",
        "altered_schema",
        "blob",
    ],
)
def test_invalid_store_rejected_without_overwriting_or_leaking_owner(tmp_path, damage):
    path = tmp_path / "memory.sqlite"
    store = SQLiteStorage(path, "session")
    store.save({"old": True})
    store.close()
    with sqlite3.connect(path) as connection:
        raw = connection.execute("SELECT envelope FROM memory_state").fetchone()[0]
        envelope = json.loads(raw)
        if damage == "kind":
            envelope["backend"] = "episodic"
        elif damage == "version":
            envelope["version"] = 3
        elif damage == "frame_era_version":
            envelope["version"] = 1
        elif damage == "boolean_version":
            envelope["version"] = True
        elif damage == "extra_field":
            envelope["extra"] = True
        elif damage == "missing_field":
            del envelope["state"]
        elif damage == "not_object":
            envelope["state"] = []
        raw = json.dumps(envelope)
        if damage == "duplicate_key":
            raw = raw[:-1] + ', "state": {}}'
        elif damage == "bad_json":
            raw = "{"
        elif damage == "nonfinite":
            raw = raw.replace('"old": true', '"old": 1e999')
        elif damage == "blob":
            raw = raw.encode()
        connection.execute("UPDATE memory_state SET envelope = ?", (raw,))
        if damage == "no_row":
            connection.execute("DELETE FROM memory_state")
        elif damage == "extra_table":
            connection.execute("CREATE TABLE unrelated(value)")
        elif damage == "altered_schema":
            connection.execute("ALTER TABLE memory_state ADD COLUMN extra TEXT")
    before = path.read_bytes()
    for _ in range(2):
        with pytest.raises(MemoryOperationError) as caught:
            SQLiteStorage(path, "session")
        assert caught.value.code != "storage.busy"
        if damage in {"version", "frame_era_version"}:
            assert caught.value.code == "storage.version"
        assert path.read_bytes() == before


@pytest.mark.parametrize("unrelated", ["table", "empty_database", "corrupt"])
def test_unrelated_files_are_preserved(tmp_path, unrelated):
    path = tmp_path / "memory.sqlite"
    if unrelated == "corrupt":
        path.write_bytes(b"caller data, not sqlite")
    else:
        with sqlite3.connect(path) as connection:
            connection.execute("CREATE TABLE unrelated(value)")
            if unrelated == "empty_database":
                connection.execute("DROP TABLE unrelated")
    before = path.read_bytes()
    with pytest.raises(MemoryOperationError):
        SQLiteStorage(path, "session")
    assert path.read_bytes() == before


@pytest.mark.parametrize("alias", [False, True])
def test_lifetime_owner_survives_commits_and_failed_competitors(tmp_path, alias):
    path = tmp_path / "memory.sqlite"
    contender = tmp_path / "alias.sqlite" if alias else path
    store = SQLiteStorage(path, "session")
    if alias:
        contender.symlink_to(path)
    try:
        for value in range(3):
            store.save({"value": value})
            with pytest.raises(MemoryOperationError) as caught:
                SQLiteStorage(contender, "session")
            _assert_error(caught.value, "busy")
            assert _child_result(_attempt_open, contender) == (
                "storage",
                "storage.busy",
                True,
            )
            assert store.load() == {"value": value}
    finally:
        store.close()
    assert _child_result(_attempt_open, contender) == "opened"


@pytest.mark.parametrize(
    "failure",
    [
        sqlite3.OperationalError("secret-host-path"),
        OSError("secret-host-path"),
        KeyboardInterrupt("stop"),
        SystemExit(7),
    ],
)
@pytest.mark.parametrize("after_commit", [False, True])
def test_failed_save_poisoning_retains_ownership_and_primary(
    tmp_path, monkeypatch, failure, after_commit
):
    connect = sqlite3.connect

    class Connection(sqlite3.Connection):
        armed = False

        def commit(self):
            if not self.armed:
                return super().commit()
            if after_commit:
                super().commit()
            raise failure

    monkeypatch.setattr(
        sqlite3, "connect", lambda *a, **kw: connect(*a, **kw, factory=Connection)
    )
    path = tmp_path / "memory.sqlite"
    store = SQLiteStorage(path, "session")
    store.save({"old": True})
    store._connection.armed = True
    try:
        with pytest.raises(BaseException) as caught:
            store.save({"new": True})
        if isinstance(failure, (sqlite3.Error, OSError)):
            assert isinstance(caught.value, MemoryOperationError)
            assert "secret-host-path" not in str(caught.value)
        else:
            assert caught.value is failure
        for operation in (store.ensure_open, store.load, lambda: store.save({})):
            with pytest.raises(MemoryOperationError) as poisoned:
                operation()
            _assert_error(poisoned.value, "poisoned")
        with pytest.raises(MemoryOperationError) as busy:
            SQLiteStorage(path, "session")
        _assert_error(busy.value, "busy")
    finally:
        store.close()
    reopened = SQLiteStorage(path, "session")
    try:
        assert reopened.load() == ({"new": True} if after_commit else {"old": True})
    finally:
        reopened.close()


def test_close_is_idempotent_and_closed_methods_do_not_touch_os_or_sqlite(
    tmp_path, monkeypatch
):
    store = SQLiteStorage(tmp_path / "memory.sqlite", "session")
    store.close()

    def forbidden(*args, **kwargs):
        pytest.fail("closed storage touched OS or SQLite")

    monkeypatch.setattr(os, "getpid", forbidden)
    monkeypatch.setattr(sqlite3, "connect", forbidden)
    store.close()
    for operation in (store.ensure_open, store.load, lambda: store.save({})):
        with pytest.raises(MemoryOperationError) as caught:
            operation()
        _assert_error(caught.value, "closed")


@pytest.mark.parametrize(
    "failure",
    [
        KeyboardInterrupt("stop"),
        SystemExit(7),
        sqlite3.OperationalError("secret-host-path"),
    ],
)
def test_failed_constructor_rolls_back_and_releases_connection(
    tmp_path, monkeypatch, failure
):
    connect = sqlite3.connect
    connections = []

    class Connection(sqlite3.Connection):
        def commit(self):
            raise failure

    def failing_connect(*args, **kwargs):
        connection = connect(*args, **kwargs, factory=Connection)
        connections.append(connection)
        return connection

    path = tmp_path / "memory.sqlite"
    with monkeypatch.context() as patcher:
        patcher.setattr(sqlite3, "connect", failing_connect)
        with pytest.raises(BaseException) as caught:
            SQLiteStorage(path, "session")
    if isinstance(failure, sqlite3.Error):
        assert isinstance(caught.value, MemoryOperationError)
        assert "secret-host-path" not in str(caught.value)
    else:
        assert caught.value is failure
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connections[0].execute("SELECT 1")
    reopened = SQLiteStorage(path, "session")
    try:
        assert reopened.load() is None
    finally:
        reopened.close()


def test_rollback_failure_does_not_replace_save_failure_or_release_owner(
    tmp_path, monkeypatch
):
    connect = sqlite3.connect
    primary = KeyboardInterrupt("original")

    class Connection(sqlite3.Connection):
        armed = False

        def commit(self):
            if self.armed:
                raise primary
            return super().commit()

        def rollback(self):
            if self.armed:
                raise SystemExit("cleanup")
            return super().rollback()

    monkeypatch.setattr(
        sqlite3, "connect", lambda *a, **kw: connect(*a, **kw, factory=Connection)
    )
    path = tmp_path / "memory.sqlite"
    store = SQLiteStorage(path, "session")
    store.save({"old": True})
    store._connection.armed = True
    try:
        with pytest.raises(KeyboardInterrupt) as caught:
            store.save({"new": True})
        assert caught.value is primary
        assert _child_result(_attempt_open, path) == ("storage", "storage.busy", True)
    finally:
        store.close()
    reopened = SQLiteStorage(path, "session")
    try:
        assert reopened.load() == {"old": True}
    finally:
        reopened.close()


def test_load_io_failure_poisoning_preserves_owner(tmp_path, monkeypatch):
    connect = sqlite3.connect

    class Connection(sqlite3.Connection):
        armed = False

        def execute(self, sql, *args):
            if self.armed and sql.startswith("SELECT id"):
                raise sqlite3.OperationalError("secret-host-path")
            return super().execute(sql, *args)

    monkeypatch.setattr(
        sqlite3, "connect", lambda *a, **kw: connect(*a, **kw, factory=Connection)
    )
    path = tmp_path / "memory.sqlite"
    store = SQLiteStorage(path, "session")
    store._connection.armed = True
    try:
        with pytest.raises(MemoryOperationError) as caught:
            store.load()
        _assert_error(caught.value, "io")
        assert "secret-host-path" not in str(caught.value)
        with pytest.raises(MemoryOperationError) as poisoned:
            store.ensure_open()
        _assert_error(poisoned.value, "poisoned")
        assert _child_result(_attempt_open, path) == ("storage", "storage.busy", True)
    finally:
        store.close()


def test_shared_connection_serializes_threads(tmp_path):
    store = SQLiteStorage(tmp_path / "memory.sqlite", "session")
    barrier = Barrier(6)

    def write(index):
        barrier.wait(5)
        for _ in range(5):
            store.save({"writer": index})
            assert store.load()["writer"] in range(6)

    try:
        with ThreadPoolExecutor(max_workers=6) as pool:
            for future in [pool.submit(write, index) for index in range(6)]:
                future.result(10)
    finally:
        store.close()


@pytest.mark.parametrize("cut", ["before", "after"])
def test_process_crash_at_commit_is_atomic(tmp_path, cut):
    path = tmp_path / "memory.sqlite"
    store = SQLiteStorage(path, "session")
    store.save({"value": "old"})
    store.close()
    process = multiprocessing.get_context("spawn").Process(
        target=_crash_at_commit, args=(path, cut)
    )
    process.start()
    _join(process, expected=23)
    reopened = SQLiteStorage(path, "session")
    try:
        expected = (
            {"value": "old"}
            if cut == "before"
            else {"value": "new", "large": "x" * 100_000}
        )
        assert reopened.load() == expected
    finally:
        reopened.close()


def test_creation_crash_leaves_recoverable_empty_file(tmp_path):
    path = tmp_path / "memory.sqlite"
    process = multiprocessing.get_context("spawn").Process(
        target=_crash_before_connect, args=(path,)
    )
    process.start()
    _join(process, expected=23)
    assert path.read_bytes() == b""
    store = SQLiteStorage(path, "session")
    try:
        assert store.load() is None
        store.save({"recovered": True})
    finally:
        store.close()


def test_process_death_releases_owner_and_preserves_commit(tmp_path):
    path = tmp_path / "memory.sqlite"
    ctx = multiprocessing.get_context("spawn")
    parent, child = ctx.Pipe()
    process = ctx.Process(target=_hold_until_killed, args=(path, child))
    process.start()
    child.close()
    try:
        assert parent.poll(10)
        assert parent.recv() == "ready"
        with pytest.raises(MemoryOperationError) as caught:
            SQLiteStorage(path, "session")
        _assert_error(caught.value, "busy")
    finally:
        process.kill()
        process.join(5)
        parent.close()
    assert not process.is_alive()
    store = SQLiteStorage(path, "session")
    try:
        assert store.load() == {"saved": True}
    finally:
        store.close()


@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(), reason="requires fork"
)
def test_postfork_guard_precedes_python_lock_and_sqlite(tmp_path):
    path = tmp_path / "memory.sqlite"
    store = SQLiteStorage(path, "session")
    store.save({"parent": True})
    try:
        with store._lock:
            assert (
                _child_result(_inherited_operations, store, context="fork")
                == ["storage.forked"] * 4
            )
        assert store.load() == {"parent": True}
        with pytest.raises(MemoryOperationError) as caught:
            SQLiteStorage(path, "session")
        _assert_error(caught.value, "busy")
    finally:
        store.close()


def test_check_storage_path_returns_an_absolute_path_without_io(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert check_storage_path("m.sqlite") == os.path.join(os.getcwd(), "m.sqlite")
    assert check_storage_path(tmp_path / "x" / "m.sqlite") == str(tmp_path / "x" / "m.sqlite")
    assert list(tmp_path.iterdir()) == []
    for bad in ("", ":memory:", "FILE:m.sqlite", "a\x00b", b"m.sqlite", 3):
        with pytest.raises(MemoryOperationError) as caught:
            check_storage_path(bad)
        _assert_error(caught.value, "path")


def test_storage_path_follows_symlinks_before_parent_steps(tmp_path):
    target = tmp_path / "target" / "sub"
    target.mkdir(parents=True)
    (tmp_path / "link").symlink_to(target)
    raw = os.path.join(str(tmp_path), "link", "..", "m.sqlite")
    assert check_storage_path(raw) == raw
    store = SQLiteStorage(raw, "session")
    try:
        store.save({"ok": True})
    finally:
        store.close()
    assert (tmp_path / "target" / "m.sqlite").is_file()
    assert not (tmp_path / "m.sqlite").exists()


def test_create_parent_makes_missing_directories(tmp_path):
    path = tmp_path / "a" / "b" / "m.sqlite"
    with pytest.raises(MemoryOperationError) as caught:
        SQLiteStorage(path, "episodic")
    _assert_error(caught.value, "io")
    assert not (tmp_path / "a").exists()
    storage = SQLiteStorage(path, "episodic", create_parent=True)
    storage.close()
    assert path.is_file()
