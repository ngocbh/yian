"""Failure-boundary tests for in-memory and locked JSONL run logs."""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Any, Callable

import pytest

import agent_composer.runtime.log as log_module
from agent_composer.runtime.log import (
    SCHEMA_VERSION,
    JsonlRunLog,
    MemoryRunLog,
    NodeExpandedRecord,
    NodeStartedRecord,
    RecordEnvelope,
    encode_record,
)


_DIGEST = "d" * 64


def test_schema_version_is_nine():
    assert SCHEMA_VERSION == 9


@pytest.mark.parametrize("schema", [7, 8])
def test_retired_schema_record_is_rejected_by_name(schema):
    message = f"run log schema {schema} predates schema 9 and is not supported"
    with pytest.raises(ValueError, match=message):
        log_module.decode_record(_wire_line(schema=schema))
    with pytest.raises(ValueError, match=message):
        RecordEnvelope(schema, "run", 0, NodeStartedRecord("work"))


@pytest.mark.parametrize("schema", [7, 8])
def test_retired_schema_log_is_rejected_on_resume(tmp_path, schema):
    path = tmp_path / "retired.jsonl"
    path.write_bytes(_wire_line(schema=schema))
    with pytest.raises(ValueError, match=f"run log schema {schema} predates schema 9"):
        JsonlRunLog.resume(path)
    assert path.read_bytes() == _wire_line(schema=schema)
    _assert_lock_available(path)


@pytest.mark.parametrize("schema", [6, 7, 8, 10, True, False, 9.0])
def test_supported_versions_do_not_weaken_validation(schema):
    with pytest.raises(ValueError):
        RecordEnvelope(schema, "run", 0, NodeStartedRecord("work"))
    envelope = RecordEnvelope(SCHEMA_VERSION, "run", 0, NodeStartedRecord("work"))
    object.__setattr__(envelope, "schema", schema)
    with pytest.raises(ValueError):
        encode_record(envelope)
    with pytest.raises(ValueError):
        log_module.decode_record(_wire_line(schema=schema))


@pytest.mark.parametrize("corruption", ["extra", "node_id", "seq", "kind"])
def test_supported_versions_do_not_weaken_payload_validation(corruption):
    wire = json.loads(_wire_line())
    if corruption == "extra":
        wire["payload"]["extra"] = None
    elif corruption == "node_id":
        wire["payload"]["node_id"] = True
    elif corruption == "seq":
        wire["seq"] = False
    else:
        wire["kind"] = "unsupported"
    with pytest.raises(ValueError):
        log_module.decode_record(json.dumps(wire))


@pytest.mark.parametrize("storage", ["jsonl", "memory"])
@pytest.mark.parametrize("corruption", ["run_id", "gap"])
def test_supported_versions_do_not_weaken_prefix_identity(tmp_path, storage, corruption):
    prefix = _wire_line()
    suffix = _wire_line(seq=2 if corruption == "gap" else 1,
                        run_id="foreign" if corruption == "run_id" else "run")
    if storage == "jsonl":
        path = tmp_path / "invalid-mixed.jsonl"
        path.write_bytes(prefix + suffix)
        with pytest.raises(ValueError):
            JsonlRunLog.resume(path)
        assert path.read_bytes() == prefix + suffix
        _assert_lock_available(path)
    else:
        records = tuple(log_module.decode_record(line) for line in (prefix, suffix))
        with pytest.raises(ValueError):
            MemoryRunLog._from_records(records)


def test_supported_versions_preserve_mixed_append_rollback(tmp_path, monkeypatch):
    path = tmp_path / "rollback-mixed.jsonl"
    prefix = _wire_line()
    path.write_bytes(prefix)
    with JsonlRunLog.resume(path) as log:
        log.append(NodeStartedRecord("second"))
        failure = OSError("fsync failed")

        def fail_sync(_descriptor):
            raise failure

        with monkeypatch.context() as patch:
            patch.setattr(log_module.os, "fsync", fail_sync)
            _assert_append_failure_is_transactional(
                log, path, lambda: log.append(NodeStartedRecord("third"))
            )
        assert path.read_bytes().startswith(prefix)
        assert [item.schema for item in log] == [SCHEMA_VERSION, SCHEMA_VERSION]
        assert log.append(NodeStartedRecord("third")).seq == 2


class _FileProxy:
    """Delegate a binary file while exposing narrow failure/counting hooks."""

    def __init__(
        self,
        wrapped: Any,
        *,
        on_read: Callable[[], None] | None = None,
        on_write: Callable[[bytes], int] | None = None,
        on_flush: Callable[[], None] | None = None,
    ) -> None:
        self._wrapped = wrapped
        self._on_read = on_read
        self._on_write = on_write
        self._on_flush = on_flush

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)

    def read(self, *args: Any, **kwargs: Any) -> bytes:
        if self._on_read is not None:
            self._on_read()
        return self._wrapped.read(*args, **kwargs)

    def write(self, data: bytes) -> int:
        if self._on_write is not None:
            return self._on_write(data)
        return self._wrapped.write(data)

    def flush(self) -> None:
        if self._on_flush is not None:
            self._on_flush()
            return
        self._wrapped.flush()


def _wire_line(
    *,
    run_id: str = "run",
    seq: int = 0,
    schema: int = SCHEMA_VERSION,
    node_id: str = "work",
) -> bytes:
    """Return one complete JSONL record with independently editable identity."""
    return (
        json.dumps(
            {
                "schema": schema,
                "run_id": run_id,
                "seq": seq,
                "kind": "node_started",
                "payload": {"node_id": node_id},
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _assert_lock_available(path: Path) -> None:
    """Prove no failed opener retained the path's exclusive advisory lock."""
    fd = os.open(path, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _baseline_log(path: Path) -> JsonlRunLog:
    """Create a durable log containing one complete authoritative record."""
    log = JsonlRunLog.create(path, "run")
    log.append(NodeStartedRecord(node_id="first"))
    return log


def _assert_append_failure_is_transactional(
    log: JsonlRunLog,
    path: Path,
    operation: Callable[[], None],
) -> None:
    """Assert a host failure leaves disk and cached authority at the old prefix."""
    prior_bytes = path.read_bytes()
    prior_records = tuple(log)
    prior_sequence = log.next_sequence

    with pytest.raises((OSError, ValueError)):
        operation()

    assert path.read_bytes() == prior_bytes
    assert tuple(log) == prior_records
    assert log.next_sequence == prior_sequence


def test_memory_log_orders_records_and_retains_a_canonical_copy():
    seed = {"items": [{"value": 1}]}
    first = NodeExpandedRecord(
        spawner_id="each",
        seed=seed,
        prune_ids=(),
        topology_digest=_DIGEST,
    )
    log = MemoryRunLog("memory-run")

    first_envelope = log.append(first)
    second_envelope = log.append(NodeStartedRecord(node_id="next"))
    seed["items"][0]["value"] = 99
    seed["items"].append({"value": 2})

    assert log.path is None
    assert first_envelope.seq == 0
    assert second_envelope.seq == 1
    assert log.next_sequence == 2
    assert tuple(log) == log.records() == (first_envelope, second_envelope)
    assert first_envelope.record.seed == {"items": [{"value": 1}]}

    log.close()
    log.close()
    assert tuple(log) == (first_envelope, second_envelope)
    with pytest.raises(RuntimeError, match="closed"):
        log.append(NodeStartedRecord(node_id="late"))


def test_memory_log_reopens_a_detached_prefix_at_the_next_sequence():
    original = MemoryRunLog("retained")
    original.append(NodeExpandedRecord("each", {"items": [{"value": 1}]}, (), _DIGEST))
    original.close()
    prefix = original.records()
    reopened = MemoryRunLog._from_records(prefix)
    try:
        prefix[0].record.seed["items"][0]["value"] = 2
        exported = reopened.records()
        exported[0].record.seed["items"].append({"value": 3})
        assert tuple(reopened)[0].record.seed == {"items": [{"value": 1}]}
        assert original.records()[0].record.seed == {"items": [{"value": 1}]}
        assert exported[0] is not prefix[0]
        accepted = reopened.append(NodeStartedRecord("next"))
        assert accepted.seq == 1 and reopened.next_sequence == 2
        assert len(original.records()) == 1
        with pytest.raises(RuntimeError, match="closed"):
            original.append(NodeStartedRecord("late"))
    finally:
        reopened.close()


@pytest.mark.parametrize("change", ["empty", "list", "tuple_subclass", "gap", "duplicate", "foreign"])
def test_memory_log_reopening_rejects_invalid_prefix_coordinates(change):
    records = (RecordEnvelope(SCHEMA_VERSION, "run", 0, NodeStartedRecord("first")),)
    if change == "empty":
        records = ()
    elif change == "list":
        records = list(records)
    elif change == "tuple_subclass":
        class OtherTuple(tuple):
            pass

        records = OtherTuple(records)
    elif change == "gap":
        records = (RecordEnvelope(SCHEMA_VERSION, "run", 1, NodeStartedRecord("first")),)
    elif change == "duplicate":
        records += records
    else:
        records += (RecordEnvelope(SCHEMA_VERSION, "other", 1, NodeStartedRecord("second")),)
    with pytest.raises(ValueError):
        MemoryRunLog._from_records(records)


@pytest.mark.parametrize("field, value", [
    ("schema", 6), ("schema", 8), ("schema", 10), ("schema", float(SCHEMA_VERSION)), ("schema", True),
    ("run_id", ""), ("run_id", 1), ("run_id", "\ud800"),
    ("seq", -1), ("seq", 0.0), ("seq", False),
    ("record", NodeStartedRecord(1)), ("envelope", None),
])
def test_memory_log_reopening_revalidates_exact_envelope_authority(field, value):
    envelope = RecordEnvelope(SCHEMA_VERSION, "run", 0, NodeStartedRecord("first"))
    if field == "envelope":
        class OtherEnvelope(RecordEnvelope):
            pass

        envelope = OtherEnvelope(SCHEMA_VERSION, "run", 0, NodeStartedRecord("first"))
    else:
        object.__setattr__(envelope, field, value)

    with pytest.raises((TypeError, ValueError)):
        MemoryRunLog._from_records((envelope,))


def test_memory_log_never_exposes_its_retained_mutable_payload():
    log = MemoryRunLog("memory-run")
    appended = log.append(
        NodeExpandedRecord(
            spawner_id="each",
            seed={"items": [{"value": 1}]},
            prune_ids=(),
            topology_digest=_DIGEST,
        )
    )

    appended.record.seed["items"].append({"from": "append"})
    iterated = next(iter(log))
    iterated.record.seed["items"][0]["value"] = 99
    snapshotted = log.records()[0]
    snapshotted.record.seed["items"].append({"from": "records"})

    assert next(iter(log)).record.seed == {"items": [{"value": 1}]}
    assert log.records()[0].record.seed == {"items": [{"value": 1}]}


@pytest.mark.parametrize("run_id", ["", 7, None])
def test_memory_log_rejects_invalid_run_ids(run_id):
    with pytest.raises((TypeError, ValueError)):
        MemoryRunLog(run_id)


def test_jsonl_create_is_exclusive_and_preserves_an_existing_file(tmp_path: Path):
    path = tmp_path / "run.jsonl"
    path.write_bytes(b"already here\n")

    with pytest.raises(FileExistsError):
        JsonlRunLog.create(path, "run")

    assert path.read_bytes() == b"already here\n"


def test_jsonl_create_sets_owner_read_write_permissions(tmp_path: Path):
    path = tmp_path / "run.jsonl"

    with JsonlRunLog.create(path, "run") as log:
        assert log.path == path

    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_jsonl_resume_locks_before_its_first_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "run.jsonl"
    with JsonlRunLog.create(path, "run") as created:
        created.append(NodeStartedRecord(node_id="work"))

    events: list[str] = []
    real_flock = fcntl.flock
    real_fdopen = os.fdopen

    def tracking_flock(fd: int, operation: int) -> None:
        if operation & fcntl.LOCK_EX:
            events.append("lock")
        real_flock(fd, operation)

    def tracking_fdopen(*args: Any, **kwargs: Any) -> _FileProxy:
        wrapped = real_fdopen(*args, **kwargs)
        return _FileProxy(wrapped, on_read=lambda: events.append("read"))

    monkeypatch.setattr(log_module.fcntl, "flock", tracking_flock)
    monkeypatch.setattr(log_module.os, "fdopen", tracking_fdopen)

    with JsonlRunLog.resume(path):
        pass

    assert events.index("lock") < events.index("read")


def test_jsonl_excludes_a_second_writer(tmp_path: Path):
    path = tmp_path / "run.jsonl"
    first = _baseline_log(path)
    try:
        with pytest.raises((BlockingIOError, OSError)):
            JsonlRunLog.resume(path)
    finally:
        first.close()

    with JsonlRunLog.resume(path) as reopened:
        assert reopened.run_id == "run"


def test_jsonl_excludes_a_writer_in_a_separate_process(tmp_path: Path):
    path = tmp_path / "run.jsonl"
    first = _baseline_log(path)
    child = """
import errno
from pathlib import Path
import sys
from agent_composer.runtime.log import JsonlRunLog

try:
    log = JsonlRunLog.resume(Path(sys.argv[1]))
except OSError as exc:
    if exc.errno not in {errno.EACCES, errno.EAGAIN, errno.EWOULDBLOCK}:
        raise
    raise SystemExit(0)
else:
    log.close()
    raise SystemExit("child unexpectedly acquired the writer lock")
"""
    try:
        completed = subprocess.run(
            [sys.executable, "-c", child, str(path)],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    finally:
        first.close()

    assert completed.returncode == 0, completed.stderr


def test_jsonl_append_uses_one_complete_write_flush_and_fsync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "run.jsonl"
    log = JsonlRunLog.create(path, "run")
    writes: list[bytes] = []
    flushes = 0
    fsyncs = 0
    wrapped = log._file
    real_fsync = os.fsync

    def write_once(data: bytes) -> int:
        writes.append(data)
        return wrapped.write(data)

    def flush_once() -> None:
        nonlocal flushes
        flushes += 1
        wrapped.flush()

    def fsync_once(fd: int) -> None:
        nonlocal fsyncs
        fsyncs += 1
        real_fsync(fd)

    log._file = _FileProxy(wrapped, on_write=write_once, on_flush=flush_once)
    monkeypatch.setattr(log_module.os, "fsync", fsync_once)
    try:
        envelope = log.append(NodeStartedRecord(node_id="work"))
    finally:
        log.close()

    assert envelope.seq == 0
    assert len(writes) == 1
    assert writes[0].endswith(b"\n")
    assert writes[0].count(b"\n") == 1
    assert flushes == 1
    assert fsyncs == 1


def test_jsonl_append_and_cache_are_isolated_from_caller_mutation(tmp_path: Path):
    path = tmp_path / "run.jsonl"
    seed = {"items": [1]}
    log = JsonlRunLog.create(path, "run")
    try:
        envelope = log.append(
            NodeExpandedRecord(
                spawner_id="each",
                seed=seed,
                prune_ids=(),
                topology_digest=_DIGEST,
            )
        )
        seed["items"].append(2)
        assert envelope.record.seed == {"items": [1]}
        assert tuple(log)[0].record.seed == {"items": [1]}
    finally:
        log.close()

    with JsonlRunLog.resume(path) as resumed:
        assert tuple(resumed)[0].record.seed == {"items": [1]}


def test_jsonl_never_exposes_its_retained_mutable_payload(tmp_path: Path):
    path = tmp_path / "run.jsonl"
    log = JsonlRunLog.create(path, "run")
    appended = log.append(
        NodeExpandedRecord(
            spawner_id="each",
            seed={"items": [{"value": 1}]},
            prune_ids=(),
            topology_digest=_DIGEST,
        )
    )
    durable_prefix = path.read_bytes()

    appended.record.seed["items"].append({"from": "append"})
    iterated = next(iter(log))
    iterated.record.seed["items"][0]["value"] = 99
    snapshotted = log.records()[0]
    snapshotted.record.seed["items"].append({"from": "records"})

    assert next(iter(log)).record.seed == {"items": [{"value": 1}]}
    assert log.records()[0].record.seed == {"items": [{"value": 1}]}
    assert path.read_bytes() == durable_prefix
    log.close()

    with JsonlRunLog.resume(path) as resumed:
        assert next(iter(resumed)).record.seed == {"items": [{"value": 1}]}
    assert path.read_bytes() == durable_prefix


@pytest.mark.parametrize(
    "tail",
    [
        b'{"partial":',
        encode_record(
            RecordEnvelope(
                schema=SCHEMA_VERSION,
                run_id="run",
                seq=1,
                record=NodeStartedRecord(node_id="valid-but-incomplete"),
            )
        ),
    ],
    ids=["partial-json", "valid-json-without-newline"],
)
def test_jsonl_resume_truncates_and_syncs_every_incomplete_tail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tail: bytes,
):
    path = tmp_path / "run.jsonl"
    with _baseline_log(path):
        pass
    complete_prefix = path.read_bytes()
    with path.open("ab") as stream:
        stream.write(tail)

    flushes = 0
    fsyncs = 0
    real_fdopen = os.fdopen
    real_fsync = os.fsync

    def tracking_fdopen(*args: Any, **kwargs: Any) -> _FileProxy:
        nonlocal flushes
        wrapped = real_fdopen(*args, **kwargs)

        def track_flush() -> None:
            nonlocal flushes
            flushes += 1
            wrapped.flush()

        return _FileProxy(wrapped, on_flush=track_flush)

    def tracking_fsync(fd: int) -> None:
        nonlocal fsyncs
        fsyncs += 1
        real_fsync(fd)

    monkeypatch.setattr(log_module.os, "fdopen", tracking_fdopen)
    monkeypatch.setattr(log_module.os, "fsync", tracking_fsync)

    with JsonlRunLog.resume(path) as resumed:
        assert [item.record.node_id for item in resumed] == ["first"]

    assert path.read_bytes() == complete_prefix
    assert flushes == 1
    assert fsyncs == 1


def test_jsonl_resume_rejects_a_corrupt_complete_line_and_releases_lock(
    tmp_path: Path,
):
    path = tmp_path / "run.jsonl"
    path.write_bytes(_wire_line() + b"{not-json}\n")

    with pytest.raises(ValueError, match="JSON"):
        JsonlRunLog.resume(path)

    _assert_lock_available(path)


@pytest.mark.parametrize(
    "contents",
    [
        _wire_line(schema=1),
        _wire_line(schema=2),
        _wire_line(seq=1),
        _wire_line(run_id="first", seq=0)
        + _wire_line(run_id="second", seq=1, node_id="other"),
    ],
    ids=["schema-1", "schema-2", "sequence", "run-id"],
)
def test_jsonl_resume_rejects_identity_mismatches_and_releases_lock(
    tmp_path: Path, contents: bytes
):
    path = tmp_path / "run.jsonl"
    path.write_bytes(contents)

    with pytest.raises(ValueError):
        JsonlRunLog.resume(path)

    _assert_lock_available(path)


def test_jsonl_resume_rejects_an_empty_file_and_releases_lock(tmp_path: Path):
    path = tmp_path / "empty.jsonl"
    path.touch()

    with pytest.raises(ValueError, match="empty"):
        JsonlRunLog.resume(path)

    _assert_lock_available(path)


def test_invalid_append_retains_the_prior_complete_prefix(tmp_path: Path):
    path = tmp_path / "run.jsonl"
    log = _baseline_log(path)
    try:
        _assert_append_failure_is_transactional(
            log,
            path,
            lambda: log.append(NodeStartedRecord(node_id=7)),  # type: ignore[arg-type]
        )
    finally:
        log.close()


def test_write_failure_rolls_back_to_the_prior_complete_prefix(tmp_path: Path):
    path = tmp_path / "run.jsonl"
    log = _baseline_log(path)
    wrapped = log._file

    def write_then_fail(data: bytes) -> int:
        wrapped.write(data[: max(1, len(data) // 2)])
        raise OSError("write failed")

    log._file = _FileProxy(wrapped, on_write=write_then_fail)
    try:
        _assert_append_failure_is_transactional(
            log, path, lambda: log.append(NodeStartedRecord(node_id="second"))
        )
    finally:
        log.close()


def test_flush_failure_rolls_back_to_the_prior_complete_prefix(tmp_path: Path):
    path = tmp_path / "run.jsonl"
    log = _baseline_log(path)
    wrapped = log._file

    def flush_then_fail() -> None:
        wrapped.flush()
        raise OSError("flush failed")

    log._file = _FileProxy(wrapped, on_flush=flush_then_fail)
    try:
        _assert_append_failure_is_transactional(
            log, path, lambda: log.append(NodeStartedRecord(node_id="second"))
        )
    finally:
        log.close()


def test_fsync_failure_rolls_back_and_syncs_the_prior_complete_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "run.jsonl"
    log = _baseline_log(path)
    real_fsync = os.fsync
    calls = 0

    def fail_once(fd: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("fsync failed")
        real_fsync(fd)

    monkeypatch.setattr(log_module.os, "fsync", fail_once)
    try:
        _assert_append_failure_is_transactional(
            log, path, lambda: log.append(NodeStartedRecord(node_id="second"))
        )
    finally:
        log.close()

    assert calls == 2


def test_short_write_is_an_append_failure_and_rolls_back(tmp_path: Path):
    path = tmp_path / "run.jsonl"
    log = _baseline_log(path)
    wrapped = log._file

    def short_write(data: bytes) -> int:
        return wrapped.write(data[:-1])

    log._file = _FileProxy(wrapped, on_write=short_write)
    try:
        _assert_append_failure_is_transactional(
            log, path, lambda: log.append(NodeStartedRecord(node_id="second"))
        )
    finally:
        log.close()


def test_tail_repair_failure_releases_the_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "run.jsonl"
    path.write_bytes(_wire_line() + b"partial")

    def fail_fsync(fd: int) -> None:
        raise OSError("repair fsync failed")

    monkeypatch.setattr(log_module.os, "fsync", fail_fsync)
    with pytest.raises(OSError, match="repair fsync"):
        JsonlRunLog.resume(path)

    _assert_lock_available(path)


def test_jsonl_create_releases_lock_when_file_adoption_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "run.jsonl"

    def fail_fdopen(*args: Any, **kwargs: Any) -> Any:
        raise OSError("fdopen failed")

    monkeypatch.setattr(log_module.os, "fdopen", fail_fdopen)
    with pytest.raises(OSError, match="fdopen failed"):
        JsonlRunLog.create(path, "run")

    _assert_lock_available(path)


def test_jsonl_close_is_idempotent_keeps_cache_and_releases_lock(tmp_path: Path):
    path = tmp_path / "run.jsonl"
    log = _baseline_log(path)
    records = tuple(log)

    log.close()
    log.close()

    assert tuple(log) == records
    with pytest.raises(RuntimeError, match="closed"):
        log.append(NodeStartedRecord(node_id="late"))
    with JsonlRunLog.resume(path) as reopened:
        assert tuple(reopened) == records


def test_jsonl_context_releases_lock_when_the_body_raises(tmp_path: Path):
    path = tmp_path / "run.jsonl"

    with pytest.raises(RuntimeError, match="body failed"):
        with JsonlRunLog.create(path, "run") as log:
            log.append(NodeStartedRecord(node_id="work"))
            raise RuntimeError("body failed")

    with JsonlRunLog.resume(path) as reopened:
        assert reopened.next_sequence == 1
