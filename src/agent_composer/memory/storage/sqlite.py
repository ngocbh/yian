"""Private single-owner SQLite storage for detached memory-backend state."""

from __future__ import annotations

import json
import os
import sqlite3
import stat
from threading import Lock

from agent_composer.memory.contracts import (
    JsonObject,
    MemoryOperationError,
    _freeze_json_object,
    _require_string,
    _thaw_json_object,
)

_VERSION = 2
_SCHEMA = (
    "CREATE TABLE memory_state ("
    "id INTEGER PRIMARY KEY CHECK (id = 1), envelope TEXT NOT NULL)"
)


def _error(reason: str, message: str) -> MemoryOperationError:
    return MemoryOperationError(
        "storage", f"storage.{reason}", message, reason == "busy"
    )


def _io_error(error: OSError | sqlite3.Error) -> MemoryOperationError:
    code = getattr(error, "sqlite_errorcode", 0) & 0xFF
    # Python 3.10 does not expose SQLite's numeric error attributes.
    busy = code in (5, 6) or (
        isinstance(error, sqlite3.OperationalError)
        and str(error) in ("database is locked", "database table is locked")
    )
    if busy:
        return _error("busy", "Memory storage is already owned or busy.")
    if code in (11, 26):
        return _error("corrupt", "Memory storage is not a valid SQLite database.")
    return _error("io", "Memory storage I/O failed.")


def _object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _encode(backend: str, state: JsonObject | None) -> str:
    envelope = _thaw_json_object(
        _freeze_json_object(
            {"version": _VERSION, "backend": backend, "state": state},
            "storage envelope",
        )
    )
    return json.dumps(
        envelope,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _decode(raw: object, backend: str) -> JsonObject | None:
    try:
        if type(raw) is not str:
            raise ValueError("envelope must be text")
        envelope = _thaw_json_object(
            _freeze_json_object(
                json.loads(raw, object_pairs_hook=_object_pairs),
                "storage envelope",
            )
        )
        if set(envelope) != {"version", "backend", "state"}:
            raise ValueError("invalid envelope fields")
        if type(envelope["version"]) is not int or envelope["version"] != _VERSION:
            raise _error("version", "Memory storage version is unsupported.")
        _require_string(envelope["backend"], "backend")
        if envelope["backend"] != backend:
            raise _error("kind", "Memory storage belongs to a different backend kind.")
        state = envelope["state"]
        if state is None:
            return None
        return _thaw_json_object(_freeze_json_object(state, "state"))
    except (TypeError, ValueError, RecursionError):
        raise _error(
            "corrupt", "Memory storage contains an invalid state envelope."
        ) from None


def check_storage_path(path: str | os.PathLike[str]) -> str:
    """Return an absolute local file path, rejecting non-file targets without I/O.

    Args:
        path (`str | PathLike[str]`): Candidate storage file.

    Returns:
        `str`: The path joined to the current directory if relative, without
        normalizing ``..``. Symlinks are resolved later, when the file is opened.

    Raises:
        `MemoryOperationError`: Empty paths, SQLite URIs, in-memory targets,
            bytes, NUL characters and non-UTF-8 text (`storage.path`).
    """
    try:
        raw_path = os.fspath(path)
        if (
            type(raw_path) is not str
            or not raw_path
            or raw_path == ":memory:"
            or raw_path.lower().startswith("file:")
            or "\x00" in raw_path
        ):
            raise ValueError("invalid file path")
        raw_path.encode("utf-8")
        if os.path.isabs(raw_path):
            return raw_path
        # Collapsing ".." here would skip a symlink the OS follows on open.
        return os.path.join(os.getcwd(), raw_path)
    except (TypeError, ValueError):
        raise _error("path", "Memory storage requires a local file path.") from None


class SQLiteStorage:
    """Own one local SQLite file until explicitly closed.

    The file contains one backend kind and one versioned JSON state envelope.
    Methods serialize one connection; transactions retain the exclusive file
    lock across commits. A failed I/O operation poisons the handle until close.
    This private adapter does not own backend sessions, queues, or callbacks.
    """

    def __init__(
        self, path: str | os.PathLike[str], backend: str, *, create_parent: bool = False
    ) -> None:
        """Open or initialize a file and acquire nonblocking lifetime ownership.

        Args:
            path (`str | PathLike[str]`): Local file path. Its parent must exist
                unless `create_parent`. Empty paths, SQLite URIs, bytes, and
                in-memory targets are invalid.
            backend (`str`): Nonempty backend-kind discriminator for this file.
            create_parent (`bool`, *optional*, defaults to `False`): Create
                missing parent directories before opening.

        Raises:
            `MemoryOperationError`: Invalid settings, an occupied file, an
                incompatible or corrupt store, or an ordinary I/O failure.
        """
        try:
            self._backend = _require_string(backend, "backend")
        except (TypeError, ValueError):
            raise _error(
                "backend", "Memory storage requires a nonempty backend kind."
            ) from None
        checked = check_storage_path(path)
        try:
            self._path = os.path.realpath(checked)
        except OSError as error:
            raise _io_error(error) from None
        self._pid = os.getpid()
        self._lock = Lock()
        self._connection: sqlite3.Connection | None = None
        self._closed = False
        self._poisoned = False
        try:
            if create_parent:
                os.makedirs(os.path.dirname(self._path), exist_ok=True)
            try:
                descriptor = os.open(
                    self._path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600
                )
            except FileExistsError:
                pass
            else:
                try:
                    os.fchmod(descriptor, 0o600)
                except BaseException:
                    try:
                        os.close(descriptor)
                    except BaseException:
                        pass
                    raise
                os.close(descriptor)
            if not stat.S_ISREG(os.stat(self._path).st_mode):
                raise _error("path", "Memory storage requires a regular file.")
            self._connection = sqlite3.connect(
                self._path,
                timeout=0,
                isolation_level=None,
                check_same_thread=False,
            )
            connection = self._connection
            connection.execute("PRAGMA locking_mode = EXCLUSIVE")
            connection.execute("PRAGMA synchronous = EXTRA")
            if connection.execute("PRAGMA journal_mode").fetchone() != ("delete",):
                raise _error(
                    "schema", "Memory storage requires rollback-journal format."
                )
            connection.execute("BEGIN EXCLUSIVE")
            # A creator can die before SQLite initialization. Only genuinely empty
            # files may initialize; an unrelated empty SQLite database is not ours.
            if os.stat(self._path).st_size == 0:
                connection.execute(_SCHEMA)
                connection.execute(
                    "INSERT INTO memory_state (id, envelope) VALUES (1, ?)",
                    (_encode(self._backend, None),),
                )
            self._validate_schema(connection)
            self._read(connection)
            connection.commit()
        except BaseException as error:
            if self._connection is not None:
                self._rollback()
                try:
                    self._connection.close()
                except BaseException:
                    pass
            self._closed = True
            if isinstance(error, (OSError, sqlite3.Error)):
                raise _io_error(error) from None
            raise

    def _validate_schema(self, connection: sqlite3.Connection) -> None:
        rows = connection.execute(
            "SELECT type, name, tbl_name, sql FROM sqlite_master"
        ).fetchall()
        if rows != [("table", "memory_state", "memory_state", _SCHEMA)]:
            raise _error("schema", "Memory storage schema is unsupported.")
        if connection.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
            raise _error("corrupt", "Memory storage failed its integrity check.")

    def _read(self, connection: sqlite3.Connection) -> JsonObject | None:
        rows = connection.execute("SELECT id, envelope FROM memory_state").fetchall()
        if len(rows) != 1 or rows[0][0] != 1:
            raise _error(
                "corrupt", "Memory storage must contain exactly one state envelope."
            )
        return _decode(rows[0][1], self._backend)

    def _before_lock(self) -> None:
        if self._closed:
            raise _error("closed", "Memory storage is closed.")
        # An inherited Python lock may be held by a vanished parent thread.
        if os.getpid() != self._pid:
            raise _error("forked", "Memory storage cannot be used after fork.")

    def _open_connection(self) -> sqlite3.Connection:
        if self._closed:
            raise _error("closed", "Memory storage is closed.")
        if self._poisoned:
            raise _error(
                "poisoned",
                "Memory storage must be closed and reopened after an I/O failure.",
            )
        assert self._connection is not None
        return self._connection

    def ensure_open(self) -> None:
        """Reject closed, poisoned, or inherited handles before backend RAM reads.

        Raises:
            `MemoryOperationError`: The handle cannot safely be used.
        """
        self._before_lock()
        with self._lock:
            self._open_connection()

    def _rollback(self) -> None:
        try:
            if self._connection is not None:
                self._connection.rollback()
        except BaseException:
            pass

    def load(self) -> JsonObject | None:
        """Return a detached state, or None before the first save.

        Returns:
            `JsonObject | None`: The last committed backend state.

        Raises:
            `MemoryOperationError`: The handle, stored data, or I/O is invalid.
        """
        self._before_lock()
        with self._lock:
            connection = self._open_connection()
            try:
                return self._read(connection)
            except BaseException as error:
                self._poisoned = True
                self._rollback()
                if isinstance(error, (OSError, sqlite3.Error)):
                    raise _io_error(error) from None
                raise

    def save(self, state: JsonObject) -> None:
        """Validate and atomically commit a detached complete backend state.

        Args:
            state (`JsonObject`): Exact finite JSON object without live handles.

        Raises:
            `MemoryOperationError`: The candidate, handle, or I/O is invalid.
                Invalid candidates do not poison the handle; uncertain writes do.
        """
        self._before_lock()
        with self._lock:
            connection = self._open_connection()
            try:
                detached = _thaw_json_object(_freeze_json_object(state, "state"))
                encoded = _encode(self._backend, detached)
            except (TypeError, ValueError, RecursionError):
                raise _error(
                    "invalid_state", "Memory storage requires a finite JSON object."
                ) from None
            try:
                connection.execute("BEGIN EXCLUSIVE")
                connection.execute(
                    "UPDATE memory_state SET envelope = ? WHERE id = 1", (encoded,)
                )
                connection.commit()
            except BaseException as error:
                self._poisoned = True
                self._rollback()
                if isinstance(error, (OSError, sqlite3.Error)):
                    raise _io_error(error) from None
                raise

    def close(self) -> None:
        """Release file ownership; repeated closes perform no I/O.

        Raises:
            `MemoryOperationError`: An inherited handle or ordinary close failure.
        """
        if self._closed:
            return
        if os.getpid() != self._pid:
            raise _error("forked", "Memory storage cannot be used after fork.")
        with self._lock:
            if self._closed:
                return
            assert self._connection is not None
            try:
                self._connection.close()
            except BaseException as error:
                self._poisoned = True
                if isinstance(error, (OSError, sqlite3.Error)):
                    raise _io_error(error) from None
                raise
            self._closed = True
            self._connection = None
