"""Closed finite-JSON framing and bounded pipes for the private SDK worker."""

from __future__ import annotations

import json
import math
import os
import queue
import select
import threading
import time
from typing import Any, Iterator

MAX_FRAME_BYTES = 8 * 1024 * 1024
MAX_QUEUE = 32
WRITE_TIMEOUT = 2.0
SDK_VERSION = "0.155.1"
_KINDS = {
    "command",
    "result",
    "event",
    "callback",
    "callback_result",
    "turn",
    "interrupt",
    "shutdown",
    "fault",
    "closed",
}
_MESSAGES = {
    "protocol": "Invalid Codex worker protocol data.",
    "capacity": "Codex worker IPC capacity was exceeded.",
    "transport": "Codex worker communication failed.",
    "closed": "Codex worker client is closed.",
    "busy": "Codex worker client already has an active operation.",
    "sdk_missing": "Codex requires the optional openai-codex dependency.",
    "sdk_version": "Codex requires openai-codex==0.155.1.",
    "sdk": "The Codex SDK operation failed.",
    "config": "Invalid private Codex worker configuration.",
    "callback": "Codex requested an unsupported host callback.",
    "shutdown": "Codex worker cleanup failed.",
    "platform": "Codex requires POSIX process-group support.",
}


class CodexWorkerError(RuntimeError):
    """A sanitized failure of the private SDK boundary."""

    def __init__(self, code: str) -> None:
        self.code = code if code in _MESSAGES else "protocol"
        super().__init__(_MESSAGES[self.code])


def _json(value: Any, active: set[int], depth: int = 0) -> Any:
    if value is None or type(value) in (bool, int):
        return value
    if type(value) is str:
        value.encode("utf-8")
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("nonfinite")
        return value
    if type(value) not in (dict, list) or depth >= 100 or id(value) in active:
        raise ValueError("invalid JSON")
    active.add(id(value))
    try:
        if type(value) is list:
            return [_json(item, active, depth + 1) for item in value]
        if any(type(key) is not str for key in value):
            raise ValueError("invalid JSON key")
        return {
            _json(key, active): _json(item, active, depth + 1)
            for key, item in value.items()
        }
    finally:
        active.remove(id(value))


def detached(value: Any) -> dict:
    """Validate and detach one exact finite JSON object."""
    try:
        if type(value) is not dict:
            raise ValueError("object required")
        return _json(value, set())
    except (TypeError, ValueError, RecursionError):
        raise CodexWorkerError("protocol") from None


def _frame(value: Any) -> dict:
    value = detached(value)
    if (
        set(value) != {"v", "kind", "id", "payload"}
        or type(value["v"]) is not int
        or value["v"] != 1
        or type(value["id"]) is not int
        or value["id"] < 0
        or type(value["kind"]) is not str
        or value["kind"] not in _KINDS
        or type(value["payload"]) is not dict
    ):
        raise CodexWorkerError("protocol")
    return value


def encode(kind: str, identity: int, payload: dict) -> bytes:
    """Encode a complete validated frame, including its nesting overhead."""
    value = _frame({"v": 1, "kind": kind, "id": identity, "payload": payload})
    try:
        body = json.dumps(
            value, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError):
        raise CodexWorkerError("protocol") from None
    if len(body) > MAX_FRAME_BYTES:
        raise CodexWorkerError("capacity")
    return len(body).to_bytes(4, "big") + body


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate key")
        result[key] = value
    return result


class Decoder:
    """Incrementally decode bounded length-prefixed JSON frames."""

    def __init__(self) -> None:
        self._buffer = bytearray()

    def feed(self, chunk: bytes) -> Iterator[dict]:
        """Yield validated frames from one bounded read before any later fault."""
        self._buffer.extend(chunk)
        while len(self._buffer) >= 4:
            size = int.from_bytes(self._buffer[:4], "big")
            if not 0 < size <= MAX_FRAME_BYTES:
                raise CodexWorkerError("capacity")
            if len(self._buffer) < size + 4:
                break
            body = bytes(self._buffer[4 : size + 4])
            del self._buffer[: size + 4]
            try:
                value = json.loads(body, object_pairs_hook=_pairs)
            except (TypeError, ValueError, RecursionError):
                raise CodexWorkerError("protocol") from None
            yield _frame(value)

    def finish(self) -> None:
        """Reject a peer that closed halfway through a frame."""
        if self._buffer:
            raise CodexWorkerError("protocol")


class Endpoint:
    """Bound pipe I/O without running application callbacks on its threads.

    File descriptors remain owned by the caller. Nonblocking OS reads/writes and
    short select waits let stop join both pumps without buffered-close deadlocks.
    """

    def __init__(self, read_fd: int, write_fd: int) -> None:
        self._read_fd, self._write_fd = read_fd, write_fd
        os.set_blocking(read_fd, False)
        os.set_blocking(write_fd, False)
        self._incoming: queue.Queue = queue.Queue(MAX_QUEUE)
        self._outgoing: queue.Queue = queue.Queue(MAX_QUEUE)
        self._stopped = threading.Event()
        self._fault_lock = threading.Lock()
        self.fault: CodexWorkerError | None = None
        self.eof = threading.Event()
        self._threads = [
            threading.Thread(target=target, daemon=True)
            for target in (self._read, self._write)
        ]
        for thread in self._threads:
            thread.start()

    def _fail(self, error: BaseException) -> None:
        with self._fault_lock:
            if self.fault is None:
                self.fault = (
                    error
                    if isinstance(error, CodexWorkerError)
                    else CodexWorkerError("transport")
                )

    def send(self, kind: str, identity: int, payload: dict) -> None:
        """Queue a validated frame without unbounded producer blocking."""
        wire = encode(kind, identity, payload)
        if self.fault is not None:
            raise self.fault
        if self._stopped.is_set():
            raise CodexWorkerError("transport")
        try:
            self._outgoing.put_nowait((wire, time.monotonic() + WRITE_TIMEOUT))
        except queue.Full:
            raise CodexWorkerError("capacity") from None

    def receive(self, timeout: float = 0.05) -> dict:
        """Return queued evidence before reporting a later pipe failure."""
        try:
            return self._incoming.get(timeout=timeout)
        except queue.Empty:
            if self.fault is not None:
                raise self.fault
            raise

    def _read(self) -> None:
        decoder = Decoder()
        try:
            while not self._stopped.is_set():
                if not select.select([self._read_fd], [], [], 0.05)[0]:
                    continue
                chunk = os.read(self._read_fd, 65536)
                if not chunk:
                    self.eof.set()
                    decoder.finish()
                    raise CodexWorkerError("transport")
                for frame in decoder.feed(chunk):
                    while not self._stopped.is_set():
                        try:
                            self._incoming.put(frame, timeout=0.05)
                            break
                        except queue.Full:
                            continue
        except BaseException as error:
            self._fail(error)

    def _write(self) -> None:
        try:
            while not self._stopped.is_set():
                try:
                    wire, deadline = self._outgoing.get(timeout=0.05)
                except queue.Empty:
                    continue
                remaining = memoryview(wire)
                while remaining and not self._stopped.is_set():
                    if time.monotonic() >= deadline:
                        raise CodexWorkerError("transport")
                    if select.select([], [self._write_fd], [], 0.05)[1]:
                        try:
                            remaining = remaining[os.write(self._write_fd, remaining) :]
                        except BlockingIOError:
                            continue
                self._outgoing.task_done()
        except BaseException as error:
            self._fail(error)

    def flush(self, timeout: float = 1.0) -> None:
        """Wait a bounded interval for queued frames to reach the pipe."""
        deadline = time.monotonic() + timeout
        while self._outgoing.unfinished_tasks:
            if self.fault is not None:
                raise self.fault
            if time.monotonic() >= deadline:
                raise CodexWorkerError("transport")
            time.sleep(0.01)

    def stop(self) -> None:
        """Stop both nonblocking pumps with bounded joins."""
        self._stopped.set()
        for thread in self._threads:
            thread.join(0.5)
