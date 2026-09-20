"""Caller-lane callbacks and supervised lifetime for one isolated SDK worker."""

from __future__ import annotations

import os
import queue
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable

from .protocol import CodexWorkerError, Endpoint, detached


def _secondary(primary: BaseException, cleanup: BaseException) -> None:
    if primary is not cleanup and hasattr(primary, "add_note"):
        try:
            primary.add_note(f"Codex cleanup also failed ({type(cleanup).__name__}).")
        except BaseException:
            pass


class CodexWorkerClient:
    """Use the pinned SDK without moving Composer callbacks into its process.

    One caller runs operations at a time. The monitor may stop the SDK while a
    caller callback is active, but the operation cannot return before that
    callback returns. Quiet inference has no deadline and is never retried.
    """

    def __init__(
        self,
        *,
        config: dict | None = None,
        callback: Callable[[str, dict | None], dict] | None = None,
        on_event: Callable[[str, dict], None] | None = None,
        on_result: Callable[[str, dict], None] | None = None,
        check_stopped: Callable[[], None] | None = None,
        _launch_args_override: tuple[str, ...] | None = None,
        _worker_command: tuple[str, ...] | None = None,
    ) -> None:
        """Configure a lazily started, process-local SDK supervisor.

        Args:
            config (`dict | None`): Private SDK cwd/env/binary/config-overrides;
                defaults to SDK launch defaults, not an authored configuration API.
            callback (`Callable | None`): Parent-owned server-request handler on
                the caller lane. Absence rejects requests; it never grants approval.
            on_event (`Callable | None`): Optional caller-lane native notification
                observer receiving method and detached JSON payload.
            on_result (`Callable | None`): Optional caller-lane observer of actual
                operation responses, invoked before honoring a concurrent stop.
            check_stopped (`Callable | None`): Thread-safe cooperative stop check,
                called by the caller and monitor; defaults to no cancellation.
            _launch_args_override (`tuple[str, ...] | None`): Private offline
                app-server launch seam; defaults to the SDK's installed runtime.
            _worker_command (`tuple[str, ...] | None`): Private fake-worker seam;
                defaults to this package's executable worker module.

        Raises:
            CodexWorkerError: Configuration is not finite JSON or is malformed.
            TypeError: An optional callback is not callable.
        """
        self._config = detached({} if config is None else config)
        if set(self._config) - {"cwd", "env", "codex_bin", "config_overrides"}:
            raise CodexWorkerError("config")
        for function in (callback, on_event, on_result, check_stopped):
            if function is not None and not callable(function):
                raise TypeError("Codex callbacks must be callable or None")
        for command in (_launch_args_override, _worker_command):
            if command is not None and (
                type(command) is not tuple
                or not command
                or any(type(part) is not str or not part for part in command)
            ):
                raise CodexWorkerError("config")
        self._callback = callback
        self._on_event = on_event
        self._on_result = on_result
        self._check = check_stopped or (lambda: None)
        self._launch_args = _launch_args_override
        self._worker_command = _worker_command or (
            sys.executable,
            "-u",
            "-m",
            "agent_composer.llm_clients.codex.worker",
        )
        self._process = None
        self._endpoint = None
        self._monitor_thread = None
        self._lock = threading.RLock()
        self._operation_lock = threading.Lock()
        self._owner = None
        self._idle = threading.Event()
        self._idle.set()
        self._close_requested = threading.Event()
        self._cleanup_done = threading.Event()
        self._closed = False
        self._primary: BaseException | None = None
        self._cleanup_error: BaseException | None = None
        self._command_id = 0
        self._callback_id = 0
        self._initialized = None
        self._thread_id = None
        self._turn_id = None

    def __enter__(self) -> CodexWorkerClient:
        """Enter this still-open client without eagerly launching a worker."""
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        """Close owned processes without replacing an initiating body failure."""
        try:
            self.close()
        except BaseException as cleanup:
            if exc is None:
                raise
            _secondary(exc, cleanup)

    def _ensure_open(self) -> None:
        if self._primary is not None:
            raise self._primary
        if self._closed or self._close_requested.is_set():
            raise CodexWorkerError("closed")

    def _latch(self, error: BaseException) -> None:
        with self._lock:
            if self._primary is None:
                self._primary = error
            elif error is not self._primary:
                _secondary(self._primary, error)

    def _start(self) -> None:
        if self._process is not None:
            return
        if os.name != "posix" or any(
            not callable(getattr(os, name, None))
            for name in ("killpg", "setsid", "getpgrp")
        ):
            raise CodexWorkerError("platform")
        try:
            # Preserve the loaded implementation and dependency search path even
            # when the caller changes cwd or supplied a relative PYTHONPATH.
            environment = os.environ.copy()
            paths = [str(Path(__file__).resolve().parents[3])]
            paths.extend(str(Path(path).resolve()) for path in sys.path if path)
            environment["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(paths))
            self._process = subprocess.Popen(
                self._worker_command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
                close_fds=True,
                start_new_session=True,
                env=environment,
            )
            self._endpoint = Endpoint(
                self._process.stdout.fileno(), self._process.stdin.fileno()
            )
            monitor = threading.Thread(target=self._monitor, daemon=True)
            monitor.start()
            self._monitor_thread = monitor
        except BaseException as error:
            if self._process is not None:
                self._terminate_group()
            if isinstance(error, OSError):
                raise CodexWorkerError("transport") from None
            raise

    def _send_control(self, kind: str) -> None:
        try:
            self._endpoint.send(kind, 0, {})
        except BaseException:
            pass

    def _monitor(self) -> None:
        stage = None
        deadline = 0.0
        try:
            while True:
                if not self._close_requested.is_set() and self._primary is None:
                    try:
                        self._check()
                    except BaseException as error:
                        self._latch(error)
                exited = self._process.poll() is not None
                broken = self._endpoint.fault is not None
                if exited:
                    break
                now = time.monotonic()
                if stage is None and (
                    self._primary is not None
                    or broken
                    or self._close_requested.is_set()
                ):
                    if self._close_requested.is_set() and self._primary is None:
                        self._send_control("shutdown")
                        stage, deadline = "shutdown", now + 3.5
                    else:
                        self._send_control("interrupt")
                        stage, deadline = "interrupt", now + 0.2
                elif stage == "interrupt" and now >= deadline:
                    self._send_control("shutdown")
                    stage, deadline = "shutdown", now + 3.5
                elif stage == "shutdown" and now >= deadline:
                    break
                time.sleep(0.02)
        except BaseException:
            self._cleanup_error = CodexWorkerError("shutdown")
        finally:
            try:
                self._terminate_group()
            except BaseException:
                self._cleanup_error = CodexWorkerError("shutdown")
            finally:
                self._cleanup_done.set()

    def _terminate_group(self) -> None:
        process = self._process
        if process is None:
            return
        # The leader can exit first; remaining members still own this group.
        for signum in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(process.pid, signum)
            except ProcessLookupError:
                pass
            if signum == signal.SIGTERM:
                try:
                    process.wait(timeout=0.2)
                except subprocess.TimeoutExpired:
                    pass
        process.wait(timeout=1)
        if self._endpoint is not None:
            self._endpoint.stop()
        for stream in (process.stdin, process.stdout):
            if stream is not None:
                stream.close()

    def _event(self, payload: dict) -> None:
        if (
            set(payload) != {"method", "params"}
            or type(payload["method"]) is not str
            or not payload["method"]
            or type(payload["params"]) is not dict
        ):
            raise CodexWorkerError("protocol")
        if self._on_event is not None:
            self._on_event(payload["method"], payload["params"])

    def _callback_request(self, payload: dict) -> None:
        if (
            set(payload) != {"callback_id", "method", "params"}
            or type(payload["callback_id"]) is not int
            or payload["callback_id"] != self._callback_id + 1
            or type(payload["method"]) is not str
            or not payload["method"]
            or (payload["params"] is not None and type(payload["params"]) is not dict)
        ):
            raise CodexWorkerError("protocol")
        self._callback_id = payload["callback_id"]

    def _host_callback(
        self, payload: dict, identity: int, native_calls: set[str] | None = None
    ) -> None:
        with self._lock:
            if (
                self._primary is not None
                or self._close_requested.is_set()
                or self._endpoint.fault is not None
                or self._process.poll() is not None
                or self._cleanup_done.is_set()
            ):
                return
            if payload["method"] == "item/tool/call":
                params = payload["params"]
                if (
                    native_calls is None
                    or self._turn_id is None
                    or type(params) is not dict
                    or any(
                        type(params.get(key)) is not str or not params[key]
                        for key in ("threadId", "turnId", "callId")
                    )
                    or params["threadId"] != self._thread_id
                    or params["turnId"] != self._turn_id
                    or params["callId"] in native_calls
                ):
                    raise CodexWorkerError("protocol")
                native_calls.add(params["callId"])
            if self._callback is None:
                raise CodexWorkerError("callback")
        # The callback and all result preparation remain on the calling lane.
        result = detached(self._callback(payload["method"], payload["params"]))
        if self._primary is None and not self._cleanup_done.is_set():
            self._endpoint.send(
                "callback_result",
                identity,
                {
                    "callback_id": payload["callback_id"],
                    "result": result,
                },
            )

    def _result(self, operation: dict, value: dict) -> None:
        """Check native response identity before publishing it to the caller."""
        if operation["op"] in {"start_thread", "resume_thread"}:
            thread = value.get("thread")
            if (
                type(thread) is not dict
                or type(thread.get("id")) is not str
                or not thread["id"]
            ):
                raise CodexWorkerError("protocol")
            if (
                operation["op"] == "resume_thread"
                and thread["id"] != operation["thread_id"]
            ):
                raise CodexWorkerError("protocol")
        elif operation["op"] == "run_turn":
            turn = value.get("turn")
            if (
                value.get("threadId") != operation["thread_id"]
                or type(turn) is not dict
                or self._turn_id is None
                or turn.get("id") != self._turn_id
                or type(turn.get("status")) is not str
                or turn.get("status") not in {"completed", "failed", "interrupted"}
            ):
                raise CodexWorkerError("protocol")

    def _call(self, payload: dict) -> dict:
        payload = detached(payload)
        if not self._operation_lock.acquire(blocking=False):
            raise CodexWorkerError("busy")
        try:
            with self._lock:
                self._ensure_open()
                self._owner = threading.get_ident()
                self._idle.clear()
        except BaseException:
            self._operation_lock.release()
            raise
        try:
            self._check()
            self._start()
            self._command_id += 1
            identity = self._command_id
            if payload["op"] == "run_turn":
                self._turn_id = None
            self._endpoint.send("command", identity, payload)
            result = None
            terminal_fault = None
            pending_callback = None
            native_calls: set[str] = set()
            turn_closed = False
            while True:
                try:
                    frame = self._endpoint.receive()
                except queue.Empty:
                    if self._cleanup_done.is_set():
                        break
                    continue
                except CodexWorkerError as error:
                    self._latch(terminal_fault or error)
                    break
                kind, frame_id, data = frame["kind"], frame["id"], frame["payload"]
                try:
                    if kind == "event" and frame_id in (0, identity):
                        if (
                            data.get("method") == "turn/completed"
                            and payload["op"] == "run_turn"
                        ):
                            params = data.get("params")
                            if (
                                type(params) is not dict
                                or params.get("threadId") != self._thread_id
                                or type(params.get("turn")) is not dict
                                or self._turn_id is None
                                or params["turn"].get("id") != self._turn_id
                            ):
                                raise CodexWorkerError("protocol")
                            turn_closed = True
                        self._event(data)
                    elif kind == "callback" and frame_id == identity:
                        if pending_callback is not None:
                            raise CodexWorkerError("protocol")
                        self._callback_request(data)
                        if data["method"] == "item/tool/call":
                            if (
                                payload["op"] != "run_turn"
                                or turn_closed
                                or result is not None
                            ):
                                raise CodexWorkerError("protocol")
                            if self._turn_id is None:
                                # The SDK reader can request a tool before its
                                # runner publishes the turn-start response.
                                pending_callback = data
                            else:
                                self._host_callback(data, identity, native_calls)
                        else:
                            self._host_callback(data, identity)
                    elif kind == "turn" and frame_id == identity:
                        if (
                            payload["op"] != "run_turn"
                            or self._turn_id is not None
                            or set(data) != {"threadId", "turnId"}
                            or data["threadId"] != self._thread_id
                            or type(data["turnId"]) is not str
                            or not data["turnId"]
                        ):
                            raise CodexWorkerError("protocol")
                        self._turn_id = data["turnId"]
                        if pending_callback is not None:
                            staged, pending_callback = pending_callback, None
                            self._host_callback(staged, identity, native_calls)
                    elif kind == "result" and frame_id == identity:
                        if (
                            result is not None
                            or set(data) != {"value"}
                            or type(data["value"]) is not dict
                        ):
                            raise CodexWorkerError("protocol")
                        self._result(payload, data["value"])
                        result = data["value"]
                        if self._on_result is not None:
                            self._on_result(payload["op"], detached(result))
                        if self._primary is None:
                            self._check()
                        with self._lock:
                            if terminal_fault is None:
                                if self._endpoint.fault is not None:
                                    terminal_fault = self._endpoint.fault
                                elif self._close_requested.is_set() or self._closed:
                                    terminal_fault = CodexWorkerError("closed")
                                elif (
                                    self._process.poll() is not None
                                    or self._cleanup_done.is_set()
                                ):
                                    terminal_fault = CodexWorkerError("transport")
                            if self._primary is None and terminal_fault is None:
                                return result
                        # Drain later observed evidence before promoting this
                        # fault; a caller publication failure remains primary.
                    elif kind == "fault" and frame_id in (0, identity):
                        if set(data) != {"code"} or type(data["code"]) is not str:
                            raise CodexWorkerError("protocol")
                        self._latch(CodexWorkerError(data["code"]))
                    elif kind == "closed" and frame_id == 0:
                        if self._primary is None:
                            self._latch(terminal_fault or CodexWorkerError("transport"))
                    else:
                        raise CodexWorkerError("protocol")
                except BaseException as error:
                    self._latch(error)
            if self._primary is None:
                self._latch(terminal_fault or CodexWorkerError("transport"))
            raise self._primary
        except BaseException as error:
            self._latch(error)
            if self._monitor_thread is not None:
                self._cleanup_done.wait()
            if self._cleanup_error is not None:
                _secondary(self._primary, self._cleanup_error)
            raise self._primary
        finally:
            with self._lock:
                self._owner = None
                self._idle.set()
            self._operation_lock.release()

    def initialize(self) -> dict:
        """Initialize once and return detached native initialization metadata.

        Raises:
            CodexWorkerError: Platform, dependency, worker, or SDK initialization fails.
            BaseException: The parent's stop or publication callback fails.
        """
        self._ensure_open()
        if self._initialized is None:
            self._initialized = self._call(
                {
                    "op": "initialize",
                    "config": self._config,
                    "launch_args": (
                        None if self._launch_args is None else list(self._launch_args)
                    ),
                }
            )
        return detached(self._initialized)

    def start_thread(self, params: dict | None = None) -> dict:
        """Start a native thread with finite SDK parameters (empty by default).

        Args:
            params (`dict | None`): Native thread-start fields, defaulting to `{}`.

        Returns:
            `dict`: Detached native thread-start response.

        Raises:
            CodexWorkerError: Parameters or the SDK operation are invalid.
        """
        params = detached({} if params is None else params)
        self.initialize()
        result = self._call({"op": "start_thread", "params": params})
        self._thread_id = result["thread"]["id"]
        return result

    def resume_thread(self, thread_id: str, params: dict | None = None) -> dict:
        """Resume the named thread without allowing raw parameters to replace it.

        Args:
            thread_id (`str`): Exact nonempty retained native thread ID.
            params (`dict | None`): Native resume fields; defaults to `{}` and
                cannot contain `threadId` or `thread_id`.

        Returns:
            `dict`: Detached native thread-resume response.

        Raises:
            CodexWorkerError: Identity, parameters, or the SDK operation are invalid.
        """
        params = detached({} if params is None else params)
        if (
            type(thread_id) is not str
            or not thread_id
            or {"threadId", "thread_id"} & set(params)
        ):
            raise CodexWorkerError("protocol")
        self.initialize()
        result = self._call(
            {"op": "resume_thread", "thread_id": thread_id, "params": params}
        )
        self._thread_id = result["thread"]["id"]
        return result

    def run_turn(self, thread_id: str, input_items, params: dict | None = None) -> dict:
        """Publish native events and return terminal disposition, including failure.

        Args:
            thread_id (`str`): The thread started or resumed by this client.
            input_items (`str | dict | list`): Finite native SDK input.
            params (`dict | None`): Native turn fields, defaulting to `{}`.

        Returns:
            `dict`: Actual `turn/completed` payload; native failed status is data.

        Raises:
            CodexWorkerError: Identity, IPC, or SDK operation fails.
            BaseException: The original parent callback or stop failure.
        """
        params = detached({} if params is None else params)
        if (
            type(thread_id) is not str
            or not thread_id
            or thread_id != self._thread_id
            or type(input_items) not in (str, dict, list)
        ):
            raise CodexWorkerError("protocol")
        return self._call(
            {
                "op": "run_turn",
                "thread_id": thread_id,
                "input": input_items,
                "params": params,
            }
        )

    def close(self) -> None:
        """Stop owned processes and drain an active parent operation before return.

        Repeated closes are inert. An active callback cannot recursively close its
        own client; a different thread may close but waits for callback completion.

        Raises:
            CodexWorkerError: Reentrant close or process cleanup fails.
        """
        with self._lock:
            if self._closed:
                return
            if self._owner == threading.get_ident():
                raise CodexWorkerError("busy")
            self._close_requested.set()
        self._idle.wait()
        if self._monitor_thread is not None:
            self._cleanup_done.wait()
        with self._lock:
            self._closed = True
        if self._cleanup_error is not None:
            raise self._cleanup_error
