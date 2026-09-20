"""Executable private worker using only the unchanged official CodexClient."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
import os
import queue
import signal
import sys
import threading
import time

from .protocol import CodexWorkerError, Endpoint, SDK_VERSION, detached


def _sdk():
    """Check the optional pinned distribution before constructing any provider."""
    try:
        installed = version("openai-codex")
    except PackageNotFoundError:
        raise CodexWorkerError("sdk_missing") from None
    if installed != SDK_VERSION:
        raise CodexWorkerError("sdk_version")
    try:
        from openai_codex.client import CodexClient, CodexConfig
    except ImportError:
        raise CodexWorkerError("sdk_missing") from None
    return CodexClient, CodexConfig


def configuration(value: dict) -> dict:
    """Validate the narrow private SDK launch configuration."""
    value = detached(value)
    if set(value) - {"cwd", "env", "codex_bin", "config_overrides"}:
        raise CodexWorkerError("config")
    for key in ("cwd", "codex_bin"):
        if key in value and (type(value[key]) is not str or not value[key]):
            raise CodexWorkerError("config")
    if "env" in value and (
        type(value["env"]) is not dict
        or any(type(item) is not str for item in value["env"].values())
    ):
        raise CodexWorkerError("config")
    if "config_overrides" in value:
        entries = value["config_overrides"]
        if type(entries) is not list or any(type(item) is not str for item in entries):
            raise CodexWorkerError("config")
        value["config_overrides"] = tuple(entries)
    return value


class _Worker:
    def __init__(self, endpoint: Endpoint) -> None:
        self.endpoint = endpoint
        self.client = None
        self.lock = threading.RLock()
        self.stopping = threading.Event()
        self.failed = threading.Event()
        self.closing = threading.Event()
        self.closed = threading.Event()
        self.close_ok = True
        self.operation = 0
        self.last_command = 0
        self.callback_sequence = 0
        self.pending_callback = None
        self.thread_id = None
        self.turn_id = None
        self.interrupt_started = False
        self.close_started_at = None

    def emit(self, kind: str, payload: dict, identity: int | None = None) -> None:
        with self.lock:
            self.endpoint.send(
                kind, self.operation if identity is None else identity, payload
            )

    def fail(self, error: BaseException) -> None:
        with self.lock:
            if self.failed.is_set():
                return
            self.failed.set()
            self.stopping.set()
            try:
                self.emit(
                    "fault",
                    {
                        "code": (
                            error.code if isinstance(error, CodexWorkerError) else "sdk"
                        )
                    },
                )
            except BaseException:
                pass

    def global_notifications(self) -> None:
        try:
            while not self.closing.is_set():
                self.notification(self.client.next_notification(), identity=0)
        except BaseException as error:
            if not self.closing.is_set():
                self.fail(error)

    def notification(
        self, notification, identity: int | None = None, *, current_turn: bool = False
    ) -> dict:
        payload = notification.payload
        if hasattr(payload, "model_dump"):
            params = payload.model_dump(mode="json", by_alias=True)
        else:
            params = payload.params
        params = detached(params)
        if current_turn:
            if (
                params.get("threadId") != self.thread_id
                or params.get("turnId", self.turn_id) != self.turn_id
            ):
                raise CodexWorkerError("protocol")
            if notification.method == "turn/completed" and (
                type(params.get("turn")) is not dict
                or params["turn"].get("id") != self.turn_id
            ):
                raise CodexWorkerError("protocol")
        self.emit("event", {"method": notification.method, "params": params}, identity)
        return params

    def callback(self, method: str, params: dict | None) -> dict:
        with self.lock:
            if (
                self.stopping.is_set()
                or not self.operation
                or self.pending_callback is not None
            ):
                raise CodexWorkerError("callback")
            self.callback_sequence += 1
            identity = self.callback_sequence
            response: queue.Queue = queue.Queue(1)
            self.pending_callback = (identity, response)
            self.emit(
                "callback",
                {"callback_id": identity, "method": method, "params": params},
            )
        try:
            while not self.closing.is_set():
                try:
                    return response.get(timeout=0.05)
                except queue.Empty:
                    continue
            raise CodexWorkerError("callback")
        finally:
            with self.lock:
                self.pending_callback = None

    def run(self, identity: int, payload: dict) -> None:
        try:
            operation = payload["op"]
            if self.stopping.is_set():
                raise CodexWorkerError("transport")
            if operation == "initialize":
                if self.client is not None or set(payload) != {
                    "op",
                    "config",
                    "launch_args",
                }:
                    raise CodexWorkerError("protocol")
                Client, Config = _sdk()
                config = configuration(payload["config"])
                launch = payload["launch_args"]
                if launch is not None:
                    if (
                        type(launch) is not list
                        or not launch
                        or any(type(item) is not str for item in launch)
                    ):
                        raise CodexWorkerError("config")
                    config["launch_args_override"] = tuple(launch)
                self.client = Client(Config(**config), approval_handler=self.callback)
                self.client.start()
                threading.Thread(target=self.global_notifications, daemon=True).start()
                if self.stopping.is_set():
                    raise CodexWorkerError("transport")
                result = self.client.initialize().model_dump(mode="json", by_alias=True)
            elif self.client is None:
                raise CodexWorkerError("protocol")
            elif operation == "start_thread":
                if set(payload) != {"op", "params"}:
                    raise CodexWorkerError("protocol")
                result = self.client.thread_start(payload["params"]).model_dump(
                    mode="json", by_alias=True
                )
                self.thread_id = result["thread"]["id"]
            elif operation == "resume_thread":
                if set(payload) != {"op", "thread_id", "params"} or any(
                    key in payload["params"] for key in ("threadId", "thread_id")
                ):
                    raise CodexWorkerError("protocol")
                result = self.client.thread_resume(
                    payload["thread_id"], payload["params"]
                ).model_dump(mode="json", by_alias=True)
                if result["thread"]["id"] != payload["thread_id"]:
                    raise CodexWorkerError("protocol")
                self.thread_id = result["thread"]["id"]
            elif operation == "run_turn":
                if (
                    set(payload) != {"op", "thread_id", "input", "params"}
                    or payload["thread_id"] != self.thread_id
                ):
                    raise CodexWorkerError("protocol")
                result = self.run_turn(payload)
            else:
                raise CodexWorkerError("protocol")
            with self.lock:
                self.operation = 0
                self.emit("result", {"value": result}, identity)
        except BaseException as error:
            self.fail(error)
        finally:
            with self.lock:
                if self.operation == identity:
                    self.operation = 0

    def run_turn(self, payload: dict) -> dict:
        started = self.client.turn_start(
            self.thread_id, payload["input"], payload["params"]
        )
        self.turn_id = started.turn.id
        self.emit("turn", {"threadId": self.thread_id, "turnId": self.turn_id})
        try:
            while True:
                event = self.client.next_turn_notification(self.turn_id)
                params = self.notification(event, current_turn=True)
                if event.method == "turn/completed":
                    return params
        finally:
            self.client.unregister_turn_notifications(self.turn_id)
            self.turn_id = None

    def interrupt(self) -> None:
        self.stopping.set()
        with self.lock:
            if (
                self.failed.is_set()
                or self.interrupt_started
                or self.client is None
                or self.turn_id is None
            ):
                return
            self.interrupt_started = True
            thread_id, turn_id = self.thread_id, self.turn_id

        def invoke():
            try:
                if self.failed.is_set() or self.closing.is_set():
                    return
                self.client.turn_interrupt(thread_id, turn_id)
            except BaseException:
                pass

        threading.Thread(target=invoke, daemon=True).start()

    def close(self) -> None:
        if self.closing.is_set():
            return
        self.stopping.set()
        self.closing.set()
        self.close_started_at = time.monotonic()

        def invoke():
            try:
                if self.client is not None:
                    self.client.close()
            except BaseException:
                self.close_ok = False
            finally:
                self.closed.set()

        threading.Thread(target=invoke, daemon=True).start()

    def accept(self, frame: dict) -> None:
        kind, identity, payload = frame["kind"], frame["id"], frame["payload"]
        if kind == "command":
            with self.lock:
                if (
                    self.stopping.is_set()
                    or self.operation
                    or identity != self.last_command + 1
                ):
                    raise CodexWorkerError("protocol")
                self.last_command = self.operation = identity
            threading.Thread(
                target=self.run, args=(identity, payload), daemon=True
            ).start()
        elif kind == "callback_result":
            with self.lock:
                pending = self.pending_callback
                if (
                    identity != self.operation
                    or pending is None
                    or set(payload) != {"callback_id", "result"}
                    or payload["callback_id"] != pending[0]
                ):
                    raise CodexWorkerError("protocol")
                pending[1].put_nowait(detached(payload["result"]))
        elif kind in {"interrupt", "shutdown"} and identity == 0 and not payload:
            self.interrupt() if kind == "interrupt" else self.close()
        else:
            raise CodexWorkerError("protocol")


def _kill_owned_group() -> None:
    # Never signal a caller's unrelated process group if invoked outside the wrapper.
    if os.getpgrp() == os.getpid():
        os.killpg(os.getpid(), signal.SIGKILL)


def main() -> None:
    """Run the private worker until shutdown or loss of its parent pipe."""
    endpoint = Endpoint(sys.stdin.fileno(), sys.stdout.fileno())
    worker = _Worker(endpoint)
    parent_gone = False
    try:
        while True:
            try:
                frame = endpoint.receive()
            except queue.Empty:
                pass
            except CodexWorkerError:
                parent_gone = True
                worker.close()
            else:
                try:
                    worker.accept(frame)
                except BaseException as error:
                    worker.fail(error)
            if worker.failed.is_set() and not worker.closing.is_set():
                worker.close()
            if worker.closed.is_set():
                if parent_gone:
                    _kill_owned_group()
                endpoint.send("closed", 0, {"ok": worker.close_ok})
                endpoint.flush()
                return
            if (
                worker.close_started_at is not None
                and time.monotonic() - worker.close_started_at > 3
            ):
                _kill_owned_group()
                return
    finally:
        endpoint.stop()


if __name__ == "__main__":
    main()
