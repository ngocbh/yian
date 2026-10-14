"""Main-thread terminal ownership for concurrent CLI controller workers.

Workers enqueue prompt and render requests into one FIFO. The process main thread
drains that FIFO with :meth:`TerminalBroker.serve`, so terminal libraries never run
on controller threads while synchronous tool calls can still wait for human input.
"""

from __future__ import annotations

from concurrent.futures import CancelledError, Future
from dataclasses import dataclass, field
from queue import Queue
from threading import Event, Lock, current_thread, main_thread
from typing import Any, Callable


__all__ = [
    "ControllerComplete",
    "PromptRequest",
    "RenderRequest",
    "TerminalBroker",
]


_PROPAGATE_EOF = object()
_STOP_POLL_SECONDS = 0.05


@dataclass(eq=False, slots=True)
class PromptRequest:
    """One worker's prompt payload and its single-assignment answer channel.

    The public metadata mirrors the run's ``ask_user`` handler. ``secret`` asks
    for a hidden answer; only the host sets it (for example, the CLI's API-key
    prompt), never a model's ``ask_user`` call. A prompt implementation may
    register a non-blocking cancellation callback while its terminal widget is
    active; cancellation invokes that callback from the cancelling thread so it
    can schedule a thread-safe widget exit.
    """

    question: str
    questions: list[dict[str, Any]] | None = None
    answer_schema: list[dict[str, Any]] | None = None
    title: str | None = None
    node_id: str | None = None
    options: list[str] | None = None
    secret: bool = False
    _future: Future[Any] = field(default_factory=Future, init=False, repr=False)
    _completed: Event = field(default_factory=Event, init=False, repr=False)
    _cancelled: Event = field(default_factory=Event, init=False, repr=False)
    _lock: Any = field(default_factory=Lock, init=False, repr=False)
    _cancel_callback: Callable[[], None] | None = field(
        default=None,
        init=False,
        repr=False,
    )

    @property
    def cancelled(self) -> bool:
        """Whether the broker cancelled this request before it completed."""
        return self._cancelled.is_set()

    def result(self) -> Any:
        """Block until the main thread answers, fails, or cancels this request."""
        return self._future.result()

    def set_cancel_callback(self, callback: Callable[[], None] | None) -> None:
        """Set or clear the active prompt's non-blocking cancellation callback.

        Registering after cancellation invokes ``callback`` immediately. Registering
        after an ordinary answer or failure is a no-op because the prompt is already
        terminal.
        """
        invoke_now = False
        with self._lock:
            if callback is None:
                self._cancel_callback = None
            elif self._cancelled.is_set():
                invoke_now = True
            elif not self._future.done():
                self._cancel_callback = callback
        if invoke_now:
            callback()

    def _resolve(self, value: Any) -> bool:
        """Publish one answer unless cancellation or another terminal state won."""
        with self._lock:
            if self._future.done():
                return False
            self._cancel_callback = None
            self._future.set_result(value)
            self._completed.set()
            return True

    def _fail(self, exception: BaseException) -> bool:
        """Publish one prompt failure unless another terminal state already won."""
        with self._lock:
            if self._future.done():
                return False
            self._cancel_callback = None
            self._future.set_exception(exception)
            self._completed.set()
            return True

    def _cancel(self) -> Callable[[], None] | None:
        """Cancel the waiter and return the active widget callback, if any."""
        with self._lock:
            if self._future.done():
                return None
            self._cancelled.set()
            self._future.cancel()
            self._completed.set()
            callback = self._cancel_callback
            self._cancel_callback = None
            return callback

    def _done(self) -> bool:
        """Return whether this request already has any terminal state."""
        return self._future.done()

    def _wait(self, timeout: float) -> bool:
        """Wait up to ``timeout`` seconds for any terminal request state."""
        return self._completed.wait(timeout)


@dataclass(frozen=True, slots=True)
class RenderRequest:
    """One deferred terminal rendering action."""

    action: Callable[[], None]


@dataclass(frozen=True, slots=True)
class ControllerComplete:
    """The exact FIFO sentinel carrying a controller's terminal outcome."""

    result: Any = None
    exception: BaseException | None = None


_TerminalMessage = PromptRequest | RenderRequest | ControllerComplete


class TerminalBroker:
    """Serialize terminal work onto the process main thread.

    ``ask_user`` is the synchronous worker-facing handler each run gets as its
    ``ask_user`` tool. ``render`` defers arbitrary terminal output.
    ``controller_complete`` closes message production with one exact sentinel, and
    ``serve`` owns invocation of both the prompt handler and render actions.
    """

    def __init__(
        self,
        prompt_handler: Callable[[PromptRequest], Any],
        *,
        eof_value: Any = _PROPAGATE_EOF,
        stop_event: Event | None = None,
    ) -> None:
        """Create a broker with prompt, EOF, and shared controller-stop policies."""
        self._prompt_handler = prompt_handler
        self._eof_value = eof_value
        self._stop_event = stop_event
        self._messages: Queue[_TerminalMessage] = Queue()
        self._lock = Lock()
        self._pending: set[PromptRequest] = set()
        self._cancelled = False
        self._completion_published = False

    def ask_user(
        self,
        *,
        question: str,
        questions: list[dict[str, Any]] | None = None,
        answer_schema: list[dict[str, Any]] | None = None,
        title: str | None = None,
        node_id: str | None = None,
        options: list[str] | None = None,
        secret: bool = False,
    ) -> Any:
        """Enqueue one prompt and block the calling worker for its terminal result."""
        if self._stop_event is not None and self._stop_event.is_set():
            self.cancel_all()
            raise CancelledError()

        request = PromptRequest(
            question=question,
            questions=questions,
            answer_schema=answer_schema,
            title=title,
            node_id=node_id,
            options=options,
            secret=secret,
        )
        with self._lock:
            if self._cancelled or self._completion_published:
                raise CancelledError()
            self._pending.add(request)
            self._messages.put(request)

        try:
            return self._wait_for_prompt(request)
        finally:
            with self._lock:
                self._pending.discard(request)

    def render(self, action: Callable[[], None]) -> None:
        """Enqueue a terminal action unless cancellation or completion closed input."""
        request = RenderRequest(action)
        with self._lock:
            if self._cancelled or self._completion_published:
                return
            self._messages.put(request)

    def controller_complete(
        self,
        *,
        result: Any = None,
        exception: BaseException | None = None,
    ) -> ControllerComplete:
        """Enqueue and return the sole controller-completion sentinel."""
        complete = ControllerComplete(result=result, exception=exception)
        with self._lock:
            if self._completion_published:
                raise RuntimeError("controller completion was already published")
            self._completion_published = True
            self._messages.put(complete)
        return complete

    def cancel_all(self) -> None:
        """Close prompt input and wake every queued or active prompt waiter."""
        with self._lock:
            self._cancelled = True
            pending = tuple(self._pending)

        # Cancel every Future before waking the active terminal widget. This guarantees
        # all worker waiters are released even if a host cancellation hook itself fails.
        callbacks = [
            callback
            for request in pending
            if (callback := request._cancel()) is not None
        ]
        for callback in callbacks:
            try:
                callback()
            except BaseException:
                # A hook is only a best-effort widget wakeup. It must never replace
                # the node, observer, or Ctrl-C cause that initiated cancellation,
                # nor prevent the remaining hooks from running.
                continue

    def serve(self) -> ControllerComplete:
        """Drain terminal work on the process main thread through completion."""
        if current_thread() is not main_thread():
            raise RuntimeError("terminal broker must be served on the process main thread")

        while True:
            message = self._messages.get()
            if isinstance(message, ControllerComplete):
                self.cancel_all()
                return message
            if isinstance(message, RenderRequest):
                if not self._is_cancelled():
                    try:
                        message.action()
                    except BaseException:
                        self.cancel_all()
                        raise
                continue
            if message._done():
                continue
            self._serve_prompt(message)

    def _is_cancelled(self) -> bool:
        """Read the broker-wide cancellation state under its lock."""
        with self._lock:
            return self._cancelled

    def _wait_for_prompt(self, request: PromptRequest) -> Any:
        """Wait for an answer while observing the engine's shared stop signal.

        The caller already occupies a controller worker, so it polls from that same
        thread instead of creating a broker-owned watcher. Detecting a stop closes the
        broker, cancels every waiter, and invokes the active main-thread prompt hook.
        """
        if self._stop_event is None:
            return request.result()

        while True:
            if self._stop_event.is_set():
                self.cancel_all()
                raise CancelledError()
            if request._wait(_STOP_POLL_SECONDS):
                # The stop may have arrived while the handler was producing an
                # answer. Recheck before returning so that late answer is discarded
                # even when completion and cancellation race within one poll.
                if self._stop_event.is_set():
                    self.cancel_all()
                    raise CancelledError()
                return request.result()

    def _serve_prompt(self, request: PromptRequest) -> None:
        """Invoke one prompt and publish exactly one terminal request state."""
        try:
            answer = self._prompt_handler(request)
        except EOFError as exception:
            if self._eof_value is _PROPAGATE_EOF:
                request._fail(exception)
                self.cancel_all()
            else:
                request._resolve(self._eof_value)
        except KeyboardInterrupt:
            self.cancel_all()
            raise
        except Exception as exception:
            request._fail(exception)
        else:
            request._resolve(answer)
