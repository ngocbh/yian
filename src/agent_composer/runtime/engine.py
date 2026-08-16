"""Append-only flow execution over one prepared run-state/log authority.

``FlowEngine`` is the sole ready-frontier consumer and state writer.  Serial mode
evaluates durably started nodes inline; pooled mode lets workers evaluate only and
returns raw outcomes to the single dispatcher.  Every yielded observer follows its
record append and reducer application.  Stream chunks are transient callback-only
messages and never enter the authoritative log.
"""

from __future__ import annotations

import queue
import threading
import traceback as _tb
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from agent_composer.compile.model import GrowthRejected
from agent_composer.diagnostics import Failure, FailureCode, OriginRef
from agent_composer.events import RunFailed, StreamChunk
from agent_composer.nodes.base import Grow, Output, Route, _HostStop
from agent_composer.runtime.eval_node import EvaluationFailure, eval_node
from agent_composer.runtime.log import RunLog
from agent_composer.runtime.state import (
    GrowBuilder,
    PreparedTransition,
    RunState,
    _NodeDataError,
    commit_prepared,
    prepare_expanded,
    prepare_node_failed,
    prepare_node_routed,
    prepare_node_started,
    prepare_node_succeeded,
)

_POLL = 0.02


@dataclass(frozen=True, slots=True)
class _RecordedStream:
    """One transient stream message transported from a worker to the dispatcher."""

    node_id: str
    chunk: StreamChunk


@dataclass(frozen=True, slots=True)
class _RecordedCompletion:
    """One completed raw evaluation accepted through the dispatcher message queue."""

    node_id: str
    future: Future


class FlowEngine:
    """Drive a prepared append-only run frontier without owning its log.

    The engine is the sole ready-queue consumer and state writer.  Workers only
    evaluate already-started nodes and transport raw outcomes or transient stream
    chunks through an in-process queue.

    ``run`` yields only observers returned after a record has been appended and its
    reducer plan applied.  The optional event callback sees the same observers first;
    stream chunks are callback-only and never durable or yielded.
    """

    def __init__(
        self,
        state: RunState,
        log: RunLog,
        *,
        grow_builder: GrowBuilder,
        num_workers: int,
        on_event: Callable[[Any], None] | None = None,
        on_stream: Callable[[StreamChunk], None] | None = None,
        stop_event: threading.Event | None = None,
    ) -> None:
        """Borrow a prepared state/log pair and configure one bounded drive."""
        if type(state) is not RunState:
            raise TypeError("FlowEngine expects a RunState")
        if not isinstance(log, RunLog):
            raise TypeError("FlowEngine expects a RunLog")
        if not callable(grow_builder):
            raise TypeError("grow_builder must be callable")
        if type(num_workers) is not int:
            raise TypeError("num_workers must be a non-negative integer")
        if num_workers < 0:
            raise ValueError("num_workers must be a non-negative integer")
        if on_event is not None and not callable(on_event):
            raise TypeError("on_event must be callable or None")
        if on_stream is not None and not callable(on_stream):
            raise TypeError("on_stream must be callable or None")
        self.state = state
        self.log = log
        self.grow_builder = grow_builder
        self.num_workers = num_workers
        self.on_event = on_event
        self.on_stream = on_stream
        self.stop_event = stop_event if stop_event is not None else threading.Event()

    def run(self) -> Iterator[Any]:
        """Drive newly ready work and yield only post-durable observer notifications."""
        self._validate_authority()
        if self.state.result is not None:
            return
        if self.state.in_flight:
            raise ValueError(
                "FlowEngine requires unfinished starts to be requeued "
                "with RunResumed before execution"
            )
        if self.num_workers == 0:
            yield from self._run_serial()
            return
        yield from self._run_pooled()

    def _validate_authority(self) -> None:
        """Reject mismatched identity, sequence, or worker authority before execution."""
        if type(self.state.run_id) is not str or not self.state.run_id:
            raise ValueError("FlowEngine requires an initialized RunState")
        if self.state.run_id != self.log.run_id:
            raise ValueError(
                "run-state/log run_id mismatch: "
                f"state {self.state.run_id!r}, log {self.log.run_id!r}"
            )
        if self.state.next_sequence != self.log.next_sequence:
            raise ValueError(
                "run-state/log sequence mismatch: "
                f"state {self.state.next_sequence}, log {self.log.next_sequence}"
            )
        if self.state.num_workers != self.num_workers:
            raise ValueError(
                "run-state/engine worker mismatch: "
                f"state {self.state.num_workers}, engine {self.num_workers}"
            )

    def _run_serial(self) -> Iterator[Any]:
        """Execute each durably started node inline on the caller's thread."""
        try:
            while self.state.result is None:
                self._raise_if_stopped()
                if not self.state.ready:
                    raise RuntimeError("FlowEngine reached nonterminal quiescence")
                node_id = self.state.ready[0]
                observers = self._commit_started(node_id)
                yield from self._publish(observers)
                self._raise_if_stopped()

                outcome = self._evaluate(
                    node_id,
                    emit_stream=self._emit_serial_stream,
                )
                self._raise_if_stopped()
                observers = self._commit_outcome(node_id, outcome)
                if isinstance(self.state.result, RunFailed):
                    self.stop_event.set()
                yield from self._publish(observers)
        except BaseException:
            self.stop_event.set()
            raise

    def _run_pooled(self) -> Iterator[Any]:
        """Execute at most ``num_workers`` starts with one state-writing dispatcher."""
        messages: queue.Queue[_RecordedStream | _RecordedCompletion] = queue.Queue()
        try:
            executor = ThreadPoolExecutor(
                max_workers=self.num_workers,
                thread_name_prefix="ac-runtime",
            )
        except BaseException:
            self.stop_event.set()
            raise
        active: dict[Future, str] = {}
        submitted: list[Future] = []
        shut_down = False
        try:
            while self.state.result is None:
                self._raise_if_stopped()
                while self.state.ready and len(active) < self.num_workers:
                    node_id = self.state.ready[0]
                    observers = self._commit_started(node_id)
                    yield from self._publish(observers)
                    self._raise_if_stopped()

                    # Submission is deliberately after the durable start and its observer.
                    # A submit failure therefore leaves one recoverable unfinished start.
                    future = executor.submit(self._evaluate_worker, node_id, messages)
                    active[future] = node_id
                    submitted.append(future)
                    future.add_done_callback(
                        lambda done, claimed=node_id: messages.put(
                            _RecordedCompletion(claimed, done)
                        )
                    )

                if not active:
                    if self.state.ready:
                        continue
                    raise RuntimeError("FlowEngine reached nonterminal quiescence")

                try:
                    message = messages.get(timeout=_POLL)
                except queue.Empty:
                    continue
                if isinstance(message, _RecordedStream):
                    if not self.stop_event.is_set() and self.on_stream is not None:
                        self.on_stream(message.chunk)
                    continue

                future = message.future
                claimed = active.pop(future, None)
                if claimed is None:
                    continue
                if claimed != message.node_id:
                    raise RuntimeError("recorded completion node identity mismatch")
                # Read the future before checking the shared event so an initiating host
                # exception is not replaced by a sibling's derived cancellation wakeup.
                outcome = future.result()
                self._raise_if_stopped()
                observers = self._commit_outcome(message.node_id, outcome)
                if isinstance(self.state.result, RunFailed):
                    self.stop_event.set()
                    self._shutdown_and_discard(executor, messages)
                    shut_down = True
                    yield from self._publish(observers)
                    return
                yield from self._publish(observers)
        except BaseException as exc:
            replacement = self._shutdown_after_error(
                executor,
                submitted,
                messages,
                exc,
            )
            shut_down = True
            if replacement is not exc:
                raise replacement
            raise
        finally:
            if not shut_down:
                executor.shutdown(wait=True, cancel_futures=False)

    def _commit_started(self, node_id: str) -> tuple[Any, ...]:
        """Append and apply the exact ready-head start transition."""
        prepared = prepare_node_started(self.state, node_id)
        return commit_prepared(self.log, self.state, prepared)

    def _commit_outcome(
        self,
        node_id: str,
        outcome: Output | Route | Grow | EvaluationFailure,
    ) -> tuple[Any, ...]:
        """Prepare node data, append its record, and apply the total reducer plan.

        Evaluator and dispatcher rejection paths both hand one canonical
        ``Failure`` to the schema-3 reducer boundary.
        """
        if isinstance(outcome, EvaluationFailure):
            failure = outcome.failure
            prepared = self._prepare_failure(node_id, failure)
            return commit_prepared(self.log, self.state, prepared)

        if not isinstance(outcome, (Output, Route, Grow)):
            raise TypeError(
                "raw evaluator returned an unsupported outcome "
                f"{type(outcome).__name__}"
            )
        if isinstance(outcome, Output):
            try:
                prepared = prepare_node_succeeded(
                    self.state,
                    node_id,
                    outcome.value,
                    commit_as=outcome.commit_as,
                )
            except _NodeDataError as wrapped:
                prepared = self._prepare_contract_failure(
                    node_id,
                    "contract.output",
                    wrapped.original,
                    origin=wrapped.origin,
                    traceback_text=self._format_exception(wrapped.original),
                )
        elif isinstance(outcome, Route):
            try:
                prepared = prepare_node_routed(
                    self.state,
                    node_id,
                    outcome.handle,
                )
            except _NodeDataError as wrapped:
                prepared = self._prepare_contract_failure(
                    node_id,
                    "contract.route",
                    wrapped.original,
                    origin=wrapped.origin,
                    traceback_text=self._format_exception(wrapped.original),
                )
        else:
            try:
                prepared = prepare_expanded(
                    self.state,
                    node_id,
                    outcome,
                    self.grow_builder,
                )
            except GrowthRejected as wrapped:
                prepared = self._prepare_contract_failure(
                    node_id,
                    "contract.growth",
                    wrapped.original,
                    traceback_text=self._format_exception(wrapped.original),
                )
        return commit_prepared(self.log, self.state, prepared)

    def _prepare_failure(
        self,
        node_id: str,
        failure: Failure,
    ) -> PreparedTransition:
        """Prepare one canonical failure without flattening its semantic fields."""
        if failure.node_id != node_id:
            raise RuntimeError(
                "raw evaluation failure node identity mismatch: "
                f"expected {node_id!r}, found {failure.node_id!r}"
            )
        return prepare_node_failed(self.state, failure)

    def _prepare_contract_failure(
        self,
        node_id: str,
        code: FailureCode,
        exception: Exception,
        *,
        origin: OriginRef | None = None,
        traceback_text: str | None = None,
    ) -> PreparedTransition:
        """Construct and project one expected dispatcher contract rejection."""
        node = self.state.flow.nodes[node_id]
        failure = Failure(
            code=code,
            message=str(exception),
            node_id=node_id,
            trace=node.failure_trace(origin),
            cause_type=type(exception).__name__,
            traceback=(
                traceback_text if traceback_text is not None else _tb.format_exc()
            ),
        )
        return self._prepare_failure(node_id, failure)

    @staticmethod
    def _format_exception(exception: Exception) -> str:
        """Format a marker's original traceback without the private wrapper tail."""
        return "".join(
            _tb.format_exception(
                type(exception),
                exception,
                exception.__traceback__,
            )
        )

    def _evaluate(
        self,
        node_id: str,
        *,
        emit_stream: Callable[[StreamChunk], None] | None,
    ) -> Output | Route | Grow | EvaluationFailure:
        """Read and execute one already-started node without mutating run state."""
        node = self.state.flow.nodes[node_id]
        return eval_node(
            node,
            self.state.flow,
            self.state.pool,
            wiring_override=self.state.wiring_overrides.get(node_id),
            stop_event=self.stop_event,
            emit_stream=emit_stream,
        )

    def _evaluate_worker(
        self,
        node_id: str,
        messages: queue.Queue,
    ) -> Output | Route | Grow | EvaluationFailure:
        """Evaluate one claimed node and enqueue only its transient stream chunks."""
        return self._evaluate(
            node_id,
            emit_stream=lambda chunk: messages.put(_RecordedStream(node_id, chunk)),
        )

    def _emit_serial_stream(self, chunk: StreamChunk) -> None:
        """Invoke the transient serial stream callback unless stop has begun."""
        if not self.stop_event.is_set() and self.on_stream is not None:
            self.on_stream(chunk)

    def _publish(self, observers: tuple[Any, ...]) -> Iterator[Any]:
        """Invoke the event callback and then yield each post-durable observer."""
        for observer in observers:
            if self.on_event is not None:
                self.on_event(observer)
            yield observer

    def _raise_if_stopped(self) -> None:
        """Translate a set cooperative event into private host-control flow."""
        if self.stop_event.is_set():
            raise _HostStop(primary=False)

    def _shutdown_and_discard(
        self,
        executor: ThreadPoolExecutor,
        messages: queue.Queue,
    ) -> None:
        """Fully join submitted work and discard every queued late message."""
        try:
            executor.shutdown(wait=True, cancel_futures=False)
        finally:
            while True:
                try:
                    messages.get_nowait()
                except queue.Empty:
                    break

    def _shutdown_after_error(
        self,
        executor: ThreadPoolExecutor,
        submitted: list[Future],
        messages: queue.Queue,
        initiating: BaseException,
    ) -> BaseException:
        """Signal, fully join, discard messages, and preserve the initiating error."""
        cleanup_errors: list[BaseException] = []
        try:
            self.stop_event.set()
        except BaseException as cleanup_error:
            cleanup_errors.append(cleanup_error)
        try:
            self._shutdown_and_discard(executor, messages)
        except BaseException as cleanup_error:
            cleanup_errors.append(cleanup_error)
        if isinstance(initiating, _HostStop) and not initiating.primary:
            for future in submitted:
                if not future.done() or future.cancelled():
                    continue
                try:
                    candidate = future.exception()
                except BaseException as cleanup_error:
                    cleanup_errors.append(cleanup_error)
                    continue
                if isinstance(candidate, _HostStop) and candidate.primary:
                    self._note_cleanup_errors(candidate, cleanup_errors)
                    return candidate
        self._note_cleanup_errors(initiating, cleanup_errors)
        return initiating

    @staticmethod
    def _note_cleanup_errors(
        initiating: BaseException,
        cleanup_errors: list[BaseException],
    ) -> None:
        """Annotate cleanup faults without changing the exception that escapes."""
        for cleanup_error in cleanup_errors:
            try:
                initiating.add_note(
                    "dispatcher cleanup also raised "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
            except BaseException:
                # Notes are diagnostics only; even a hostile exception implementation
                # cannot replace the exact control/callback error being propagated.
                pass
