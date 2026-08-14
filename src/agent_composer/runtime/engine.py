"""FlowEngine — one engine, one knob (`num_workers`).

A producer/consumer engine with two drive modes behind a single `num_workers`
knob, sharing one set of state-mutation helpers (`_on_success`/`_on_pause`/
`_advance`/`_branch`/`_skip_edge`):

- `num_workers=0` (default) — the single-threaded inline drain: the caller's
  thread pops the ready queue, runs each node's generator inline, applies the
  consequences, and forwards events. This is the deterministic path (exact event
  ordering, no `event_q` hop); golden-locked in F0.
- `num_workers>=1` — a fixed worker pool with a single-writer dispatcher: N
  daemon workers pull ids off `ready_q`, run `eval_node`, and push events onto
  `event_q`; the dispatcher (`run()`'s generator) drains `event_q`, forwards each
  event, and applies the *same* mutation helpers. The dispatcher is the sole
  writer of graph/edge/pool state.

Both modes capture all of graphon's *correctness* (3-state edge join, exact-once
fan-in, outputs-before-successors, branch skip-flood). `resume()` drives under the
same `num_workers` mode the engine carries (serial or pooled), via the shared
`_drive_to_terminal`.

Load-bearing orderings (do not reorder):
- A node's outputs are written to the pool **before** any successor is scheduled.
- A successor is scheduled only when `disposition` (via `is_node_ready`) says it is
  ready — an edge-class-aware join: a diamond fires exactly once, a control edge
  hard-gates (veto) and a required data group co-skips; a `dead` head is
  skip-flooded by `_skip_edge`.
"""

import copy
import queue
import threading
import traceback as _tb
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Optional

from agent_composer.events import (
    RunAborted,
    RunFailed,
    RunPaused,
    RunResumed,
    RunStarted,
    RunSucceeded,
    SourceRange,
    SourceSpan,
    NodeExpanded,
    NodeFailed,
    NodeRouted,
    NodeSucceeded,
    PauseRequested,
    StreamChunk,
)
from agent_composer.compile.model import END_ID, START_ID, CompiledFlow, Edge, NodeState
from agent_composer.nodes.end import EndNode
from agent_composer.nodes.base import Grow, Output, Route, _HostStop
from agent_composer.runtime.eval_node import EvaluationFailure, _eval_node_raw, eval_node
from agent_composer.runtime.log import RunLog
from agent_composer.runtime.state import (
    RunState,
    commit_prepared,
    prepare_expanded,
    prepare_node_failed,
    prepare_node_routed,
    prepare_node_started,
    prepare_node_succeeded,
)
from agent_composer.runtime.state_manager import StateManager
from agent_composer.typesys import TypeCheckError
from agent_composer.suspension.expansions import GrowRecord
from agent_composer.typesys.pool import VariablePool

DEFAULT_HANDLE = "default"

# Runtime expansion bounds. Enforced in `_apply_grow` (the dispatcher mints
# every node); both raises funnel to RunFailed via the boundary wrap (clean status=="failed").
MAX_TOTAL_NODES = 10_000   # the load-bearing runtime-size bound (nested MAP breadth multiplies)
MAX_REF_DEPTH = 5          # defense-in-depth depth bound (the static call graph is acyclic+finite)

_POLL = 0.02  # queue poll timeout (s); keeps shutdown responsive


class _Aborted(Exception):
    """Internal signal raised out of the drain when a cooperative abort is requested."""


@dataclass(frozen=True)
class _WorkerHostStop:
    """Worker-to-dispatcher signal that host-stop control flow unwound one executor."""

    node_id: str
    exception: _HostStop


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


class _RecordedDispatcher:
    """Drive a prepared append-only run frontier without owning its log.

    This private migration seam coexists with the legacy public ``FlowEngine`` until
    the atomic runtime cutover.  The dispatcher is the sole ready-queue consumer and
    state writer.  Workers only evaluate already-started nodes and transport raw
    outcomes or transient stream chunks back through an in-process queue.

    ``run`` yields only observers returned after a record has been appended and its
    reducer plan applied.  The optional event callback sees the same observers first;
    stream chunks are callback-only and never durable or yielded.
    """

    def __init__(
        self,
        state: RunState,
        log: RunLog,
        *,
        grow_builder: Callable,
        num_workers: int,
        on_event: Callable[[Any], None] | None = None,
        on_stream: Callable[[StreamChunk], None] | None = None,
        stop_event: threading.Event | None = None,
    ) -> None:
        """Borrow a prepared state/log pair and configure one bounded drive."""
        if type(state) is not RunState:
            raise TypeError("recorded dispatcher expects a RunState")
        if not isinstance(log, RunLog):
            raise TypeError("recorded dispatcher expects a RunLog")
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
                "recorded dispatcher requires unfinished starts to be requeued "
                "with RunResumed before execution"
            )
        if self.num_workers == 0:
            yield from self._run_serial()
            return
        yield from self._run_pooled()

    def _validate_authority(self) -> None:
        """Reject mismatched identity, sequence, or worker authority before execution."""
        if type(self.state.run_id) is not str or not self.state.run_id:
            raise ValueError("recorded dispatcher requires an initialized RunState")
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
                "run-state/dispatcher worker mismatch: "
                f"state {self.state.num_workers}, dispatcher {self.num_workers}"
            )

    def _run_serial(self) -> Iterator[Any]:
        """Execute each durably started node inline on the caller's thread."""
        try:
            while self.state.result is None:
                self._raise_if_stopped()
                if not self.state.ready:
                    raise RuntimeError(
                        "recorded dispatcher reached nonterminal quiescence"
                    )
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
                thread_name_prefix="ac-recorded",
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
                    raise RuntimeError(
                        "recorded dispatcher reached nonterminal quiescence"
                    )

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
        """Prepare node data, append its record, and apply the total reducer plan."""
        if isinstance(outcome, EvaluationFailure):
            if outcome.node_id != node_id:
                raise RuntimeError(
                    "raw evaluation failure node identity mismatch: "
                    f"expected {node_id!r}, found {outcome.node_id!r}"
                )
            prepared = prepare_node_failed(
                self.state,
                node_id,
                outcome.error,
                error_type=outcome.error_type,
                locator=outcome.locator,
                range=outcome.range,
                traceback=outcome.traceback,
            )
            return commit_prepared(self.log, self.state, prepared)

        if not isinstance(outcome, (Output, Route, Grow)):
            raise TypeError(
                "raw evaluator returned an unsupported outcome "
                f"{type(outcome).__name__}"
            )
        try:
            if isinstance(outcome, Output):
                prepared = prepare_node_succeeded(
                    self.state,
                    node_id,
                    outcome.value,
                    commit_as=outcome.commit_as,
                )
            elif isinstance(outcome, Route):
                prepared = prepare_node_routed(
                    self.state,
                    node_id,
                    outcome.handle,
                )
            else:
                prepared = prepare_expanded(
                    self.state,
                    node_id,
                    outcome,
                    self.grow_builder,
                )
        except _HostStop:
            raise
        except Exception as exc:  # noqa: BLE001 - node-result validation boundary
            # Invalid output, route, seed, or built topology is a failure of this
            # already-started node.  Failure preparation/append itself stays outside
            # this catch so reducer/programming and durability errors remain exact.
            locator = getattr(exc, "locator", None)
            range_ = getattr(exc, "range", None)
            prepared = prepare_node_failed(
                self.state,
                node_id,
                str(exc),
                error_type=type(exc).__name__,
                locator=locator,
                range=range_,
                traceback=_tb.format_exc(),
            )
        return commit_prepared(self.log, self.state, prepared)

    def _evaluate(
        self,
        node_id: str,
        *,
        emit_stream: Callable[[StreamChunk], None] | None,
    ) -> Output | Route | Grow | EvaluationFailure:
        """Read and execute one already-started node without mutating run state."""
        node = self.state.flow.nodes[node_id]
        return _eval_node_raw(
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
        executor.shutdown(wait=True, cancel_futures=False)
        while True:
            try:
                messages.get_nowait()
            except queue.Empty:
                return

    def _shutdown_after_error(
        self,
        executor: ThreadPoolExecutor,
        submitted: list[Future],
        messages: queue.Queue,
        initiating: BaseException,
    ) -> BaseException:
        """Signal, fully join, discard messages, and preserve the initiating error."""
        self.stop_event.set()
        self._shutdown_and_discard(executor, messages)
        if isinstance(initiating, _HostStop) and not initiating.primary:
            for future in submitted:
                if not future.done() or future.cancelled():
                    continue
                candidate = future.exception()
                if isinstance(candidate, _HostStop) and candidate.primary:
                    return candidate
        return initiating


class NodeExecutionError(RuntimeError):
    """A node emitted NodeFailed and no error strategy recovered it (abort).

    `locator` is an optional `SourceSpan` pinning the failure to a YAML line — set at
    the typed write boundary (a value that fails its node's declared `output:` Type)
    so the CLI boxes the `output:` field rather than printing a plain message. `range` is
    the matching concrete `SourceRange` (the caret span), carried from the NodeFailed event
    so it survives the wrap to RunFailed.
    """

    def __init__(
        self,
        node_id: str,
        error: str,
        error_type: str = "",
        locator: Optional[SourceSpan] = None,
        traceback: Optional[str] = None,
        range: Optional["SourceRange"] = None,
    ) -> None:
        super().__init__(f"node {node_id!r} failed: {error}")
        self.node_id = node_id
        self.error = error
        self.error_type = error_type
        self.locator = locator
        self.range = range
        # Formatted Python traceback of the raising call, carried from the NodeFailed event
        # so run() can attach it to RunFailed (CLI surfaces it under `--engine-trace`).
        self.traceback = traceback


class FlowEngine:
    """
    The flow execution engine: drive a `CompiledFlow` to a terminal, emitting events.

    A producer/consumer engine with two drive modes behind one `num_workers` knob: the
    default single-threaded inline drain (deterministic event order) and a worker pool
    with a single-writer dispatcher. Both capture the same correctness (3-state edge
    join, exact-once fan-in, outputs-before-successors, branch skip-flood). Suspended
    runs can be captured with [`snapshot`][agent_composer.FlowEngine.snapshot] and
    rebuilt in a fresh process with [`restore`][agent_composer.FlowEngine.restore].

    Most callers should use [`run_flow`][agent_composer.run_flow] rather than driving the
    engine directly.

    Args:
        flow (`CompiledFlow`):
            The compiled graph to execute. The engine mutates it in place when a spawner
            node (CALL/MAP) grows the graph at run time.
        pool (`VariablePool`, *optional*, defaults to `None`):
            The variable pool to read/write. A fresh empty pool is created when `None`.
        num_workers (`int`, *optional*, defaults to `0`):
            `0` runs the deterministic inline drain on the caller's thread; `>=1` spawns
            that many daemon workers behind a single-writer dispatcher. Clamped to `>=0`.
        run_inputs (`dict`, *optional*, defaults to `None`):
            The flow's run arguments. START is scheduled as an ordinary node at run init; the
            engine primes `store[START_ID]` with these raw args (so START's `${input.X}` wiring
            resolves) and narrows START's wiring to the passed keys via a per-engine override
            (`self.wiring_overrides`, NOT the shared IR). `None` for direct-engine tests that
            hand-seed the START store entry (the compiled `${input.X}` wiring reads it). The
            flow's boundary (`${input}`) asserts ride START's ordinary `post_asserts` and fire
            through eval_node against START's coerced output — the engine takes no assert list.
        grow_builder (`Callable`, *optional*, defaults to `None`):
            Pure `(spawner, canonical_seed) -> Flow` constructor supplied by the run boundary.
            Required only when a node returns `Grow` or a checkpoint replays growth; static direct
            engines may omit it. Runtime treats the callable generically and never imports the
            compile-owned implementation.
        stop_event (`threading.Event`, *optional*, defaults to `None`):
            Host-owned cooperative execution stop injected only into nodes declaring
            `needs_stop_event`. A fresh private event is created when omitted. This transient
            process capability is not persisted in checkpoints.
    """

    def __init__(
        self,
        flow: CompiledFlow,
        pool: Optional[VariablePool] = None,
        *,
        num_workers: int = 0,
        run_inputs: Optional[dict] = None,
        grow_builder=None,
        stop_event=None,
    ) -> None:
        self.flow = flow
        self.pool = pool if pool is not None else VariablePool()
        self.sm = StateManager(flow)
        self.num_workers = max(0, num_workers)
        # The top-level START_ID is scheduled as an ordinary node at run init (`_seed_start_and_
        # advance`): the engine primes store[START_ID] with the raw run args + narrows START's
        # wiring to the passed keys (via `self.wiring_overrides`, not the shared IR), then enqueues
        # START so it runs through eval_node (emitting NodeStarted/NodeSucceeded, coercing, and
        # firing the flow's boundary asserts via its own post_asserts). `run_inputs` defaults to
        # None for direct-FlowEngine tests that hand-seed.
        self.run_inputs = run_inputs
        # Per-engine, per-run wiring overrides (`node_id -> wiring`) consulted by eval_node in
        # place of the shared compiled `flow.wiring[node_id]`. Populated for top-level START by
        # `_seed_start_and_advance` to narrow `${input.X}` to only the passed run args WITHOUT
        # mutating the shared IR (a ComposedFlow is reusable across runs). Empty otherwise.
        self.wiring_overrides: dict[str, dict] = {}
        # Pure expansion constructor injected from the package-root run boundary. None is valid
        # for a static direct engine; `_apply_grow` fails loudly before mutation if a dynamic node
        # is driven without the seam.
        self.grow_builder = grow_builder
        # Host-owned cooperative execution stop. This is intentionally distinct from the
        # pooled worker-lifecycle `_stop`: `stop_event` is injected into opted-in nodes and may
        # be set by the host or a primary node failure; `_stop` only tells idle executor threads
        # that one pooled drive has finished and they should exit.
        self.stop_event = stop_event if stop_event is not None else threading.Event()
        self.ready: deque[str] = deque()  # serial path; also the serial _ready_snapshot arm
        if self.num_workers >= 1:
            self.ready_q: "queue.Queue[str]" = queue.Queue()
            self.event_q: "queue.Queue" = queue.Queue()
            self._stop = threading.Event()
        self.paused: list[tuple[str, Any]] = []  # (node_id, PauseReason)
        self.deferred: list[str] = []  # became ready while suspending
        # Runtime graph-expansion bookkeeping: the commit redirect a cloned filler carries
        # (CALL child END_ID / MAP END_ID-list / loop-body END) now
        # lives on the node itself as `Node.commit_as` (baked by the `_grow_*` helpers), so
        # `_on_success` writes the filler's value under the spawner id + fires its out-edges; no
        # side dict. `depth` carries each cloned spawner-eligible id's expansion depth for
        # MAX_REF_DEPTH.
        self.depth: dict[str, int] = {}
        self.expansions: list = []  # durable ledger: top-level GrowRecords in grow order
            # (nested grows ride under their parent record's flat `children`; element/iteration
            # placement is re-derived on replay from each child's namespaced spawner_id).
        self._spawner_expansion: dict[str, Any] = {}  # cloned spawner_id -> the GrowRecord
            # that contains it. The SOLE lookup for "which record does this spawner belong to"
            # — no scan over `self.expansions` ever happens — and the branch key for
            # "nest under parent vs new top-level" in `_apply_grow`.
        # Loop ledger bookkeeping: each loop keeps exactly ONE live GrowRecord, keyed by its ORIGIN
        # id (the compiled loop `L`). `_apply_grow` attributes every iteration's grow to `origin`
        # (`flow.nodes[spawner_id].origin_id or spawner_id`) and drops the superseded iteration's
        # record from its container, so the durable ledger stays single-record + bounded. The
        # iteration index lives on the driver clone's `iteration` field + the record seed
        # `(carried, k)`, not the engine — there is no engine-side loop step or iteration counter.
        self._origin_record: dict[str, Any] = {}  # origin loop id -> its live GrowRecord
        self._cancel = False

    def request_abort(self) -> None:
        """Request a legacy abort and wake any cooperative blocking node."""
        self._cancel = True
        self.stop_event.set()

    # --- lifecycle ---------------------------------------------------------- #

    def run(self):
        """Drive the run, yielding run events. One terminal event at the end.

        `num_workers==0` is the inline drain (byte-identical event order, the
        deterministic path); `num_workers>=1` spawns the worker pool + dispatcher.
        """
        yield RunStarted()
        # Init: schedule START_ID as an ordinary node (seeded from run_inputs), then drive to a
        # terminal. START runs through eval_node — it emits NodeStarted/NodeSucceeded, coerces +
        # type-checks + defaults its inputs, and fires the flow's boundary asserts through its
        # ordinary `post_asserts` seam. An input-type / boundary-assert failure surfaces as an
        # ordinary NodeFailed on START -> RunFailed (no body node ran), preserving "no node ran".
        self._seed_start_and_advance()
        yield from self._drive_to_terminal()

    def _drive_to_terminal(self):
        """Drive an ALREADY-SEEDED ready frontier to a terminal, yielding every event
        incl. the terminal one. Picks serial vs pooled on `self.num_workers` — the SOLE
        drive block, shared by run() (after START seed) and resume() (after the resume
        seed). Both modes capture the same correctness; pooled reorders events but the
        result is worker-count-independent."""
        if self.num_workers == 0:
            try:
                yield from self._drain()
            except _Aborted:
                yield RunAborted(); return
            except _HostStop:
                # request_abort uses the host event only as its worker wakeup. The authored
                # public outcome remains the legacy RunAborted rather than private host control.
                if self._cancel:
                    yield RunAborted(); return
                raise
            except NodeExecutionError as exc:
                self.stop_event.set()
                yield RunFailed(error=exc.error, error_type=exc.error_type, locator=exc.locator,
                                traceback=exc.traceback, range=exc.range); return
            if self.paused:
                yield RunPaused(reasons=[reason for _, reason in self.paused]); return
            yield self._terminal_event(); return

        # Pooled: N daemon workers + single-writer dispatcher. Clear _stop first — a prior
        # run()/resume() pooled pass set it in its finally; a fresh pass must re-enable workers.
        self._stop.clear()
        workers = [
            threading.Thread(target=self._worker, name=f"ac-worker-{i}", daemon=True)
            for i in range(self.num_workers)
        ]
        host_stop: Optional[_HostStop] = None
        for w in workers:
            w.start()
        try:
            yield from self._dispatch()
            terminal = (
                RunPaused(reasons=[r for _, r in self.paused])
                if self.paused
                else self._terminal_event()  # co-skipped terminal -> RunFailed (shared helper)
            )
        except _Aborted:
            terminal = RunAborted()
        except _HostStop as exc:
            # A request_abort/AbortCommand wakeup can reach the dispatcher as an internal
            # worker stop before its next `_cancel` check. Preserve the abort as first cause.
            if not self._cancel:
                host_stop = exc
            else:
                terminal = RunAborted()
        except NodeExecutionError as exc:
            # First-cause wins: publish cooperative stop before draining sibling workers, but
            # retain this exact node failure as the run terminal even if a sibling unwinds with
            # _HostStop while the pool joins.
            self.stop_event.set()
            terminal = RunFailed(error=exc.error, error_type=exc.error_type, locator=exc.locator,
                                 traceback=exc.traceback, range=exc.range)
        except BaseException:
            # An abandoned or externally interrupted drive still owns its live executors.
            # Wake cooperative blockers before the mandatory full join, then preserve the
            # initiating exception exactly (including GeneratorExit from ``stream.close()``).
            self.stop_event.set()
            raise
        finally:
            self._stop.set()
            for w in workers:
                # Opted-in blocking nodes observe `stop_event`; join fully so a completed drive
                # never leaks an executor into the next run. An uncooperative external call is a
                # process boundary and intentionally has no pretend timeout-based cleanup.
                w.join()
        if host_stop is not None:
            # All workers are now quiescent, so every exact stop signal is present. Prefer an
            # initiating exception to a sibling that merely observed the already-set event.
            while True:
                try:
                    message = self.event_q.get_nowait()
                except queue.Empty:
                    break
                if (
                    isinstance(message, _WorkerHostStop)
                    and message.exception.primary
                    and not host_stop.primary
                ):
                    host_stop = message.exception
            raise host_stop
        yield terminal

    def _seed_start_and_advance(self):
        """Schedule the top-level START_ID as an ordinary node.

        START now runs through `eval_node` like any other node: it emits NodeStarted/NodeSucceeded,
        binds its inputs, coerces + type-checks + defaults them, and fires the flow's boundary
        asserts through its ordinary `post_asserts` seam (a false one -> NodeFailed -> RunFailed).

        Seeding: `store[START_ID]` is primed with the RAW `run_inputs` so START's `${input.X}`
        wiring resolves (it self-references `store[START_ID]`); the wiring is narrowed to ONLY the
        passed keys so an OMITTED input is ABSENT (bind_params then fills START's declared default
        instead of binding a self-referential null). The narrowed wiring is a PER-ENGINE override
        (`self.wiring_overrides`) consulted by eval_node — the shared compiled `flow.wiring` is
        NOT mutated (a ComposedFlow is reusable across runs). START's own eval_node run re-commits
        the COERCED record under START_ID. Direct-FlowEngine tests that hand-seed `store[START_ID]`
        pass no `run_inputs`; the compiled `${input.X}` wiring reads that hand-seeded record."""
        start_id = self.flow.start_id
        if start_id not in self.flow.nodes:
            return
        if self.run_inputs is not None:
            # Prime store[START_ID] with the raw run args so `${input.X}` resolves for START's own
            # bind, and narrow the wiring (per-engine override, NOT the shared IR) to the passed
            # keys so an omitted input binds absent -> bind_params fills START's declared default
            # (its ParamDecls carry the default), which START's `_run_impl` then coerces/type-checks.
            self.pool.set(start_id, dict(self.run_inputs))
            self.wiring_overrides[start_id] = {
                k: f"${{input.{k}}}" for k in self.run_inputs
            }
        self._schedule(start_id)

    def _terminal_event(self):
        """The terminal event for a completed (non-paused/aborted/failed) run: the
        run result IS the END_ID node's committed value. END_ID is an ordinary tail — when it RAN
        (`END_ID in pool.store`) the run SUCCEEDED with `store[END_ID]`; when it was skip-flooded
        (a required `outputs:` group co-skipped -> END_ID's disposition `dead` -> never committed)
        the run FAILED with the byte-stable `terminal output {name!r} skipped`, the name recovered
        from the SKIPPED required group's `input_group` (the output name keyed onto each
        producer->END_ID edge). Shared by serial run()/resume() + the parallel run() so the
        co-skip path is identical on both engines."""
        if END_ID in self.pool.store:
            return RunSucceeded(output=self.pool.get(END_ID))
        return RunFailed(error=f"terminal output {self._coskipped_output_name()!r} skipped",
                         error_type="TerminalSkipped")

    def _coskipped_output_name(self) -> Optional[str]:
        """The declared-output name of END_ID's dead required group: the `input_group` of a
        required (`optional=False`) producer->END_ID data edge whose every edge in the group SKIPPED.
        Recovers the exact name for the byte-stable terminal message."""
        st = self.sm.edge_state
        groups: dict[Optional[str], list] = {}
        for e in self.flow.incoming(END_ID):
            if e.source_handle is None and not e.ordering:
                groups.setdefault(e.input_group, []).append(e)
        for group, edges_g in groups.items():
            if edges_g[0].optional:
                continue
            if all(st.get(e.id) == NodeState.SKIPPED for e in edges_g):
                return group
        return None

    # --- durable suspend / resume ------------------------------------------ #

    def snapshot(self):
        """Capture the suspended run as a serializable RunCheckpoint.

        Captures pool + ready + node_state + edge_state + paused_nodes +
        deferred_nodes + pause_reasons + num_workers + expansions (the ledger of
        GrowRecords for runtime-grown CALL/MAP/LOOP subgraphs, which
        `restore()` replays top-down to re-grow the cloned subgraphs).

        Call after `run()` yields `RunPaused`. The checkpoint can be persisted
        (dumps/loads) and resumed in a FRESH process via `restore` + `resume`,
        including a run paused mid-expansion (a grown CALL/MAP/LOOP subgraph).
        """
        from agent_composer.suspension.checkpoint import RunCheckpoint

        # Capture by VALUE — a point-in-time snapshot the holder can serialize later. The
        # pool and the GrowRecord ledger are mutable pydantic models that the live engine keeps
        # advancing (e.g. a nested CALL grows under its parent's record); a shallow
        # `self.pool` / `list(self.expansions)` would let later live progress retro-mutate an
        # already-taken checkpoint. node_state/edge_state are dict() copies of immutable NodeState
        # enum values, so they need no deep copy.
        return RunCheckpoint(
            pool=self.pool.model_copy(deep=True),
            ready=self._ready_snapshot(),
            node_state=dict(self.sm.node_state),
            edge_state=dict(self.sm.edge_state),
            paused_nodes=[node_id for node_id, _ in self.paused],
            deferred_nodes=list(self.deferred),
            pause_reasons=[reason for _, reason in self.paused],
            num_workers=self.num_workers,
            expansions=[d.model_copy(deep=True) for d in self.expansions],
        )

    @classmethod
    def restore(
        cls,
        flow: CompiledFlow,
        checkpoint,
        *,
        num_workers: Optional[int] = None,
        grow_builder=None,
        stop_event=None,
    ) -> "FlowEngine":
        """Rebuild a resumable engine on `flow` from a (deserialized) checkpoint.

        `num_workers=None` (default) rebuilds the engine at the checkpoint's recorded
        drive mode; pass an int to OVERRIDE it — a run checkpointed serial can resume
        pooled and vice-versa (workers are pure executors; correctness is
        worker-count-independent). `grow_builder` is the same optional pure constructor
        accepted by `__init__`; it is required when the checkpoint contains growth records.
        `stop_event` supplies the new process-local cooperative-stop capability; when omitted,
        the restored engine owns a fresh unset event.

        Order: build the engine on the pool (at the resolved drive mode) → replay the
        expansions descriptor tree
        (re-grows flow + sm, re-derives alias/depth/_spawner_expansion) → OVERWRITE
        node_state/edge_state from the checkpoint (now covers the re-grown nodes too) →
        re-seed paused/deferred/ready. Order matters: replay must register the cloned nodes
        BEFORE the node_state overwrite restores their TAKEN/SKIPPED/EXPANDED states.

        `flow` MUST be CLEAN — a fresh compile with NO namespaced ids. `add_subgraph`
        is non-idempotent (it `extend`s edges + `append`s adjacency), so replaying over an
        already-grown flow would duplicate edges/adjacency and double-run a side-effecting
        node. restore() mutates `flow` in place, so it must not be re-invoked on the same
        object. A hand-built flow passed here must carry the SAME baked `.child` on its
        CALL/MAP spawners as a loader compile (the replay needs it)."""
        # Defense-in-depth version gate: a checkpoint may reach restore() without
        # passing through RunCheckpoint.loads() (which also gates). The current blob
        # version is a breaking migration over older blobs.
        from agent_composer.suspension.checkpoint import CHECKPOINT_VERSION
        if getattr(checkpoint, "version", None) != CHECKPOINT_VERSION:
            raise ValueError(
                f"incompatible checkpoint version {getattr(checkpoint, 'version', None)!r}; "
                f"this build reads {CHECKPOINT_VERSION!r} "
                "(AGENT continuation topology was removed)"
            )
        # Clean-flow guard (BEFORE replay): a cloned id carries `/` or `#`, so a flow that
        # already has any is a re-grown one — replaying onto it duplicates the overlay.
        bad = [n for n in flow.nodes if "/" in n or "#" in n]
        if bad:
            raise ValueError(
                f"restore() requires a clean flow (fresh compile); found namespaced/cloned "
                f"node ids {bad[:5]!r} — pass a freshly recompiled flow, not a re-grown one"
            )
        # Consume the checkpoint BY VALUE (symmetric with snapshot()'s write-side deep-copy):
        # a held checkpoint stays a point-in-time value even on the READ side, so a host that
        # reuses a retained snapshot()/loads() object across resume_flow() retries is not
        # retro-mutated by later execution.
        workers = checkpoint.num_workers if num_workers is None else num_workers
        engine = cls(
            flow,
            pool=checkpoint.pool.model_copy(deep=True),
            num_workers=max(0, workers),
            grow_builder=grow_builder,
            stop_event=stop_event,
        )
        # Replay re-grows the live topology + sm overlay + alias/depth/_spawner_expansion and
        # rebuilds self.expansions from OUR OWN descriptor copies. schedule=False.
        engine._replay_expansions([d.model_copy(deep=True) for d in checkpoint.expansions])
        # Overwrite node/edge state from the checkpoint — now covers the re-grown nodes too.
        engine.sm.node_state = dict(checkpoint.node_state)
        engine.sm.edge_state = dict(checkpoint.edge_state)
        # Re-seed the suspend frontier: self.paused (zip nodes+reasons), self.deferred, and
        # self.ready as a PLAIN frontier (set directly, NOT via _schedule — _schedule would
        # route through the paused-check into deferred). resume()'s `seed = deferred + ready`
        # then _enqueue(seed) consumes it.
        engine.paused = list(zip(checkpoint.paused_nodes, checkpoint.pause_reasons))
        engine.deferred = list(checkpoint.deferred_nodes)
        engine.ready = deque(checkpoint.ready)
        return engine

    def resume(self, commands=None):
        """Continue a paused run by DELIVERING each command's answer as the parked leaf's
        Output. ORDERING INVARIANT: apply commands WHILE self.paused is still set, so
        a delivered node's newly-ready successors are held in self.deferred (via _schedule);
        only THEN clear paused/deferred and seed = deferred + ready. This makes a
        multi-command resume drop no successor and double-run none — a fan-in fires exactly
        once after all its predecessors are delivered. NO re-enqueue of paused nodes (the
        re-run model is gone). Resume drives under the SAME drive mode the engine carries
        (serial or pooled), via the shared _drive_to_terminal."""
        yield RunResumed()
        # A commandless resume of a STILL-paused run re-emits RunPaused and returns
        # WITHOUT clearing self.paused. An idempotent poll / watcher tick / partial multi-wait
        # delivery must not destroy the pause (a no-op resume stays paused, never falls
        # through to a state-destroying terminal). Guard before the clear below.
        if self.paused and not (commands or []):
            yield RunPaused(reasons=[reason for _, reason in self.paused])
            return
        # Apply commands WHILE self.paused is still set (successors route to deferred). A
        # type-invalid answer raises NodeExecutionError here -> RunFailed (resume never crashes).
        try:
            for command in commands or []:
                self._apply_command(command)
        except NodeExecutionError as exc:
            self.stop_event.set()
            yield RunFailed(error=exc.error, error_type=exc.error_type, locator=exc.locator,
                            traceback=exc.traceback, range=exc.range)
            return
        seed = list(self.deferred) + list(self.ready)
        self.paused = []
        self.deferred = []
        self.ready = deque()           # seed already captured it; _enqueue re-appends once
        for node_id in seed:
            self._enqueue(node_id)     # serial -> self.ready; pooled -> self.ready_q
        yield from self._drive_to_terminal()

    def _apply_command(self, command) -> None:
        from agent_composer.suspension.commands import (
            AbortCommand,
            DeliverAnswerCommand,
        )

        if isinstance(command, DeliverAnswerCommand):
            # Deliver-as-Output: write the answer as the parked leaf's value and fire
            # its existing out-edges. The node is resolved against the LIVE graph, so a
            # runtime-namespaced id resolves. Wrapped in the SAME TypeCheckError -> NodeExecutionError
            # guard _on_success uses, so a type-invalid answer FAILS the run (it does not crash
            # resume). A compatibility scheduled-release command may deliver value=None.
            node = self.flow.nodes[command.node_id]
            try:
                self.pool.set(command.node_id, command.value, declared=node.output_type)
            except TypeCheckError as exc:
                self.sm.finish_executing(command.node_id)
                raise NodeExecutionError(
                    command.node_id, str(exc), type(exc).__name__,
                    locator=SourceSpan(node=command.node_id, kind="field", key="output"),
                )
            self.sm.finish_executing(command.node_id)  # idempotent (already finished on pause)
            for nid in self._advance(command.node_id):
                self._schedule(nid)
        elif isinstance(command, AbortCommand):
            self._cancel = True
            self.stop_event.set()

    # --- drain -------------------------------------------------------------- #

    def _drain(self):
        while self.ready:
            if self._cancel:
                raise _Aborted
            if self.stop_event.is_set():
                raise _HostStop(primary=False)
            node_id = self.ready.popleft()
            yield from self._run_node(node_id)

    def _run_node(self, node_id: str):
        node = self.flow.nodes[node_id]
        succeeded: Optional[NodeSucceeded] = None
        routed: Optional[NodeRouted] = None
        for event in eval_node(
            node,
            self.flow,
            self.pool,
            wiring_override=self.wiring_overrides.get(node_id),
            stop_event=self.stop_event,
        ):
            # A cooperative node may return normally after its event was set. Its late outcome
            # must not mutate the pool/topology or become externally visible.
            if self.stop_event.is_set():
                raise _HostStop(primary=False)
            if isinstance(event, NodeExpanded):
                # Growth is committed before its event becomes observable. A direct generator
                # consumer can mutate an event payload while this generator is suspended at
                # `yield`; exposing NodeExpanded first would therefore let the observer change
                # the builder input (or the spawner itself) and make live topology diverge from
                # replay. Failed construction likewise must not claim that expansion happened.
                try:
                    self._apply_grow(node_id, event.grow)
                except NodeExecutionError:
                    raise
                except Exception as exc:  # noqa: BLE001 — boundary: any apply error -> RunFailed
                    self.sm.finish_executing(node_id)
                    raise NodeExecutionError(node_id, str(exc), type(exc).__name__)
                yield event
                return
            yield event
            if self.stop_event.is_set():
                raise _HostStop(primary=False)
            if isinstance(event, NodeSucceeded):
                succeeded = event
            elif isinstance(event, NodeRouted):
                routed = event
            elif isinstance(event, NodeFailed):
                self.sm.finish_executing(node_id)
                raise NodeExecutionError(
                    node_id, event.error, event.error_type, locator=event.locator,
                    traceback=event.traceback, range=event.range,
                )
            elif isinstance(event, PauseRequested):
                self._on_pause(node_id, event.reason)
                return
        if succeeded is not None:
            self._on_success(node_id, succeeded)
        elif routed is not None:
            self._on_route(node_id, routed.handle)

    # --- pooled path (num_workers>=1): dispatcher + workers ----------------- #

    def _dispatch(self):
        # Single writer: drains event_q, forwards each event, applies the shared
        # mutation helpers. Completion = ready_q empty AND no node executing.
        while not self.sm.is_complete(self.ready_q.empty()):
            if self._cancel:
                raise _Aborted
            if self.stop_event.is_set():
                raise _HostStop(primary=False)
            try:
                event = self.event_q.get(timeout=_POLL)
            except queue.Empty:
                continue
            if isinstance(event, _WorkerHostStop):
                raise event.exception
            # Drop every node outcome produced after another path initiated host stop. An
            # initiating failure reaches this point before the dispatcher sets the event.
            if self.stop_event.is_set():
                raise _HostStop(primary=False)
            if isinstance(event, NodeExpanded):
                # Match the inline path: construction/application is atomic with respect to
                # external observation, and a failed construction emits no NodeExpanded.
                try:
                    self._apply_grow(event.node_id, event.grow)
                except NodeExecutionError:
                    raise
                except Exception as exc:  # noqa: BLE001 — boundary: any apply error -> RunFailed
                    self.sm.finish_executing(event.node_id)
                    raise NodeExecutionError(event.node_id, str(exc), type(exc).__name__)
                yield event
                continue
            yield event  # forward to the caller (streaming)
            if self.stop_event.is_set():
                raise _HostStop(primary=False)
            if isinstance(event, NodeSucceeded):
                self._on_success(event.node_id, event)
            elif isinstance(event, NodeRouted):
                self._on_route(event.node_id, event.handle)
            elif isinstance(event, NodeFailed):
                self.sm.finish_executing(event.node_id)
                raise NodeExecutionError(
                    event.node_id, event.error, event.error_type, locator=event.locator,
                    range=event.range,
                )
            elif isinstance(event, PauseRequested):
                self._on_pause(event.node_id, event.reason)

    def _worker(self) -> None:
        # Pure executor: pulls a node id, runs eval_node, pushes events. Never
        # mutates graph/edge/pool state.
        while not self._stop.is_set():
            try:
                node_id = self.ready_q.get(timeout=_POLL)
            except queue.Empty:
                continue
            if self.stop_event.is_set():
                self.event_q.put(
                    _WorkerHostStop(node_id, _HostStop(primary=False))
                )
                return
            node = self.flow.nodes[node_id]
            try:
                for event in eval_node(
                    node,
                    self.flow,
                    self.pool,
                    wiring_override=self.wiring_overrides.get(node_id),
                    stop_event=self.stop_event,
                ):
                    if self.stop_event.is_set():
                        self.event_q.put(
                            _WorkerHostStop(node_id, _HostStop(primary=False))
                        )
                        return
                    self.event_q.put(event)
            except _HostStop as exc:
                self.event_q.put(_WorkerHostStop(node_id, exc))
                # A derived stop only observed cancellation already initiated elsewhere. In
                # particular, an EOF-cancelled sibling must not set the event before the active
                # prompt worker has recovered and transported the exact EOF-backed exception.
                if exc.primary:
                    self.stop_event.set()
                return
            except Exception as exc:  # noqa: BLE001 — never let a worker die silently
                self.event_q.put(NodeFailed(node_id, str(exc), type(exc).__name__))

    # --- state mutation (shared by the inline loop and the pooled dispatcher) #

    def _enqueue(self, node_id: str) -> None:
        # Runs on the dispatcher / inline loop only (single writer).
        self.sm.mark_node(node_id, NodeState.TAKEN)
        self.sm.add_executing(node_id)
        if self.num_workers == 0:
            self.ready.append(node_id)
        else:
            self.ready_q.put(node_id)

    def _ready_snapshot(self) -> list[str]:
        """The queued-ready ids for a checkpoint — branches strictly on
        `num_workers==0` (the serial deque) vs the pooled `ready_q`, so snapshot()
        captures the queued ids under either path."""
        return list(self.ready) if self.num_workers == 0 else list(self.ready_q.queue)

    def _schedule(self, node_id: str) -> None:
        # While the run is suspending, hold newly-ready nodes as deferred rather
        # than starting fresh work; they re-enter the queue on resume.
        if self.paused:
            self.deferred.append(node_id)
        else:
            self._enqueue(node_id)

    def _on_pause(self, node_id: str, reason: Any) -> None:
        # Park the node: the engine delivers its answer as an Output on resume (no
        # re-run, so NO UNKNOWN reset). The node stays TAKEN (set by _enqueue); finish_executing
        # only discards it from the `executing` set, it does not touch node_state.
        self.sm.finish_executing(node_id)
        self.paused.append((node_id, reason))

    def _prune(self, ids: "frozenset[str]") -> None:
        """Kind-BLIND retirement of a named id-set — the generic inverse of `_apply_grow`'s splice.

        `ids` is whatever node-id set the caller wants gone from the LIVE overlay (e.g. a finished
        loop iteration's `f"{spawner}#{i}/"` namespace). Removes exactly those ids and everything
        keyed by them: their nodes + any edge touching one of them (the dead-edge set is DERIVED
        from `ids`), their state-manager entries, pool values, and `depth`/`_spawner_expansion`
        bookkeeping. Takes `self.sm.lock` (a re-entrant RLock) so topology and state mutate
        atomically. A no-op on the empty set. The engine stays kind-blind: the OUTCOME decides
        which ids retire, never `node.kind`."""
        if not ids:
            return
        with self.sm.lock:                                   # mutate topology + state atomically
            dead_edges = {e.id for e in self.flow.edges
                          if e.from_ in ids or e.to in ids}
            self.flow.remove_subgraph(ids)                   # topology inverse of add_subgraph
            self.sm.drop(ids, dead_edges)                    # state inverse of register
            for nid in ids:
                self.pool.store.pop(nid, None)
                self.depth.pop(nid, None)
                self._spawner_expansion.pop(nid, None)

    def _replay_expansions(self, records: list, *, is_top_level: bool = True) -> None:
        """Deterministic fold over the persisted `GrowRecord` tree: re-grow the live topology +
        bookkeeping a paused run had, with all live effects suppressed (`schedule=False` — no
        boundary assert, no budget raise, no scheduling). Kind-BLIND: each record names its
        spawner, and the injected builder rebuilds that spawner's OWN subgraph from the canonical
        seed. `_apply_grow` re-splices it and re-applies the generic growth
        bookkeeping (trait-driven depth/stamps/loop retention). The pure
        clone (`ns(callsite, child_id)`) re-keys every cloned node identically, so the rebuilt
        overlay matches the live engine's byte-for-byte.

        REBUILDS `self.expansions` by REUSING each deserialized record — a top-level record
        (`is_top_level`) is appended here; a nested one is ALREADY attached under its parent's
        `children` by deserialization, so the fold only walks it. `_apply_grow` is called with
        `record=rec` so it does NOT re-append (this top-level append owns the ledger); the stamping
        step stamps `_spawner_expansion` at THESE SAME objects, so a later in-place `children.append` (an
        LOOP next iteration after a durable hop) grows a record that IS in the
        ledger."""
        for rec in records:
            if is_top_level:
                self.expansions.append(rec)     # rebuild the ledger; nested ones ride their parent
            self._apply_grow(
                rec.spawner_id,
                Grow(seed=rec.seed),
                schedule=False,
                record=rec,
            )
            self._replay_expansions(rec.children, is_top_level=False)

    def _apply_grow(self, spawner_id, grow, *, schedule=True, record=None):
        """Generic growth core (kind-BLIND): build a `Flow`, validate its bounds, splice/register
        it, record ONE uniform `GrowRecord` in the durable ledger, apply the trait-driven growth
        bookkeeping (boundary asserts, REF depth, `_spawner_expansion` stamps), finish/mark the
        spawner, apply `grow.prune` (enabling loop self-prune), and
        schedule its `start_id` (the single entry, which fans out downstream via `_advance`).
        `schedule=False` suppresses scheduling+budget on replay; `record` reuses
        a deserialized ledger entry on replay (so no new record is minted).

        Ledger attach is kind-blind for non-loop kinds: mint a `GrowRecord(origin, seed, [])`, then
        nest it under the enclosing spawner's record if this spawner is INSIDE one (`_spawner_expansion`
        names it), else append it top-level. A LOOP takes the origin-keyed single-record path: the
        record is attributed to the compiled loop `origin` and supersedes the prior iteration's
        record in the same container (see the `node.is_loop` branch below)."""
        if self.grow_builder is None:
            raise RuntimeError(
                "dynamic graph growth requires an injected grow_builder; "
                "pass grow_builder=(spawner, canonical_seed) -> Flow"
            )
        spawner_node = self.flow.nodes[spawner_id]
        # Preserve one canonical value for the durable ledger. The event payload remains owned by
        # eval_node/its observer, while the builder gets a separate disposable copy: neither an
        # observer's nested mutation nor an impure builder can retroactively alter replay input.
        canonical_seed = copy.deepcopy(grow.seed)
        sg = self.grow_builder(spawner_node, copy.deepcopy(canonical_seed))
        with self.sm.lock:
            projected_node_count = len(self.flow.nodes) + sum(
                node_id not in self.flow.nodes for node_id in sg.nodes
            )
            if schedule and projected_node_count > MAX_TOTAL_NODES:
                raise RuntimeError(
                    f"expansion exceeded node budget ({MAX_TOTAL_NODES}) at spawner {spawner_id!r}")
            grow_depth = self._validated_grow_depth(spawner_id, schedule=schedule)
            self.flow.add_subgraph(sg.nodes, sg.edges, sg.wiring)
            self.sm.register(list(sg.nodes), sg.edges)
        # Attribution origin (kind-blind): a loop driver clone `L~k` attributes its grow/commit to the
        # compiled loop `L` via `origin_id`; every other node has `origin_id is None`, so `origin` is
        # just the spawner id (no behavior change). The durable ledger keys on `origin`, so a loop's
        # iterations fold into a single record under `L`. `is_loop` is read HERE (before the moved
        # prune below) because a loop driver may SELF-prune `spawner_id`, after which
        # `flow.nodes[spawner_id]` no longer exists.
        origin = spawner_node.origin_id or spawner_id
        spawner_is_loop = spawner_node.is_loop
        # Mint the one GrowRecord for this grow (live), OR reuse the deserialized one on replay.
        rec = record if record is not None else GrowRecord(
            spawner_id=origin, seed=copy.deepcopy(canonical_seed), children=[])
        # A nested grow whose spawner was stamped at an enclosing record rides under that record's
        # `children`; a top-level grow is a `self.expansions` root. On replay (`record is not None`)
        # the record is already attached by deserialization, so this is skipped.
        parent = self._spawner_expansion.get(spawner_id) if record is None else None
        # REF-depth stamping (kind-blind, driven by the `grow_depth_delta` trait): the depth was
        # derived and bounded before the splice; now stamp it on every spawner in the subgraph AND
        # its terminal(s). None (a non-spawner) is a no-op; zero lets LOOP inherit
        # the enclosing REF depth without consuming another bounded level.
        self._stamp_grow_depth(spawner_id, sg, grow_depth)
        # `_spawner_expansion` stamping (kind-blind): point every spawner in the spliced subgraph +
        # its derived terminal(s) at THIS record, so a nested grow nests under it and the CALL
        # post-assert commit-site recovery finds it.
        self._stamp_spawner_expansion(spawner_id, sg, rec)
        # Finish/mark is UNIFORM across every spawner (kind-blind): a spawner that returned a Grow
        # has run and expanded, so it finishes executing and enters EXPANDED. Idempotent for a loop
        # whose repeated iteration grows re-mark the same spawner; runs on replay too (the legacy
        # arms did it unconditionally). Placed AFTER the boundary/budget stamping steps so a
        # boundary-assert / depth-budget raise above leaves the spawner un-finished for the located
        # failure.
        self.sm.finish_executing(spawner_id)
        self.sm.mark_node(spawner_id, NodeState.EXPANDED)
        # Prune is the generic inverse of the splice — the outcome names which ids to retire; the
        # engine stays kind-blind (∅ for call/map). Moved AFTER finish/mark (was before the splice
        # bookkeeping) so a loop driver can SELF-prune: `grow.prune` may name the spawner id itself
        # (`L~k` retires itself in its own grow), and finish/mark must run on the still-live spawner
        # first. Safe because only loops ever set a non-empty `prune`, and the origin `L` (which the
        # ledger attach below reads) is NEVER in the prune set (self-prune excludes the origin).
        if grow.prune:
            self._prune(grow.prune)
        # Ledger attach AFTER the splice bookkeeping + prune (attach-after-grow): a boundary-assert /
        # budget raise above on the live path leaves NO orphan record in the ledger.
        if spawner_is_loop:
            # Single-record-by-origin (replaces the deleted `_apply_loop_bookkeeping`). The
            # parent/container is STABLE across iterations because `origin` is the compiled `L`
            # (never pruned, never re-stamped by its own iterations): top-level -> `self.expansions`,
            # nested in a CALL -> that CALL record's `children`. Drop the superseded iteration's
            # record from the container and append the new one; the map is set on replay too so a
            # post-restore continuation finds the live record.
            parent = self._spawner_expansion.get(origin)
            container = parent.children if isinstance(parent, GrowRecord) else self.expansions
            prev = self._origin_record.get(origin)
            if record is None:                         # live path: replace prev with rec
                if prev is not None and prev in container:
                    container.remove(prev)
                container.append(rec)
            self._origin_record[origin] = rec          # set on replay too (post-restore continuation)
        elif record is None:                           # generic CALL/MAP — unchanged
            if isinstance(parent, GrowRecord):
                parent.children.append(rec)
            else:
                self.expansions.append(rec)
        if schedule:
            self._schedule(sg.start_id)

    def _stamp_spawner_expansion(self, spawner_id, sg, rec) -> None:
        """Kind-blind `_spawner_expansion` stamping: point every `is_spawner` node in the spliced
        subgraph AND its derived terminal(s) at `rec`, so a grow nested inside one of them nests its
        GrowRecord under `rec` and the commit-site post-assert recovery (`_on_success`) finds the
        seed.

        A loop CONTINUATION driver `L~(k+1)` in the subgraph is SKIPPED: it carries `origin_id` set
        to the compiled loop and `!= its own id`, so its own grows attribute to the origin (§6b). If
        it were stamped here, `_spawner_expansion[L~(k+1)]` would shadow the stable origin lookup and
        break the single-record collapse. A compiled loop / genuine nested spawner (`origin_id` None,
        or `== id`) is still stamped."""
        for nid, node in sg.nodes.items():
            if getattr(node, "origin_id", None) is not None and node.origin_id != nid:
                continue
            if node.is_spawner:
                self._spawner_expansion[nid] = rec
        for term in self._derived_terminals(sg, spawner_id):
            self._spawner_expansion[term] = rec

    def _derived_terminals(self, sg, spawner_id) -> list:
        """The subgraph's terminal filler id(s) — kind-BLIND: the node(s) whose baked
        `commit_as == spawner_id` (a CALL/MAP/LOOP self-commit terminal). The growth core reuses
        this handle for depth stamping and `_spawner_expansion` stamping, so neither
        names the terminal by kind (`ns(spawner, END_ID)`, for example)."""
        return [nid for nid, node in sg.nodes.items() if node.commit_as == spawner_id]

    def _validated_grow_depth(self, spawner_id, *, schedule) -> Optional[int]:
        """Derive and validate a grow's REF depth without mutating runtime bookkeeping."""
        spawner = self.flow.nodes[spawner_id]
        delta = spawner.grow_depth_delta
        if delta is None:
            return None
        d = self.depth.get(spawner_id, 0) + delta
        if schedule and delta > 0 and d > MAX_REF_DEPTH:
            raise RuntimeError(
                f"expansion exceeded MAX_REF_DEPTH ({MAX_REF_DEPTH}) at {spawner_id!r}"
            )
        return d

    def _stamp_grow_depth(self, spawner_id, sg, depth: Optional[int]) -> None:
        """Stamp a prevalidated REF depth on a grown subgraph's spawners and terminals."""
        if depth is None:
            return
        for nid, node in sg.nodes.items():
            if node.is_spawner:
                self.depth[nid] = depth
        for term in self._derived_terminals(sg, spawner_id):
            self.depth[term] = depth

    def _on_success(self, node_id: str, event: NodeSucceeded) -> None:
        # The commit target: `commit_as` redirects a subflow terminal's value under its spawner
        # id (a cloned child END_ID / MAP END_ID-list / loop-body END),
        # else the node commits under its own id. This one line replaces the former `alias` /
        # `loop_alias` dict lookups.
        target = event.commit_as or node_id
        # Commit the value under `target` (the spawner id on a redirect, else `node_id`) with the
        # target's declared Type (same TypeCheckError -> NodeExecutionError guard the tail uses),
        # then advance the target's out-edges. For a redirect the filler's own pool.set is SKIPPED;
        # `finish_executing` still runs on the FILLER `node_id` (the node that actually ran).
        #
        # A spawner that returned `Output` instead of `Grow` (a 0-iteration `while` loop committing
        # its seed unchanged) flows through HERE, so it ends in the `NodeState.TAKEN` set by its
        # `_enqueue` — NOT `EXPANDED`. That is intentional and correct: EXPANDED means "a spawner that
        # ran and returned a Grow", and a 0-iteration loop grew nothing. Marking it EXPANDED would
        # need a kind-aware special-case here, which the kind-agnostic commit path deliberately avoids.
        target_node = self.flow.nodes[target]
        try:
            self.pool.set(target, event.output, declared=target_node.output_type)
        except TypeCheckError as exc:
            self.sm.finish_executing(node_id)
            raise NodeExecutionError(
                node_id, str(exc), type(exc).__name__,
                locator=SourceSpan(node=target, kind="field", key="output"),
            )
        # A spawner whose value is committed HERE (at the filler/redirect site), not in eval_node —
        # the spawner only yielded a Grow. Its node-local POST asserts (which read `${output}`) do
        # NOT fire here: a CALL lowers them onto the cloned child END (in `call_subgraph`), so they
        # fire through the ordinary `eval_node` post-assert seam on that END (naming the child END on
        # failure). MAP rejects `asserts:` at load; a LOOP body-END carries no `commit_as` on a
        # CONTINUE (it feeds the next driver by wiring) while a STOP commits directly from `run` and
        # fires its post_asserts in eval_node — so no redirect-commit spawner needs a firing site here.
        self.sm.finish_executing(node_id)
        for nid in self._advance(target):
            self._schedule(nid)

    def _on_route(self, node_id: str, handle: str) -> None:
        # A router (CASE) selected an out-edge handle: take it and skip-flood the siblings.
        # Dispatch rides the NodeRouted event, not node.kind — routing writes no pool value.
        self.sm.finish_executing(node_id)
        for nid in self._branch(node_id, handle):
            self._schedule(nid)

    def _advance(self, node_id: str) -> list[str]:
        ready: list[str] = []
        for edge in self.flow.outgoing(node_id):
            self.sm.mark_edge(edge.id, NodeState.TAKEN)
            # END_ID is a REAL node now — it must be scheduled, run, and committed (the run
            # result), so no `edge.to != END_ID` guard. END_ID participates in readiness/skip-flood.
            if self.sm.is_node_ready(edge.to):
                ready.append(edge.to)
        return ready

    def _branch(self, node_id: str, handle: str) -> list[str]:
        ready: list[str] = []
        for edge in self.flow.outgoing(node_id):
            if (edge.source_handle or DEFAULT_HANDLE) == handle:
                self.sm.mark_edge(edge.id, NodeState.TAKEN)
                if self.sm.is_node_ready(edge.to):
                    ready.append(edge.to)
            else:
                ready += self._skip_edge(edge)
        return ready

    def _skip_edge(self, edge) -> list[str]:
        """Skip an edge, then resolve the head via the unified disposition: ready -> schedule;
        dead -> skip-flood; wait -> leave for a later edge. The veto/data-co-skip live in disposition,
        so a skipped control edge that leaves all-control-skipped (or a fully-skipped required
        data group) floods the head even if another data edge is TAKEN."""
        self.sm.mark_edge(edge.id, NodeState.SKIPPED)
        head = edge.to
        # END_ID participates in skip-flood + disposition (its required output group can die).
        disp = self.sm.disposition(head)
        if disp == "ready":
            return [head]
        if disp == "dead":
            self.sm.mark_node(head, NodeState.SKIPPED)
            ready: list[str] = []
            for out in self.flow.outgoing(head):
                ready += self._skip_edge(out)
            return ready
        return []  # "wait" — a predecessor edge is still pending; decide later
