"""Node evaluation at the engine's pure read/execution boundary.

A node is a pure function of its bound input record; the ENGINE owns both boundaries —
the read (bind the inputs from the pool) and the write (the dispatcher stores `Output`).
``_eval_node_raw`` is the append-only dispatcher's private seam: it binds the record,
pre-resolves reserved keys, builds trait-gated capabilities, runs the node through its
assert-owning ``run`` wrapper, forwards transient chunks through ``emit_stream``, and
returns exactly ``Output | Route | Grow | EvaluationFailure``.  It emits no observer
notifications; durable observer ordering belongs to the dispatcher.

``eval_node`` is the temporary legacy event adapter.  It drives the same internal
evaluation generator but retains ``NodeStarted`` / terminal events and artificial
``PauseRequested`` until the atomic runtime cutover deletes suspension.  Keeping one
evaluation implementation prevents the two runtime paths from drifting while the old
engine still owns public execution.

Every node-side failure path funnels to ``EvaluationFailure``: a node ``raise``, a
``Grow`` returned by a non-spawner, and a non-``NodeResult`` return.  ``_HostStop`` and
an injected stream callback exception are deliberate host-side exceptions and propagate
unchanged.  The legacy adapter maps the failure carrier to ``NodeFailed``.

The bind is PURE: it reads the node's sources ONLY from `flow.wiring[node.id]` joined to the
node's `params` — the node carries no source. Direct-drive tests supply a stub `flow` with
`wiring` (the test helpers derive it). Layer: runtime imports nodes.* freely.

Accepted bind ordering vs the old per-node `_run` on an unreachable loaded-flow state: a
`call` that is both unbaked and has a bad `over`/binding surfaces the bind error before the
not-baked guard; the loader always bakes calls, and either path terminates in `NodeFailed`.
"""

from __future__ import annotations

import inspect
import threading
import traceback as _tb
from collections.abc import Callable, Generator
from dataclasses import dataclass, replace
from typing import Any, TypeAlias

from agent_composer.events import (
    NodeExpanded,
    NodeFailed,
    NodeRouted,
    NodeStarted,
    NodeSucceeded,
    PauseRequested,
    SourceRange,
    SourceSpan,
    StreamChunk,
)
from agent_composer.nodes.base import Grow, Output, Pause, Route, _HostStop
from agent_composer.nodes.binding import bind_params
from agent_composer.typesys.pool import VariablePool


@dataclass(frozen=True, slots=True)
class EvaluationFailure:
    """One raw node-evaluation failure for durable dispatcher preparation.

    ``node_id`` identifies the executing node; ``error`` and ``error_type`` are the
    stable diagnostic strings; ``locator`` and ``range`` retain symbolic and concrete
    YAML locations; ``traceback`` is the formatted Python stack captured while the
    exception is active.  Missing optional diagnostics are ``None``.
    """

    node_id: str
    error: str
    error_type: str = ""
    locator: SourceSpan | None = None
    range: SourceRange | None = None
    traceback: str | None = None


WorkerOutcome: TypeAlias = Output | Route | Grow | EvaluationFailure
"""Closed raw worker result consumed by the append-only dispatcher."""

_EvaluationOutcome: TypeAlias = WorkerOutcome | Pause
"""Raw result plus the legacy-only Pause retained by the temporary adapter."""


def _evaluate_node_stream(
    node,
    flow,
    pool: VariablePool,
    *,
    wiring_override=None,
    stop_event=None,
) -> Generator[StreamChunk, None, _EvaluationOutcome]:
    """Bind and run one node, yielding chunks and returning one normalized result.

    `wiring_override` is a per-run wiring dict the engine may supply for THIS node instead of
    the compiled `flow.wiring[node.id]` — used to narrow top-level START's `${input.X}` sources
    to only the passed run args (so an OMITTED input binds absent and bind_params fills START's
    declared default) WITHOUT mutating the shared compiled IR. `None` falls back to flow.wiring.

    `stop_event` is the host-owned cooperative-stop signal. Every evaluation checks it before
    running, but only a node declaring `needs_stop_event` receives it as a capability. A direct
    opted-in evaluation that omits it receives a fresh unset private event.

    Returns:
        `_EvaluationOutcome`:
            One ``Output``, ``Route``, ``Grow``, legacy-only ``Pause``, or frozen
            ``EvaluationFailure``.  Stream chunks are yielded before that return value.

    Raises:
        `_HostStop`:
            Host-control flow is outside the node failure sum and propagates unchanged.
    """
    try:
        # Host cancellation is control flow outside the node-result sum. The legacy adapter
        # emits NodeStarted before entering this generator; the raw seam has no observer event.
        if stop_event is not None and stop_event.is_set():
            raise _HostStop(primary=False)
        # The flow-owned wiring for this node (the node/flow split): every kind's sources live here
        # (leaf/WAIT, CALL, CASE). The node holds NO source. A direct-driver must supply a
        # stub `flow.wiring` with the reserved keys; `flow is None` gives empty wiring, so a timed
        # WAIT / mapped call driven that way would KeyError on `until`/`over` (caught as NodeFailed).
        # A `wiring_override` (engine-supplied, per-run) wins over the compiled entry.
        if wiring_override is not None:
            node_wiring = wiring_override
        else:
            node_wiring = {} if flow is None else flow.wiring.get(node.id, {})
        # Read boundary (pure): bind the node from its `params` + the flow-owned
        # wiring — never the node's own `inputs`. A per-item node (MAP) binds per-element via
        # bind_item, so its record starts empty. (`params or []` covers a no-input node / a
        # direct-construction test fake; loader-built nodes always carry params.)
        per_item = node.binds_per_item          # MAP = per-element bind (trait, not a kind read)
        if per_item:
            record = {}
        else:
            record = bind_params(node.params or [], node_wiring, pool)
        # Reserved-key pre-resolve (node-owned): WAIT `until` -> ISO deadline; mapped-call `over`
        # -> list (validated in the hook -> NodeFailed). The hook reads its sources from
        # `node_wiring` (the node/flow split) + `pool`; default `{}` for an ordinary node.
        record.update(node.bind_reserved(node_wiring, pool))
        # Narrow caps built by the engine — never the pool itself. A mapped call gets
        # `bind_item`; a cooperative blocker gets `stop_event` below.
        caps: dict[str, Any] = {}
        if per_item:
            # Per-element bind from params + flow.wiring (pure). No system cap — the
            # cloned children share the one live pool, so `${system.X}` resolves directly.
            caps["bind_item"] = lambda el: bind_params(node.params or [], node_wiring, pool, item=el)
        if node.needs_stop_event:
            # Engine-driven evaluation always passes its run-owned event. Direct evaluation has
            # no host owner, so create one only for an opted-in node; ordinary nodes allocate and
            # receive nothing.
            if stop_event is None:
                stop_event = threading.Event()
            caps["stop_event"] = stop_event
        # Drive the base `run` WRAPPER (not `_run_impl`): the wrapper owns assert firing.
        # Kind-specific execution diagnostics, such as CODE's in-process watchdog, stay in
        # the owning node package rather than timing every engine operation.
        outcome = node.run(record, **caps)
        if isinstance(outcome, (Output, Route, Pause, Grow)):
            result = outcome
        elif inspect.isgenerator(outcome):  # a streaming kind: yields StreamChunk, returns a NodeResult
            result = yield from node._drain_node_generator(outcome)
        else:
            raise RuntimeError(
                f"node {node.id!r} run() returned {type(outcome).__name__}, not a NodeResult"
            )
        if isinstance(result, Grow):
            # Any non-spawner returning Grow is a clear error. A valid raw Grow is returned
            # untouched; the dispatcher owns construction, durability, and observer emission.
            if not node.is_spawner:
                # `node.kind.value` here is MESSAGE-ONLY (a diagnostic string), not dispatch —
                # the gate is the kind-blind `is_spawner` trait, so the census stays 0.
                raise RuntimeError(
                    f"node {node.id!r} (kind {node.kind.value}) returned a Grow but is "
                    f"not a spawner (CALL/MAP/LOOP); only spawner kinds may grow the graph"
                )
        return result
    except _HostStop:
        # A host stop is neither a node failure nor a durable flow outcome.
        raise
    except Exception as exc:  # noqa: BLE001 — boundary: any node error -> EvaluationFailure
        # A failure may carry a `locator` (BindingError stamps a node-less input SourceSpan;
        # the binding layer has no node identity). Fill the node id here — this funnel is the
        # single point that knows it.
        loc = getattr(exc, "locator", None)
        if loc is not None and loc.node is None:
            loc = replace(loc, node=node.id)
        # Dual-path: ALONGSIDE the symbolic locator, resolve a concrete SourceRange. An exception
        # that already carries its own `range` (any node's own `run` wrapper fires its pre/post
        # asserts and raises `NodeAssertFailed` with the assert's span — located by position) wins;
        # otherwise key the node's recorded `field_ranges` on the locator's `(kind, key)` (e.g.
        # `("input", name)` / `("input_decl", name)`) — NOT on `node.kind`, and never on the bare key
        # (which would collide across namespaces). None when neither is present.
        rng = getattr(exc, "range", None)
        if rng is None and loc is not None:
            rng = node.field_ranges.get((loc.kind, loc.key))
        # Capture the full formatted traceback here, while the exception is live, so the CLI
        # can surface the raising call's Python stack under `--engine-trace`. The terse
        # default (message + boxed YAML frame) never shows it.
        return EvaluationFailure(
            node.id,
            error=str(exc),
            error_type=type(exc).__name__,
            locator=loc,
            range=rng,
            traceback=_tb.format_exc(),
        )


def _eval_node_raw(
    node,
    flow,
    pool: VariablePool,
    *,
    wiring_override=None,
    stop_event=None,
    emit_stream: Callable[[StreamChunk], None] | None = None,
) -> WorkerOutcome:
    """Evaluate one node into a raw worker result and transient stream callback.

    ``emit_stream`` is called synchronously for each chunk yielded by the node and is
    deliberately invoked outside the node-failure funnel.  If it raises, that exact host
    exception escapes; it must never become a durable node failure.  When omitted, chunks
    are discarded.  ``_HostStop`` likewise propagates unchanged.

    A returned ``Pause`` is converted to ``EvaluationFailure`` only during this migration:
    the legacy ``eval_node`` adapter still supports artificial pause, while the final
    append-only runtime deletes ``Pause`` entirely.
    """
    stream = _evaluate_node_stream(
        node,
        flow,
        pool,
        wiring_override=wiring_override,
        stop_event=stop_event,
    )
    while True:
        try:
            chunk = next(stream)
        except StopIteration as stopped:
            result = stopped.value
            break
        if emit_stream is not None:
            # Host-owned transport/callback boundary: keep this OUTSIDE the evaluator's
            # exception funnel so callback failures propagate with identity intact.
            emit_stream(chunk)

    if isinstance(result, Pause):
        return EvaluationFailure(
            node.id,
            error=(
                f"node {node.id!r} returned Pause, which is unsupported by raw "
                "append-only evaluation"
            ),
            error_type="RuntimeError",
        )
    return result


def eval_node(
    node,
    flow,
    pool: VariablePool,
    *,
    wiring_override=None,
    stop_event=None,
):
    """Temporarily adapt raw evaluation to the legacy engine event stream.

    The old engine still consumes ``NodeStarted`` followed by stream chunks and one
    terminal observer (or ``PauseRequested``).  This wrapper preserves that contract
    while all binding, execution, normalization, and failure diagnostics are shared with
    ``_eval_node_raw`` through ``_evaluate_node_stream``.
    """
    yield NodeStarted(node.id)
    result = yield from _evaluate_node_stream(
        node,
        flow,
        pool,
        wiring_override=wiring_override,
        stop_event=stop_event,
    )

    if isinstance(result, EvaluationFailure):
        yield NodeFailed(
            result.node_id,
            error=result.error,
            error_type=result.error_type,
            locator=result.locator,
            range=result.range,
            traceback=result.traceback,
        )
        return

    if isinstance(result, Pause):
        yield PauseRequested(node.id, result.reason)  # suspended; no terminal
        return
    if isinstance(result, Grow):
        yield NodeExpanded(node.id, result)
        return
    if isinstance(result, Route):
        yield NodeRouted(node.id, result.handle)
        return
    # Fold the commit redirect onto the terminal: a node-chosen `Output.commit_as` (roadmap)
    # wins over the engine-baked `node.commit_as` (the subflow-terminal redirect); both default
    # None so an ordinary node commits under its own id. This is the sole channel `_on_success`
    # reads to decide the commit target.
    yield NodeSucceeded(node.id, output=result.value,
                        commit_as=(result.commit_as or node.commit_as))
