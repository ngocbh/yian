"""eval_node — the engine's node-evaluation seam.

A node is a pure function of its bound input record; the ENGINE owns both boundaries —
the read (bind the inputs from the pool) and the write (the dispatcher stores `Output`).
This generator is that read seam plus the dispatch normalization, lifted out of
the node (it replaces the temporary `Node._emit`). It yields `NodeStarted`, binds the
record, pre-resolves reserved keys (timed WAIT `until` / mapped-call `over`), builds the per-kind
narrow caps (a mapped call's `bind_item`, the engine-owned `caps['llm']` model factory, and the
host-owned `caps['stop_event']`, each gated on its node trait), runs the node through its `run`
wrapper (which owns pre/post assert firing — record-only), and turns the returned `NodeResult`
into one terminal `NodeSucceeded`/`NodeFailed` — or a single `PauseRequested` for a returned
`Pause`.

Every node-side failure path funnels to `NodeFailed` so the two engines (serial + parallel)
agree byte-for-byte on the same input: a node `raise`, a `Grow` returned by a NON-spawner
kind, and a non-`NodeResult` return all become `NodeFailed` inside the `try` — none
escape the generator uncaught. `_HostStop` is the deliberate exception: private host control
flow bypasses the failure funnel. (A spawner's `Grow` instead becomes `NodeExpanded`.)

The bind is PURE: it reads the node's sources ONLY from `flow.wiring[node.id]` joined to the
node's `params` — the node carries no source. Direct-drive tests supply a stub `flow` with
`wiring` (the test helpers derive it). Layer: runtime imports nodes.* freely.

Accepted bind ordering vs the old per-node `_run` (both on states a loaded flow cannot reach,
both still terminating in `NodeFailed`): a timed WAIT re-resolves `until` on every path including
release/resume (harmless given the monotonic pool), and a `call` that is BOTH unbaked AND has a
bad `over`/binding surfaces the over/bind error before the not-baked guard (loader always bakes).
"""

import inspect
import threading
import traceback as _tb
from dataclasses import replace
from typing import Any

from agent_composer.events import (
    NodeExpanded,
    NodeFailed,
    NodeRouted,
    NodeStarted,
    NodeSucceeded,
    PauseRequested,
)
from agent_composer.nodes.base import Grow, Output, Pause, Route, _HostStop
from agent_composer.nodes.binding import bind_params
from agent_composer.runtime.watchdog import node_watchdog
from agent_composer.typesys.pool import VariablePool


def _default_llm(config):
    """Default LLM provider for the caps['llm'] seam: a call-time lazy lookup of the package
    factory, so monkeypatching `agent_composer.llm_clients.model_from_config` is honored."""
    from agent_composer.llm_clients import model_from_config

    return model_from_config(config)


def eval_node(
    node,
    flow,
    pool: VariablePool,
    llm=_default_llm,
    *,
    wiring_override=None,
    stop_event=None,
):
    """Evaluate one node through the engine read/dispatch seam; yield its event stream.

    `llm` is the engine-owned LLM-client provider (a `model_from_config`-shaped callable)
    handed to LLM-backed nodes via `caps['llm']` (gated on `node.needs_llm`); it defaults to
    the lazy package-lookup thunk so a monkeypatched factory is honored on the direct path.

    `wiring_override` is a per-run wiring dict the engine may supply for THIS node instead of
    the compiled `flow.wiring[node.id]` — used to narrow top-level START's `${input.X}` sources
    to only the passed run args (so an OMITTED input binds absent and bind_params fills START's
    declared default) WITHOUT mutating the shared compiled IR. `None` falls back to flow.wiring.

    `stop_event` is the host-owned cooperative-stop signal. Every evaluation checks it before
    running, but only a node declaring `needs_stop_event` receives it as a capability. A direct
    opted-in evaluation that omits it receives a fresh unset private event."""
    yield NodeStarted(node.id)
    try:
        # Host cancellation is control flow outside the node-result sum. Check only after the
        # observable NodeStarted boundary, then let _HostStop escape this generator unchanged.
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
        # Reserved-key pre-resolve (node-owned): timed WAIT `until` -> ISO ts; mapped-call `over`
        # -> list (validated in the hook -> NodeFailed). The hook reads its sources from
        # `node_wiring` (the node/flow split) + `pool`; default `{}` for an ordinary node.
        record.update(node.bind_reserved(node_wiring, pool))
        # Per-kind narrow caps, built by the engine — never the pool itself. A mapped call
        # gets `bind_item`; an LLM-backed node gets `llm`. HUMAN_INPUT/WAIT receive neither.
        caps: dict[str, Any] = {}
        if per_item:
            # Per-element bind from params + flow.wiring (pure). No system cap — the
            # cloned children share the one live pool, so `${system.X}` resolves directly.
            caps["bind_item"] = lambda el: bind_params(node.params or [], node_wiring, pool, item=el)
        if node.needs_llm:
            # LLM-backed node (AGENT): hand the engine-driven path its model factory. A
            # direct no-cap node driver owns a lower-layer llm_clients fallback instead.
            caps["llm"] = llm
        if node.needs_stop_event:
            # Engine-driven evaluation always passes its run-owned event. Direct evaluation has
            # no host owner, so create one only for an opted-in node; ordinary nodes allocate and
            # receive nothing.
            if stop_event is None:
                stop_event = threading.Event()
            caps["stop_event"] = stop_event
        # A kind-blind watchdog logs (does not kill) a node whose run overruns a soft
        # budget — so a runaway in-process CODE body is diagnosed, not a silent hang.
        with node_watchdog(node.id):
            # eval_node drives the base `run` WRAPPER (not `_run_impl`): the wrapper owns
            # assert firing — it fills declared defaults, fires pre-asserts against the bound
            # record, runs the kind's `_run_impl`, then fires post-asserts against
            # `{**record, output}` (all record-only, no pool). Firing happens EXACTLY ONCE,
            # here in the node. A streaming kind's `run` returns the generator unchanged (the
            # wrapper fires no post on it), so eval_node still drains it below.
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
            # A spawner grows the live graph: hand its seed-first `Grow` to the dispatcher's
            # `_apply_grow` via NodeExpanded. The injected builder constructs topology there. Any
            # non-spawner kind returning a Grow is a clear error (the graph only grows from
            # spawners).
            if not node.is_spawner:
                # `node.kind.value` here is MESSAGE-ONLY (a diagnostic string), not dispatch —
                # the gate is the kind-blind `is_spawner` trait, so the census stays 0.
                raise RuntimeError(
                    f"node {node.id!r} (kind {node.kind.value}) returned a Grow but is "
                    f"not a spawner (CALL/MAP/AGENT/LOOP); only spawner kinds may grow the graph"
                )
            yield NodeExpanded(node.id, result)
            return
    except _HostStop:
        # A host stop is neither a node failure nor a public abort outcome. The serial engine lets
        # it unwind; a pooled worker translates it to an internal dispatcher signal.
        raise
    except Exception as exc:  # noqa: BLE001 — boundary: any node error -> NodeFailed (both engines)
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
        yield NodeFailed(node.id, error=str(exc), error_type=type(exc).__name__, locator=loc,
                         range=rng, traceback=_tb.format_exc())
        return

    if isinstance(result, Pause):
        yield PauseRequested(node.id, result.reason)  # suspended; no terminal
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
