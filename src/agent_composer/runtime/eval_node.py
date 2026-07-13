"""eval_node — the engine's node-evaluation seam.

A node is a pure function of its bound input record; the ENGINE owns both boundaries —
the read (bind the inputs from the pool) and the write (the dispatcher stores `Output`).
This generator is that read seam plus the assert + dispatch normalization, lifted out of
the node (it replaces the temporary `Node._emit`). It yields `NodeStarted`, binds the
record, pre-resolves reserved keys (timed WAIT `until` / mapped-call `over`), builds the per-kind
narrow caps (a mapped call's `bind_item`, and `caps['llm']` — the engine-owned model factory —
for an LLM-backed node gated on `node.needs_llm`), runs the node,
and turns the returned `NodeResult` into one terminal `NodeSucceeded`/`NodeFailed` — or a
single `PauseRequested` for a returned `Pause`.

Every node-side failure path funnels to `NodeFailed` so the two engines (serial + parallel)
agree byte-for-byte on the same input: a node `raise`, a `Grow` returned by a NON-spawner
kind, and a non-`NodeResult` return all become `NodeFailed` inside the `try` — none
escape the generator uncaught. (A spawner's `Grow` instead becomes `NodeExpanded`.)

The bind is PURE: it reads the node's sources ONLY from `flow.wiring[node.id]` joined to the
node's `params` — the node carries no source. Direct-drive tests supply a stub `flow` with
`wiring` (the test helpers derive it). Layer: runtime imports nodes.* freely.

Accepted bind ordering vs the old per-node `_run` (both on states a loaded flow cannot reach,
both still terminating in `NodeFailed`): a timed WAIT re-resolves `until` on every path including
release/resume (harmless given the monotonic pool), and a `call` that is BOTH unbaked AND has a
bad `over`/binding surfaces the over/bind error before the not-baked guard (loader always bakes).
"""

import copy
import inspect
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
    SourceSpan,
)
from agent_composer.expr import resolve_reference
from agent_composer.expr.expressions import ExpressionError, _evaluate, _resolve_in_record
from agent_composer.nodes.base import Grow, Output, Pause, Route
from agent_composer.nodes.binding import bind_params
from agent_composer.runtime.watchdog import node_watchdog
from agent_composer.typesys.pool import VariablePool


def _first_failing_assert(asserts, record: dict, pool: VariablePool):
    """The first assert in `asserts` that does not hold against a bound scope, else None.

    ONE generic assert-check for both pre- and post-asserts, kind-blind. Per assert, each
    reference resolves from the widest scope that owns it, mirroring the two historical
    resolution rules with no `node.kind` branch:
      - head already in `record` (a declared input, or the synthetic `output` the caller
        injects for post-asserts) -> `record`, via the dotted `_resolve_in_record` walk
        (the node-local rule);
      - any other head -> the pool, via full-path `resolve_reference` (the END pool-fallback
        for namespaced cross-node refs like `${each#0/n.output.X}` that ride END from expand).

    A raising assert (an ordered/arith comparison over a None/non-scalar ref) does NOT hold —
    the same clean-NodeFailed contract as `Node._assert_holds`; only `ExpressionError` is
    swallowed (the domain the condition engine raises in), so a real bug still surfaces.

    Args:
        asserts (`list[str]`): the assert expressions (any `when:`/`asserts:` spelling).
        record (`dict`): the bound scope — input params (+ synthetic `output` for post).
        pool (`VariablePool`): the live pool for cross-node/namespaced ref fallback.

    Returns:
        `str | None`: the first assert that does not hold, or None if all hold.
    """
    for a in asserts:
        # Resolve each ref of THIS assert record-first / pool-fallback (byte-parity with the
        # retired END `_resolve_end` and the node-local `_resolve_in_record` paths). The record
        # head wins so a declared input / the synthetic `output` never reads a stale pool value;
        # an unknown head (a namespaced END ref) falls to the pool as END did.
        def _resolve(path: str, _record=record) -> Any:
            head = path.split(".", 1)[0].strip()
            if head in _record:
                return _resolve_in_record(path, _record)
            return resolve_reference(path, pool)

        try:
            holds = _evaluate(a, _resolve)
        except ExpressionError:
            holds = False
        if not holds:
            return a
    return None


def _default_llm(config):
    """Default LLM provider for the caps['llm'] seam: a call-time lazy lookup of the package
    factory, so monkeypatching `agent_composer.llm_clients.model_from_config` is honored."""
    from agent_composer.llm_clients import model_from_config

    return model_from_config(config)


def eval_node(node, flow, pool: VariablePool, llm=_default_llm, *, wiring_override=None):
    """Evaluate one node through the engine read/dispatch seam; yield its event stream.

    `llm` is the engine-owned LLM-client provider (a `model_from_config`-shaped callable)
    handed to LLM-backed nodes via `caps['llm']` (gated on `node.needs_llm`); it defaults to
    the lazy package-lookup thunk so a monkeypatched factory is honored on the direct path.

    `wiring_override` is a per-run wiring dict the engine may supply for THIS node instead of
    the compiled `flow.wiring[node.id]` — used to narrow top-level START's `${input.X}` sources
    to only the passed run args (so an OMITTED input binds absent and bind_params fills START's
    declared default) WITHOUT mutating the shared compiled IR. `None` falls back to flow.wiring."""
    yield NodeStarted(node.id)
    try:
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
        bad = _first_failing_assert(node.pre_asserts, record, pool)
        if bad is not None:
            display = node.assert_texts.get(bad, bad)
            yield NodeFailed(node.id, error=f"node {node.id!r} pre-assert failed: {display}",
                             error_type="NodeAssertFailed",
                             locator=SourceSpan(node.id, "assert", display),
                             range=node.field_ranges.get(bad))
            return
        # Per-kind narrow caps, built by the engine — never the pool itself.
        # HUMAN_INPUT/WAIT are deliver-as-Output: they always Pause and the engine
        # delivers the answer. AGENT lowers a control pause to a continuation `Grow`,
        # carrying its memo as graph data — so a mapped call's `bind_item` is the only cap now.
        caps: dict[str, Any] = {}
        if per_item:
            # Per-element bind from params + flow.wiring (pure). No system cap — the
            # cloned children share the one live pool, so `${system.X}` resolves directly.
            caps["bind_item"] = lambda el: bind_params(node.params or [], node_wiring, pool, item=el)
        if node.needs_llm:
            # LLM-backed node (AGENT): hand it the engine-owned model factory. The node no
            # longer imports `model_from_config` — it builds its model from this provider.
            caps["llm"] = llm
        # Pristine snapshot for the POST asserts: a leaf may mutate the dict it receives
        # (e.g. a CODE function transforming in place), which must not corrupt the contract
        # check — restores the isolation an earlier double-bind gave. Only paid when declared.
        post_input = copy.deepcopy(record) if node.post_asserts else None
        # A kind-blind watchdog logs (does not kill) a node whose run overruns a soft
        # budget — so a runaway in-process CODE body is diagnosed, not a silent hang.
        with node_watchdog(node.id):
            # Transition: eval_node calls _run_impl directly + fires asserts itself (the pre
            # block above, the post block below) through Step 7; Step 8 flips this to node.run
            # (the base wrapper) and deletes these firing blocks atomically. Calling _run_impl
            # here bypasses the dormant wrapper so asserts fire EXACTLY ONCE.
            outcome = node._run_impl(record, **caps)
            if isinstance(outcome, (Output, Route, Pause, Grow)):
                result = outcome
            elif inspect.isgenerator(outcome):  # a streaming kind: yields StreamChunk, returns a NodeResult
                result = yield from node._drain_node_generator(outcome)
            else:
                raise RuntimeError(
                    f"node {node.id!r} run() returned {type(outcome).__name__}, not a NodeResult"
                )
        if isinstance(result, Grow):
            # A spawner grows the live graph: hand the self-describing `Grow` to the dispatcher's
            # `_apply_grow` via NodeExpanded. Any non-spawner kind returning a Grow is a clear error
            # (the graph only grows from spawners).
            if not node.is_spawner:
                raise RuntimeError(
                    f"node {node.id!r} (kind {node.kind.value}) returned a Grow but is "
                    f"not a spawner (CALL/MAP/AGENT/LOOP); only spawner kinds may grow the graph"
                )
            yield NodeExpanded(node.id, result)
            return
    except Exception as exc:  # noqa: BLE001 — boundary: any node error -> NodeFailed (both engines)
        # A failure may carry a `locator` (BindingError stamps a node-less input SourceSpan;
        # the binding layer has no node identity). Fill the node id here — this funnel is the
        # single point that knows it.
        loc = getattr(exc, "locator", None)
        if loc is not None and loc.node is None:
            loc = replace(loc, node=node.id)
        # Dual-path: ALONGSIDE the symbolic locator, resolve a concrete SourceRange. An exception
        # that already carries its own `range` (a child START's NodeAssertFailed, which fires the
        # boundary asserts itself and knows the assert's span) wins; otherwise key the node's
        # recorded `field_ranges` on the locator KEY (an input name / assert expr) — NOT on
        # `node.kind`. None when neither is present.
        rng = getattr(exc, "range", None)
        if rng is None and loc is not None:
            rng = node.field_ranges.get(loc.key)
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
    if node.post_asserts:
        # Post-asserts check against the pristine input record plus the synthetic `${output}`
        # (the just-produced terminal value), injected EXACTLY as `{"output": value}` (NOT
        # spread) so the `${output}` selector wins over any declared field literally named
        # `output` (the precedence rule in agent-compose-principles.md). `_first_failing_assert`
        # resolves each ref record-first with a pool fallback, so a flow terminal's namespaced
        # cross-node refs (`${each#0/n.output.X}` riding END from expand) and a named node's
        # `${X.output}` flow-post assert resolve from the pool exactly as before — no END special
        # case, no `node.kind` read.
        post_record = {**post_input, "output": result.value}
        bad = _first_failing_assert(node.post_asserts, post_record, pool)
        if bad is not None:
            display = node.assert_texts.get(bad, bad)
            yield NodeFailed(node.id, error=f"node {node.id!r} post-assert failed: {display}",
                             error_type="NodeAssertFailed",
                             locator=SourceSpan(node.id, "assert", display),
                             range=node.field_ranges.get(bad))
            return
    # Fold the commit redirect onto the terminal: a node-chosen `Output.commit_as` (roadmap)
    # wins over the engine-baked `node.commit_as` (the subflow-terminal redirect); both default
    # None so an ordinary node commits under its own id. This is the sole channel `_on_success`
    # reads to decide the commit target.
    yield NodeSucceeded(node.id, output=result.value,
                        commit_as=(result.commit_as or node.commit_as))
