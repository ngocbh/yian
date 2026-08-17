"""Runtime graph-expansion machinery — the pure half.

When a spawner (CALL / MAP / LOOP) runs, it does not run a child engine; it
returns a seed-first `Grow`. The injected `build_grow` interpreter constructs the fragment and
the dispatcher prepares a canonical expansion record whose reducer plan splices the fragment.
This module holds the
**pure** machinery that growth keys off:

- `ns` / `map_callsite` (imported from `nodes.ids`): deterministic id minting. Every
  cloned node/edge id is a pure function of `(callsite, child static id, element index)` —
  NO emission counter — so prefix recovery reconstructs identical ids.

The pure cloner (`clone_child`) splices the child's own
`START_ID..END_ID` (every flow is `START_ID -> body -> END_ID`) into a `Flow`: the child `START_ID`
is the alias-seed point — SEEDED WITH THE CALL-ARGS AS EDGES (no `_rens` literal-baking) — and the
child `END_ID` is the alias filler. A child node reading `${input.X}` is re-pointed to the namespaced
child START_ID's output object (`${<callsite>/<start>.output.X}`). Transition preparation
validates the fragment and records its canonical seed/digest; the reducer then splices it,
initializes its dispositions, and queues its START node.

Layer: compile — imports `expr`/`nodes` plus the shared growth-schema leaf;
never `runtime` or the package-root run boundary.
"""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import Any, Optional

from agent_composer.compile.model import (
    END_ID,
    START_ID,
    Edge,
    Flow,
    GrowthRejected,
)
from agent_composer.diagnostics import OriginRef
from agent_composer.expr import rewrite_template_refs
from agent_composer.expr.expressions import rewrite_condition_refs
from agent_composer.nodes.base import Node, NodeKind
from agent_composer.nodes.contracts import Assertion, Binding, Parameter
from agent_composer.nodes.end.node import EndNode
from agent_composer.nodes.ids import map_callsite, ns
from agent_composer.nodes.start.node import StartNode

# Reserved prefix for END params synthesized from a lowered spawner post-assert. Mirrors the
# BUILD-time `_SYNTH_PREFIX` in `compose/validate/asserts.py` — kept a separate literal here
# because the layer ladder forbids `compile` importing from `compose`. Authors cannot write
# `__`-prefixed ids, so a synth name never collides with an author id.
_SYNTH_PREFIX = "__assert_"


# --------------------------------------------------------------------------- #
# clone_child — the pure deep-flatten + partial-eval + arity cloner
# --------------------------------------------------------------------------- #


def _whole_span(src: str) -> Optional[str]:
    """If `src` is EXACTLY one `${...}` span, return its interior; else None."""
    if not isinstance(src, str) or not (src.startswith("${") and src.endswith("}")):
        return None
    interior = src[2:-1]
    if "${" in interior or "}" in interior:  # embedded / nested — not a single bare span
        return None
    return interior


def _rens_internal(binding: Binding, callsite: str) -> Binding:
    """Re-namespace one canonical source while retaining its authored origin.

    Exactly one ``Binding`` layer is semantic IR: its source is rewritten and a
    new carrier retains the same origin. A nested carrier is an opaque payload
    and is not recursively unwrapped.

    Singular only:
    - `${input.<k>...}`   -> `${<callsite>/<start>.output.<k>...}` (namespaced child
      START_ID's output object read via the node-first head)
    - `${<X>.output...}`  -> `${<callsite>/<X>.output...}` (node-first re-namespaced in place)
    - `${system.X}` (run-global) and other heads untouched.

    Legacy plural heads (`outputs.X` / `inputs.X`) are rejected at parse time.

    Routes every `${...}` span through `rewrite_template_refs` and the shared parse tree, so
    embedded, coalesced, and whole-expression spans are re-namespaced leaf by leaf."""
    if not isinstance(binding, Binding):
        raise TypeError(
            "compiled flow wiring must contain Binding values, "
            f"got {type(binding).__name__}"
        )
    source = binding.source
    if not isinstance(source, str):
        return Binding(source, binding.origin)

    def _rename(path: str) -> "str | None":
        parts = path.split(".")
        if len(parts) >= 2 and parts[0] == "input":
            # input.k[.rest] -> <callsite>/<start>.output.k[.rest]
            key = parts[1]
            rest = ".".join(parts[2:])
            new = f"{ns(callsite, START_ID)}.output.{key}"
            return new + (f".{rest}" if rest else "")
        # Node-first: <X>.output[.rest] -> <callsite>/<X>.output[.rest]
        if len(parts) >= 2 and parts[1] == "output":
            new = f"{ns(callsite, parts[0])}.output"
            return new + ("." + ".".join(parts[2:]) if len(parts) > 2 else "")
        return None  # system.X / any other head untouched

    rewritten = rewrite_template_refs(source, _rename)
    return Binding(rewritten, binding.origin)


def _lower_spawner_post_asserts(
    end: Node,
    end_wiring: dict[str, Any],
    end_edges: list[Edge],
    *,
    callsite: str,
    post_asserts: list[Assertion],
    input_names: "set[str]",
) -> None:
    """Stamp a spawner's node-level POST `asserts:` onto the CLONED CHILD END, in place.

    A `call` spawner may carry node-level `asserts:` reading `${output}` (its own committed
    output) and its declared call inputs (bare `${X}`). Since `run()` returns a `Grow`, the
    spawner's `${output}` is only known when the cloned child END commits under the callsite —
    so its POST asserts must ride that END and fire through the ordinary `eval_node` seam. This
    LOWERS each post-assert onto `end`, mirroring build-time
    flow-assert lowering:

    - `${output}` / `${output.field}` is kept unchanged — the cloned END's own record value
      (== the spawner's committed output) is what eval_node injects as `{"output": value}`.
    - each declared call input `${X}` becomes a synth END param (`__assert_input_<X>`) wired
      off the cloned child START's coerced output (`${<callsite>/__start__.output.X:-null}`),
      with a structural `<callsite>/__start__ -> <callsite>/__end__` edge — the SAME shape the
      build-time producer/input lowering uses (see `compose/validate/asserts.py`). Repeated
      refs to the same input SHARE one synth param.

    The synth name is allocated COLLISION-FREE against names already on the cloned END: the
    child def's OWN flow-level post-asserts (lowered at build) may already occupy
    `__assert_input_<X>` on the deep-copied END (via its `params`/`end_wiring`). Mirroring
    build-time `_alloc`, a taken base bumps to `<base>_1`, `<base>_2`, ... so the spawner's
    synth never duplicates a child param decl or its producer->END edge's `input_group`.

    Args:
        end: the cloned child END node (deep-copied, namespaced) — mutated in place.
        end_wiring: the cloned END's wiring dict (`param -> ${...}` source) — mutated in place.
        end_edges: the flow's edge list — synth producer edges are appended.
        callsite: the spawner id (== the clone namespace).
        post_asserts: the spawner's `${output}`-reading node-level asserts (may be empty).
        input_names: the spawner's declared call-input names (its param names).
    """
    if not post_asserts:
        return
    start_ns = ns(callsite, START_ID)
    synth_for_input: dict[str, str] = {}       # input name -> allocated synth END param
    # Names ALREADY on the cloned END (survivors of the deep-copy: the child def's own lowered
    # synth params + record params) — the spawner's synth must not collide with any of them, nor
    # with a name allocated earlier in THIS pass. Mirrors build-time `_alloc`'s collision guard.
    taken: set[str] = {p.name for p in end.params} | set(end_wiring.keys())

    def _alloc(base: str) -> str:
        """Allocate a collision-free synth name from `base`: bump `<base>_1`, `<base>_2`, ...
        until it is free of every name already on the cloned END (and this pass), then reserve
        it. Mirrors the build-time `_alloc` in `compose/validate/asserts.py`."""
        name = base
        i = 1
        while name in taken:
            name = f"{base}_{i}"
            i += 1
        taken.add(name)
        return name

    def _rename(path: str) -> "str | None":
        # Keep `${output[.field]}` (the END's own value); map a declared input `${X[.rest]}` to a
        # synth param bound off the cloned child START's coerced output.
        segs = path.split(".")
        head = segs[0]
        if head == "output":
            return None                        # the END's own output — unchanged
        if head in input_names:
            synth = synth_for_input.get(head)
            if synth is None:
                synth = _alloc(f"{_SYNTH_PREFIX}input_{head}")
                synth_for_input[head] = synth
                end.params.append(Parameter(name=synth))
                # `:-null` keeps the END input edge OPTIONAL so END waits for START to settle
                # and binds None if the value was omitted, instead of co-skipping END to dead
                # (the SAME structural shape build-time flow-assert lowering uses).
                end_wiring[synth] = f"${{{start_ns}.output.{head}:-null}}"
                end_edges.append(Edge(
                    id=f"{start_ns}->{end.id}#{synth}",
                    from_=start_ns, to=end.id, input_group=synth, optional=True))
            return ".".join([synth, *segs[1:]])
        return None                            # anything else — unchanged

    for assertion in post_asserts:
        lowered = rewrite_condition_refs(assertion.expression, _rename)
        end.post_asserts.append(replace(assertion, expression=lowered))


def clone_child(child, callsite: str, record: dict, *,
                origin_context: "tuple[OriginRef, ...]" = (),
                post_asserts: "list[Assertion] | None" = None,
                input_names: "set[str] | None" = None) -> Flow:
    """Splice a child `CompiledFlow`'s `START_ID..END_ID` at `callsite`. Every child node
    (incl. its `START_ID`/`END_ID`) is cloned deep-namespaced; the child `START_ID` is SEEDED with the
    call-args as edges (no baking); the child `END_ID` is the alias filler. Pure — the growth
    reducer applies the returned fragment and updates its dispositions/frontier.

    Returns a `Flow` whose `start_id` is the namespaced child START (the sole seed point) and whose
    `end_id` is the namespaced child END (the alias filler for a call / one map element).

    `origin_context` is the enclosing spawner's ordered failure trace. It is prefixed to each
    clone's existing generated ancestry without deduplication. `post_asserts`/
    `input_names` (default empty) carry a SPAWNER's node-level POST `asserts:`
    — a `call` node may declare `asserts:` reading `${output}` (its committed output) and its
    declared inputs. They are LOWERED onto the cloned child END (via
    `_lower_spawner_post_asserts`) so they fire through the inherited `run` template, NOT a
    bespoke redirect-commit site. Empty for MAP/LOOP and for a CALL without node asserts."""
    nodes: dict[str, Node] = {}
    for nid, node in child.nodes.items():
        clone = copy.deepcopy(node)
        clone.id = ns(callsite, nid)
        clone.origin_context = origin_context + clone.origin_context
        # Re-namespace a self-attribution origin that pointed at THIS node's pre-clone id (a
        # compiled loop `L` inside the child: `origin_id == old id`). The cloned loop is a NEW
        # compiled origin at the namespaced callsite, so its origin must follow to `clone.id`;
        # otherwise its `run`/`commit_as=origin` and body callsite key the un-namespaced id and
        # success-transition preparation cannot resolve that target (a loop nested in CALL/MAP).
        if getattr(node, "origin_id", None) == nid:
            clone.origin_id = clone.id
        nodes[clone.id] = clone

    # Re-namespace EVERY node's wiring (internal ${X.output}/${input.X} re-pointed; no baking).
    wiring: dict[str, dict[str, Any]] = {}
    for nid, w in child.wiring.items():
        wiring[ns(callsite, nid)] = {p: _rens_internal(src, callsite) for p, src in w.items()}

    # Re-key ALL internal edges (incl. START_ID->body and body->END_ID) identically — START_ID/END_ID are
    # ordinary nodes with reserved ids now (no __start__/__end__ sentinel special-cases).
    edges: list[Edge] = []
    for e in child.edges:
        edges.append(Edge(
            id=ns(callsite, e.id),
            from_=ns(callsite, e.from_),
            to=ns(callsite, e.to),
            source_handle=e.source_handle,
            input_group=e.input_group,
            optional=e.optional,
            ordering=e.ordering,
        ))

    # Seed the child START_ID with the call-args AS EDGES: a ${...} forward-ref value mints
    # a producer->START_ID edge; a literal is a constant seed with no edge. The child START_ID is the
    # sole seed point (it OVERRIDES the provisional ${input.X} wiring the loader left on it).
    start_ns = ns(callsite, child.start_id)
    start_wiring = wiring.setdefault(start_ns, {})
    start_wiring.clear()                       # drop the provisional `{name: ${input.name}}`
    for param, value in record.items():
        start_wiring[param] = value
        if isinstance(value, str) and "${" in value:
            producer = _producer_of(value)
            if producer is not None:
                edges.append(Edge(
                    id=f"{producer}->{start_ns}#0",
                    from_=producer,
                    to=start_ns,
                    input_group=param,
                ))

    # The child END_ID is the alias filler (its producer->END_ID edges are already re-keyed above).
    out_id = ns(callsite, child.end_id)

    # The child END's POST asserts (the def's flow-level `asserts:` lowered onto END at build) ride
    # the deep-copied node UNCHANGED — their synth params, re-namespaced wiring (`${<callsite>/pid
    # .output:-null}`), and re-keyed producer->END edges all survive the clone above. Each synth
    # param binds its producer's value INTO the cloned END's record (the assertion expression
    # reads the node-local `${__assert_*}` head), so the END wrapper fires them record-only. No re-homing
    # needed (and none possible without deleting the structural synth edges that gate the assert on
    # its producer).
    #
    # A SPAWNER's OWN node-level POST asserts (a `call` node's `asserts:`) are LOWERED onto the
    # cloned END here (net-new — the child does not carry them): `${output}` stays, a declared call
    # input `${X}` becomes a synth END param wired off the cloned child START's coerced output. They
    # then fire through the SAME END post-assert seam (the node's `run` wrapper) as the def's own asserts.
    _lower_spawner_post_asserts(
        nodes[out_id], wiring.setdefault(out_id, {}), edges,
        callsite=callsite,
        post_asserts=post_asserts or [],
        input_names=set(input_names or ()),
    )
    return Flow(nodes=nodes, edges=edges, wiring=wiring, start_id=start_ns, end_id=out_id)


def _producer_of(src: str) -> Optional[str]:
    """The producer node id of a forward-ref record value, else None.
    Singular only: `${<producer>.output[.…]}`."""
    whole = _whole_span(src)
    if whole is None:
        return None
    parts = whole.split(".")
    if len(parts) >= 2 and parts[1] == "output":
        return parts[0]
    return None


def call_subgraph(child, callsite: str, record: dict, *,
                  origin_context: "tuple[OriginRef, ...]" = (),
                  post_asserts: "list[Assertion] | None" = None,
                  input_names: "set[str] | None" = None) -> Flow:
    """The pure CALL builder: the self-describing fragment a CALL spawner grows into.

    Wraps `clone_child` (the deep-namespaced clone of the child's `START_ID..END_ID` at `callsite`,
    seeded with `record`), bakes `commit_as=callsite` on the cloned child END filler (so its Output
    commits under the spawner id on the ordinary success path), and returns a `Flow` whose `start_id`
    is the single cloned child START and whose `end_id` is the cloned child END filler. The child's
    asserts (a def's `asserts:` lowered onto its START/END at build) ride the cloned boundary nodes
    and fire through the ordinary `Node.run` wrapper — boundary asserts after the cloned
    `StartNode._run_impl()` coerces its record, and post asserts after the cloned
    `EndNode._run_impl()` returns `Output`.

    `post_asserts`/`input_names` carry the SPAWNER's OWN node-level POST `asserts:`
    (a `call` node's `asserts:` reading `${output}` + declared inputs). They are lowered onto the
    cloned child END by `clone_child`, so the END's ordinary `Node.run` wrapper evaluates them.
    `origin_context` is the spawner's outer-to-inner failure trace and is prefixed to every clone,
    including both child boundaries. The CALL node passes its ordered assertion carriers and
    declared-input names; durable `build_grow` uses the same inputs so a resumed run's rebuilt
    child END is identical."""
    cloned = clone_child(child, callsite=callsite, record=record,
                         origin_context=origin_context,
                         post_asserts=post_asserts,
                         input_names=input_names)
    cloned.nodes[cloned.end_id].commit_as = callsite
    return Flow(nodes=cloned.nodes, edges=cloned.edges, wiring=cloned.wiring,
                start_id=cloned.start_id, end_id=cloned.end_id)


def map_subgraph(
    child,
    spawner_id: str,
    records: list,
    *,
    origin_context: "tuple[OriginRef, ...]" = (),
) -> Flow:
    """The pure MAP builder: the self-describing fragment a MAP spawner grows into.

    MAP is CALL × N plus a synthesized `map#/__start__` fan-out and a list-`END` fan-in. For each
    element `i`, `clone_child` deep-namespaces the child at `map_callsite(spawner_id, i)`
    (`f"{spawner}#{i}"`), seeded with element `i`'s `record`. A single synthetic `map#/__start__`
    StartNode (`ns(spawner_id, START_ID)`, empty decls — it runs once and emits `{}`) fans out to
    every element's namespaced child START via an ORDERING edge (`ordering=True, optional=False`):
    a data edge with `input_group=None` would form a phantom data group, so the fan-out is pure
    ordering. Element STARTs have zero real incoming data edges (MAP records are resolved values), so
    the ordering edge is their sole gate; when `map#/__start__` runs (always TAKEN — it has no
    predecessors), each element START's disposition becomes `ready`, mirroring the top-level
    `__start__ → body-root` seed. The N child END fillers fan into ONE `EndNode.list_(map_end_id, n=N)`
    (`map_end_id = ns(spawner_id, END_ID)`) via one `e{i}` input group per element. The list-END
    carries `commit_as=spawner_id`, so its list Output commits under the spawner on the success path.

    Returns a `Flow` with `start_id = ns(spawner_id, START_ID)` (the synthetic start) and
    `end_id = ns(spawner_id, END_ID)` (the list collector).

    `origin_context` is the spawner's outer-to-inner failure trace. Every element clone and both
    generated fan-out/fan-in boundaries receive that same authored context.

    N=0 (empty map): the only body node is the list-END; `map#/__start__ → map#/__end__` is wired
    with the same ordering edge so the collector still schedules and emits `[]`."""
    map_start_id = ns(spawner_id, START_ID)              # the synthetic fan-out start
    map_end_id = ns(spawner_id, END_ID)                  # the END_ID-list filler
    nodes: dict[str, Node] = {}
    edges: list[Edge] = []
    wiring: dict[str, dict[str, Any]] = {}
    end_wiring: dict[str, str] = {}
    end_edges: list[Edge] = []
    start_edges: list[Edge] = []                         # map_start -> each element start (ordering)

    for i, record in enumerate(records):
        cloned = clone_child(
            child,
            callsite=map_callsite(spawner_id, i),
            record=record,
            origin_context=origin_context,
        )
        nodes.update(cloned.nodes)
        edges.extend(cloned.edges)
        wiring.update(cloned.wiring)
        # Ordering fan-out: map_start -> this element's START (the element's sole incoming gate).
        start_edges.append(Edge(
            id=f"{map_start_id}->{cloned.start_id}#0",
            from_=map_start_id, to=cloned.start_id, ordering=True))
        end_wiring[f"e{i}"] = f"${{{cloned.end_id}.output}}"   # node-first
        end_edges.append(Edge(
            id=f"{cloned.end_id}->{map_end_id}#{i}",
            from_=cloned.end_id, to=map_end_id, input_group=f"e{i}"))

    # The synthetic fan-out start: a StartNode with no parameters — it runs once, emits `{}`, and its
    # out-edges make every element START ready (the single-start entry, like CALL and the top level).
    map_start = StartNode(map_start_id, params=[])
    map_start.origin_context = origin_context
    nodes[map_start_id] = map_start
    wiring[map_start_id] = {}

    # ONE EndNode in LIST mode — the MAP fan-in over the N child ENDs (still built + stamped at
    # N=0). Bake the commit redirect so its list Output commits under the spawner id.
    map_end = EndNode.list_(map_end_id, n=len(records))
    map_end.commit_as = spawner_id
    map_end.origin_context = origin_context
    nodes[map_end_id] = map_end
    edges.extend(start_edges)
    edges.extend(end_edges)
    wiring[map_end_id] = end_wiring
    # N=0: no element clones, so the list-END has 0 incoming data edges. Wire the synthetic start to
    # it (same ordering edge) so the collector schedules off the fan-out and emits [].
    if not records:
        edges.append(Edge(
            id=f"{map_start_id}->{map_end_id}#0",
            from_=map_start_id, to=map_end_id, ordering=True))

    return Flow(nodes=nodes, edges=edges, wiring=wiring,
                start_id=map_start_id, end_id=map_end_id)


def loop_continue_subgraph(
    child,
    origin: str,
    carried: dict,
    k: int,
    driver,
    *,
    origin_context: "tuple[OriginRef, ...]" = (),
) -> Flow:
    """The pure LOOP CONTINUE builder: body_k + the fresh next-iteration driver.

    `origin` is the ORIGIN loop id `L` (NOT the running driver id) — bodies are always keyed
    on the origin so live construction (from `L~k`) and durable construction (from `L`) build the SAME
    `L#k/…` namespace. `k` is THIS iteration's index; `carried` seeds body_k. `driver` is the
    fresh `L~(k+1)` LoopNode the node minted via `respawn(k+1)`.

    Splices: body_k (`clone_child` at `map_callsite(origin, k)`, NO baked `commit_as` — its
    Output commits under its own id `L#k/END` and feeds the next driver by plain wiring), the
    driver `L~(k+1)`, the producer edge `body_k.END -> L~(k+1)` (one input_group per carried
    field, or one ordering edge for an empty record), and the driver's wiring
    `{field: "${L#k/END.output.field}"}`. The returned `Flow`'s `start_id` is body_k's START
    (the sole seed point) and `end_id` is body_k's END; the driver is scheduled by normal
    readiness when the edge fires.

    Unlike `call_subgraph`/`map_subgraph`, the body-END carries NO `commit_as`: the CONTINUE arm
    never commits under the origin (only the STOP arm's `Output(carried, commit_as=origin)` does),
    so `_derived_terminals` returns `[]` for a continue grow and the terminal stamping no-ops.

    `origin_context` is the running driver's complete failure trace and prefixes only the body
    clones. The supplied next driver already owns its copied outer context; its authored LOOP
    origin remains local and is appended by `Node.failure_trace`."""
    # The loop's carried record may be wider than the body domain: the authoring
    # contract permits body inputs to be a subset of carried fields. Seed only those
    # declared child inputs into its START, while the full record continues through
    # the next driver below.
    body_input_names = {
        param.name for param in (child.nodes[child.start_id].params or [])
    }
    body_seed = {
        name: value for name, value in carried.items() if name in body_input_names
    }
    cloned = clone_child(
        child,
        callsite=map_callsite(origin, k),
        record=body_seed,
        origin_context=origin_context,
    )
    body_end = cloned.end_id                               # ns(map_callsite(origin, k), END_ID)
    nodes = dict(cloned.nodes)
    edges = list(cloned.edges)
    wiring = dict(cloned.wiring)
    nodes[driver.id] = driver
    # Wire the fresh driver's carried params off body_k.END by field name (the carried record is
    # the body codomain == the driver's params). One producer edge per field.
    driver_wiring: dict[str, str] = {}
    driver_edges: list[Edge] = []
    for i, field_name in enumerate(carried.keys()):
        driver_wiring[field_name] = f"${{{body_end}.output.{field_name}}}"
        driver_edges.append(Edge(
            id=f"{body_end}->{driver.id}#{i}",
            from_=body_end, to=driver.id, input_group=field_name))
    if not carried:
        driver_edges.append(Edge(
            id=f"{body_end}->{driver.id}#0",
            from_=body_end, to=driver.id, ordering=True))
    wiring[driver.id] = driver_wiring
    edges.extend(driver_edges)
    return Flow(nodes=nodes, edges=edges, wiring=wiring,
                start_id=cloned.start_id, end_id=body_end)


def _growth_record(seed: Any, *, label: str) -> dict[str, Any]:
    """Require one exact string-keyed record from a legal spawner seed."""
    if type(seed) is not dict:
        error = TypeError(f"{label} growth seed must be a record")
        raise GrowthRejected(error) from error
    if any(type(key) is not str for key in seed):
        error = TypeError(f"{label} growth seed record keys must be strings")
        raise GrowthRejected(error) from error
    if any(not key for key in seed):
        error = ValueError(f"{label} growth seed record keys must be non-empty")
        raise GrowthRejected(error) from error
    return seed


def _map_growth_records(seed: Any) -> list[dict[str, Any]]:
    """Require MAP's exact ordered list of per-item records."""
    if type(seed) is not list:
        error = TypeError("MAP growth seed must be a list of records")
        raise GrowthRejected(error) from error
    return [
        _growth_record(record, label=f"MAP item {index}")
        for index, record in enumerate(seed)
    ]


def _loop_growth_seed(seed: Any) -> tuple[dict[str, Any], int]:
    """Require LOOP's carried record and exact non-negative iteration index."""
    if type(seed) not in (list, tuple) or len(seed) != 2:
        error = ValueError(
            "LOOP growth seed must be a two-item (record, iteration) pair"
        )
        raise GrowthRejected(error) from error
    carried = _growth_record(seed[0], label="LOOP carried")
    iteration = seed[1]
    if type(iteration) is not int:
        error = TypeError("LOOP growth seed iteration must be an integer")
        raise GrowthRejected(error) from error
    if iteration < 0:
        error = ValueError("LOOP growth seed iteration must be non-negative")
        raise GrowthRejected(error) from error
    return carried, iteration


def build_grow(spawner: Node, canonical_seed: Any) -> Flow:
    """Build the deterministic fragment described by one spawner's canonical seed.

    This is the sole kind-aware growth-construction path. Nodes calculate only their pure seed;
    the package-root run boundary injects this callable into the kind-blind runtime for both live
    growth and durable replay.

    Args:
        spawner: The CALL, MAP, or LOOP node whose seed is being interpreted.
        canonical_seed: The persisted builder input produced by that node. LOOP accepts the live
            tuple and its JSON-round-tripped two-element list form.

    Returns:
        The self-contained `Flow` fragment the runtime splices into its private live graph.

    Raises:
        GrowthRejected: If a legal spawner kind produced a malformed seed.
        ValueError: If `spawner` is not one of the closed legal growth kinds.
    """
    match spawner.kind:
        case NodeKind.CALL:
            record = _growth_record(canonical_seed, label="CALL")
            return call_subgraph(
                spawner.child,
                callsite=spawner.id,
                record=dict(record),
                origin_context=spawner.failure_trace(),
                post_asserts=spawner.post_asserts,
                input_names={param.name for param in (spawner.params or [])},
            )
        case NodeKind.MAP:
            records = _map_growth_records(canonical_seed)
            return map_subgraph(
                spawner.child,
                spawner_id=spawner.id,
                records=[dict(record) for record in records],
                origin_context=spawner.failure_trace(),
            )
        case NodeKind.LOOP:
            carried, iteration = _loop_growth_seed(canonical_seed)
            return loop_continue_subgraph(
                spawner.child,
                spawner.origin_id,
                dict(carried),
                iteration,
                spawner.respawn(iteration + 1),
                origin_context=spawner.failure_trace(),
            )
        case _:
            raise ValueError(
                f"node {spawner.id!r} ({spawner.kind.value}) cannot build graph growth"
            )
