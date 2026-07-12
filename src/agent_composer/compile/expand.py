"""Runtime graph-expansion machinery — the pure half.

When a spawner (REF / MAP / agent-pause) runs, it does not run a child engine; it
returns a *description* (`Grow(Flow)`) and the engine GROWS the live graph by cloning the
target child(ren) deep-namespaced into the running `CompiledFlow`. This module holds the
**pure** machinery that growth keys off:

- `ns` / `map_callsite` / `ask_resume_edge_id`: deterministic id minting. Every
  cloned node/edge id is a pure function of `(callsite, child static id, element index)` —
  NO emission counter — so a re-clone on kill-recovery re-keys identically.

The pure cloner (`clone_child`) splices the child's own
`START_ID..END_ID` (every flow is `START_ID -> body -> END_ID`) into a `Flow`: the child `START_ID`
is the alias-seed point — SEEDED WITH THE CALL-ARGS AS EDGES (no `_rens` literal-baking) — and the
child `END_ID` is the alias filler. A child node reading `${input.X}` is re-pointed to the namespaced
child START_ID's output object (`${<callsite>/<start>.output.X}`); the dispatcher
consumes the descriptions and performs the (impure) `add_subgraph` + `register` + seed.

Layer: compile — imports `nodes`/`model`/`expr` (ladder-legal); never `runtime`.
"""

from __future__ import annotations

import copy
from typing import Any, Optional

from agent_composer.compile.model import Edge, END_ID, Flow, START_ID
from agent_composer.expr import rewrite_template_refs
from agent_composer.nodes.base import Node
from agent_composer.nodes.end.node import EndNode
from agent_composer.nodes.start.node import StartNode


def ns(callsite: str, child_id: str) -> str:
    """Namespace a child node/edge id under its callsite: `<callsite>/<child_id>`.
    `callsite` = spawner id (REF/agent) or `f"{spawner}#{i}"` (MAP element i); nests."""
    return f"{callsite}/{child_id}"


def map_callsite(spawner_id: str, i: int) -> str:
    """The per-element callsite for MAP element `i`: `f"{spawner}#{i}"`."""
    return f"{spawner_id}#{i}"


def ask_resume_edge_id(callsite: str) -> str:
    """The agent continuation edge id: `f"{callsite}/__ask_resume#0"`."""
    return f"{callsite}/__ask_resume#0"


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


def _rens_internal(src: Any, callsite: str) -> Any:
    """Re-namespace one binding source under `callsite` — NO baking.

    Singular only:
    - `${input.<k>...}`   -> `${<callsite>/<start>.output.<k>...}` (namespaced child
      START_ID's output object read via the node-first head)
    - `${<X>.output...}`  -> `${<callsite>/<X>.output...}` (node-first re-namespaced in place)
    - `${system.X}` (run-global) and other heads untouched.

    Legacy plural heads (`outputs.X` / `inputs.X`) are rejected at parse time.

    Routes every `${...}` span through `rewrite_template_refs` + the ONE parse tree, so an
    embedded / coalesce / whole-expression span is re-namespaced leaf-by-leaf (the old flat
    `${...}` regex only handled a simple single-path span)."""
    if not isinstance(src, str):
        return src

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

    return rewrite_template_refs(src, _rename)


def clone_child(child, callsite: str, record: dict) -> Flow:
    """Splice a child `CompiledFlow`'s `START_ID..END_ID` at `callsite`. Every child node
    (incl. its `START_ID`/`END_ID`) is cloned deep-namespaced; the child `START_ID` is SEEDED with the
    call-args as edges (no baking); the child `END_ID` is the alias filler. Pure — the dispatcher
    performs the impure `add_subgraph`/`register`/seed.

    Returns a `Flow` whose `start_id` is the namespaced child START (the sole seed point) and whose
    `end_id` is the namespaced child END (the alias filler for REF / one element input for MAP)."""
    nodes: dict[str, Node] = {}
    for nid, node in child.nodes.items():
        clone = copy.deepcopy(node)
        clone.id = ns(callsite, nid)
        # Re-namespace a self-attribution origin that pointed at THIS node's pre-clone id (a
        # compiled loop `L` inside the child: `origin_id == old id`). The cloned loop is a NEW
        # compiled origin at the namespaced callsite, so its origin must follow to `clone.id`;
        # otherwise its `run`/`commit_as=origin` and body callsite key the un-namespaced id and
        # `_on_success` cannot find `flow.nodes[origin]` (a loop nested in a CALL/MAP).
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
    # .output:-null}`), and re-keyed producer->END edges all survive the clone above, so eval_node
    # fires them on the cloned END with pool-fallback resolution. No re-homing needed (and none
    # possible without deleting the structural synth edges that gate the assert on its producer).
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


def call_subgraph(child, callsite: str, record: dict) -> Flow:
    """The pure CALL builder: the self-describing fragment a CALL spawner grows into.

    Wraps `clone_child` (the deep-namespaced clone of the child's `START_ID..END_ID` at `callsite`,
    seeded with `record`), bakes `commit_as=callsite` on the cloned child END filler (so its Output
    commits under the spawner id on the ordinary success path), and returns a `Flow` whose `start_id`
    is the single cloned child START and whose `end_id` is the cloned child END filler. The child
    boundary asserts are re-derived engine-side (the engine reads them off the spawner node), so they
    are not carried on the returned `Flow`."""
    cloned = clone_child(child, callsite=callsite, record=record)
    cloned.nodes[cloned.end_id].commit_as = callsite
    return Flow(nodes=cloned.nodes, edges=cloned.edges, wiring=cloned.wiring,
                start_id=cloned.start_id, end_id=cloned.end_id)


def map_subgraph(child, spawner_id: str, records: list) -> Flow:
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
        cloned = clone_child(child, callsite=map_callsite(spawner_id, i), record=record)
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

    # The synthetic fan-out start: a StartNode with empty decls — it runs once, emits `{}`, and its
    # out-edges make every element START ready (the single-start entry, like CALL and the top level).
    nodes[map_start_id] = StartNode(map_start_id, input_decls=[])
    wiring[map_start_id] = {}

    # ONE EndNode in LIST mode — the MAP fan-in over the N child ENDs (still built + stamped at
    # N=0). Bake the commit redirect so its list Output commits under the spawner id.
    map_end = EndNode.list_(map_end_id, n=len(records))
    map_end.commit_as = spawner_id
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


def loop_continue_subgraph(child, origin: str, carried: dict, k: int, driver) -> Flow:
    """The pure LOOP CONTINUE builder: body_k + the fresh next-iteration driver.

    `origin` is the ORIGIN loop id `L` (NOT the running driver id) — bodies are always keyed
    on the origin so live `run` (on `L~k`) and durable `replay_grow` (on `L`) build the SAME
    `L#k/…` namespace. `k` is THIS iteration's index; `carried` seeds body_k. `driver` is the
    fresh `L~(k+1)` LoopNode the node minted via `respawn(k+1)`.

    Splices: body_k (`clone_child` at `map_callsite(origin, k)`, NO baked `commit_as` — its
    Output commits under its own id `L#k/END` and feeds the next driver by plain wiring), the
    driver `L~(k+1)`, the producer edge `body_k.END -> L~(k+1)` (one input_group per carried
    field), and the driver's wiring `{field: "${L#k/END.output.field}"}`. The returned `Flow`'s
    `start_id` is body_k's START (the sole seed point) and `end_id` is body_k's END; the driver is
    scheduled by normal readiness when the edge fires.

    Unlike `call_subgraph`/`map_subgraph`, the body-END carries NO `commit_as`: the CONTINUE arm
    never commits under the origin (only the STOP arm's `Output(carried, commit_as=origin)` does),
    so `_derived_terminals` returns `[]` for a continue grow and the terminal stamping no-ops."""
    cloned = clone_child(child, callsite=map_callsite(origin, k), record=dict(carried))
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
    wiring[driver.id] = driver_wiring
    edges.extend(driver_edges)
    return Flow(nodes=nodes, edges=edges, wiring=wiring,
                start_id=cloned.start_id, end_id=body_end)



# --------------------------------------------------------------------------- #
# clone_continuation_pair — the agent-pause continuation cloner
# --------------------------------------------------------------------------- #


def clone_continuation_pair(pair, callsite: str, *, output_type=None, retries: int = 2,
                            max_tool_iterations: Optional[int] = None) -> Flow:
    """Materialize the agent-pause continuation PAIR namespaced at `callsite`.

    `pair` is `[human_input_desc, resume_desc]` from `agent_step`'s continuation `Grow`. The
    `human_input` leaf is the `start_id` (no incoming edge), so the engine's leaf-pause path applies
    and its `HumanInputRequired.node_id` is the namespaced `hi_id`. The resume node is an `AgentNode`
    with a `Resume` entry (the continuation arm — same `kind = AGENT`, no separate kind); it
    reads the human's `answer` via the BARE forward-ref `${<hi_id>.output}` bound to
    its single `answer` param; the data edge for that ref is synthesized via the SAME producer
    derivation `clone_child`/`compose/edges/data.py` use (`f"{producer}->{consumer}#{i}"`), so the pool ref
    and the edge agree on `hi_id`. Returns a `Flow` (`start_id = hi_id`, `end_id = resume_id`). Pure
    — the dispatcher performs the impure append/register/seed.

    `output_type`/`retries` carry the SPAWNER's declared output Type and self-correction cap
    onto the resume node so a resumed agent with a non-text `output:` still emits the declared
    type on its final turn (the dispatcher reads them off `flow.nodes[spawner_id]`; they are
    not serialized — restore re-grows from the compiled spawner). For a multi-pause chain each
    resume node becomes the next segment's spawner, so the type propagates segment to segment.

    `max_tool_iterations` carries the spawner's per-node tool-iteration cap (from its `env:`)
    onto the resume node's own `env` so a resumed agent keeps the same cap across the pause; a
    `None` falls back to the module default `MAX_TOOL_ITERATIONS`."""
    from agent_composer.nodes.agent.node import AgentNode, Resume
    from agent_composer.nodes.agent.modes.tool_calling import MAX_TOOL_ITERATIONS
    from agent_composer.nodes.human_input import HumanInputNode

    if max_tool_iterations is None:
        max_tool_iterations = MAX_TOOL_ITERATIONS
    hi_desc, resume_desc = pair
    hi_id = ns(callsite, hi_desc["node_id"])              # e.g. agent/__ask#q1
    resume_id = ns(callsite, "__resume#" + hi_desc["slot"])

    hi_node = HumanInputNode(hi_id, prompt=hi_desc["prompt"])
    resume_node = AgentNode(
        resume_id,
        entry=Resume(
            memo=resume_desc["memo"],
            iterations=resume_desc["iterations"],
            pending=resume_desc["pending"],
        ),
        llm_config=resume_desc.get("llm_config"),
        tools=resume_desc.get("tools"),
        controls=resume_desc.get("controls"),
        mode=resume_desc.get("mode", "tool_calling"),
        retries=retries,
    )
    # The Resume entry's declared output Type is set as node DATA (AgentNode.__init__ takes no
    # output_type param — the compiler stamps it; here the dispatcher supplies the spawner's).
    resume_node.output_type = output_type
    # Carry the per-node tool-iteration cap forward so the resumed loop uses the same bound.
    resume_node.env = {"max_tool_iterations": max_tool_iterations}

    # Rewrite the answer forward-ref to the NAMESPACED node-first ref.
    answer_ref = f"${{{hi_id}.output}}"
    producer = _producer_of(answer_ref)                  # == hi_id
    assert producer is not None                          # a well-formed `${id.output}` always resolves
    edge = Edge(
        id=f"{producer}->{resume_id}#0",                 # same shape as compose/edges/data.py's producer edges
        from_=producer,
        to=resume_id,
        input_group="answer",
    )

    return Flow(
        nodes={hi_id: hi_node, resume_id: resume_node},
        edges=[edge],
        wiring={hi_id: {}, resume_id: {"answer": answer_ref}},
        start_id=hi_id,
        end_id=resume_id,
    )


def agent_segment_subgraph(pair, callsite: str, *, output_type=None, retries: int = 2,
                           max_tool_iterations: Optional[int] = None) -> Flow:
    """The pure AGENT-pause builder: the continuation fragment an AGENT grows into when it pauses.

    Wraps `clone_continuation_pair` (the deep-namespaced clone of the `[human_input, resume]` PAIR
    at `callsite`) and returns a `Flow`. Its `start_id` is the `human_input` leaf (a 0-incoming
    entry, so the engine's leaf-pause path applies and the `HumanInputRequired.node_id` is the
    namespaced `hi_id`); its `end_id` is the resume terminal. Every cloned id is `ns(callsite, …)`-
    prefixed (`hi_id = ns(callsite, hi_desc["node_id"])`, `resume_id = ns(callsite,
    "__resume#" + hi_desc["slot"])`).

    Bakes a PROVISIONAL `commit_as=callsite` on the resume terminal. This is NOT the final
    commit target: a multi-pause chain routes the FINAL non-pausing Output back to the ORIGINAL
    spawner, so the engine residual OVERRIDES this provisional value with the true origin (read
    off the previous segment's baked `commit_as`). The builder is pure and cannot see the prior
    segment, so it bakes the local callsite and lets the engine chain the origin.

    `output_type`/`retries` carry the SPAWNER's declared output Type + self-correction cap onto
    the resume node (see `clone_continuation_pair`), so a resumed agent with a non-text `output:`
    still emits the declared type on its final turn and the type propagates segment to segment.
    `max_tool_iterations` carries the spawner's per-node tool-iteration cap the same way, so the
    resumed loop keeps the same bound across the pause."""
    cloned = clone_continuation_pair(pair, callsite=callsite, output_type=output_type, retries=retries,
                                     max_tool_iterations=max_tool_iterations)
    cloned.nodes[cloned.end_id].commit_as = callsite   # PROVISIONAL: the engine residual overrides it to the true origin
    return Flow(nodes=cloned.nodes, edges=cloned.edges, wiring=cloned.wiring,
                start_id=cloned.start_id, end_id=cloned.end_id)
