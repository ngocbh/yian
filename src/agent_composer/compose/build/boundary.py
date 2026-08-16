"""DAG boundary synthesis: `START_ID`/`END_ID` nodes + their edges over ALL edges.

The `build_leaf_node` funnel is dissolved: every node kind now builds through
its own `descriptor.build(ctx) -> BuildResult` (see `compose/nodes/<kind>.py`), and the
conductor loop drives them kind-blind. What remains here is the boundary synthesis —
`synthesize_boundary_graph` — which is representation-neutral graph plumbing, not per-kind
build.
"""

from __future__ import annotations

from typing import Any, Optional

from agent_composer.compile.model import END_ID, START_ID, Edge, FlowOutput
from agent_composer.expr import binding_co_skips
from agent_composer.nodes.base import Node
from agent_composer.nodes.contracts import Parameter, binding_source
from agent_composer.nodes.end import EndNode
from agent_composer.nodes.start import StartNode
from agent_composer.compose.edges.producers import _binding_producers
# Forward edge only: `validate/asserts.py` (and its own deps) never import this module,
# so importing the lowered-asserts carrier here adds no import cycle.
from agent_composer.compose.validate.asserts import LoweredAsserts


# --------------------------------------------------------------------------- #
# DAG assembly: root/`__start__` + terminal/`__end__` over ALL edges
#
# Runs AFTER both the data edges and the `case` control edges are
# reconciled — so a `case` branch target carrying only a `gate -> target` control
# edge is correctly demoted from root. A root is a node with NO incoming edge of ANY
# kind (data OR control); the engine seeds every root unconditionally, so a wrongly-
# marked root makes BOTH branches of a `case` run. The flow `outputs:` bindings give
# the terminals: each `${P.output[...]}` producer gets a `P -> __end__` edge (a
# coalesce flow-output -> one terminal edge per producer; only the taken branch's
# value survives, resolved by `terminal_output`).
# --------------------------------------------------------------------------- #


def synthesize_boundary_graph(
    input_parameters: list[Parameter],
    outputs: list[FlowOutput],
    node_ids: "set[str]",
    body_edges: list[Edge],
    start_field_ranges: "Optional[dict]" = None,
    lowered: "Optional[LoweredAsserts]" = None,
) -> tuple[dict[str, Node], list[Edge], dict[str, dict[str, Any]]]:
    """Synthesize the `START_ID`/`END_ID` boundary nodes + their edges + wiring.

    Every flow becomes `START_ID -> body -> END_ID` with ORDINARY edges (the `__start__`/`__end__`
    sentinels are retired — the strings persist only as the reserved node ids). Returns:

    - `boundary_nodes`: `{START_ID: StartNode, END_ID: EndNode(record)}`.
    - `edges`: the END_ID producer edges (one per producer per declared output, `input_group`
      = the output name, `optional = not binding_co_skips` so a coalesce/`:-`/`:?` output
      never co-skips) + the lowered-assert synth END edges (one per producer per synthesized
      END input, same numbering scheme so ids never collide) + the START_ID root edges (a bare
      `START_ID->nid` for every body node with no incoming of any kind, over the FULL set
      incl. the END_ID producer AND synth edges) + a bare `START_ID->END_ID` root edge for a
      0-output END_ID with no synth edges (so the 0-output END_ID runs and commits `None`).
    - `boundary_wiring`: parity-clean wiring for both — `START_ID` keyed by input name
      (provisional `${input.X}` source so `check_wiring_parity` passes; the engine seeds
      START_ID's record directly at runtime), `END_ID` keyed by output name -> the output's
      `from_`, merged with the lowered synth wiring (`<synth> -> ${pid.output:-null}` / `${input.X}`).

    The START_ID-side `${input.X}`-producer DATA edges + the demote-input-reader-from-root
    rule live here too (every input read resolves through the START_ID namespace).

    `start_field_ranges` (optional) is the legacy flow-boundary input-declaration
    source map stamped onto StartNode (`("input", name) -> SourceRange`). Runtime
    diagnosis resolves Parameter origins instead. Assertion ranges travel in the
    positional lists on `lowered`. `None` for a spanless flow (compact / in-memory / def).

    `lowered` (optional) is the flow's `asserts:` lowered into node-local `LoweredAsserts`.
    When given, its PRE (boundary) asserts are stamped on START's ordinary
    `post_asserts` (they read `${output.X}` — START's coerced output), its POST asserts on END
    (`end.post_asserts`), each synthesized END input becomes an extra `Parameter` on END plus a
    wiring entry and a producer edge — so a cross-node post assert's `${producer.output}`
    dependency is STRUCTURAL (a real END input) and survives child cloning. `None` leaves the
    boundary nodes assert-free (a def / in-memory flow with no flow-level asserts).
    """
    start = StartNode(
        START_ID,
        params=input_parameters,
        field_ranges=start_field_ranges,
    )
    end = EndNode.record(END_ID, output_names=[o.name for o in outputs])
    boundary_nodes: dict[str, Node] = {START_ID: start, END_ID: end}

    # END_ID producer edges — one per producer per output, keyed by the output NAME, with the
    # coalesce/optional stance (reuses the infer_data_edges emit_for convention).
    end_edges: list[Edge] = []
    counts: dict[tuple[str, str], int] = {}
    for o in outputs:
        optional = not binding_co_skips(binding_source(o.binding))
        for producer in _binding_producers(o.binding):
            i = counts.get((producer, END_ID), 0)
            counts[(producer, END_ID)] = i + 1
            end_edges.append(Edge(id=f"{producer}->{END_ID}#{i}", from_=producer, to=END_ID,
                                  input_group=o.name, optional=optional))

    # Stamp the lowered flow-level asserts onto the boundary nodes (Node is not frozen —
    # validate_node_asserts mutates the assert slots the same way).
    if lowered is not None:
        # START fires its boundary asserts through the ORDINARY `post_asserts` seam: the lowered
        # PRE asserts read `${output.X}` (see asserts._pre_rename), so the node's `run` wrapper fires
        # them against `{**inputs, "output": <START's coerced record>}` — the coerced/defaulted
        # value, not the wire string. This unifies START with every other node (no bespoke slot);
        # eval_node stays kind-blind. END fires its POST asserts through the same seam; any
        # `${producer.output}` head is lowered to a synth END param (see below) that binds the
        # producer's value INTO END's record, so the assert fires record-only. Extend-safe: START
        # carries no post_asserts today, but merge rather than assign so a future START-owned
        # post-assert is not clobbered.
        start.post_asserts = list(start.post_asserts) + list(lowered.start_pre)
        end.post_asserts = list(lowered.end_post)
        # Legacy positional ranges remain aligned 1:1 with lowered assertions while the
        # compatibility fields exist. Runtime diagnosis resolves each Assertion.origin instead.
        start.post_assert_ranges = list(start.post_assert_ranges) + list(lowered.pre_ranges)
        end.post_assert_ranges = list(lowered.post_ranges)
        # Each synthesized END input is an extra param + a wiring source + a producer edge, so
        # the value the post-assert reads is present in END's record when eval_node fires it.
        # EndNode.run() reads only its declared output names, so extra params are inert there.
        end.params = list(end.params) + list(lowered.synth_params)
        # Continue the SAME per-(producer, END) counter as the output-edge loop so a producer
        # feeding both an output and a synth input never mints a colliding edge id. An `${input.X}`
        # synth is a pure ref -> binding_co_skips True -> optional=False (a required START->END
        # edge; START always runs). A `${pid.output:-null}` synth -> binding_co_skips False ->
        # optional=True: END waits for the producer to settle, binding None if it was skipped.
        for synth_name, src in lowered.synth_wiring.items():
            optional = not binding_co_skips(binding_source(src))
            for producer in _binding_producers(src):
                i = counts.get((producer, END_ID), 0)
                counts[(producer, END_ID)] = i + 1
                end_edges.append(Edge(id=f"{producer}->{END_ID}#{i}", from_=producer, to=END_ID,
                                      input_group=synth_name, optional=optional))
    # START_ID root edges over the FULL set (body + END_ID producer edges): a body node with no
    # incoming of any kind is a bare root. A 0-output END_ID (no producer edge) is itself a root.
    full = body_edges + end_edges
    has_incoming = {e.to for e in full}
    root_edges = [Edge(id=f"{START_ID}->{nid}", from_=START_ID, to=nid)
                  for nid in sorted(node_ids) if nid not in has_incoming]
    if END_ID not in has_incoming:
        root_edges.append(Edge(id=f"{START_ID}->{END_ID}", from_=START_ID, to=END_ID))

    boundary_wiring: dict[str, dict[str, Any]] = {
        # START_ID wiring: input names -> `${input.X}` sources. At runtime the engine schedules
        # START through eval_node, priming store[START_ID] with the raw run-args record and
        # consulting a PER-ENGINE override that narrows this compiled wiring to only the passed
        # keys (an omitted input then binds absent so START's wrapper fills its declared default).
        # This compiled entry is what a direct-driver / hand-seed test (no run_inputs) reads.
        START_ID: {
            parameter.name: f"${{input.{parameter.name}}}"
            for parameter in input_parameters
        },
        END_ID: {o.name: o.binding for o in outputs},
    }
    # Merge each synthesized END param's source so bind_params finds a source per synth param
    # (key parity with the extra params + edges above). A synth name never collides with an
    # output name (it is `__assert_`-prefixed; authors cannot write `__` ids).
    if lowered is not None:
        boundary_wiring[END_ID].update(lowered.synth_wiring)
    return boundary_nodes, end_edges + root_edges, boundary_wiring
