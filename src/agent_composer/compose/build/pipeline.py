"""The post-parse conductor: assembled sections -> a runnable `ComposedFlow`.

`_assemble` is the pipeline shared by the top-level flow AND every in-file `defs:`
callable:

  flow-input decls -> build every node kind-blind (`desc.build(ctx)`; a case lowers
  in its own build) -> cross-flow type check + case exhaustiveness -> infer data
  edges + append each case's own edges -> validate references + classify asserts
  -> synthesize the START_ID/END_ID boundary over ALL edges -> reject cycles + check
  if/else handles -> CompiledFlow.from_parts.

Holds `ComposedFlow` (the compiled+validated flow the engine runs) plus the flow
`outputs:` reader (`_flow_outputs`). Ladder position: the LAST phase before the
loader — it may import every earlier phase (helpers/nodes/parse/desugar/edges/
validate/build.boundary); the loader imports `_assemble`/`ComposedFlow` back from
here. It imports NOTHING from `loader.py` (that would be a back-edge).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from agent_composer.compile.model import CompiledFlow, Edge, FlowOutput
from agent_composer.events import SourceRange
from agent_composer.nodes.base import Node
from agent_composer.typesys.values import Type
from agent_composer.compose.validate.asserts import lower_flow_asserts
from agent_composer.compose.build.boundary import synthesize_boundary_graph
from agent_composer.compose.edges.data import infer_data_edges
from agent_composer.compose.nodes.base import BuildContext, BuildResult, NodeDescriptor
from agent_composer.compose.nodes.resolver import ChildResolver
from agent_composer.compose.desugar.inline_calls import _synth_id_gen, desugar_inline_calls
from agent_composer.compose.desugar.case_outputs import expand_case_outputs
from agent_composer.compose.helpers.errors import LoadError
from agent_composer.compose.desugar.adaptive_questions import desugar_adaptive_questions
from agent_composer.compose.desugar.case_targets import desugar_case_call_targets
from agent_composer.compose.validate.case_exhaustive import check_case_exhaustive
from agent_composer.compose.validate.cycles import reject_cycles
from agent_composer.compose.validate.human_questions import validate_human_questions
from agent_composer.compose.validate.loop_types import check_loop_type_contract
from agent_composer.compose.edges.ordering import infer_ordering_edges
from agent_composer.compose.validate.call_types import check_ref_map_types
from agent_composer.compose.validate.references import validate_references
from agent_composer.compose.validate.wiring_parity import check_wiring_parity
from agent_composer.compose.helpers.reader import NodeReader
from agent_composer.compose.helpers.source import SourceFrame
from agent_composer.compose.parse.nodes import parse_nodes
from agent_composer.compose.helpers.types import InputDecl, read_flow_inputs
from agent_composer.compose.validate.handles import check_case_handles
from agent_composer.compose.validate.asserts import validate_node_asserts

# Name for the single output of a bare whole-string `outputs:` binding. With one
# `FlowOutput`, `terminal_output` returns the bare resolved value, so the name
# only labels the carrier.
_SINGLE_OUTPUT_NAME = "result"


@dataclass(frozen=True)
class BuildRequest:
    """One active descriptor build and the dependency edge that requested it."""

    node_id: str
    via_label: str | None
    via_range: SourceRange | None


class _DescriptorBuildCache:
    """Build descriptors lazily once while retaining dependency-cycle provenance.

    A descriptor may request a later descriptor while deriving its own build product.
    Successful products are memoized; failures are not. The final source-order walk
    therefore reuses forward-built products without changing assembled node order.
    """

    def __init__(
        self,
        descriptors: dict[str, NodeDescriptor],
        context: BuildContext,
        node_lines: dict[str, int],
    ) -> None:
        self._descriptors = descriptors
        self._context = context
        self._node_lines = node_lines
        self._built: dict[str, BuildResult] = {}
        self._building: list[BuildRequest] = []

    def build_one(
        self,
        node_id: str,
        via_label: str | None = None,
        via_range: SourceRange | None = None,
    ) -> BuildResult:
        """Build one descriptor, reusing success and rejecting an active dependency cycle."""
        if node_id in self._built:
            return self._built[node_id]

        request = BuildRequest(node_id, via_label, via_range)
        repeated_at = next(
            (
                index
                for index, active in enumerate(self._building)
                if active.node_id == node_id
            ),
            None,
        )
        if repeated_at is not None:
            cycle = self._building[repeated_at:]
            cycle_edges = [*cycle[1:], request]
            if all(edge.via_label is not None for edge in cycle_edges):
                path = [edge.via_label for edge in cycle_edges]
                path.append(node_id)
            else:
                path = [active.node_id for active in cycle]
                path.append(node_id)
            error_range = request.via_range or next(
                (
                    edge.via_range
                    for edge in cycle_edges
                    if edge.via_range is not None
                ),
                None,
            )
            raise LoadError(
                f"node build dependency cycle: {' -> '.join(path)}",
                range=error_range,
            )

        self._building.append(request)
        try:
            try:
                result = self._descriptors[node_id].build(self._context)
            except LoadError as exc:
                if exc.line is None:
                    exc.line = self._node_lines.get(node_id)
                raise
            self._built[node_id] = result
            return result
        finally:
            self._building.pop()

    def build_in_source_order(self) -> dict[str, BuildResult]:
        """Return every successful build in descriptor insertion order."""
        return {
            node_id: self.build_one(node_id)
            for node_id in self._descriptors
        }


@dataclass(frozen=True)
class ComposedFlow:
    """
    A compiled, validated flow ready to run.

    Produced by [`load_flow`][agent_composer.load_flow] and consumed by
    [`run_flow`][agent_composer.run_flow] / the engine. Immutable so it can be reused
    across runs and resumes.

    Attributes:
        compiled (`CompiledFlow`):
            The runnable IR (nodes + edges + outputs + wiring) the `FlowEngine` executes.
        input (`list[InputDecl]`):
            The declared flow inputs `run_flow` coerces and defaults run arguments
            against. Singular for symmetry with the `input:` YAML keyword.
        version (`str`, *optional*, defaults to `None`):
            The flow's declared `version:`, or `None` if unversioned. Used to validate a
            `uses: ref@<version>` pin against the resolved file.
        name (`str`, *optional*, defaults to `None`):
            The flow's declared `name:` — human-facing metadata (e.g. the CLI's
            interactive-prompt header). `None` for a `defs:` sub-flow, which has no name.
        description (`str`, *optional*, defaults to `None`):
            The flow's declared `description:`, or `None` if absent — human-facing
            metadata (e.g. the CLI's interactive-prompt header).
        source (`SourceFrame`, *optional*, defaults to `None`):
            Render-only source-location metadata for THIS flow (its own text + node-line
            maps). Set for the top-level flow, each `defs:` callable, and each external
            `uses:` file; copied onto the built `CallNode`/`MapNode` as `child_source` so
            the CLI can box a frame for the child a call descends into. `None` only when a
            flow was assembled without a known source (defensive).
    """

    compiled: CompiledFlow
    input: list[InputDecl]
    version: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    source: Optional[SourceFrame] = None


def _flow_outputs(outputs) -> list[FlowOutput]:
    """Read the flow `outputs:` section into the `FlowOutput` carriers `terminal_output` reads.

    A name -> binding map becomes one `FlowOutput` per entry
    (>=2 -> `terminal_output` returns an object keyed by name); a bare whole-string
    binding (`${note.output}`) is one output named `result` (1 ->
    `terminal_output` returns the bare value). A `None` section -> no declared outputs
    (`terminal_output` returns `None`).

    Args:
        outputs (`dict | str | None`):
            The raw `outputs:` section: a name -> binding map, a single binding string,
            or `None`.
    """
    if outputs is None:
        return []
    if isinstance(outputs, dict):
        return [FlowOutput(name=str(name), from_=value) for name, value in outputs.items()]
    return [FlowOutput(name=_SINGLE_OUTPUT_NAME, from_=outputs)]


def _assemble(
    *,
    inputs_section: Optional[dict] = None,
    input_decls: Optional[list[InputDecl]] = None,
    nodes_reader: NodeReader,
    outputs_section,
    asserts_section: list,
    registry,
    resolver: ChildResolver,
    n_lines: dict,
    s_lines: dict,
    uses_aliases: set = frozenset(),
    version: Optional[str] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
    flow_llm_config: Optional[dict] = None,
    flow_env: Optional[dict] = None,
    source: Optional[SourceFrame] = None,
    boundary_field_ranges: Optional[dict] = None,
    boundary_assert_ranges: Optional[list] = None,
) -> ComposedFlow:
    """Assemble parsed sections into a `ComposedFlow` — the post-parse pipeline shared by
    the top-level flow and every in-file def:

      flow-input decls -> build every node kind-blind (`desc.build(ctx)`; a case lowers
      in its own build) -> cross-flow type check + case exhaustiveness -> infer data
      edges + append each case's own edges -> validate references + classify asserts
      -> synthesize roots/terminals -> reject cycles + check if/else handles
      -> CompiledFlow.from_parts.

    Exactly one input representation is required: `inputs_section` is the authored
    mapping that still needs parsing, while `input_decls` carries declarations already
    resolved by an earlier compiler pass. `resolver` is the composite call resolver
    (defs-first); `registry` is the file's
    `read_typedefs(...)`. `n_lines`/`s_lines` are the source maps — each node id / top-level
    section key -> its 1-based HEADER line (`{}` for a nested def, whose errors are unlocated).
    Inline-CODE traceback anchors are derived from each descriptor's own `code:` range."""
    if (inputs_section is None) == (input_decls is None):
        raise ValueError("_assemble requires exactly one of inputs_section or input_decls")

    # The flow's file label frames inline-CODE tracebacks; each inline node's content start
    # line is derived from its recorded `code:` range in the CODE build. Inert for reference-mode CODE.
    flow_label = source.label if source is not None else "<inline-code>"
    if inputs_section is not None:
        inputs = read_flow_inputs(inputs_section, registry)
    else:
        assert input_decls is not None
        inputs = list(input_decls)
    flow_inputs = {decl.name for decl in inputs}
    descriptors = parse_nodes(nodes_reader)

    # A `uses:` alias shares the callable namespace, so it must not collide with a node
    # id (the resolver only sees a bare name; the author conflates them). Checked here, where
    # node ids exist, and BEFORE eager resolution so the message is the precise collision.
    collide = sorted(uses_aliases & set(descriptors))
    if collide:
        raise LoadError(f"uses: alias {collide} collides with a node id")

    # Eager `uses:` resolution (mirrors eager defs): resolve every alias now so a
    # declared-but-uncalled broken/`hub:`/missing target is loud at load, not at first call.
    for alias in uses_aliases:
        try:
            resolver(alias)
        except LoadError as exc:
            if exc.line is None:
                exc.line = s_lines.get("uses")
            raise

    # Desugar inline `${ f(...) }` call expressions: each becomes a synth `call`
    # node (CallDescriptor, over=None) and its host binding is rewritten to
    # `${<synth>.output}`. Pure sugar — the rest of the pipeline treats the synth
    # nodes as ordinary `call` nodes (build/edges/validate/DAG run unchanged).
    mint = _synth_id_gen()  # ONE synth-id minter shared by both inline-call desugars (no collision)
    descriptors, outputs_section, asserts_section = desugar_inline_calls(
        descriptors, outputs_section, asserts_section=asserts_section,
        outputs_line=s_lines.get("output") or s_lines.get("outputs"), asserts_line=s_lines.get("asserts"), next_id=mint,
    )

    # Lower each `human_input` `adaptive_questions:` block into a synth composer agent
    # + the rewritten gate. A descriptor-level pass (the `desugar_inline_calls` precedent
    # above): runs BEFORE the build loop so the synth agent flows through build + prompt-
    # scope validation + ref wiring like any author-written agent.
    descriptors = desugar_adaptive_questions(descriptors)

    # `then:/else: ${call}`: a `case` branch target that is an inline call becomes a
    # synth `call` node, the then:/else: rewritten to its id. Runs BEFORE expand_case_outputs
    # (so the synth args + the rewritten then-targets are visible to it) and shares `mint`.
    descriptors = desugar_case_call_targets(descriptors, mint)

    # Expand `${<case>.output}` value refs to a coalesce over the case's branch targets.
    # Runs after the inline-call desugar (so synth-node bindings are covered too)
    # and before the build loop — downstream sees only ordinary branch-target refs.
    descriptors, outputs_section = expand_case_outputs(
        descriptors,
        outputs_section,
        asserts_section,
        node_lines=n_lines,
        outputs_line=s_lines.get("output") or s_lines.get("outputs"),
        asserts_line=s_lines.get("asserts"),
    )

    # Build EVERY node kind-blind through its own descriptor's `build(ctx)` — leaf, call/map
    # (MapDescriptor rides CallDescriptor, no branch), loop, AND case (which lowers in its own
    # `.build()`, returning its data+control edges in `BuildResult.edges`). The conductor never
    # branches on the descriptor kind. Flow-owned wiring (the split): `flow_wiring[node_id][param]
    # -> source`, a `{}` entry for every node so the key-parity check ranges over all nodes.
    ctx_base = BuildContext(registry=registry, resolver=resolver, source_file=flow_label)
    build_results = _DescriptorBuildCache(
        descriptors, ctx_base, n_lines
    ).build_in_source_order()
    nodes_built: dict[str, Node] = {
        nid: result.node for nid, result in build_results.items()
    }
    flow_wiring: dict[str, dict] = {
        nid: result.wiring for nid, result in build_results.items()
    }
    # A case's own data+control edges; every other kind returns an empty tuple.
    case_edges: list[Edge] = [
        edge
        for result in build_results.values()
        for edge in result.edges
    ]
    # Stamp each built node's effective static config: the flow-level `env:` default merged
    # under the node's own `env:` (node override wins), flow-local (a called child flow has its
    # own flow env). The engine core never reads `Node.env` — only the node's `_run_impl()` path does.
    _flow_env = flow_env or {}
    for nid, node in nodes_built.items():
        node.env = {**_flow_env, **getattr(descriptors[nid], "env", {})}
    # producers: each built node's declared output_type (drives the case `on:` enum
    # exhaustiveness + the dotted-field walk in ref/assert validation). A case node has no
    # output_type, so it is naturally excluded by the filter.
    producers: dict[str, Type] = {
        nid: node.output_type
        for nid, node in nodes_built.items()
        if node.output_type is not None
    }

    # cross-flow type check: each `call` binding's source type vs the callable
    # input type (needs producers, so it runs after all nodes are built).
    flow_input_types = {decl.name: decl.type for decl in inputs}
    check_ref_map_types(nodes_built, producers, flow_input_types, flow_wiring, n_lines)
    # loop `'a -> 'a` type contract (types): each carried field's seed-source Type vs the
    # body output's same-named field Type (needs producers, so it runs alongside the
    # cross-flow call/map type check above).
    check_loop_type_contract(nodes_built, producers, flow_input_types, flow_wiring, n_lines)
    # case `on:` enum-exhaustiveness (a validate-phase check — needs the assembled producer
    # types). Runs right after the type contracts, the same phase-position as the old
    # desugar-time check, so error ordering is preserved.
    check_case_exhaustive(descriptors, producers, node_lines=n_lines)

    nodes: dict[str, Node] = dict(nodes_built)

    # flow.wiring/params key parity over EVERY node (catches an orphan/
    # missing source or a node absent from the wiring). Runs once the wiring is complete.
    check_wiring_parity(nodes, flow_wiring, n_lines)

    # Data edges — the projection of flow.wiring (leaf + WAIT + REF/MAP); a case owns its own
    # data+control edges (`case_edges`, from each `CaseDescriptor.build`), appended here.
    data_edges = infer_data_edges(descriptors, flow_wiring)
    reconciled = data_edges + case_edges

    # Reference + assert validation over the built flow. Errors locate
    # at the node's line (a bad node-input ref) or the `outputs:` section line (a
    # dangling flow-output ref).
    validate_references(
        nodes,
        flow_inputs,
        producers,
        outputs_section,
        flow_wiring,
        node_lines=n_lines,
        outputs_line=s_lines.get("output") or s_lines.get("outputs"),
    )
    # human_input gates: structurally check a literal `questions:` list at load, and
    # pin a ref-form `questions: ${name}` to a declared input (the prompt-scope rule).
    validate_human_questions(nodes, node_lines=n_lines)
    # Node-local `asserts:` (per-node contract): validate node-locally + classify pre/post +
    # stamp onto each built leaf/call node. Shared by defs (a def's internal nodes too).
    validate_node_asserts(nodes_built, descriptors, n_lines)
    # Lower the flow-level `asserts:` into node-local asserts stamped on the synthesized
    # START/END. Fires uniformly via eval_node — a cross-node post
    # assert's `${producer.output}` becomes a real END input edge, so it survives child cloning.
    # `boundary_assert_ranges` is the POSITIONAL flow-`asserts:` source list (aligned 1:1 with the
    # authored asserts), so each lowered START/END assert and each bad-ref `LoadError` carets by
    # index — never by (non-unique) text; `None`/`[]` when spanless.
    lowered = lower_flow_asserts(
        asserts_section, flow_inputs, set(descriptors), producers,
        assert_ranges=boundary_assert_ranges,
    )

    # Run-ordering edges: depends_on (co-skip) / runs_after (pure order). Folded in
    # BEFORE root synthesis so an ordering-only dependent is demoted from root, and into
    # the cycle check below so an ordering cycle is loud.
    ordering_edges = infer_ordering_edges(descriptors, node_ids := set(descriptors))

    # DAG assembly: synthesize the START_ID/END_ID boundary NODES + their edges over the
    # FULL reconciled body-edge set (the __start__/__end__ sentinels retire; the strings
    # persist as the boundary node ids). synthesize_boundary_graph mints the END_ID producer edges
    # (one per producer per output, keyed by output name, with the coalesce/optional stance)
    # + the START_ID root edges + (0-output) the bare START_ID->END_ID root edge,
    # and returns parity-clean START_ID/END_ID wiring.
    flow_outputs = _flow_outputs(outputs_section)
    structural = reconciled + ordering_edges
    boundary_nodes, boundary_edges, boundary_wiring = synthesize_boundary_graph(
        inputs, flow_outputs, node_ids, structural, boundary_field_ranges, lowered=lowered
    )
    nodes.update(boundary_nodes)
    flow_wiring.update(boundary_wiring)
    edges: list[Edge] = structural + boundary_edges

    # A cycle spans nodes; anchor the error on the first stuck node's range and highlight every
    # stuck node's line. START_ID/END_ID are acyclic by construction (START_ID has no incoming,
    # END_ID no outgoing).
    node_ranges = {
        nid: d.range for nid, d in descriptors.items() if getattr(d, "range", None)
    }
    reject_cycles(edges, node_ids, n_lines, node_ranges)
    check_case_handles(nodes, edges)

    compiled = CompiledFlow.from_parts(nodes, edges, outputs=flow_outputs, wiring=flow_wiring,
                                       flow_llm_config=flow_llm_config or {})
    return ComposedFlow(
        compiled=compiled, input=inputs,
        version=version, name=name, description=description, source=source,
    )
