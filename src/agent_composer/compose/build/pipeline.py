"""The post-parse conductor: assembled sections -> a runnable `ComposedFlow`.

`_assemble` is the pipeline shared by the top-level flow AND every in-file `defs:`
callable:

  flow parameters + authored parameter/contract/default origins -> build every node
  kind-blind (`desc.build(ctx)`; physical node-output contracts compile at their
  existing leaf-owned build position; a case lowers
  in its own build) -> cross-flow type check + case exhaustiveness -> infer data
  edges + append each case's own edges -> validate references + classify asserts
  -> attach authored assertion/flow-output binding origins -> synthesize the START_ID/END_ID
  boundary over ALL edges -> reject cycles + check if/else handles ->
  CompiledFlow.from_parts. One retained assertion item cursor supplies its carrier
  origin and the legacy aligned range; rewrites preserve both fields.

Holds `ComposedFlow` (the compiled+validated flow the engine runs) plus the flow
`outputs:` parser and source-origin attachment. Ladder position: the LAST phase
before the loader — it may import every earlier phase (helpers/nodes/parse/desugar/
edges/validate/build.boundary); the loader imports `_assemble`/`ComposedFlow` back
from here. It imports NOTHING from `loader.py` (that would be a back-edge).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Optional, Protocol

from agent_composer.compile.model import CompiledFlow, Edge, FlowOutput
from agent_composer.diagnostics import (
    Diagnostic,
    DiagnosticFrame,
    Failure,
    OriginRef,
    OriginResolver,
)
from agent_composer.events import SourceRange
from agent_composer.nodes.base import Node
from agent_composer.nodes.contracts import Parameter, ValueContract, as_binding
from agent_composer.typesys.values import Type
from agent_composer.compose.validate.asserts import lower_flow_asserts
from agent_composer.compose.build.boundary import synthesize_boundary_graph
from agent_composer.compose.build.inline_inputs import (
    canonicalize_inline_type,
    infer_inline_binding_type,
)
from agent_composer.compose.edges.data import infer_data_edges
from agent_composer.compose.nodes.base import BuildContext, BuildResult, NodeDescriptor
from agent_composer.compose.nodes.call import CallDescriptor
from agent_composer.compose.nodes.loop import LoopDescriptor
from agent_composer.compose.nodes.map import MapDescriptor
from agent_composer.compose.desugar.inline_bodies import (
    InlineBody,
    extract_inline_bodies,
)
from agent_composer.compose.desugar.inline_calls import _synth_id_gen, desugar_inline_calls
from agent_composer.compose.desugar.case_outputs import expand_case_outputs
from agent_composer.compose.helpers.errors import LoadDiagnostics, LoadError
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
from agent_composer.compose.parse.nodes import (
    normalize_node_signature_cursor,
    parse_nodes,
)
from agent_composer.compose.source import SourceAccessError, SourceCursor
from agent_composer.compose.source.catalog import _SourceCatalogBuilder
from agent_composer.compose.helpers.types import (
    read_flow_inputs,
    read_source_flow_inputs,
    read_value_contract,
)
from agent_composer.compose.validate.handles import check_case_handles
from agent_composer.compose.validate.asserts import validate_node_asserts

# Name for the single output of a bare whole-string `outputs:` binding. With one
# `FlowOutput`, `terminal_output` returns the bare resolved value, so the name
# only labels the carrier.
_SINGLE_OUTPUT_NAME = "result"

_PRESENTATION_UNAVAILABLE_NOTE = "Source presentation was unavailable."


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
    ) -> None:
        self._descriptors = descriptors
        self._context = context
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
            raise self._context.diagnostics.error(
                f"node build dependency cycle: {' -> '.join(path)}",
                range=error_range,
            )

        self._building.append(request)
        try:
            try:
                result = self._descriptors[node_id].build(self._context)
            except LoadError as exc:
                if exc.diagnostic.frames:
                    raise
                descriptor_range = getattr(
                    self._descriptors[node_id], "range", None
                )
                if descriptor_range is None:
                    raise
                raise self._context.diagnostics.prepend(
                    exc,
                    range=descriptor_range,
                ) from exc
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
        input (`list[Parameter]`):
            A property view of the synthesized START node's canonical signature.
        version (`str`, *optional*, defaults to `None`):
            The flow's declared `version:`, or `None` if unversioned. Used to validate a
            `uses: ref@<version>` pin against the resolved file.
        name (`str`, *optional*, defaults to `None`):
            The flow's declared `name:` — human-facing metadata (e.g. the CLI's
            interactive-prompt header). `None` for a `defs:` sub-flow, which has no name.
        description (`str`, *optional*, defaults to `None`):
            The flow's declared `description:`, or `None` if absent — human-facing
            metadata (e.g. the CLI's interactive-prompt header).
        origin_resolver (`OriginResolver`, *optional*, defaults to `None`):
            Source-neutral authority for every reachable origin in a flow returned by
            `load_flow`. `None` means the compiled graph is source-free. The private
            assembly pipeline leaves this unset while one physical document is still
            being built; the loader installs the exact immutable resolver before the
            flow is returned or cached.
    """

    compiled: CompiledFlow
    version: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    origin_resolver: OriginResolver | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    @property
    def input(self) -> list[Parameter]:
        """Return the one canonical flow signature stored on the START node."""
        start = self.compiled.nodes[self.compiled.start_id]
        parameters = start.params
        if type(parameters) is not list:
            raise TypeError("compiled START params must be an exact list")
        for parameter in parameters:
            if type(parameter) is not Parameter:
                raise TypeError(
                    "compiled START params must contain exact Parameter values"
                )
        return parameters

    def diagnose(self, failure: Failure) -> Diagnostic:
        """Resolve one semantic failure through this flow's source authority.

        Resolution is presentation-only. A missing, incomplete, malformed, or
        failing resolver produces an unlocated diagnostic without changing the
        semantic failure fields.

        Args:
            failure (`Failure`):
                Valid source-neutral failure whose trace is ordered outermost to
                innermost.

        Returns:
            `Diagnostic`:
                The unchanged semantic fields plus every resolved frame in trace
                order, or no frames and one generic note when presentation is
                unavailable.

        Raises:
            TypeError: Raised when `failure` is not a `Failure`.
        """
        if not isinstance(failure, Failure):
            raise TypeError("diagnose requires a Failure")

        diagnostic = Diagnostic(
            code=failure.code,
            message=failure.message,
            node_id=failure.node_id,
            cause_type=failure.cause_type,
            traceback=failure.traceback,
        )
        if not failure.trace:
            return diagnostic

        unavailable = replace(
            diagnostic,
            notes=(_PRESENTATION_UNAVAILABLE_NOTE,),
        )
        resolver = self.origin_resolver
        if resolver is None:
            return unavailable

        frames: list[DiagnosticFrame] = []
        for origin in failure.trace:
            try:
                frame = resolver.resolve(origin)
                if not isinstance(frame, DiagnosticFrame):
                    return unavailable

                label = frame.label
                text = frame.text
                source_range = frame.range
                note = frame.note
                if (
                    type(label) is not str
                    or type(text) is not str
                    or (note is not None and type(note) is not str)
                    or not isinstance(source_range, SourceRange)
                ):
                    return unavailable

                coordinates = (
                    source_range.start_line,
                    source_range.start_col,
                    source_range.end_line,
                    source_range.end_col,
                )
                if any(
                    type(coordinate) is not int or coordinate <= 0
                    for coordinate in coordinates
                ):
                    return unavailable
                start_line, start_col, end_line, end_col = coordinates
                if (end_line, end_col) < (start_line, start_col):
                    return unavailable

                frames.append(
                    DiagnosticFrame(
                        label=label,
                        text=text,
                        range=SourceRange(
                            start_line,
                            start_col,
                            end_line,
                            end_col,
                        ),
                        note=note,
                    )
                )
            except Exception:
                return unavailable

        return replace(diagnostic, frames=tuple(frames))


class _PrivateRegistrar(Protocol):
    """Structural seam for registering a lazy compiler-owned child callable."""

    def register_private(
        self,
        flow_id: str,
        owner: tuple[str, ...],
        label: str,
        builder: Callable[[], ComposedFlow],
    ) -> None:
        """Register one private callable builder under its generated identity."""


class _CallableScopeLike(_PrivateRegistrar, Protocol):
    """File-owned callable scope used by assembly and anonymous child lowering."""

    @property
    def catalog_builder(self) -> _SourceCatalogBuilder:
        """Return the one mutable origin registry owned by this source file."""

    def resolve(
        self, flow_id: str, version: str | None = None
    ) -> ComposedFlow:
        """Resolve one named or compiler-owned callable in this file scope."""


def _register_inline_bodies(
    descriptors: dict[str, NodeDescriptor],
    bodies: dict[str, InlineBody],
    *,
    registrar: _PrivateRegistrar,
    diagnostics: LoadDiagnostics | None = None,
    build_child: Callable[
        [InlineBody, CallDescriptor | LoopDescriptor], ComposedFlow
    ],
) -> dict[str, NodeDescriptor]:
    """Associate retained inline bodies with descriptors and register lazy builders.

    Every association is validated before the registrar is touched. The returned map
    replaces only matching frozen descriptors with copies carrying an author-facing child
    label; the input descriptors remain unchanged. Child construction stays lazy until the
    registrar invokes a captured zero-argument builder.
    """
    if diagnostics is None:
        first_body = next(iter(bodies.values()), None)
        diagnostics = LoadDiagnostics.from_cursor(
            None if first_body is None else first_body.nodes_cursor
        )
    associated = dict(descriptors)
    registrations: list[
        tuple[InlineBody, CallDescriptor | LoopDescriptor, str]
    ] = []

    for body in bodies.values():
        label = f"inline body of node '{body.node_id}'"
        descriptor = descriptors.get(body.node_id)
        if descriptor is None:
            raise diagnostics.error(
                f"{label}: no matching composite node descriptor",
                range=body.nodes_cursor.range,
            )
        if not isinstance(descriptor, (CallDescriptor, LoopDescriptor)):
            raise diagnostics.error(
                f"{label}: final node descriptor is not CALL, MAP, or LOOP",
                range=body.nodes_cursor.range,
            )
        if descriptor.call != body.flow_id:
            raise diagnostics.error(
                f"{label}: final node descriptor does not target its retained inline body",
                range=body.nodes_cursor.range,
            )

        labelled_descriptor = replace(descriptor, child_label=label)
        associated[body.node_id] = labelled_descriptor
        registrations.append((body, labelled_descriptor, label))

    for body, labelled_descriptor, label in registrations:
        def builder(
            body: InlineBody = body,
            descriptor: CallDescriptor | LoopDescriptor = labelled_descriptor,
        ) -> ComposedFlow:
            return build_child(body, descriptor)

        registrar.register_private(body.flow_id, body.owner, label, builder)

    return associated


def _assertion_cursors(
    assertions: list,
    asserts_cursor: SourceCursor | None,
    diagnostics: LoadDiagnostics,
) -> tuple[SourceCursor | None, ...]:
    """Align one retained assertion sequence with its semantic source order.

    Args:
        assertions (`list`):
            Raw or carrierized assertion values in authored order.
        asserts_cursor (`SourceCursor | None`):
            Retained authored sequence, or ``None`` for a generated/source-free list.

    Returns:
        `tuple[SourceCursor | None, ...]`:
            One item cursor per assertion. Source-free lists contain ``None`` entries.

    Raises:
        `LoadError`:
            Raised when the retained sequence does not align with the semantic list.
    """
    if asserts_cursor is None:
        return (None,) * len(assertions)
    try:
        cursors = asserts_cursor.children()
    except SourceAccessError as exc:
        raise diagnostics.error(str(exc), range=exc.range) from exc
    if len(cursors) != len(assertions):
        raise diagnostics.error(
            "retained assertion cursor does not align with semantic assertions",
            range=asserts_cursor.range,
        )
    return cursors


def _register_assertion_origin(
    cursor: SourceCursor | None,
    site_path: tuple[str | int, ...],
    catalog_builder: _SourceCatalogBuilder,
    diagnostics: LoadDiagnostics,
) -> OriginRef | None:
    """Register one authored assertion item without exposing source to consumers.

    Args:
        cursor (`SourceCursor | None`):
            Authored list-item cursor, or ``None`` for a source-free assertion.
        site_path (`tuple[str | int, ...]`):
            Complete lexical semantic path for the assertion occurrence.
        catalog_builder (`_SourceCatalogBuilder`):
            File-owned origin catalog builder shared by all lexical children.

    Returns:
        `OriginRef | None`:
            Opaque registered origin, or ``None`` when no physical site exists.

    Raises:
        `LoadError`:
            Raised when the catalog site collides with another registration.
    """
    if cursor is None:
        return None
    try:
        return catalog_builder.register(site_path, cursor)
    except SourceAccessError as exc:
        raise diagnostics.error(str(exc), range=exc.range) from exc


def _stamp_node_wiring_origins(
    descriptors: dict[str, NodeDescriptor],
    nodes_cursor: SourceCursor,
    nodes: dict[str, Node],
    flow_wiring: dict[str, dict],
    *,
    owner: tuple[str, ...],
    catalog_builder: _SourceCatalogBuilder,
    diagnostics: LoadDiagnostics,
) -> None:
    """Attach authored node-wiring sites to their canonical carriers.

    Desugaring has already produced each final binding source, while
    ``nodes_cursor`` still retains the original occurrence. Each ordinary input
    entry contributes two meanings: its key owns the formal ``Parameter`` and
    its value owns the flow ``Binding``. A reserved MAP ``over`` or WAIT
    ``until`` field contributes only a flow-owned ``Binding`` origin. For names
    absent from the retained cursor, Compose mints no origin and preserves any
    existing carrier origin.

    Args:
        descriptors (`dict[str, NodeDescriptor]`):
            Final descriptors keyed by node id after every graph desugar.
        nodes_cursor (`SourceCursor`):
            Retained source view of the flow's authored node mapping.
        nodes (`dict[str, Node]`):
            Built runtime nodes keyed by final descriptor id.
        flow_wiring (`dict[str, dict]`):
            Mutable final binding maps keyed by node id and wiring key.
        owner (`tuple[str, ...]`):
            Complete lexical owner path used to namespace catalog sites.
        catalog_builder (`_SourceCatalogBuilder`):
            File-owned origin catalog builder shared by all lexical children.

    Raises:
        `LoadError`:
            Raised when a retained source cursor cannot be traversed.
    """
    for node_id, descriptor in descriptors.items():
        node = nodes.get(node_id)
        wiring = flow_wiring.get(node_id)
        if node is None or wiring is None:
            continue

        try:
            if not nodes_cursor.has(node_id):
                continue
            node_cursor = normalize_node_signature_cursor(
                nodes_cursor.child(node_id)
            )

            for key in sorted(node.reserved_wiring_keys()):
                if key not in wiring or not node_cursor.has(key):
                    continue
                binding_origin = catalog_builder.register(
                    (*owner, "node", node_id, "reserved", key, "binding"),
                    node_cursor.child(key),
                )
                if binding_origin is not None:
                    wiring[key] = replace(
                        as_binding(wiring[key]),
                        origin=binding_origin,
                    )

            if (
                not descriptor.inputs
                or node.params is None
                or not node_cursor.has("inputs")
            ):
                continue
            inputs_cursor = node_cursor.child("inputs")
            parameter_indexes = {
                parameter.name: index
                for index, parameter in enumerate(node.params)
            }
            parameters = list(node.params)
            parameters_changed = False
            for name in descriptor.inputs:
                if (
                    name not in parameter_indexes
                    or name not in wiring
                    or not inputs_cursor.has(name)
                ):
                    continue
                entry_cursor = inputs_cursor.child(name)
                parameter_origin = catalog_builder.register(
                    (*owner, "node", node_id, "input", name, "parameter"),
                    entry_cursor,
                    anchor="key",
                )
                binding_origin = catalog_builder.register(
                    (*owner, "node", node_id, "input", name, "binding"),
                    entry_cursor,
                )

                parameter_index = parameter_indexes[name]
                if parameter_origin is not None:
                    parameters[parameter_index] = replace(
                        parameters[parameter_index],
                        origin=parameter_origin,
                    )
                    parameters_changed = True
                if binding_origin is not None:
                    wiring[name] = replace(
                        as_binding(wiring[name]),
                        origin=binding_origin,
                    )
            if parameters_changed:
                node.params = parameters
        except SourceAccessError as exc:
            raise diagnostics.error(str(exc), range=exc.range) from exc


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
        return [
            FlowOutput(name=str(name), binding=value)
            for name, value in outputs.items()
        ]
    return [FlowOutput(name=_SINGLE_OUTPUT_NAME, binding=outputs)]


def _stamp_flow_output_origins(
    flow_outputs: list[FlowOutput],
    outputs_section,
    outputs_cursor: SourceCursor | None,
    *,
    owner: tuple[str, ...],
    catalog_builder: _SourceCatalogBuilder,
    diagnostics: LoadDiagnostics,
) -> list[FlowOutput]:
    """Attach authored output values to their final canonical bindings.

    Desugaring has already produced each final binding source, while the cursor
    retains the original occurrence. Mapping keys identify their effective raw
    source children, but only values receive origins. A generated or source-free
    output has no cursor; an explicitly originated binding remains untouched.

    Args:
        flow_outputs (`list[FlowOutput]`):
            Final normalized output carriers in section order.
        outputs_section (`dict | object | None`):
            Final post-desugar output values used to align mapping keys.
        outputs_cursor (`SourceCursor | None`):
            Retained authored ``output:`` value, or ``None`` for generated output.
        owner (`tuple[str, ...]`):
            Complete lexical owner path used to namespace output sites.
        catalog_builder (`_SourceCatalogBuilder`):
            File-owned origin catalog builder shared by all lexical children.

    Returns:
        `list[FlowOutput]`:
            Output carriers whose physically authored bindings own their origin.

    Raises:
        `ValueError`:
            Raised when the normalized carriers do not align with the section.
        `LoadError`:
            Raised when the retained cursor cannot be traversed or a catalog site
            collides with another registration.
    """
    if outputs_cursor is None or not flow_outputs:
        return flow_outputs

    if isinstance(outputs_section, dict):
        raw_keys = tuple(outputs_section)
        if len(raw_keys) != len(flow_outputs):
            raise ValueError("flow output carriers do not align with their section")
        try:
            cursors = tuple(outputs_cursor.child(key) for key in raw_keys)
        except SourceAccessError as exc:
            raise diagnostics.error(str(exc), range=exc.range) from exc
    else:
        if len(flow_outputs) != 1:
            raise ValueError("a bare flow output requires exactly one carrier")
        cursors = (outputs_cursor,)

    stamped: list[FlowOutput] = []
    for index, (flow_output, cursor) in enumerate(zip(flow_outputs, cursors)):
        if flow_output.binding.origin is not None:
            stamped.append(flow_output)
            continue
        try:
            origin = catalog_builder.register(
                (*owner, "output", index, "binding"),
                cursor,
            )
        except SourceAccessError as exc:
            raise diagnostics.error(str(exc), range=exc.range) from exc
        stamped.append(
            flow_output
            if origin is None
            else replace(
                flow_output,
                binding=replace(flow_output.binding, origin=origin),
            )
        )
    return stamped


def _assemble(
    *,
    inputs_section: Optional[dict] = None,
    input_parameters: Optional[list[Parameter]] = None,
    inputs_cursor: SourceCursor | None = None,
    nodes_cursor: SourceCursor,
    outputs_section,
    asserts_section: list,
    registry,
    scope: _CallableScopeLike,
    owner: tuple[str, ...],
    uses_aliases: set = frozenset(),
    version: Optional[str] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
    flow_llm_config: Optional[dict] = None,
    flow_env: Optional[dict] = None,
    boundary_field_ranges: Optional[dict] = None,
    asserts_cursor: SourceCursor | None = None,
    outputs_range: Optional[SourceRange] = None,
    outputs_cursor: SourceCursor | None = None,
) -> ComposedFlow:
    """Assemble parsed sections into a `ComposedFlow` — the post-parse pipeline shared by
    the top-level flow and every in-file def:

      flow parameters -> build every node kind-blind (`desc.build(ctx)`; physical
      node-output contracts compile at their existing leaf-owned position; a case lowers
      in its own build) -> cross-flow type check + case exhaustiveness -> infer data
      edges + append each case's own edges -> validate references + classify asserts
      -> attach authored flow-output binding origins -> synthesize roots/terminals -> reject cycles
      + check if/else handles -> CompiledFlow.from_parts.

    Exactly one input representation is required: `inputs_section` is the authored
    mapping that still needs parsing, while `input_parameters` carries parameters already
    resolved by an earlier compiler pass. `inputs_cursor` is the optional retained
    occurrence for the authored mapping; when present, input parameters and their
    recursive contracts compile opaque origins from it. A Compose-owned callable in
    `BuildContext` compiles a physical node `output:` cursor into its canonical contract
    only when the leaf reaches its established output-contract step; source-free builds
    use the descriptor's ordinary type reader. `scope` owns composite call resolution
    (defs-first) and private child registration; `registry` is the file's
    `read_typedefs(...)`. Retained ranges and `nodes_cursor` own diagnostic locations; after
    the common header stamp each descriptor may immutably attach kind-owned opaque origins.
    `owner` is this flow's complete lexical path and seeds stable identities for anonymous
    children. `boundary_field_ranges` locates synthesized boundary-input behavior;
    `asserts_cursor` is the single authority for each flow assertion's opaque origin and
    legacy aligned range. `outputs_range` is the retained authored codomain range when one exists,
    while `outputs_cursor` retains each physically authored output binding occurrence.
    `scope.catalog_builder` is the one registry shared by every assembly originating
    from this physical document, so root, definitions, and inline bodies mint
    collision-free local origins under `owner`. After kind-owned attachment, CODE may
    present a syntax failure through the builder's registered-site lookup without
    freezing the catalog. `_assemble` performs one
    construction attempt; retriable callable attempts run through
    `_CallableScope.resolve`, while a failed top-level load discards its scope."""
    if (inputs_section is None) == (input_parameters is None):
        raise ValueError(
            "_assemble requires exactly one of inputs_section or input_parameters"
        )
    if input_parameters is not None and inputs_cursor is not None:
        raise ValueError("pre-resolved input parameters cannot carry an inputs cursor")

    diagnostics = LoadDiagnostics.from_cursor(nodes_cursor)

    if outputs_range is None and outputs_cursor is not None:
        outputs_range = outputs_cursor.range

    # Inline-body extraction masks child-only fields for descriptor parsing. Assertion
    # provenance still belongs to the unmodified authored occurrence tree.
    authored_nodes_cursor = nodes_cursor
    try:
        nodes_cursor, inline_bodies = extract_inline_bodies(
            nodes_cursor,
            owner=owner,
        )
    except SourceAccessError as exc:
        raise diagnostics.error(str(exc), range=exc.range) from exc

    catalog_builder = scope.catalog_builder
    flow_assertion_cursors = _assertion_cursors(
        asserts_section,
        asserts_cursor,
        diagnostics,
    )
    flow_assertion_ranges = [
        cursor.range if cursor is not None else None
        for cursor in flow_assertion_cursors
    ]

    def flow_assertion_origin_at(index: int) -> OriginRef | None:
        """Return the registered origin for one authored flow assertion."""
        return _register_assertion_origin(
            flow_assertion_cursors[index],
            (*owner, "assert", index),
            catalog_builder,
            diagnostics,
        )

    def flow_assertion_range_at(index: int) -> SourceRange | None:
        """Return one authored flow assertion's retained positional range."""
        return flow_assertion_ranges[index]

    def node_assertion_site_at(
        node_id: str,
        index: int,
    ) -> tuple[OriginRef | None, SourceRange | None] | None:
        """Return the origin and range from one authored assertion cursor."""
        try:
            if not authored_nodes_cursor.has(node_id):
                return None
            node_cursor = normalize_node_signature_cursor(
                authored_nodes_cursor.child(node_id)
            )
            if not node_cursor.has("asserts"):
                return None
            cursor = node_cursor.child("asserts").child(index)
        except SourceAccessError as exc:
            raise diagnostics.error(str(exc), range=exc.range) from exc
        return (
            _register_assertion_origin(
                cursor,
                (*owner, "node", node_id, "assert", index),
                catalog_builder,
                diagnostics,
            ),
            cursor.range,
        )

    def read_authored_node_output(
        node_id: str,
        output_spec: Any,
    ) -> ValueContract | None:
        """Compile one physical node-output declaration at its build position.

        Args:
            node_id (`str`): Node whose normalized declaration should be selected.
            output_spec (`Any`): Semantic type shape already retained by its descriptor.

        Returns:
            `ValueContract | None`: Located contract, or ``None`` for a virtual site.

        Raises:
            `LoadError`: Raised when source traversal or type compilation fails.
        """
        try:
            if not nodes_cursor.has(node_id):
                return None
            node_cursor = normalize_node_signature_cursor(nodes_cursor.child(node_id))
            if not node_cursor.has("outputs"):
                return None
            output_cursor = node_cursor.child("outputs")
            if output_cursor.range is None:
                return None
            return read_value_contract(
                output_cursor.with_value(output_spec),
                registry,
                site_path=(*owner, "node", node_id, "output", "contract"),
                catalog_builder=catalog_builder,
                diagnostics=diagnostics,
            )
        except SourceAccessError as exc:
            raise diagnostics.error(str(exc), range=exc.range) from exc

    # Inline CODE receives the owning document label only as an opaque Python
    # traceback filename; authored locations come from semantic origins. Inert for
    # reference-mode CODE.
    flow_label = diagnostics.label or "<inline-code>"
    if inputs_section is not None:
        if inputs_cursor is not None:
            inputs = read_source_flow_inputs(
                inputs_cursor.with_value(inputs_section),
                registry,
                site_path=owner,
                catalog_builder=catalog_builder,
                diagnostics=diagnostics,
            )
        else:
            input_ranges = {
                name: (
                    boundary_field_ranges.get(("input", name))
                    if boundary_field_ranges is not None
                    else None
                )
                for name in inputs_section
            }
            inputs = read_flow_inputs(
                inputs_section,
                registry,
                input_ranges,
                diagnostics=diagnostics,
            )
    else:
        assert input_parameters is not None
        inputs = list(input_parameters)
    flow_input_contracts: dict[str, ValueContract] = {}
    for parameter in inputs:
        contract = parameter.contract
        if contract is None:
            continue
        if type(contract) is not ValueContract:
            raise TypeError(
                "flow input contracts must be exact ValueContract values, "
                f"got {type(contract).__name__} for {parameter.name!r}"
            )
        flow_input_contracts[parameter.name] = contract
    flow_input_types = {
        name: contract.type for name, contract in flow_input_contracts.items()
    }
    flow_inputs = {parameter.name for parameter in inputs}
    descriptors = parse_nodes(nodes_cursor)
    try:
        located_descriptors: dict[str, NodeDescriptor] = {}
        for node_id, descriptor in descriptors.items():
            authored_cursor = nodes_cursor.child(node_id)
            node_path = (*owner, "node", node_id)
            located = replace(
                descriptor,
                origin=catalog_builder.register(
                    node_path,
                    authored_cursor,
                    anchor="key",
                ),
            )
            located_descriptors[node_id] = located.with_source_origins(
                normalize_node_signature_cursor(authored_cursor),
                site_path=node_path,
                catalog_builder=catalog_builder,
            )
        descriptors = located_descriptors
    except SourceAccessError as exc:
        raise diagnostics.error(str(exc), range=exc.range) from exc

    # A `uses:` alias shares the callable namespace, so it must not collide with a node
    # id (the resolver only sees a bare name; the author conflates them). Checked here, where
    # node ids exist, and BEFORE eager resolution so the message is the precise collision.
    collide = sorted(uses_aliases & set(descriptors))
    if collide:
        raise diagnostics.error(f"uses: alias {collide} collides with a node id")

    # Eager `uses:` resolution (mirrors eager defs): resolve every alias now so a
    # declared-but-uncalled broken/`hub:`/missing target is loud at load, not at first call.
    for alias in uses_aliases:
        scope.resolve(alias)

    # Desugar inline `${ f(...) }` call expressions: each becomes a synth `call`
    # node (CallDescriptor, over=None) and its host binding is rewritten to
    # `${<synth>.output}`. Pure sugar — the rest of the pipeline treats the synth
    # nodes as ordinary `call` nodes (build/edges/validate/DAG run unchanged).
    mint = _synth_id_gen()  # ONE synth-id minter shared by both inline-call desugars (no collision)
    descriptors, outputs_section, asserts_section = desugar_inline_calls(
        descriptors,
        outputs_section,
        asserts_section=asserts_section,
        outputs_range=outputs_range,
        assertion_range_at=flow_assertion_range_at,
        assertion_origin_at=flow_assertion_origin_at,
        next_id=mint,
        diagnostics=diagnostics,
    )

    # Lower each `human_input` `adaptive_questions:` block into a synth composer agent
    # + the rewritten gate. A descriptor-level pass (the `desugar_inline_calls` precedent
    # above): runs BEFORE the build loop so the synth agent flows through build + prompt-
    # scope validation + ref wiring like any author-written agent.
    descriptors = desugar_adaptive_questions(
        descriptors,
        diagnostics=diagnostics,
    )

    # `then:/else: ${call}`: a `case` branch target that is an inline call becomes a
    # synth `call` node, the then:/else: rewritten to its id. Runs BEFORE expand_case_outputs
    # (so the synth args + the rewritten then-targets are visible to it) and shares `mint`.
    descriptors = desugar_case_call_targets(
        descriptors,
        mint,
        diagnostics=diagnostics,
    )

    # Expand `${<case>.output}` value refs to a coalesce over the case's branch targets.
    # Runs after the inline-call desugar (so synth-node bindings are covered too)
    # and before the build loop — downstream sees only ordinary branch-target refs.
    descriptors, outputs_section = expand_case_outputs(
        descriptors,
        outputs_section,
        asserts_section,
        outputs_range=outputs_range,
        assertion_range_at=flow_assertion_range_at,
        diagnostics=diagnostics,
    )

    descriptor_cache: _DescriptorBuildCache | None = None

    def build_inline_child(
        body: InlineBody,
        descriptor: CallDescriptor | LoopDescriptor,
    ) -> ComposedFlow:
        """Assemble one retained body lazily in this file's callable scope."""

        def producer_type(
            producer_id: str,
            binding_label: str,
            binding_range: SourceRange | None,
        ) -> Type:
            """Resolve one parent producer through this assembly's build cache."""
            def unavailable_type() -> LoadError:
                return diagnostics.error(
                    f"cannot infer type for inline binding {binding_label!r}: "
                    f"producer {producer_id!r} has no available static type; use an "
                    "explicitly typed named `defs:` body",
                    range=binding_range,
                )

            if producer_id not in descriptors:
                raise unavailable_type()

            cache = descriptor_cache
            if cache is None:
                raise RuntimeError(
                    "descriptor cache was not initialized before inline child resolution"
                )
            output_contract = cache.build_one(
                producer_id,
                via_label=binding_label,
                via_range=binding_range,
            ).node.output_contract
            if output_contract is None:
                raise unavailable_type()
            if type(output_contract) is not ValueContract:
                raise TypeError(
                    "output producer contracts must be exact ValueContract values, "
                    f"got {type(output_contract).__name__} for {producer_id!r}"
                )
            return output_contract.type

        item_type: Callable[[str, SourceRange | None], Type] | None = None
        if isinstance(descriptor, MapDescriptor):
            cached_item_type: Type | None = None

            def map_item_type(
                binding_label: str,
                binding_range: SourceRange | None,
            ) -> Type:
                """Prove the MAP element type once, only when an item walk requests it."""
                nonlocal cached_item_type
                if cached_item_type is not None:
                    return cached_item_type

                over_type = infer_inline_binding_type(
                    descriptor.over,
                    parent_input_types=flow_input_types,
                    producer_type=producer_type,
                    binding_label=binding_label,
                    binding_range=binding_range,
                    diagnostics=diagnostics,
                )
                if not over_type.kind.is_list() or over_type.element is None:
                    raise diagnostics.error(
                        f"cannot infer type for inline binding {binding_label!r}: "
                        "MAP `over:` must have a proven list element type; use an "
                        "explicitly typed named `defs:` body",
                        range=binding_range,
                    )
                cached_item_type = over_type.element
                return cached_item_type

            item_type = map_item_type

        child_inputs: list[Parameter] = []
        for key, value in (descriptor.inputs or {}).items():
            binding_label = f"{descriptor.id}.input.{key}"
            binding_range = body.input_ranges.get(key)
            inferred = infer_inline_binding_type(
                value,
                parent_input_types=flow_input_types,
                producer_type=producer_type,
                item_type=item_type,
                binding_label=binding_label,
                binding_range=binding_range,
                diagnostics=diagnostics,
            )
            canonical = canonicalize_inline_type(inferred)
            child_inputs.append(
                Parameter(
                    name=key,
                    contract=ValueContract(canonical),
                    required=not canonical.nullable,
                    coerce_strings=True,
                )
            )

        return _assemble(
            input_parameters=child_inputs,
            nodes_cursor=body.nodes_cursor,
            outputs_section=body.output,
            asserts_section=[],
            registry=registry,
            scope=scope,
            owner=body.owner,
            flow_llm_config={},
            flow_env={},
            boundary_field_ranges={
                ("input", name): binding_range
                for name, binding_range in body.input_ranges.items()
            },
            outputs_range=body.output_range,
            outputs_cursor=body.output_cursor,
        )

    descriptors = _register_inline_bodies(
        descriptors,
        inline_bodies,
        registrar=scope,
        diagnostics=diagnostics,
        build_child=build_inline_child,
    )

    # Build EVERY node kind-blind through its own descriptor's `build(ctx)` — leaf, call/map
    # (MapDescriptor rides CallDescriptor, no branch), loop, AND case (which lowers in its own
    # `.build()`, returning its data+control edges in `BuildResult.edges`). The conductor never
    # branches on the descriptor kind. Flow-owned wiring (the split): `flow_wiring[node_id][param]
    # -> source`, a `{}` entry for every node so the key-parity check ranges over all nodes.
    ctx_base = BuildContext(
        registry=registry,
        resolver=scope.resolve,
        source_file=flow_label,
        resolve_origin=catalog_builder.resolve_registered,
        diagnostics=diagnostics,
        output_contract_reader=read_authored_node_output,
    )
    descriptor_cache = _DescriptorBuildCache(descriptors, ctx_base)
    build_results = descriptor_cache.build_in_source_order()
    nodes_built: dict[str, Node] = {
        nid: result.node for nid, result in build_results.items()
    }
    flow_wiring: dict[str, dict] = {
        nid: result.wiring for nid, result in build_results.items()
    }
    _stamp_node_wiring_origins(
        descriptors,
        nodes_cursor,
        nodes_built,
        flow_wiring,
        owner=owner,
        catalog_builder=catalog_builder,
        diagnostics=diagnostics,
    )
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
    # producers: each built node's declared output contract (drives the case `on:` enum
    # exhaustiveness + the dotted-field walk in ref/assert validation). A case node has no
    # contract, so it is naturally excluded.
    producers: dict[str, ValueContract] = {}
    for nid, node in nodes_built.items():
        contract = node.output_contract
        if contract is None:
            continue
        if type(contract) is not ValueContract:
            raise TypeError(
                "output producer contracts must be exact ValueContract values, "
                f"got {type(contract).__name__} for {nid!r}"
            )
        producers[nid] = contract

    # cross-flow type check: each `call` binding's source type vs the callable
    # input type (needs producers, so it runs after all nodes are built).
    check_ref_map_types(
        nodes_built,
        producers,
        flow_input_contracts,
        flow_wiring,
        diagnostics=diagnostics,
    )
    # loop `'a -> 'a` type contract (types): each carried field's seed-source Type vs the
    # body output's same-named field Type (needs producers, so it runs alongside the
    # cross-flow call/map type check above).
    check_loop_type_contract(
        nodes_built,
        producers,
        flow_input_contracts,
        flow_wiring,
        diagnostics=diagnostics,
    )
    # case `on:` enum-exhaustiveness (a validate-phase check — needs the assembled producer
    # types). Runs right after the type contracts, the same phase-position as the old
    # desugar-time check, so error ordering is preserved.
    check_case_exhaustive(
        descriptors,
        producers,
        diagnostics=diagnostics,
    )

    nodes: dict[str, Node] = dict(nodes_built)

    # flow.wiring/params key parity over EVERY node (catches an orphan/
    # missing source or a node absent from the wiring). Runs once the wiring is complete.
    check_wiring_parity(
        nodes,
        flow_wiring,
        diagnostics=diagnostics,
    )

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
        outputs_range=outputs_range,
        diagnostics=diagnostics,
    )
    # human_input gates: structurally check a literal `questions:` list at load, and
    # pin a ref-form `questions: ${name}` to a declared input (the prompt-scope rule).
    validate_human_questions(
        nodes,
        diagnostics=diagnostics,
    )
    # Node-local `asserts:` (per-node contract): validate node-locally + classify pre/post +
    # stamp onto each built leaf/call node. Shared by defs (a def's internal nodes too).
    validate_node_asserts(
        nodes_built,
        descriptors,
        assertion_site_at=node_assertion_site_at,
        diagnostics=diagnostics,
    )
    # Lower the flow-level `asserts:` into node-local asserts stamped on the synthesized
    # START/END. Fires uniformly via eval_node — a cross-node post
    # assert's `${producer.output}` becomes a real END input edge, so it survives child cloning.
    # `flow_assertion_ranges` comes from the same item cursors as each origin and stays aligned
    # 1:1 with the authored assertions, so lowered START/END asserts and bad-ref errors locate by
    # index — never by non-unique text. Source-free assertions carry `None`.
    lowered = lower_flow_asserts(
        asserts_section, flow_inputs, set(descriptors), producers,
        assert_ranges=flow_assertion_ranges,
        diagnostics=diagnostics,
    )

    # Run-ordering edges: depends_on (co-skip) / runs_after (pure order). Folded in
    # BEFORE root synthesis so an ordering-only dependent is demoted from root, and into
    # the cycle check below so an ordering cycle is loud.
    ordering_edges = infer_ordering_edges(
        descriptors,
        node_ids := set(descriptors),
        diagnostics=diagnostics,
    )

    # DAG assembly: synthesize the START_ID/END_ID boundary NODES + their edges over the
    # FULL reconciled body-edge set (the __start__/__end__ sentinels retire; the strings
    # persist as the boundary node ids). synthesize_boundary_graph mints the END_ID producer edges
    # (one per producer per output, keyed by output name, with the coalesce/optional stance)
    # + the START_ID root edges + (0-output) the bare START_ID->END_ID root edge,
    # and returns parity-clean START_ID/END_ID wiring.
    flow_outputs = _stamp_flow_output_origins(
        _flow_outputs(outputs_section),
        outputs_section,
        outputs_cursor=outputs_cursor,
        owner=owner,
        catalog_builder=catalog_builder,
        diagnostics=diagnostics,
    )
    structural = reconciled + ordering_edges
    boundary_nodes, boundary_edges, boundary_wiring = synthesize_boundary_graph(
        inputs, flow_outputs, node_ids, structural, boundary_field_ranges, lowered=lowered
    )
    nodes.update(boundary_nodes)
    flow_wiring.update(boundary_wiring)
    edges: list[Edge] = structural + boundary_edges

    # A cycle spans nodes; every actual member contributes its authored range.
    # START_ID/END_ID are acyclic by construction (START_ID has no incoming,
    # END_ID has no outgoing).
    node_ranges = {
        nid: d.range for nid, d in descriptors.items() if getattr(d, "range", None)
    }
    reject_cycles(
        edges,
        node_ids,
        node_ranges=node_ranges,
        diagnostics=diagnostics,
    )
    check_case_handles(nodes, edges, diagnostics=diagnostics)

    compiled = CompiledFlow.from_parts(nodes, edges, outputs=flow_outputs, wiring=flow_wiring,
                                       flow_llm_config=flow_llm_config or {})
    return ComposedFlow(
        compiled=compiled,
        version=version,
        name=name,
        description=description,
    )
