"""The post-parse conductor: assembled sections -> a runnable `ComposedFlow`.

`_assemble` is the pipeline shared by the top-level flow AND every in-file `defs:`
callable:

  flow parameters + authored parameter/contract/default origins -> parse nodes, lowering
  each tagged `nodes:` entry to a CODE node -> build every node
  kind-blind (`desc.build(ctx)`; physical node-output contracts compile at their
  existing leaf-owned build position; a case lowers
  in its own build) -> cross-flow type check + case exhaustiveness -> infer data
  edges + append each case's own edges + copy each host's route/ordering edges and data
  gates onto the nodes lifted out of it -> validate references + classify asserts
  -> attach authored assertion/flow-output binding origins -> synthesize the START_ID/END_ID
  boundary over ALL edges -> resolve each node's `isinstance` type table -> reject cycles +
  check if/else handles -> CompiledFlow.from_parts. One retained assertion item cursor
  supplies its carrier origin and Compose-only validation range; only the origin crosses
  into the semantic assertion.

Holds `ComposedFlow` (the compiled+validated flow the engine runs) plus the flow
`output:` parser and source-origin attachment. Ladder position: the LAST phase
before the loader — it may import every earlier phase (helpers/nodes/parse/desugar/
edges/validate/build.boundary); the loader imports `_assemble`/`ComposedFlow` back
from here. It imports NOTHING from `loader.py` (that would be a back-edge).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Optional, Protocol

from yian.compile.model import CompiledFlow, Edge, FlowOutput
from yian.diagnostics import (
    Diagnostic,
    DiagnosticFrame,
    Failure,
    OriginRef,
    OriginResolver,
    SourceRange,
)
from yian.nodes.base import Node, NodeKind
from yian.nodes.contracts import (
    Assertion,
    Binding,
    OutputContract,
    Parameter,
    PartialRecordContract,
    ValueContract,
    as_binding,
)
from yian.expr import ExpressionError
from yian.memory.contracts import DeclSite, MemoryDecl
from yian.typesys import TypeCheckError
from yian.typesys.types import TypeRegistry
from yian.typesys.values import Type
from yian.compose.validate.asserts import lower_flow_asserts
from yian.compose.build.boundary import synthesize_boundary_graph
from yian.compose.build.inline_inputs import (
    canonicalize_inline_type,
    explicit_type_hint,
    infer_inline_binding_type,
)
from yian.compose.edges.data import infer_data_edges
from yian.compose.edges.producers import _binding_producers
from yian.compose.nodes.base import BuildContext, BuildResult, NodeDescriptor
from yian.compose.nodes.call import CallDescriptor
from yian.compose.nodes.case import CaseDescriptor
from yian.compose.nodes.loop import LoopDescriptor
from yian.compose.nodes.map import MapDescriptor
from yian.compose.desugar.inline_bodies import (
    InlineBody,
    extract_inline_bodies,
)
from yian.compose.desugar.inline_calls import _synth_id_gen, desugar_inline_calls
from yian.compose.desugar.case_outputs import case_output_reads, expand_case_outputs
from yian.compose.desugar.value_entries import ValueEntryDescriptor, read_value_entry
from yian.compose.helpers.errors import LoadDiagnostics, LoadError
from yian.compose.helpers.isinstance_types import (
    collect_constructor_names,
    collect_isinstance_names,
    constructor_type_fn,
    expression_trees,
    resolve_constructor_type,
    resolve_isinstance_type,
)
from yian.compose.helpers.output_contracts import resolve_output_binding_contract
from yian.compose.helpers.source import nth, one_line_text
from yian.compose.helpers.tags import (
    QUOTED_STYLES,
    old_form_suggestion,
    old_typed_text,
    read_tagged_value,
    reject_tagged_output,
    suggest_tag,
    yaml_quoted,
)
from yian.compose.desugar.adaptive_questions import QPARAM, desugar_adaptive_questions
from yian.compose.desugar.case_targets import desugar_case_call_targets
from yian.compose.validate.case_exhaustive import (
    check_case_exhaustive,
    check_case_when_values,
    exhaustive_cases,
)
from yian.compose.validate.case_branches import check_case_branches
from yian.compose.validate.cycles import reject_cycles
from yian.compose.validate.human_questions import validate_human_questions
from yian.compose.validate.loop_types import check_loop_type_contract
from yian.compose.validate.narrowing import compute_narrowing
from yian.compose.edges.hosts import host_control_edges
from yian.compose.edges.ordering import infer_ordering_edges
from yian.compose.validate.call_types import check_call_map_types
from yian.compose.validate.references import validate_references
from yian.compose.validate.secret_literals import (
    check_authored_bindings,
    check_value_entry,
)
from yian.compose.validate.skips import (
    SkipModel,
    check_skipped_arithmetic,
    null_skipped_outputs,
)
from yian.compose.validate.tool_inputs import stamp_tool_input_types
from yian.compose.validate.typed_bindings import check_typed_inputs, check_typed_outputs
from yian.compose.validate.wiring_parity import check_wiring_parity
from yian.compose.parse.nodes import (
    normalize_node_signature_cursor,
    parse_nodes,
)
from yian.compose.source import SourceAccessError, SourceCursor
from yian.compose.source.catalog import _SourceCatalogBuilder
from yian.compose.helpers.types import (
    read_flow_inputs,
    read_source_flow_inputs,
    read_source_parameters,
    read_value_contract,
)
from yian.compose.validate.handles import check_case_handles
from yian.compose.validate.asserts import validate_node_asserts
from yian.compose.validate.constructors import constructor_errors

# Name for the single output of a bare whole-string `output:` binding. The value-mode
# EndNode returns the bare resolved value, so the name only labels the carrier.
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

    Produced by [`load_flow`][yian.load_flow] and consumed by
    [`run_flow`][yian.run_flow] / the engine. Immutable so it can be reused
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
            interactive-prompt header). `None` for a `defs:` child flow, which has no name.
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


def _stamp_binding_labels(build_results: dict[str, BuildResult]) -> None:
    """Give each param a desugar named for load errors the same phrase for runtime errors.

    A CASE ``on:`` or ``cases[i].when`` param takes its ``wiring_labels`` site name; a
    value entry's input takes its ``field_labels`` phrase. Other params keep the plain
    ``input '<param>'`` wording.

    Args:
        build_results (`dict[str, BuildResult]`):
            Built nodes keyed by final descriptor id; each node's ``binding_labels`` is
            filled in place.
    """
    for node_id, result in build_results.items():
        node = result.node
        for parameter in node.params or ():
            name = parameter.name
            wiring_label = node.wiring_labels.get(name)
            if wiring_label is not None:
                # `on:` names the YAML key; a runtime message adds its own colon.
                node.binding_labels[name] = result.site_name(
                    node_id, wiring_label.removesuffix(":")
                )
            elif (site := f"input {name!r}") in result.field_labels:
                node.binding_labels[name] = result.field_labels[site]


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
    its value owns the flow ``Binding``; a value entry's one input is the entry
    itself, so a runtime type failure points at it. A reserved MAP ``over`` or WAIT
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
            if isinstance(descriptor, ValueEntryDescriptor):
                input_cursors = {node_id: nodes_cursor.child(node_id)}
            else:
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

                if not descriptor.inputs or not node_cursor.has("inputs"):
                    continue
                inputs_cursor = node_cursor.child("inputs")
                input_cursors = {
                    name: inputs_cursor.child(name)
                    for name in descriptor.inputs
                    if inputs_cursor.has(name)
                }
            if node.params is None:
                continue
            parameter_indexes = {
                parameter.name: index
                for index, parameter in enumerate(node.params)
            }
            parameters = list(node.params)
            parameters_changed = False
            for name, entry_cursor in input_cursors.items():
                if name not in parameter_indexes or name not in wiring:
                    continue
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


def _split_typed_value(
    value: Any,
    cursor: SourceCursor | None,
    *,
    binding_label: str,
    site_path: tuple[str | int, ...],
    registry,
    catalog_builder: _SourceCatalogBuilder,
    diagnostics: LoadDiagnostics,
) -> tuple[Any, ValueContract | None] | None:
    """Split one binding value into its value and declared contract.

    A tagged value (`!int ${size.output.qty}`) is read through its tag. Untagged
    plain text in the old `Type = value` form is an error that shows the tag; a
    quoted or block value is text. Every authored value is read with the spans and
    tags inside it, so a container holding them comes back as a `ValueTemplate`,
    typed or not.

    Args:
        value (`Any`): The authored binding value.
        cursor (`SourceCursor | None`): Retained view of the value, or `None` for a
            value with no authored occurrence.
        binding_label (`str`): The binding's key name, used in error messages.
        site_path (`tuple[str | int, ...]`): Catalog site prefix for this binding.
        registry (`TypeRegistry`): The file's typedefs the type text resolves against.
        catalog_builder (`_SourceCatalogBuilder`): File-owned origin catalog builder.
        diagnostics (`LoadDiagnostics`): Factory for located load errors.

    Returns:
        `tuple[Any, ValueContract | None] | None`: The binding value and its declared
            contract (`None` for an untyped mapping or list), or `None` when the
            value is untyped and kept as it is: not a mapping or list, or not the
            value its cursor shows.

    Raises:
        `LoadError`: Raised when a tag is malformed, has no value after it, or does
            not fit its tagged literal, or the value is in the old `Type = value`
            form.
    """
    if cursor is not None and cursor.tag is not None:
        slot = read_tagged_value(
            cursor,
            registry,
            label=binding_label,
            site_path=site_path,
            catalog_builder=catalog_builder,
            diagnostics=diagnostics,
            ledger=catalog_builder.consumed_tags,
        )
        assert slot.contract is not None
        if not slot.has_value:
            spelled = suggest_tag(
                binding_label,
                slot.contract.type,
                "${...}",
                registry=registry,
                type_text=cursor.tag,
            )
            raise diagnostics.error(
                f"{binding_label!r}: a binding needs a value: {spelled}",
                range=cursor.range,
            )
        return slot.value, slot.contract
    style = cursor.scalar_style if cursor is not None else None
    old = old_typed_text(value, registry) if style not in QUOTED_STYLES else None
    if old is not None:
        typ, spelled, right = old
        suggestion = old_form_suggestion(
            binding_label, typ, spelled, right, registry=registry, placeholder="${...}"
        )
        # Builtin names such as `date` or `file` are also words, so the text may be
        # meant literally.
        raise diagnostics.error(
            f"binding {binding_label!r}: write the type as a tag: {suggestion}. "
            f"If you meant the text, quote it: "
            f"`{binding_label}: {yaml_quoted(value.strip())}`",
            range=cursor.range if cursor is not None else None,
        )
    # Only an authored mapping or list can hold nested spans or tags; a string
    # stays as written, and a carrier or a value replaced after parsing is not
    # what the cursor shows, so it is kept as given.
    if cursor is None or type(value) not in (dict, list) or not cursor.holds(value):
        return None
    slot = read_tagged_value(
        cursor,
        registry,
        label=binding_label,
        site_path=site_path,
        catalog_builder=catalog_builder,
        diagnostics=diagnostics,
        ledger=catalog_builder.consumed_tags,
    )
    return slot.value, None


def _split_typed_inputs(
    descriptors: dict[str, NodeDescriptor],
    nodes_cursor: SourceCursor,
    *,
    registry,
    owner: tuple[str, ...],
    catalog_builder: _SourceCatalogBuilder,
    diagnostics: LoadDiagnostics,
) -> dict[str, NodeDescriptor]:
    """Split each typed `input:` binding into its value and declared contract.

    A binding tagged `qty: !int ${size.output.qty}` keeps `${size.output.qty}` as
    its value, and the compiled `int` contract lands in the descriptor's
    `input_types` under the parameter name; the retained cursor locates both errors
    and the contract origin.

    Args:
        descriptors (`dict[str, NodeDescriptor]`): Located descriptors keyed by node id.
        nodes_cursor (`SourceCursor`): Retained view of the flow's node mapping.
        registry (`TypeRegistry`): The file's typedefs the type text resolves against.
        owner (`tuple[str, ...]`): Complete lexical owner path for catalog sites.
        catalog_builder (`_SourceCatalogBuilder`): File-owned origin catalog builder.
        diagnostics (`LoadDiagnostics`): Factory for located load errors.

    Returns:
        `dict[str, NodeDescriptor]`: The descriptors, with typed inputs split; a value
            entry, already read through its tag, is returned unchanged.

    Raises:
        `LoadError`: Raised when a typed binding's type or value is malformed, a
            binding is in the old `Type = value` form, or a retained cursor cannot be
            traversed.
    """
    split: dict[str, NodeDescriptor] = {}
    for node_id, descriptor in descriptors.items():
        # A value entry's one input was read through its tag when it was parsed.
        if not descriptor.inputs or isinstance(descriptor, ValueEntryDescriptor):
            split[node_id] = descriptor
            continue
        try:
            inputs_cursor: SourceCursor | None = None
            if nodes_cursor.has(node_id):
                node_cursor = normalize_node_signature_cursor(nodes_cursor.child(node_id))
                if node_cursor.has("inputs"):
                    inputs_cursor = node_cursor.child("inputs")
            inputs = dict(descriptor.inputs)
            input_types = dict(descriptor.input_types)
            changed = False
            for name, value in descriptor.inputs.items():
                cursor = (
                    inputs_cursor.child(name)
                    if inputs_cursor is not None and inputs_cursor.has(name)
                    else None
                )
                typed = _split_typed_value(
                    value,
                    cursor,
                    binding_label=name,
                    site_path=(*owner, "node", node_id, "input", name),
                    registry=registry,
                    catalog_builder=catalog_builder,
                    diagnostics=diagnostics,
                )
                if typed is None:
                    continue
                inputs[name], contract = typed
                if contract is not None:
                    input_types[name] = contract
                changed = True
        except SourceAccessError as exc:
            raise diagnostics.error(str(exc), range=exc.range) from exc
        split[node_id] = (
            replace(descriptor, inputs=inputs, input_types=input_types)
            if changed
            else descriptor
        )
    return split


def _binding_style(
    nodes_cursor: SourceCursor,
    node_id: str,
    field: str,
    key: str,
    *,
    value_entries: frozenset[str],
    diagnostics: LoadDiagnostics,
) -> str | None:
    """Return the authored YAML scalar style of one node binding.

    A value entry's one binding is the entry itself, so its style is the entry's.

    Args:
        nodes_cursor (`SourceCursor`): Retained view of the flow's node mapping.
        node_id (`str`): The node that owns the binding.
        field (`str`): The binding field (`inputs`, or a scalar field such as `over`).
        key (`str`): The binding name in a mapping field; equals `field` for a scalar one.
        value_entries (`frozenset[str]`): The ids of the flow's value entries.
        diagnostics (`LoadDiagnostics`): Factory for located load errors.

    Returns:
        `str | None`: The style, or `None` for a synthesized node or binding.

    Raises:
        `LoadError`: Raised when the retained cursor cannot be traversed.
    """
    try:
        if not nodes_cursor.has(node_id):
            return None
        if node_id in value_entries:
            if (field, key) != ("inputs", node_id):
                return None
            return nodes_cursor.child(node_id).scalar_style
        cursor = normalize_node_signature_cursor(nodes_cursor.child(node_id))
        if not cursor.has(field):
            return None
        cursor = cursor.child(field)
        if key != field:
            if not cursor.has(key):
                return None
            cursor = cursor.child(key)
        return cursor.scalar_style
    except SourceAccessError as exc:
        raise diagnostics.error(str(exc), range=exc.range) from exc


def _output_style(
    outputs_cursor: SourceCursor | None,
    key: Any,
    *,
    diagnostics: LoadDiagnostics,
) -> str | None:
    """Return the authored YAML scalar style of one flow output binding.

    Args:
        outputs_cursor (`SourceCursor | None`): Retained authored `output:` value.
        key (`Any`): The output's record field name, or `None` for a single output.
        diagnostics (`LoadDiagnostics`): Factory for located load errors.

    Returns:
        `str | None`: The style, or `None` for a generated section.

    Raises:
        `LoadError`: Raised when the retained cursor cannot be traversed.
    """
    if outputs_cursor is None:
        return None
    try:
        if key is None:
            return outputs_cursor.scalar_style
        if not outputs_cursor.has(key):
            return None
        return outputs_cursor.child(key).scalar_style
    except SourceAccessError as exc:
        raise diagnostics.error(str(exc), range=exc.range) from exc


def _case_output_cursor(
    nodes_cursor: SourceCursor,
    node_id: str,
    *,
    diagnostics: LoadDiagnostics,
) -> SourceCursor | None:
    """Return the authored `output:` of one node, or `None` when it has none.

    Args:
        nodes_cursor (`SourceCursor`): Retained view of the flow's node mapping.
        node_id (`str`): The case node's id.
        diagnostics (`LoadDiagnostics`): Factory for located load errors.

    Returns:
        `SourceCursor | None`: The `output:` value, or `None` for a synthesized node or a
        node without `output:`.

    Raises:
        `LoadError`: Raised when the retained cursor cannot be traversed.
    """
    try:
        if not nodes_cursor.has(node_id):
            return None
        cursor = normalize_node_signature_cursor(nodes_cursor.child(node_id))
        return cursor.child("outputs") if cursor.has("outputs") else None
    except SourceAccessError as exc:
        raise diagnostics.error(str(exc), range=exc.range) from exc


def _attach_input_contracts(
    descriptors: dict[str, NodeDescriptor],
    flow_wiring: dict[str, dict],
) -> None:
    """Put each declared input type on its node's final wiring binding.

    Desugaring may rewrite a typed binding's value (an inline `call(...)` becomes a
    synthetic node's output, a `${case.output}` becomes a coalesce), so the contract
    is keyed by parameter name rather than by value.

    Args:
        descriptors (`dict[str, NodeDescriptor]`): Final descriptors keyed by node id.
        flow_wiring (`dict[str, dict]`): Mutable final binding maps keyed by node id.

    Raises:
        RuntimeError: A node has a declared input type but no wiring binding by
            that name.
    """
    for node_id, descriptor in descriptors.items():
        wiring = flow_wiring.get(node_id)
        if wiring is None:
            continue
        for name, contract in descriptor.input_types.items():
            # Every kind keeps its authored inputs in its wiring, so a declared type
            # with no binding to land on would be silently unchecked at runtime.
            if name not in wiring:
                raise RuntimeError(
                    f"node {node_id!r}: typed input {name!r} has no wiring binding"
                )
            wiring[name] = replace(as_binding(wiring[name]), contract=contract)


def _check_output_names(
    outputs_section: dict,
    outputs_cursor: SourceCursor | None,
    *,
    diagnostics: LoadDiagnostics,
) -> None:
    """Reject a flow `output:` record whose field name is not a string.

    YAML reads an unquoted `true`, `null`, or `1` key as a bool, None, or
    number; naming the output after its `str()` would rename it, and `1:` beside
    `'1':` would drop one output.

    Raises:
        `LoadError`: Raised at the offending key when a field name is not a string.
    """
    for key in outputs_section:
        if isinstance(key, str):
            continue
        key_range = None
        authored = str(key)
        if outputs_cursor is not None and outputs_cursor.has(key):
            key_range = outputs_cursor.child(key).key_range
            authored = one_line_text(outputs_cursor.document, key_range) or authored
        raise diagnostics.error(
            f"flow output name `{authored}` reads as {key!r} ({type(key).__name__}), "
            f"not a string; quote the key: `'{authored}':`",
            range=key_range or (outputs_cursor.range if outputs_cursor else None),
        )


def _split_typed_outputs(
    outputs_section,
    outputs_cursor: SourceCursor | None,
    *,
    registry,
    owner: tuple[str, ...],
    catalog_builder: _SourceCatalogBuilder,
    diagnostics: LoadDiagnostics,
) -> tuple[Any, dict[str, ValueContract]]:
    """Split each typed flow output into its value and declared contract.

    A record field written `decision: !Decision ${d.output}` keeps `${d.output}`
    as its value; a bare
    `output: !Size ${size.output}` is the single output named `result`. A tagged
    whole section is that single output even when its value is a mapping
    (`output: !T {a: 1}`), so it comes back as a `Binding` carrier rather than a
    dict. Recognition and locations follow the retained cursor exactly as for node
    inputs.

    Args:
        outputs_section (`dict | object | None`): The raw `output:` section.
        outputs_cursor (`SourceCursor | None`): Retained authored `output:` value, or
            `None` for a generated section.
        registry (`TypeRegistry`): The file's typedefs the type text resolves against.
        owner (`tuple[str, ...]`): Complete lexical owner path for catalog sites.
        catalog_builder (`_SourceCatalogBuilder`): File-owned origin catalog builder.
        diagnostics (`LoadDiagnostics`): Factory for located load errors.

    Returns:
        `tuple[Any, dict[str, ValueContract]]`: The section with typed values split
            (a `Binding` for a tagged whole section), and the declared contracts
            keyed by output name.

    Raises:
        `LoadError`: Raised when a record field name is not a string, a typed output's
            type or value is malformed, or the retained cursor cannot be traversed.
    """
    def split_one(value, cursor, binding_label: str, index: int):
        """Split one output value at its positional catalog site."""
        return _split_typed_value(
            value,
            cursor,
            binding_label=binding_label,
            site_path=(*owner, "output", index),
            registry=registry,
            catalog_builder=catalog_builder,
            diagnostics=diagnostics,
        )

    try:
        if outputs_cursor is not None and outputs_cursor.tag is not None:
            # A tagged whole section is one typed value, even a mapping: the carrier
            # is not a dict, so every later pass takes it as the single output.
            typed = split_one(outputs_section, outputs_cursor, "output", 0)
            assert typed is not None and typed[1] is not None
            value, contract = typed
            carrier = Binding(source=value, contract=contract)
            return carrier, {_SINGLE_OUTPUT_NAME: contract}
        if outputs_section is None:
            return None, {}
        if not isinstance(outputs_section, dict):
            typed = split_one(outputs_section, outputs_cursor, "output", 0)
            if typed is None:
                return outputs_section, {}
            value, contract = typed
            if contract is None:
                return value, {}
            return value, {_SINGLE_OUTPUT_NAME: contract}
        _check_output_names(outputs_section, outputs_cursor, diagnostics=diagnostics)
        split = dict(outputs_section)
        output_types: dict[str, ValueContract] = {}
        for index, (key, value) in enumerate(outputs_section.items()):
            cursor = (
                outputs_cursor.child(key)
                if outputs_cursor is not None and outputs_cursor.has(key)
                else None
            )
            typed = split_one(value, cursor, str(key), index)
            if typed is None:
                continue
            split[key], contract = typed
            if contract is not None:
                output_types[str(key)] = contract
    except SourceAccessError as exc:
        raise diagnostics.error(str(exc), range=exc.range) from exc
    return split, output_types


def _flow_outputs(outputs) -> list[FlowOutput]:
    """Read the flow `output:` section into carriers for the boundary EndNode.

    A name -> binding map becomes one `FlowOutput` per entry (a record-mode
    END returns a dict keyed by every name, `{}` for an empty map); a bare
    whole-string binding (`${note.output}`) or a tagged section's `Binding` carrier
    is one output named `result` (a value-mode END returns the bare value). A
    `None` section -> no declared outputs (a value-mode END returns `None`).

    Args:
        outputs (`dict | str | Binding | None`):
            The raw `output:` section: a name -> binding map, a single binding string,
            a tagged single output's `Binding`, or `None`.
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


def _site_range(
    item: SourceRange | None,
    field_ranges: dict[tuple[str, str], SourceRange],
    field: str,
    node_range: SourceRange | None,
) -> SourceRange | None:
    """Return a site's own range, else its node field's range, else the node's range.

    Args:
        item (`SourceRange | None`):
            The site's own range, `None` when it has none.
        field_ranges (`dict[tuple[str, str], SourceRange]`):
            The node's field ranges, keyed like `BuildResult.field_ranges`.
        field (`str`):
            The node field holding the site, e.g. `"asserts"`.
        node_range (`SourceRange | None`):
            The node's range.

    Returns:
        `SourceRange | None`:
            The narrowest range available.
    """
    return item or field_ranges.get(("field", field)) or node_range


# The `isinstance` type names and the constructor names one expression site uses.
TypeNames = tuple[list[str], list[str]]


def _fill_type_tables(
    nodes: dict[str, Node],
    flow_wiring: dict[str, dict[str, Any]],
    build_results: dict[str, BuildResult],
    descriptors: dict[str, NodeDescriptor],
    flow_asserts: list[Assertion],
    flow_assertion_ranges: list[SourceRange | None],
    outputs_cursor: SourceCursor | None,
    outputs_range: SourceRange | None,
    registry: TypeRegistry,
    diagnostics: LoadDiagnostics,
) -> None:
    """Set each node's `types` to the type names its `isinstance` calls and constructors use.

    A node evaluates its wiring sources, its asserts, a CASE's `when:`s and a LOOP's
    predicate with its own table. Names resolve against this flow's `registry`; a
    constructor name must be a `typedefs:` name.

    Args:
        nodes (`dict[str, Node]`):
            Every node of the flow, START/END included, keyed by id. Mutated in place.
        flow_wiring (`dict[str, dict[str, Any]]`):
            Each node's `param -> source` wiring.
        build_results (`dict[str, BuildResult]`):
            The body nodes' build results, for their field ranges and input site names.
        descriptors (`dict[str, NodeDescriptor]`):
            The body nodes' descriptors, for their authored asserts and ranges.
        flow_asserts (`list[Assertion]`):
            The flow's authored `asserts:` carriers, aligned with `flow_assertion_ranges`.
        flow_assertion_ranges (`list[SourceRange | None]`):
            Each flow assert's range.
        outputs_cursor (`SourceCursor | None`):
            The authored `output:` value; a mapping locates each named output.
        outputs_range (`SourceRange | None`):
            The whole `output:` range, used when an output has no range of its own.
        registry (`TypeRegistry`):
            The flow's `typedefs:` registry.
        diagnostics (`LoadDiagnostics`):
            The flow document's diagnostic factory, for located errors.

    Raises:
        `LoadError`:
            Raised at the offending site for an unknown type or constructor name, or
            for `isinstance` or a constructor in a reserved wiring key (WAIT `until:`,
            MAP `over:`).
    """
    resolved: dict[str, Type] = {}

    def table(names: TypeNames, where: str, rng: SourceRange | None) -> dict[str, Type]:
        """Resolve `names`, raising at `rng` for an unknown one."""
        isinstance_names, constructors = names
        for name in isinstance_names:
            if name not in resolved:
                try:
                    resolved[name] = resolve_isinstance_type(name, registry)
                except TypeCheckError as exc:
                    raise diagnostics.error(f"{where}: {exc}", range=rng) from exc
        for name in constructors:
            if name in resolved and name in registry:
                continue  # a typedef name resolves the same for isinstance and constructors
            try:
                resolved[name] = resolve_constructor_type(name, registry)
            except TypeCheckError as exc:
                raise diagnostics.error(f"{where}: {exc}", range=rng) from exc
        return {name: resolved[name] for name in (*isinstance_names, *constructors)}

    def output_range(name: str) -> SourceRange | None:
        """One flow output's authored range, else the whole `output:` range."""
        try:
            if outputs_cursor is not None and outputs_cursor.has(name):
                return outputs_cursor.child(name).range
        except SourceAccessError:
            pass  # a bare `output:` value has no named children
        return outputs_range

    def checked(
        names: TypeNames,
        source: Any,
        where: str,
        rng: SourceRange | None,
        *,
        condition: bool = False,
    ) -> dict[str, Type]:
        """Resolve `names` as `table` does, then reject a constructor call in `source`
        that can never succeed whatever the references hold.

        Only for condition sites: `validate_references` already checked every binding's
        constructor calls, with the references typed."""
        resolved_names = table(names, where, rng)
        if names[1]:
            for tree in expression_trees(source, condition=condition):
                problems = constructor_errors(tree, registry)
                if problems:
                    raise diagnostics.error(f"{where}: {problems[0]}", range=rng)
        return resolved_names

    def names_in(source: Any, *, condition: bool = False) -> TypeNames:
        """The `isinstance` and constructor names `source` uses.

        Earlier passes report a source that won't parse."""
        try:
            return (
                collect_isinstance_names(source, condition=condition),
                collect_constructor_names(source, condition=condition),
            )
        except ExpressionError:
            return [], []

    # Authored flow asserts first, so an unknown name is located at its own assert. The
    # lowered START/END asserts below use the same names.
    for index, assertion in enumerate(flow_asserts):
        where = f"flow assert {assertion.display!r}"
        rng = flow_assertion_ranges[index]
        checked(
            names_in(assertion.expression, condition=True),
            assertion.expression,
            where,
            rng,
            condition=True,
        )

    for nid, node in nodes.items():
        result = build_results.get(nid)
        desc = descriptors.get(nid)
        franges = result.field_ranges if result is not None else {}
        node_range = result.node_range if result is not None else None
        types: dict[str, Type] = {}
        reserved = node.reserved_wiring_keys()
        for param, source in flow_wiring.get(nid, {}).items():
            if node.kind is NodeKind.END:  # END's wiring carries the flow outputs
                where = node.binding_labels.get(param, f"flow output {param!r}")
                rng = output_range(param)
            else:
                # A synthesized param's label may end in ':' (`on:`); the message adds one.
                site = node.wiring_ref_label(param).removesuffix(":")
                where = (
                    result.site_name(nid, site) if result is not None else f"node {nid!r} {site}"
                )
                key = ("field", param) if param in reserved else ("input", param)
                rng = _site_range(franges.get(key), franges, "inputs", node_range)
            names = names_in(source)
            if param in reserved and (names[0] or names[1]):
                used = "isinstance" if names[0] else f"{names[1][0]}(...)"
                raise diagnostics.error(
                    f"node {nid!r}: {used} is not supported in a {node.kind.value} "
                    f"node's {param}:",
                    range=rng,
                )
            types.update(table(names, where, rng))
        authored = list(getattr(desc, "asserts", None) or [])
        assert_ranges = list(getattr(desc, "assert_ranges", None) or [])
        for index, expression in enumerate(authored):
            where = f"node {nid!r} assert {expression!r}"
            rng = _site_range(nth(assert_ranges, index), franges, "asserts", node_range)
            names = names_in(expression, condition=True)
            types.update(checked(names, expression, where, rng, condition=True))
        if node.kind in (NodeKind.START, NodeKind.END):
            # START/END evaluate the lowered flow asserts, whose names were located above.
            for assertion in [*node.pre_asserts, *node.post_asserts]:
                names = names_in(assertion.expression, condition=True)
                types.update(table(names, f"flow assert {assertion.display!r}", None))
        when_ranges = list(getattr(desc, "when_ranges", None) or [])
        for index, case in enumerate(getattr(node, "cases", None) or ()):
            where = f"case node {nid!r} cases[{index}].when"
            rng = _site_range(nth(when_ranges, index), franges, "cases", node_range)
            names = names_in(case.when, condition=True)
            types.update(checked(names, case.when, where, rng, condition=True))
        predicate = getattr(node, "predicate", None)
        if isinstance(predicate, str):
            keyword = getattr(node, "predicate_kind", "while")
            where = f"loop node {nid!r} {keyword}"
            rng = _site_range(None, franges, keyword, node_range)
            names = names_in(predicate, condition=True)
            types.update(checked(names, predicate, where, rng, condition=True))
        node.types = types


def _value_cases(descriptors: dict, read: frozenset[str]) -> frozenset[str]:
    """Return the ids of the cases that produce a value.

    A case produces a value when it is in `read`; a case that is a `then:`/`else:` target
    of a value-producing case produces one too.

    Args:
        descriptors (`dict`): Node descriptors keyed by node id.
        read (`frozenset[str]`): The case ids some binding reads as `${<case>.output}`.

    Returns:
        `frozenset[str]`: The value-producing case ids.
    """
    cases = {
        nid: d for nid, d in descriptors.items() if isinstance(d, CaseDescriptor)
    }
    pending = [nid for nid in cases if nid in read]
    found: set[str] = set()
    while pending:
        nid = pending.pop()
        if nid in found:
            continue
        found.add(nid)
        case = cases[nid]
        targets = [c.get("then") for c in case.cases] + [case.else_]
        pending.extend(t for t in targets if isinstance(t, str) and t in cases)
    return frozenset(found)


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
    uses_aliases: tuple[str, ...] = (),
    version: Optional[str] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
    flow_llm_config: Optional[dict] = None,
    flow_env: Optional[dict] = None,
    boundary_field_ranges: Optional[dict] = None,
    asserts_cursor: SourceCursor | None = None,
    outputs_range: Optional[SourceRange] = None,
    outputs_cursor: SourceCursor | None = None,
    flow_id: str | None = None,
    memories: tuple[MemoryDecl, ...] = (),
    memories_cursor: SourceCursor | None = None,
) -> ComposedFlow:
    """Assemble parsed sections into a `ComposedFlow` — the post-parse pipeline shared by
    the top-level flow and every in-file def:

      flow parameters -> parse nodes (a tagged entry lowers to a CODE node returning its
      value, after a secret-literal check) -> split tagged node inputs and flow outputs
      into value + declared contract -> check case branches (a node or a call; a value or
      a case `output:` is a load error showing the value entry) -> desugar (inline calls,
      adaptive questions, case call targets, `${case.output}` to a coalesce over the
      case's branch targets) -> build
      every node kind-blind
      (`desc.build(ctx)`; physical node-output contracts compile at their existing
      leaf-owned position; a case lowers in its own build) -> attach declared input
      contracts to the node wiring -> cross-flow type check + case exhaustiveness ->
      infer data edges + append each case's own edges + copy each host's route/ordering
      edges and data gates onto the nodes lifted out of it -> validate references +
      classify asserts -> attach authored flow-output binding origins + declared output
      contracts
      -> synthesize roots/terminals -> resolve `isinstance` type tables -> reject cycles +
      check if/else handles -> CompiledFlow.from_parts.

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
    children. `flow_id` retains the authored or generated semantic identity;
    `memories` and its optional cursor carry the flow-owned declarations; each is
    restamped with this flow's `DeclSite` and gets one origin per entry.
    `boundary_field_ranges` locates authored flow-input declarations during Compose type
    compilation; `asserts_cursor` is the single authority for each flow assertion's
    opaque origin and Compose-only validation range. `outputs_range` is the retained
    authored codomain range when one exists, while `outputs_cursor` retains each
    physically authored output binding occurrence.
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
    if type(memories) is not tuple or any(type(decl) is not MemoryDecl for decl in memories):
        raise TypeError("memories must be a tuple of exact MemoryDecl values")
    if memories:
        stamped: list[MemoryDecl] = []
        for decl in memories:
            origin = decl.origin
            if memories_cursor is not None:
                path = (*owner, "memories", decl.name)
                if memories_cursor.has(decl.name):
                    origin = catalog_builder.register(
                        path, memories_cursor.child(decl.name), anchor="key"
                    )
                else:  # sugar: the whole scalar value is the one entry
                    origin = catalog_builder.register(path, memories_cursor)
            stamped.append(
                MemoryDecl(
                    decl.name, decl.type, decl.config, DeclSite((), owner, None), origin
                )
            )
        memories = tuple(stamped)
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
            if node_id in value_entries or not authored_nodes_cursor.has(node_id):
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
            if node_id in value_entries or not nodes_cursor.has(node_id):
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

    def read_authored_node_variables(
        node_id: str,
        declarations: Any,
    ) -> list[Parameter] | None:
        """Compile one physical node's variables through its normalized cursor."""
        try:
            if node_id in value_entries or not nodes_cursor.has(node_id):
                return None
            node_cursor = normalize_node_signature_cursor(nodes_cursor.child(node_id))
            if not node_cursor.has("variables"):
                return None
            variables_cursor = node_cursor.child("variables")
            if variables_cursor.range is None:
                return None
            return read_source_parameters(
                variables_cursor.with_value(declarations),
                registry,
                site_path=(*owner, "node", node_id),
                catalog_builder=catalog_builder,
                diagnostics=diagnostics,
                field_name="variables",
                declaration_label="variable",
                require_default=True,
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
    # Static inference types a `T(...)` constructor by this flow's own typedefs.
    constructor_type = constructor_type_fn(registry)
    flow_inputs = {parameter.name for parameter in inputs}

    def read_value(node_id: str, cursor: SourceCursor) -> ValueEntryDescriptor:
        """Lower one tagged `nodes:` entry, after rejecting a secret literal in it."""
        check_value_entry(node_id, cursor, diagnostics)
        return read_value_entry(
            node_id,
            cursor,
            registry=registry,
            owner=owner,
            catalog_builder=catalog_builder,
            diagnostics=diagnostics,
        )

    descriptors = parse_nodes(nodes_cursor, read_value=read_value)
    # Value entries have no node body, so every pass below that re-reads a node's body
    # from `nodes_cursor` skips them or reads the entry itself.
    value_entries = frozenset(
        node_id
        for node_id, descriptor in descriptors.items()
        if isinstance(descriptor, ValueEntryDescriptor)
    )
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
            if node_id in value_entries:
                located_descriptors[node_id] = located
                continue
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
    collide = sorted(set(uses_aliases) & set(descriptors))
    if collide:
        raise diagnostics.error(f"uses: alias {collide} collides with a node id")

    # Eager `uses:` resolution (mirrors eager defs): resolve every alias now so a
    # declared-but-uncalled broken/`hub:`/missing target is loud at load, not at first call.
    # Source order, so the first broken alias is the one reported, the same on every run.
    for alias in uses_aliases:
        scope.resolve(alias)

    def output_range(name: Any) -> SourceRange | None:
        """One flow output's authored range, else the whole `output:` range."""
        try:
            if outputs_cursor is not None and outputs_cursor.has(name):
                return outputs_cursor.child(name).range
        except SourceAccessError:
            pass  # a bare `output:` value has no named children
        return outputs_range

    # Reject a literal under a secret name while values are still as authored: the
    # tagged split and the desugars below may echo a value in their own errors.
    check_authored_bindings(
        descriptors,
        outputs_section,
        binding_style_at=lambda node_id, field, key: _binding_style(
            nodes_cursor,
            node_id,
            field,
            key,
            value_entries=value_entries,
            diagnostics=diagnostics,
        ),
        output_style_at=lambda key: _output_style(
            outputs_cursor, key, diagnostics=diagnostics
        ),
        output_range=output_range,
        diagnostics=diagnostics,
    )

    # A node's `output:` is a type declaration; a tag there is rejected before a
    # declaration with nothing after its tag reads as no output at all. A case has no
    # `output:`; the case-branch check below rejects it, tagged or not, with the rewrite.
    try:
        for node_id, descriptor in descriptors.items():
            if isinstance(descriptor, CaseDescriptor):
                continue
            if node_id not in value_entries and nodes_cursor.has(node_id):
                node_cursor = normalize_node_signature_cursor(nodes_cursor.child(node_id))
                if node_cursor.has("outputs"):
                    reject_tagged_output(node_cursor.child("outputs"), diagnostics)
    except SourceAccessError as exc:
        raise diagnostics.error(str(exc), range=exc.range) from exc

    # Split tagged (`name: !Type value`) input and output bindings before any
    # desugar reads their values, so an inline `call(...)` or
    # `${case.output}` in the value desugars as usual.
    descriptors = _split_typed_inputs(
        descriptors,
        nodes_cursor,
        registry=registry,
        owner=owner,
        catalog_builder=catalog_builder,
        diagnostics=diagnostics,
    )
    outputs_section, output_types = _split_typed_outputs(
        outputs_section,
        outputs_cursor,
        registry=registry,
        owner=owner,
        catalog_builder=catalog_builder,
        diagnostics=diagnostics,
    )

    # A case branch names a node or calls a flow, and a case has no `output:`. Runs
    # BEFORE the inline-call desugar so a branch's text is still as authored.
    check_case_branches(
        descriptors,
        registry=registry,
        input_types=flow_input_types,
        output_at=lambda node_id: _case_output_cursor(
            nodes_cursor, node_id, diagnostics=diagnostics
        ),
        diagnostics=diagnostics,
    )

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
        binding_style_at=lambda node_id, field, key: _binding_style(
            nodes_cursor,
            node_id,
            field,
            key,
            value_entries=value_entries,
            diagnostics=diagnostics,
        ),
        output_style_at=lambda key: _output_style(
            outputs_cursor, key, diagnostics=diagnostics
        ),
        typed_outputs=(
            set(output_types) if isinstance(outputs_section, dict) else {None}
        )
        if output_types
        else (),
        next_id=mint,
        type_names=frozenset(registry),
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

    # The cases whose value is used, taken before the expansion below rewrites the reads;
    # each must be exhaustive (checked once producer types are known).
    value_cases = _value_cases(
        descriptors, case_output_reads(descriptors, outputs_section)
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
        ) -> Type | OutputContract:
            """Resolve one parent producer through this assembly's build cache."""
            def unavailable_type() -> LoadError:
                return diagnostics.error(
                    f"cannot infer type for inline binding {binding_label!r}: "
                    f"producer {producer_id!r} has no available static type; "
                    + explicit_type_hint(binding_label),
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
            if type(output_contract) not in (ValueContract, PartialRecordContract):
                raise TypeError(
                    "output producer contracts must be ValueContract or "
                    "PartialRecordContract values, "
                    f"got {type(output_contract).__name__} for {producer_id!r}"
                )
            return output_contract

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
                    type_fn=constructor_type,
                    binding_label=binding_label,
                    binding_range=binding_range,
                    diagnostics=diagnostics,
                )
                if not over_type.kind.is_list() or over_type.element is None:
                    raise diagnostics.error(
                        f"cannot infer type for inline binding {binding_label!r}: "
                        "MAP `over:` must have a proven list element type; "
                        + explicit_type_hint(binding_label),
                        range=binding_range,
                    )
                cached_item_type = over_type.element
                return cached_item_type

            item_type = map_item_type

        child_inputs: list[Parameter] = []
        for key, value in (descriptor.inputs or {}).items():
            binding_label = f"{descriptor.id}.input.{key}"
            binding_range = body.input_ranges.get(key)
            declared = descriptor.input_types.get(key)
            # A typed binding states the parameter's type, and the binder checks the
            # value against it at runtime, so its value need not be inferable.
            if declared is not None:
                parameter_type = declared.type
            else:
                parameter_type = infer_inline_binding_type(
                    value,
                    parent_input_types=flow_input_types,
                    producer_type=producer_type,
                    type_fn=constructor_type,
                    item_type=item_type,
                    binding_label=binding_label,
                    binding_range=binding_range,
                    diagnostics=diagnostics,
                )
            canonical = canonicalize_inline_type(parameter_type)
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
            flow_id=body.flow_id,
        )

    def read_binding_contract(
        binding_label: str,
        binding: Any,
        binding_range: SourceRange | None,
    ) -> OutputContract | None:
        """Resolve one parent binding's contract, building referenced producers on demand."""
        cache = descriptor_cache
        if cache is None:
            raise RuntimeError(
                "descriptor cache was not initialized before binding contract resolution"
            )
        producers: dict[str, OutputContract] = {}
        for producer_id in dict.fromkeys(_binding_producers(binding)):
            if producer_id not in descriptors:
                continue
            contract = cache.build_one(
                producer_id,
                via_label=binding_label,
                via_range=binding_range,
            ).node.output_contract
            if contract is not None:
                producers[producer_id] = contract
        return resolve_output_binding_contract(
            binding,
            producers,
            flow_input_contracts,
            type_fn=constructor_type,
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
        variable_declaration_reader=read_authored_node_variables,
        binding_contract_reader=read_binding_contract,
        owner=owner,
    )
    descriptor_cache = _DescriptorBuildCache(descriptors, ctx_base)
    build_results = descriptor_cache.build_in_source_order()
    nodes_built: dict[str, Node] = {
        nid: result.node for nid, result in build_results.items()
    }
    flow_wiring: dict[str, dict] = {
        nid: result.wiring for nid, result in build_results.items()
    }
    _stamp_binding_labels(build_results)
    _stamp_node_wiring_origins(
        descriptors,
        nodes_cursor,
        nodes_built,
        flow_wiring,
        owner=owner,
        catalog_builder=catalog_builder,
        diagnostics=diagnostics,
    )
    _attach_input_contracts(descriptors, flow_wiring)
    # A case's own data+control edges; every other kind returns an empty tuple.
    case_edges: list[Edge] = [
        edge
        for result in build_results.values()
        for edge in result.edges
    ]
    # Stamp each built node's effective static config: the flow-level `env:` default merged
    # under the node's own `env:` (node override wins), flow-local (a called child flow has its
    # own flow env). The engine core never reads `Node.env` — only the node's `forward()` path does.
    _flow_env = flow_env or {}
    for nid, node in nodes_built.items():
        node.env = {**_flow_env, **getattr(descriptors[nid], "env", {})}
    # producers: each built node's declared output contract (drives the case `on:` enum
    # exhaustiveness + the dotted-field walk in ref/assert validation). A case node has no
    # contract, so it is naturally excluded.
    producers: dict[str, OutputContract] = {}
    for nid, node in nodes_built.items():
        contract = node.output_contract
        if contract is None:
            continue
        if type(contract) not in (ValueContract, PartialRecordContract):
            raise TypeError(
                "output producer contracts must be ValueContract or "
                "PartialRecordContract values, "
                f"got {type(contract).__name__} for {nid!r}"
            )
        producers[nid] = contract

    # Data edges — the projection of flow.wiring (leaf + WAIT + call/map); a case owns its own
    # data+control edges (`case_edges`, from each `CaseDescriptor.build`), appended below.
    data_edges = infer_data_edges(descriptors, flow_wiring)
    # A node lifted out of an authored node (an inline `call(...)`, an adaptive composer) runs
    # only on its host's routes, after its host's ordering, and once its host's other data
    # inputs settle, skipping when they would skip the host. So it gets copies of the routes
    # and ordering plus data gates here, before narrowing and the skip model read the graph.
    host_edges = host_control_edges(descriptors, data_edges + case_edges)
    # What each node may assume about union values from the CASE branches that route it.
    narrowing = compute_narrowing(
        descriptors,
        data_edges + case_edges + host_edges,
        producers,
        flow_input_contracts,
        registry=registry,
    )
    # Runs before every pass that reads narrowing, so a typo'd tag is reported first.
    check_case_when_values(
        descriptors,
        producers,
        flow_input_contracts,
        narrowing=narrowing,
        diagnostics=diagnostics,
    )

    # Typed node inputs that can never fit their declared type. The CALL, MAP, and LOOP
    # checks below take a typed binding's declared type as its source, so a value that
    # cannot fit that type must be reported here first, against the binding itself.
    check_typed_inputs(
        flow_wiring,
        producers,
        flow_input_contracts,
        build_results,
        narrowing=narrowing,
        type_fn=constructor_type,
        registry=registry,
        resolve_origin=catalog_builder.resolve_registered,
        diagnostics=diagnostics,
    )
    # The run-start tool check compares these with the host tool's declared input types.
    stamp_tool_input_types(
        flow_wiring,
        producers,
        flow_input_contracts,
        build_results,
        narrowing=narrowing,
        type_fn=constructor_type,
    )
    # cross-flow type check: each `call` binding's source type vs the callable
    # input type (needs producers, so it runs after all nodes are built).
    check_call_map_types(
        build_results,
        producers,
        flow_input_contracts,
        flow_wiring,
        narrowing=narrowing,
        type_fn=constructor_type,
        diagnostics=diagnostics,
    )
    # loop `'a -> 'a` type contract (types): each field the body returns is compared with
    # its carried seed-source Type (needs producers, so it runs alongside the cross-flow
    # call/map type check above). An inline body's returned fields must also fit the
    # parameters generated from its seeds, since they are bound to them on the next turn.
    check_loop_type_contract(
        build_results,
        producers,
        flow_input_contracts,
        flow_wiring,
        inline_loops={
            nid
            for nid, descriptor in descriptors.items()
            if isinstance(descriptor, LoopDescriptor)
            and descriptor.child_label is not None
        },
        registry=registry,
        diagnostics=diagnostics,
    )
    # Case `on:` enum exhaustiveness needs the assembled producer types, so it runs after
    # the cross-flow type contracts while preserving validation error order.
    check_case_exhaustive(
        descriptors,
        producers,
        narrowing=narrowing,
        value_cases=value_cases,
        diagnostics=diagnostics,
    )

    nodes: dict[str, Node] = dict(nodes_built)

    # flow.wiring/params key parity over EVERY node (catches an orphan/
    # missing source or a node absent from the wiring). Runs once the wiring is complete.
    check_wiring_parity(
        build_results,
        flow_wiring,
        diagnostics=diagnostics,
    )

    reconciled = data_edges + case_edges + host_edges

    # Reference + assert validation over the built flow. Errors locate
    # at the node's line (a bad node-input ref) or the `output:` section line (a
    # dangling flow-output ref).
    validate_references(
        build_results,
        flow_inputs,
        producers,
        outputs_section,
        flow_wiring,
        input_contracts=flow_input_contracts,
        outputs_range=outputs_range,
        narrowing=narrowing,
        registry=registry,
        resolve_origin=catalog_builder.resolve_registered,
        diagnostics=diagnostics,
    )
    # human_input gates: structurally check a literal `questions:` list at load, and
    # pin a ref-form `questions: ${name}` to a declared input (the prompt-scope rule).
    validate_human_questions(
        build_results,
        diagnostics=diagnostics,
    )
    # Node-local `asserts:` (per-node contract): validate node-locally + classify pre/post +
    # stamp onto each built leaf/call node. Shared by defs (a def's internal nodes too).
    validate_node_asserts(
        nodes_built,
        descriptors,
        assertion_site_at=node_assertion_site_at,
        type_names=frozenset(registry),
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
    flow_outputs = [
        replace(output, binding=replace(output.binding, contract=output_types[output.name]))
        if output.name in output_types
        else output
        for output in flow_outputs
    ]
    # An output that reads only nodes CASE routing may skip would leave END unreachable;
    # the explicit null default makes its END edge optional, so the run returns null.
    structural = reconciled + ordering_edges
    skip_model = SkipModel(
        set(descriptors),
        structural,
        exhaustive_cases(descriptors, producers, flow_input_contracts, narrowing=narrowing),
    )
    flow_outputs = null_skipped_outputs(flow_outputs, skip_model)
    value_mode = outputs_section is not None and not isinstance(outputs_section, dict)
    # Arithmetic on such a value would fail at runtime with null; ask for a `:-` default.
    # A searched `when:` is checked as authored: its wiring holds only its `__rN` reads.
    searched_whens = {
        node_id: [
            (
                index,
                case.get("when"),
                desc.when_origins[index] if index < len(desc.when_origins) else None,
                desc.when_ranges[index] if index < len(desc.when_ranges) else None,
            )
            for index, case in enumerate(desc.cases)
        ]
        for node_id, desc in descriptors.items()
        if isinstance(desc, CaseDescriptor) and desc.on is None
    }
    check_skipped_arithmetic(
        flow_wiring,
        flow_outputs,
        skip_model,
        build_results,
        value_mode=value_mode,
        outputs_range=outputs_range,
        resolve_origin=catalog_builder.resolve_registered,
        diagnostics=diagnostics,
        searched_whens=searched_whens,
    )
    # Typed flow outputs that can never fit their declared type (needs the output
    # contracts attached above).
    check_typed_outputs(
        flow_outputs,
        producers,
        flow_input_contracts,
        value_mode=value_mode,
        outputs_range=outputs_range,
        type_fn=constructor_type,
        registry=registry,
        resolve_origin=catalog_builder.resolve_registered,
        diagnostics=diagnostics,
    )
    boundary_nodes, boundary_edges, boundary_wiring = synthesize_boundary_graph(
        inputs,
        flow_outputs,
        node_ids,
        structural,
        lowered=lowered,
        record=isinstance(outputs_section, dict),
    )
    nodes.update(boundary_nodes)
    flow_wiring.update(boundary_wiring)
    edges: list[Edge] = structural + boundary_edges

    # Type tables for `isinstance`, once START/END carry the lowered flow asserts and outputs.
    _fill_type_tables(
        nodes,
        flow_wiring,
        build_results,
        descriptors,
        asserts_section,
        flow_assertion_ranges,
        outputs_cursor,
        outputs_range,
        registry,
        diagnostics,
    )

    # A cycle spans nodes; every actual member contributes its authored range.
    # START_ID/END_ID are acyclic by construction (START_ID has no incoming,
    # END_ID has no outgoing).
    node_ranges = {
        nid: d.range for nid, d in descriptors.items() if getattr(d, "range", None)
    }
    # A synth node whose text all lives under its host (it has `field_labels`) is reported
    # as that host, unless the cycle runs through an author's read of it.
    synth_hosts = {
        nid: d.host
        for nid, d in descriptors.items()
        if d.host is not None and d.field_labels
    }
    synth_wires = frozenset(
        e.id
        for e in edges
        if synth_hosts.get(e.from_) == e.to and e.input_group == QPARAM
    )
    reject_cycles(
        edges,
        node_ids,
        node_ranges=node_ranges,
        diagnostics=diagnostics,
        gate_ids=frozenset(e.id for e in host_edges if e.input_group),
        synth_hosts=synth_hosts,
        synth_wires=synth_wires,
    )
    check_case_handles(nodes, edges, diagnostics=diagnostics)

    compiled = CompiledFlow.from_parts(
        nodes,
        edges,
        outputs=flow_outputs,
        wiring=flow_wiring,
        flow_llm_config=flow_llm_config or {},
        flow_id=flow_id,
        memories=memories,
    )
    return ComposedFlow(
        compiled=compiled,
        version=version,
        name=name,
        description=description,
    )
