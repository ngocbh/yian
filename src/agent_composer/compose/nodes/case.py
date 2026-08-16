"""kind=case — the branch-routing node's descriptor + parse/build handle.

One of the two-axis `compose/nodes/<kind>.py` modules: it owns the `case` node's frozen
`CaseDescriptor`. It inherits the template `parse` from `NodeDescriptor` (which records each
field's `SourceRange`), overriding only the `_normalize_cursor` hook — case is the sole kind
whose surface carries a key YAML 1.1 coerces (`on:`->`True`), so it undoes that coercion before
the template's key-legality check runs. Registered into `KINDS`/`BINDING_FIELDS` at import so the
shared dispatch routes here.

Unlike the leaf kinds, a `case` LOWERS in its own `.build()`: pattern-match compilation (the
OCaml analogue of lowering `match e with | p -> e1 | _ -> e2` to nested conditionals). A `case`
carries no inputs and no leaf Node; `.build()` returns a `BuildResult` whose `node` is the strict
`CaseNode` and whose flat `edges` tuple is the case's DATA edges followed by its CONTROL edges —
no new node kind. Two surface forms:

- **searched** (`cases: [{when: "<bool>", then: <t>}]`, `else: <e>`): scan each `when:` for its
  distinct `${...}` refs -> allocate one `CaseNode` input `__rN` per distinct ref -> rewrite the
  `when:` to the bare local `${__rN}`. Each case routes to its `then:` (the edge handle); `else:`
  routes to the shipped internal `default` handle.

- **`on:`** (`on: ${ref}`, `cases: [{when: <value>, then: <t>}]`): one input `__on = ${ref}`;
  each `when: <value>` becomes `${__on} == "<value>"`.

Because the desugared `when:`s interpolate ONLY the node-local `__rN`/`__on` names, the strict
`CaseNode` evaluates them via the EXISTING `evaluate_when_record` against its bound input record —
no new evaluator, and the strict declared-input check accepts them.

**Exhaustiveness is a SEPARATE validate-phase check** (`validate/case_exhaustive.py`): it needs
the flow's producer types, which are only assembled after every node is built, so it never lives
in `.build()`. `.build()` lowers unconditionally; the loader runs `check_case_exhaustive` in the
validate phase.

Imports flow down/peer only: `expr` (ref scan/rewrite), `nodes` (Case/CaseNode/Parameter),
`compile.model` (Edge), and the compose peer `source`/`helpers.errors`/`nodes.base`.
It imports NOTHING from parse/desugar/edges/validate/build/loader (the ladder), so it stays a
leaf of the nodes package.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Optional

from agent_composer.compile.model import Edge
from agent_composer.diagnostics import OriginRef
from agent_composer.expr import (
    ExpressionError,
    binding_co_skips,
    condition_refs,
    expr_refs_of,
    rewrite_condition_refs,
)
from agent_composer.nodes.case import DEFAULT_HANDLE, Case, CaseNode
from agent_composer.nodes.contracts import Binding, Parameter, binding_source
from agent_composer.compose.helpers.errors import LoadDiagnostics
from agent_composer.compose.nodes.base import BuildContext, BuildResult, NodeDescriptor
from agent_composer.compose.source import SourceCursor

if TYPE_CHECKING:
    from agent_composer.compose.source.catalog import _SourceCatalogBuilder
    from agent_composer.events import SourceRange

# YAML 1.1 (PyYAML `safe_load`) coerces the bare key `on:` to the boolean `True` (likewise
# `off`/`yes`/`no`). The case-node surface key `on:` is the only such key in any node body, so
# this pure key projection undoes the coercion. Case-local: no other kind body
# carries a YAML-bool-coerced key.
_YAML_BOOL_KEYS = {True: "on", False: "off"}


def _normalized_key_names(cursor: SourceCursor) -> dict[Any, Any]:
    """Return effective YAML 1.1 boolean keys mapped to schema names.

    Args:
        cursor (`SourceCursor`):
            Node-body cursor whose effective source pairs determine whether a
            Python-equal key was authored as a YAML boolean or a number.

    Returns:
        `dict[Any, Any]`:
            Displayed constructed keys mapped to their replacement schema
            keys, usually ``"on"`` or ``"off"``.
    """

    body = cursor._value_view()
    if not isinstance(body, Mapping):
        return {}
    names: dict[Any, Any] = {}
    for key in body:
        child = cursor.child(key)
        source_key = child.source_key_value
        if type(source_key) is bool:
            names[key] = _YAML_BOOL_KEYS[source_key]
        elif child.key_range is not None and type(key) is not type(source_key):
            names[key] = source_key
        elif child.document is None and type(key) is bool:
            names[key] = _YAML_BOOL_KEYS[key]
    return names


@dataclass(frozen=True, kw_only=True)
class CaseDescriptor(NodeDescriptor):
    """kind=case — branch routing (searched `when:` or `on:`-value form).

    A `case` routes only — it carries NO `inputs:` (the lowering maps it onto the
    inputs-bearing `CaseNode`). `else_` is the `else:` fallback.

    Inherits the common node fields (`id`/`node_name`/`depends_on`/`runs_after`/`env` + the
    `range`/`field_ranges` parse metadata) and the template `parse` from `NodeDescriptor`.
    It overrides ONLY the `_normalize_cursor` hook (to undo the `on:`->`True` YAML coercion
    before key legality runs); its field legality is otherwise the default allowed/required key
    check plus the default asserts-list check. Only the case-specific fields are declared here.

    Class attrs: `kind="case"`; `allowed` adds `cases`/`else`/`on` to the commons; `required`
    is `{"cases"}`; `binding_fields=()` — a `case` carries CONDITIONS (`on:` + each `when:`),
    not `${...}` sink bindings, so the kind-blind binding walk in `bindings.py` touches no field.
    `keyword_remap` is case-owned (the base's is empty): `else` is a Python keyword, so the
    template `parse` reads the surface `else:` into the `else_` descriptor field.

    `asserts` stays a declared field (the lowering path may set it) but is deliberately NOT in
    `allowed` — a `case` never authors `asserts`, so the surface key stays rejected.
    """

    kind = "case"
    allowed = NodeDescriptor.allow("cases", "else", "on")
    required = NodeDescriptor.require("cases")
    binding_fields = ()
    # Case owns this remap (the base's is empty): `else` is a Python keyword, so the template
    # `parse` reads the surface `else:` into the `else_` descriptor field.
    keyword_remap = {"else": "else_"}

    cases: list[dict[str, Any]] = field(default_factory=list)
    else_: Optional[str] = None
    on: Optional[str] = None
    asserts: list[str] = field(default_factory=list)
    # Positional per-case source ranges of the `when:` / `then:` VALUES, aligned 1:1 with `cases`
    # (None where no span exists). Captured at parse so a synth CASE param (`__rN` from a `when:`
    # ref) or a `then:/else: call(...)` target can locate at its own line, not the node header.
    # Render-only, out of equality (like `field_ranges`); `cases[N].when/then` has no unique text
    # key, so position — not text — is the identity.
    when_ranges: list[Optional[SourceRange]] = field(
        default_factory=list, compare=False
    )
    then_ranges: list[Optional[SourceRange]] = field(
        default_factory=list, compare=False
    )
    # Source-neutral sites follow the same positional identity as the legacy ranges.
    # An empty tuple means direct/source-free construction; otherwise it aligns 1:1
    # with `cases`. `on_origin` belongs to the generated `__on` binding, not a Case.
    when_origins: tuple[OriginRef | None, ...] = field(default=(), compare=False)
    on_origin: OriginRef | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        """Reject mutable, misaligned, or non-opaque CASE provenance."""
        super().__post_init__()
        if type(self.when_origins) is not tuple:
            raise TypeError("CASE predicate origins must be a tuple")
        self._validate_origin_alignment()
        for origin in self.when_origins:
            if origin is not None and not isinstance(origin, OriginRef):
                raise TypeError("CASE predicate origin must be an OriginRef or None")
        if self.on_origin is not None and not isinstance(self.on_origin, OriginRef):
            raise TypeError("CASE on origin must be an OriginRef or None")

    def _validate_origin_alignment(self) -> None:
        """Keep optional predicate origins aligned with a valid case sequence."""
        if not self.when_origins:
            return
        if (
            not isinstance(self.cases, Sequence)
            or isinstance(self.cases, (str, bytes))
            or len(self.when_origins) != len(self.cases)
        ):
            raise ValueError("CASE predicate origins must align with cases")

    @classmethod
    def parse(
        cls,
        node_id: str,
        cursor: SourceCursor,
    ) -> "NodeDescriptor":
        """Template `parse` + capture each `cases[N].when` / `.then` VALUE range positionally.

        `cases:` is a sequence of `{when:, then:}` mappings whose per-item fields carry no unique
        text key, so neither `field_ranges` nor `item_range_list` can reach them; the positional
        `sequence_field_ranges` does. `_desugar_searched` / `desugar_case_call_targets` read these
        to give a synth CASE param / call target its own caret."""
        cursor = cls._normalize_cursor(cursor)
        desc = super().parse(node_id, cursor)
        cases = cursor._value_view().get("cases")
        if isinstance(cases, Sequence) and not isinstance(cases, (str, bytes)):
            when_ranges = cursor.sequence_field_ranges("cases", "when")
            then_ranges = cursor.sequence_field_ranges("cases", "then")
        else:
            when_ranges = []
            then_ranges = []
        if when_ranges or then_ranges:
            desc = replace(desc, when_ranges=when_ranges, then_ranges=then_ranges)
        return desc

    @classmethod
    def _normalize_cursor(cls, cursor: SourceCursor) -> SourceCursor:
        """Project YAML 1.1's boolean `on:` key back to its surface spelling."""

        body = cursor._value_view()
        if not isinstance(body, Mapping):
            return cursor
        names = _normalized_key_names(cursor)
        return cursor.renamed(names) if names else cursor

    def with_source_origins(
        self,
        cursor: SourceCursor,
        *,
        site_path: tuple[str | int, ...],
        catalog_builder: "_SourceCatalogBuilder",
    ) -> "CaseDescriptor":
        """Attach each physical predicate and optional ``on:`` binding site.

        Predicate sites are indexed, so duplicate expression text remains
        distinct. Missing/malformed entries retain ``None`` and continue to the
        established build-time validation instead of changing error order.
        """
        cursor = self._normalize_cursor(cursor)
        case_specs = (
            self.cases
            if isinstance(self.cases, Sequence)
            and not isinstance(self.cases, (str, bytes))
            else ()
        )
        when_origins = (
            list(self.when_origins) if self.when_origins else [None] * len(case_specs)
        )
        cases_cursor = cursor.child("cases")
        for index, case_spec in enumerate(case_specs):
            if not isinstance(case_spec, Mapping):
                continue
            item_cursor = cases_cursor.child(index)
            if not item_cursor.has("when"):
                continue
            registered = catalog_builder.register(
                (*site_path, "case", index, "when"),
                item_cursor.child("when"),
            )
            if registered is not None:
                when_origins[index] = registered

        on_origin = self.on_origin
        if self.on is not None and cursor.has("on"):
            registered = catalog_builder.register(
                (*site_path, "field", "on"),
                cursor.child("on"),
            )
            if registered is not None:
                on_origin = registered

        return replace(
            self,
            when_origins=tuple(when_origins),
            on_origin=on_origin,
        )

    def build(self, ctx: BuildContext) -> BuildResult:
        """Lower this `case` to a strict `CaseNode` + its control + data edges.

        The case's `.build()` is a pattern-match compilation, not a leaf instantiation: it
        dispatches on the surface form (`on:` vs searched), builds the `CaseNode` + the
        flow-owned `{__rN|__on -> source}` wiring, and returns a `BuildResult` whose flat `edges`
        tuple is the DATA edges (per `${<id>.output}` ref a `__rN`/`__on` source reads, keyed by
        the param name) FOLLOWED BY the CONTROL edges (`gate -> <then|else target>`, keyed by the
        branch handle). Exhaustiveness is NOT checked here — it is the loader's separate
        `check_case_exhaustive` validate pass (it needs the assembled producer types). `ctx` is
        unused (a case resolves nothing and reads no typedefs); it rides the shared contract.
        """
        self._validate_origin_alignment()
        if self.on is not None:
            case_node, wiring, data_edges, control_edges, param_ranges, param_labels = (
                _desugar_on(
                    self,
                    ctx.diagnostics,
                )
            )
        else:
            case_node, wiring, data_edges, control_edges, param_ranges, param_labels = (
                _desugar_searched(
                    self,
                    ctx.diagnostics,
                )
            )
        result = self._finish(
            case_node, wiring, edges=tuple(data_edges) + tuple(control_edges)
        )
        # A synth CASE param (`__rN` hoisted from a `when:` ref, `__on` from `on:`) owns no
        # authored field, so the ref-scan would fall back to the node header and print the
        # synthetic name. Stamp its own source range and author-facing label here so the caret
        # lands on the `cases[N].when` / `on:` line and the message names it, not `__rN`.
        result.node.field_ranges.update(param_ranges)
        result.node.wiring_labels = param_labels
        return result


def _outputs_producer(ref: str) -> Optional[str]:
    """The producing node id of a ref path. Singular only: accepts
    `<id>.output[.…]`. Legacy `outputs.<id>` is rejected at parse time."""
    parts = ref.split(".")
    if len(parts) >= 2 and parts[1] == "output":
        return parts[0]
    return None


def _refs_in(expr: str) -> list[str]:
    """The distinct reference paths a `when:` expression reads, in first-seen order.

    Routes through `condition_refs` (parse the WHOLE `when:` value, all three spellings —
    bare / mixed / whole-span) so it agrees with `_rewrite_when`, which rewrites the same
    parsed leaves. Raises `ExpressionError` on a malformed `when:` (the caller frames it
    as a located `LoadError`)."""
    out: list[str] = []
    for path in condition_refs(expr):
        if path not in out:
            out.append(path)
    return out


def _quote(
    value: str,
    diagnostics: LoadDiagnostics | None = None,
    *,
    range: SourceRange | None = None,
) -> str:
    """A `when:` match value quoted as a string literal for `${__on} == "<v>"`.

    The `when:`/`asserts:` grammar's STRING terminal is `/"[^"]*"/ | /'[^']*'/` with
    NO escape support (the evaluator strips the outer quotes verbatim), so we cannot
    escape — we pick the quote style the value allows: double-quote when it has no `"`,
    single-quote when it has no `'`. A value containing BOTH quote characters is
    unrepresentable in this grammar → a loud, located load error rather than a
    route-time crash. (Backslashes pass through `[^"]*` fine — no escaping needed.)
    """
    diagnostics = diagnostics or LoadDiagnostics()
    s = str(value)
    if '"' not in s:
        return f'"{s}"'
    if "'" not in s:
        return f"'{s}'"
    raise diagnostics.error(
        f"case `on:` match value {s!r} contains both ' and \" — unrepresentable in a "
        f"`when:` string literal (no escape support); rename the matched value",
        range=range,
    )


def _data_edges(node_id: str, wiring: dict[str, Any]) -> list[Edge]:
    """Reconciled data edges: one per `${<id>.output}` ref a `__rN`/`__on` source reads.

    Reads the flow-owned `wiring` (`{__rN|__on -> source}`); the `input_group` is the
    param name (`__rN`/`__on`) so per-input readiness keys on the desugared input. Each edge
    carries `input_group=name` and leaves `source_handle=None` — the invariant that keeps the
    flat `BuildResult.edges` split back into data-vs-control by attribute (paired with
    `_control_edges`, which sets `source_handle` and leaves `input_group=None`).
    """
    edges: list[Edge] = []
    counts: dict[str, int] = {}
    for name, source in wiring.items():
        source = binding_source(source)
        if not isinstance(source, str):
            continue
        try:
            refs = expr_refs_of(source)
        except ExpressionError:
            continue  # malformed refs surface, located, in the ref-wiring pass
        for ref in refs:
            producer = _outputs_producer(ref)
            if producer is None:
                continue
            i = counts.get(producer, 0)
            counts[producer] = i + 1
            edges.append(
                Edge(
                    id=f"{producer}->{node_id}#{i}",
                    from_=producer,
                    to=node_id,
                    input_group=name,
                    # a `when:`/`on:` ref can carry an escape (`:-`/`:?`); a
                    # co-skipping condition would otherwise wrongly skip-flood the gate.
                    optional=not binding_co_skips(source),
                )
            )
    return edges


def _control_edges(node_id: str, handle_targets: list[tuple[str, str]]) -> list[Edge]:
    """`gate -> target` edges, each carrying its `source_handle` (case id or default).

    Each edge sets `source_handle=handle` and leaves `input_group=None` — the invariant that
    keeps the flat `BuildResult.edges` split back into data-vs-control by attribute (paired with
    `_data_edges`, which sets `input_group` and leaves `source_handle=None`).
    """
    edges: list[Edge] = []
    counts: dict[str, int] = {}
    for handle, target in handle_targets:
        i = counts.get(target, 0)
        counts[target] = i + 1
        edges.append(
            Edge(
                id=f"{node_id}->{target}#{i}",
                from_=node_id,
                to=target,
                source_handle=handle,
            )
        )
    return edges


def _desugar_on(
    desc: CaseDescriptor,
    diagnostics: LoadDiagnostics,
) -> tuple[CaseNode, dict[str, Any], list[Edge], list[Edge], dict, dict[str, str]]:
    """`on: ${ref}` form -> one `__on` param + a `${__on} == "<value>"` per case.

    Returns `(node, wiring, data_edges, control_edges, param_ranges, param_labels)` — the last
    two locate the synth `__on` param at the `on:` line and label it `on:` (not `__on`).
    """
    cases: list[Case] = []
    handle_targets: list[tuple[str, str]] = []
    for index, c in enumerate(desc.cases):
        value = c.get("when")
        then = c.get("then")
        if value is None or then is None:
            missing_range = (
                desc.when_ranges[index]
                if value is None and index < len(desc.when_ranges)
                else (
                    desc.then_ranges[index] if index < len(desc.then_ranges) else None
                )
            )
            raise diagnostics.error(
                f"case node {desc.id!r}: each on:-form case needs a `when:` value and a `then:`",
                range=missing_range
                or desc.field_ranges.get(("field", "cases"))
                or desc.range,
            )
        when_range = desc.when_ranges[index] if index < len(desc.when_ranges) else None
        cases.append(
            Case(
                handle=then,
                when=f"${{__on}} == {_quote(value, diagnostics, range=when_range or desc.range)}",
                origin=(desc.when_origins[index] if desc.when_origins else None),
            )
        )
        handle_targets.append((then, then))
    if desc.else_ is not None:
        handle_targets.append((DEFAULT_HANDLE, desc.else_))
    node = CaseNode(desc.id, cases, title=desc.node_name)
    node.params = [Parameter(name="__on")]
    wiring = {
        "__on": (
            desc.on
            if desc.on_origin is None
            else Binding(desc.on, origin=desc.on_origin)
        )
    }
    on_rng = desc.field_ranges.get(("field", "on"))  # the `on:` value's own range
    param_ranges = {("input", "__on"): on_rng} if on_rng is not None else {}
    return (
        node,
        wiring,
        _data_edges(desc.id, wiring),
        _control_edges(desc.id, handle_targets),
        param_ranges,
        {"__on": "on:"},
    )


def _desugar_searched(
    desc: CaseDescriptor,
    diagnostics: LoadDiagnostics,
) -> tuple[CaseNode, dict[str, Any], list[Edge], list[Edge], dict, dict[str, str]]:
    """Searched form -> one `__rN` input per distinct `${...}` ref; `when:` rewritten.

    Returns `(node, wiring, data_edges, control_edges, param_ranges, param_labels)` — the last
    two locate each synth `__rN` at the `cases[N].when` it was first seen in and label it
    `cases[N].when` (not the synthetic `__rN`)."""
    # Allocate one local per distinct ref across ALL when:s (first-seen order), remembering the
    # case index each was first seen in (its caret / label site).
    ref_to_local: dict[str, str] = {}
    local_case_index: dict[str, int] = {}
    for ci, c in enumerate(desc.cases):
        when = c.get("when")
        if not isinstance(when, str):
            raise diagnostics.error(
                f"case node {desc.id!r}: each searched case needs a string `when:` expression",
                range=(
                    desc.when_ranges[ci]
                    if ci < len(desc.when_ranges)
                    else desc.field_ranges.get(("field", "cases")) or desc.range
                ),
            )
        try:
            refs = _refs_in(when)
        except ExpressionError as exc:
            raise diagnostics.error(
                f"case node {desc.id!r}: `when:` {when!r} is not a valid boolean expression: {exc}",
                range=(
                    desc.when_ranges[ci]
                    if ci < len(desc.when_ranges)
                    else desc.field_ranges.get(("field", "cases")) or desc.range
                ),
            ) from exc
        for ref in refs:
            if ref not in ref_to_local:
                local = f"__r{len(ref_to_local)}"
                ref_to_local[ref] = local
                local_case_index[local] = ci

    wiring: dict[str, Any] = {}
    for ref, local in ref_to_local.items():
        source = f"${{{ref}}}"
        case_index = local_case_index[local]
        origin = desc.when_origins[case_index] if desc.when_origins else None
        wiring[local] = source if origin is None else Binding(source, origin=origin)

    cases: list[Case] = []
    handle_targets: list[tuple[str, str]] = []
    for index, c in enumerate(desc.cases):
        when = c["when"]
        then = c.get("then")
        if then is None:
            raise diagnostics.error(
                f"case node {desc.id!r}: each searched case needs a `then:`",
                range=(
                    desc.then_ranges[index]
                    if index < len(desc.then_ranges)
                    else desc.field_ranges.get(("field", "cases")) or desc.range
                ),
            )
        rewritten = _rewrite_when(when, ref_to_local)
        cases.append(
            Case(
                handle=then,
                when=rewritten,
                origin=(desc.when_origins[index] if desc.when_origins else None),
            )
        )
        handle_targets.append((then, then))
    if desc.else_ is not None:
        handle_targets.append((DEFAULT_HANDLE, desc.else_))

    node = CaseNode(desc.id, cases, title=desc.node_name)
    node.params = [Parameter(name=local) for local in ref_to_local.values()]
    param_ranges: dict = {}
    param_labels: dict[str, str] = {}
    for local, ci in local_case_index.items():
        rng = desc.when_ranges[ci] if ci < len(desc.when_ranges) else None
        if rng is not None:
            param_ranges[("input", local)] = rng
        param_labels[local] = f"cases[{ci}].when"
    return (
        node,
        wiring,
        _data_edges(desc.id, wiring),
        _control_edges(desc.id, handle_targets),
        param_ranges,
        param_labels,
    )


def _rewrite_when(when: str, ref_to_local: dict[str, str]) -> str:
    """Rewrite each reference in `when` to its bare local `__rN`, honouring all three
    condition spellings. A `${<ref>}` (mixed) becomes `${__rN}`, a bare `<ref>`
    (whole-span / bare) becomes `__rN`; the surrounding expression is preserved. A ref
    with no allocated local is left untouched. Shares `rewrite_condition_refs` with
    `_refs_in`, so extraction and rewrite parse the SAME leaves."""
    return rewrite_condition_refs(when, lambda path: ref_to_local.get(path))
