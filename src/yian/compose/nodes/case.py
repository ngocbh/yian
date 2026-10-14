"""kind=case — the branch-routing node's descriptor + parse/build handle.

One of the two-axis `compose/nodes/<kind>.py` modules: it owns the `case` node's frozen
`CaseDescriptor`. Its `parse` wraps the `NodeDescriptor` template to retain positional
`cases[N].when`/`then` ranges. Its immutable source-origin hook registers each indexed
predicate and optional `on:` site before lowering. Registered into `KINDS`/`BINDING_FIELDS`
at import so the shared dispatch routes here.

Unlike the leaf kinds, a `case` LOWERS in its own `.build()`: pattern-match compilation (the
OCaml analogue of lowering `match e with | p -> e1 | _ -> e2` to nested conditionals). A `case`
carries no inputs and no leaf Node; `.build()` returns a `BuildResult` whose `node` is the strict
`CaseNode` and whose flat `edges` tuple is the case's DATA edges followed by its CONTROL edges —
no new node kind. Two surface forms:

- **searched** (`cases: [{when: "<bool>", then: <t>}]`, `else: <e>`): scan each `when:` for its
  distinct `${...}` refs -> allocate one `CaseNode` input `__rN` per distinct ref -> rewrite the
  `when:` to the bare local `${__rN}`. Each case routes to its `then:` (the edge handle); `else:`
  routes to the internal else handle (`ELSE_HANDLE`).

- **`on:`** (`on: ${ref}`, `cases: [{when: <value>, then: <t>}]`): one input `__on = ${ref}`;
  each `when: <value>` is a typed literal compared exactly: text (quoted, or an unquoted date
  as its ISO text) becomes `__on == "<value>"`, a number
  `__on == <n> and not isinstance(__on, bool)`, a bool
  `__on == true and isinstance(__on, bool)`. The `isinstance` guard keeps a bool from
  equalling a number even when the `on:` type is unknown.

Because the desugared `when:`s interpolate ONLY the node-local `__rN`/`__on` names, the strict
`CaseNode` evaluates them via the EXISTING `evaluate_when_record` against its bound input record —
no new evaluator, and the strict declared-input check accepts them.

**Exhaustiveness is a SEPARATE validate-phase check** (`validate/case_exhaustive.py`): it needs
the flow's producer types, which are only assembled after every node is built, so it never lives
in `.build()`. `.build()` lowers unconditionally; the loader runs `check_case_exhaustive` in the
validate phase.

Imports flow down/peer only: `lark` (parse-tree types), `expr` (ref scan/rewrite and the
`isinstance` tree helpers), `nodes` (Case/CaseNode/Parameter), `compile.model` (Edge),
`typesys` (the non-finite check), and the compose peers
`source`/`helpers.errors`/`helpers.output_contracts`/`helpers.typed_bindings`/`nodes.base`.
It imports NOTHING from parse/desugar/edges/validate/build/loader (the ladder), so it stays a
leaf of the nodes package.
"""

from __future__ import annotations

import datetime
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Optional

from lark import Token, Tree

from yian.compile.model import Edge
from yian.diagnostics import OriginRef
from yian.expr import (
    ExpressionError,
    binding_co_skips,
    expr_refs,
    expr_refs_of,
    isinstance_subject,
    isinstance_type_names,
    parse_condition,
    parse_expr,
    rewrite_condition_refs,
)
from yian.nodes.case import ELSE_HANDLE, Case, CaseNode
from yian.nodes.contracts import Binding, Parameter, binding_source
from yian.typesys import TypeCheckError
from yian.typesys.values import reject_nonfinite
from yian.compose.helpers.errors import LoadDiagnostics, source_text
from yian.compose.helpers.output_contracts import direct_reference_parts
from yian.compose.helpers.typed_bindings import yaml_quoted
from yian.compose.nodes.base import BuildContext, BuildResult, NodeDescriptor
from yian.compose.source import SourceCursor

if TYPE_CHECKING:
    from yian.diagnostics import SourceRange
    from yian.compose.source.catalog import _SourceCatalogBuilder

@dataclass(frozen=True, kw_only=True)
class CaseDescriptor(NodeDescriptor):
    """kind=case — branch routing (searched `when:` or `on:`-value form).

    A `case` carries no authored `input:` (the lowering maps it onto the inputs-bearing
    `CaseNode`). It may declare `output:` (normalized to `outputs`): the type of the value it
    produces. `else_` is the `else:` fallback.

    `then_styles` (aligned 1:1 with `cases`), `else_style`, and `has_else` exist so branch
    classification can tell an unquoted node name from quoted text, and `else: null` from no
    `else:` at all. A style is the YAML scalar style of the authored value (non-string scalars
    report ``"plain"``; mappings, lists, missing values, and strings read through a source-free
    cursor report None). Downstream treats an unknown (None) style as unquoted (plain).

    Inherits the common node fields (`id`/`node_name`/`depends_on`/`runs_after`/`env` + the
    `range`/`field_ranges` parse metadata) and delegates common parsing to `NodeDescriptor`.
    Its `parse` wrapper captures positional predicate/target ranges, and
    `with_source_origins` attaches indexed predicate and optional `on:` origins. Its parse
    wrapper also checks the `cases:` shape; field legality otherwise uses the default
    allowed/required-key and asserts-list checks.
    Only the case-specific fields are declared here.

    Class attrs: `kind="case"`; `allowed` adds `cases`/`else`/`on`/`outputs` to the commons;
    `required` is `{"cases"}`; `binding_fields=()` — a `case` carries CONDITIONS (`on:` + each
    `when:`), not `${...}` sink bindings, so the kind-blind binding walk in `bindings.py` touches
    no field.
    `keyword_remap` is case-owned (the base's is empty): `else` is a Python keyword, so the
    template `parse` reads the surface `else:` into the `else_` descriptor field.

    `asserts` stays a declared field (the lowering path may set it) but is deliberately NOT in
    `allowed` — a `case` never authors `asserts`, so the surface key stays rejected.
    """

    kind = "case"
    allowed = NodeDescriptor.allow("cases", "else", "on", "outputs")
    required = NodeDescriptor.require("cases")
    binding_fields = ()
    # Case owns this remap (the base's is empty): `else` is a Python keyword, so the template
    # `parse` reads the surface `else:` into the `else_` descriptor field.
    keyword_remap = {"else": "else_"}

    cases: list[dict[str, Any]] = field(default_factory=list)
    else_: Any = None
    # True when `else:` was written, even as `else: null` (where `else_` stays None).
    has_else: bool = False
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
    # YAML scalar style of each `cases[N].then` value (aligned 1:1 with `cases`) and of the
    # `else:` value, so an unquoted bare word (a node name) is told apart from quoted text.
    # Out of equality like the ranges: a directly constructed descriptor has no source styles.
    then_styles: list[Optional[str]] = field(default_factory=list, compare=False)
    else_style: Optional[str] = field(default=None, compare=False)
    # Synthetic value-node id -> the authored branch value it was lifted from, filled by
    # `desugar_case_values` so later checks can read the value a branch produces.
    value_branches: dict[str, Any] = field(default_factory=dict, compare=False)
    # Source-neutral sites follow the same positional identity as the Compose-only ranges.
    # An empty tuple means direct/source-free construction; otherwise it aligns 1:1
    # with `cases`. `on_origin` belongs to the generated `__on` binding, not a Case.
    when_origins: tuple[OriginRef | None, ...] = field(default=(), compare=False)
    on_origin: OriginRef | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        """Derive `has_else` from a constructed `else_`, then reject mutable, misaligned, or
        non-opaque CASE provenance."""
        super().__post_init__()
        if self.else_ is not None and not self.has_else:
            # Frozen dataclass: a directly constructed `else_=` implies the key was present.
            object.__setattr__(self, "has_else", True)
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
        to give a synth CASE param / call target its own caret.

        It also records each `then:` value's YAML scalar style (aligned with `cases`), the
        `else:` value's style, and whether `else:` was written at all. A `cases:` that is not a
        list of mappings is a located load error before any of this runs."""
        desc = super().parse(node_id, cursor)
        cls._check_cases_shape(desc, cursor)
        body = cursor._value_view()
        has_else = "else" in body
        updates: dict[str, Any] = {
            "has_else": has_else,
            "else_style": cursor.child("else").scalar_style if has_else else None,
        }
        when_ranges = cursor.sequence_field_ranges("cases", "when")
        then_ranges = cursor.sequence_field_ranges("cases", "then")
        if when_ranges or then_ranges:
            updates["when_ranges"] = when_ranges
            updates["then_ranges"] = then_ranges
        cases_cursor = cursor.child("cases")
        updates["then_styles"] = [
            cases_cursor.child(index).child("then").scalar_style
            if "then" in item
            else None
            for index, item in enumerate(desc.cases)
        ]
        return replace(desc, **updates)

    @staticmethod
    def _check_cases_shape(desc: "CaseDescriptor", cursor: SourceCursor) -> None:
        """Reject a `cases:` that is not a list of mappings, at the offending field or item.

        Later passes index `cases:` as a list of mappings, so a malformed shape would
        otherwise surface as a bare Python error with no location."""
        diagnostics = LoadDiagnostics.from_cursor(cursor)
        where = f"case node {desc.id!r}"
        if not isinstance(desc.cases, list):
            hint = "; start each case with `- `" if isinstance(desc.cases, Mapping) else ""
            raise diagnostics.error(
                f"{where}: `cases:` must be a list of {{when:, then:}} mappings "
                f"(got {desc.cases!r}){hint}",
                range=desc.field_ranges.get(("field", "cases")) or desc.range,
            )
        cases_cursor = cursor.child("cases")
        for index, item in enumerate(desc.cases):
            if not isinstance(item, Mapping):
                raise diagnostics.error(
                    f"{where}: `cases[{index}]` must be a {{when:, then:}} mapping "
                    f"(got {item!r})",
                    range=cases_cursor.child(index).range or desc.range,
                )

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
        # synthetic name. Retain its range in the Compose-only build product so the caret lands
        # on the `cases[N].when` / `on:` line and the message names it, not `__rN`.
        result.node.wiring_labels = param_labels
        return replace(
            result,
            field_ranges={**result.field_ranges, **param_ranges},
        )


def _outputs_producer(ref: str) -> Optional[str]:
    """The producing node id of a ref path. Singular only: accepts
    `<id>.output[.…]`. Legacy `outputs.<id>` is rejected at parse time."""
    parts = ref.split(".")
    if len(parts) >= 2 and parts[1] == "output":
        return parts[0]
    return None


def _quote(
    value: str,
    diagnostics: LoadDiagnostics | None = None,
    *,
    range: SourceRange | None = None,
) -> str:
    """A `when:` match value quoted as a string literal for `__on == "<v>"`.

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


def on_match_value(value: Any) -> Any:
    """Return the value an `on:`-form `when:` matches.

    Args:
        value (`Any`):
            The authored `when:` value.

    Returns:
        `Any`:
            A date as its ISO text, anything else unchanged. A datetime is returned
            unchanged; `_match_condition` rejects it before matching.
    """
    if isinstance(value, datetime.date) and not isinstance(value, datetime.datetime):
        return value.isoformat()
    return value


def _number_text(value: int | float) -> str:
    """Return a number as expression-literal digits: no exponent, and a float keeps a `.`."""
    # repr is the shortest round-tripping form; Decimal spells it positionally, since the
    # NUMBER terminal has no exponent.
    text = format(Decimal(repr(value)), "f")
    if isinstance(value, float) and "." not in text:
        text += ".0"
    return text


def _match_condition(
    value: Any,
    diagnostics: LoadDiagnostics,
    *,
    node_id: str,
    index: int,
    range: SourceRange | None,
) -> str:
    """Return the `__on` condition one `on:`-form `when:` value lowers to.

    Text compares as a string literal; a number compares numerically and never with a bool;
    a bool compares only with a bool.

    Raises:
        `LoadError`:
            The value is a datetime (its text has several spellings, so it must be quoted),
            a non-finite number, a list, a record, or text with both quote characters.
    """
    if isinstance(value, datetime.datetime):
        text = source_text(diagnostics, range) or value.isoformat()
        raise diagnostics.error(
            f"case node {node_id!r}: quote a datetime when: to match its text exactly: "
            f"`when: {yaml_quoted(text)}`",
            range=range,
        )
    value = on_match_value(value)
    if isinstance(value, str):
        return f"__on == {_quote(value, diagnostics, range=range)}"
    if isinstance(value, bool):
        # Python's == has True == 1; the isinstance guard keeps a bool from equalling a number.
        literal = "true" if value else "false"
        return f"__on == {literal} and isinstance(__on, bool)"
    if isinstance(value, (int, float)):
        try:
            reject_nonfinite(value, f"case node {node_id!r} cases[{index}].when")
        except TypeCheckError as exc:
            raise diagnostics.error(str(exc), range=range) from exc
        return f"__on == {_number_text(value)} and not isinstance(__on, bool)"
    if isinstance(value, list):
        kind = "a list"
    elif isinstance(value, Mapping):
        kind = "a record"
    else:
        kind = type(value).__name__
    raise diagnostics.error(
        f"case node {node_id!r}: an on: case's when: takes text, a number, or a bool, "
        f"got {kind}",
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
    """`gate -> target` edges, each carrying its `source_handle` (case id or the else handle).

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
    """`on: ${ref}` form -> one `__on` param + one typed `__on` match condition per case.

    Each `when:` lowers through `_match_condition`, so it compares exactly with the `on:`
    value: text as text, numbers numerically, bools only with bools.

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
                when=_match_condition(
                    value,
                    diagnostics,
                    node_id=desc.id,
                    index=index,
                    range=when_range or desc.range,
                ),
                origin=(desc.when_origins[index] if desc.when_origins else None),
            )
        )
        handle_targets.append((then, then))
    if desc.else_ is not None:
        handle_targets.append((ELSE_HANDLE, desc.else_))
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


IsinstanceCheck = tuple[str, str]
"""A top-level `isinstance(<base>, <T>)` conjunct: the dotted base path and the type name."""


def _isinstance_check(conjunct: Tree | Token) -> Optional[IsinstanceCheck]:
    """The check a conjunct makes when it is `isinstance(<direct ref>, T)`, else `None`."""
    subject = isinstance_subject(conjunct)
    if isinstance(subject, Token) and subject.type == "WRAPPED_REF":
        # The mixed spelling `isinstance(${d.output}, T)` wraps the reference in braces.
        try:
            subject = parse_expr(str(subject)[2:-1])
        except ExpressionError:
            return None
    if subject is None:
        return None
    parts = direct_reference_parts(subject)
    if parts is None:
        return None
    return ".".join(parts), isinstance_type_names(conjunct)[0]


GuardedRead = tuple[int, tuple[IsinstanceCheck, ...]]
"""One read of a searched-case param: its case index and the checks guarding it."""


def _conjuncts(tree: Tree | Token) -> list[Tree | Token]:
    """The operands of a top-level `and` chain; a lone operand otherwise."""
    if isinstance(tree, Tree) and tree.data == "and_expr":
        return [
            child
            for child in tree.children
            if not (isinstance(child, Token) and child.type == "AND")
        ]
    return [tree]


@dataclass(frozen=True)
class _SearchedParams:
    """The `__rN` params of a searched case, from one parse of its `when:`s.

    Args:
        ref_to_local (`dict[str, str]`):
            Each distinct ref path and its local, in first-seen order.
        local_case_index (`dict[str, int]`):
            The case index each local was first seen in (its caret / label site).
        guards (`dict[str, list[GuardedRead]]`):
            For each local, the case index of each occurrence and the checks guarding it.
    """

    ref_to_local: dict[str, str]
    local_case_index: dict[str, int]
    guards: dict[str, list[GuardedRead]]


def _searched_params(desc: CaseDescriptor, diagnostics: LoadDiagnostics) -> _SearchedParams:
    """Allocate one `__rN` local per distinct ref across all searched `when:`s.

    Locals and guards come from the same parse, so every guarded occurrence has a local.
    An occurrence inside a top-level `and` operand is guarded by the
    `isinstance(<direct ref>, T)` operands before it; any other occurrence has no guards.

    Raises:
        `LoadError`:
            A `when:` is not a string or does not parse.
    """
    ref_to_local: dict[str, str] = {}
    local_case_index: dict[str, int] = {}
    guards: dict[str, list[GuardedRead]] = {}
    for ci, c in enumerate(desc.cases):
        when = c.get("when")
        when_range = (
            desc.when_ranges[ci]
            if ci < len(desc.when_ranges)
            else desc.field_ranges.get(("field", "cases")) or desc.range
        )
        if not isinstance(when, str):
            raise diagnostics.error(
                f"case node {desc.id!r}: each searched case needs a string `when:` expression",
                range=when_range,
            )
        try:
            tree = parse_condition(when)
        except ExpressionError as exc:
            raise diagnostics.error(
                f"case node {desc.id!r}: `when:` {when!r} is not a valid boolean expression: {exc}",
                range=when_range,
            ) from exc
        # Parsing the whole value once covers all three spellings, matching `_rewrite_when`.
        for ref in expr_refs(tree):
            if ref not in ref_to_local:
                local = f"__r{len(ref_to_local)}"
                ref_to_local[ref] = local
                local_case_index[local] = ci
                guards[local] = []
        for ref, checks in guarded_refs(tree):
            guards[ref_to_local[ref]].append((ci, checks))
    return _SearchedParams(ref_to_local, local_case_index, guards)


def guarded_refs(tree: Tree | Token) -> list[tuple[str, tuple[IsinstanceCheck, ...]]]:
    """Return each reference occurrence in a parsed condition with the checks guarding it.

    An occurrence inside a top-level `and` operand is guarded by the
    `isinstance(<direct ref>, T)` operands before it: `and` short-circuits, so the read
    only runs once those checks pass. Any other occurrence has no guards.

    Args:
        tree (`Tree | Token`):
            A condition parsed with `parse_condition`, in any spelling.

    Returns:
        `list[tuple[str, tuple[IsinstanceCheck, ...]]]`:
            One `(ref path, guarding checks)` pair per occurrence, in source order.

    Example:
        For `isinstance(r, Done) and r.answer == 'ok'`, the pairs are
        `("r", ())` and `("r.answer", (("r", "Done"),))`.
    """
    reads: list[tuple[str, tuple[IsinstanceCheck, ...]]] = []
    checks: list[IsinstanceCheck] = []
    for conjunct in _conjuncts(tree):
        reads.extend((ref, tuple(checks)) for ref in expr_refs(conjunct))
        check = _isinstance_check(conjunct)
        if check is not None:
            checks.append(check)
    return reads


def isinstance_conjuncts(when: Any) -> list[IsinstanceCheck]:
    """Return the `isinstance(<direct ref>, T)` checks among a `when:`'s top-level conjuncts.

    Only top-level `and` operands count: a check under `or` or `not` does not hold
    whenever the case matches.

    Args:
        when (`Any`):
            A searched case's original `when:` value, in any spelling. A non-string or
            unparsable value makes no checks.

    Returns:
        `list[IsinstanceCheck]`:
            `(base path, type name)` pairs in source order.

    Example:
        `isinstance_conjuncts("isinstance(d.output, Buy) and x > 1")` is
        `[("d.output", "Buy")]`.
    """
    if not isinstance(when, str):
        return []
    try:
        tree = parse_condition(when)
    except ExpressionError:
        return []
    checks = (_isinstance_check(conjunct) for conjunct in _conjuncts(tree))
    return [check for check in checks if check is not None]


def searched_param_guards(desc: CaseDescriptor) -> dict[str, list[GuardedRead]]:
    """Return, per searched-case `__rN` param, each occurrence's case and guarding checks.

    An occurrence of a param's ref inside a top-level `and` operand is guarded by the
    `isinstance(<direct ref>, T)` operands before it in the same chain: `and`
    short-circuits, so the read only runs once those checks pass. An occurrence anywhere
    else has no guards. The params are allocated exactly as the build allocates them.

    Args:
        desc (`CaseDescriptor`):
            A searched case that has already built. An `on:` case has no guards.

    Returns:
        `dict[str, list[GuardedRead]]`:
            For each local, one `(case index, guarding checks)` pair per occurrence,
            across all `when:`s in source order.

    Raises:
        `LoadError`:
            `desc` has a `when:` its build would reject. The loader only calls this after
            the build succeeded, so the error is unlocated.

    Example:
        For a first case `isinstance(d.output, Buy) and d.output.qty > 5`, the local
        of `d.output.qty` maps to `[(0, (("d.output", "Buy"),))]`.
    """
    if desc.on is not None:
        return {}
    return _searched_params(desc, LoadDiagnostics()).guards


def _desugar_searched(
    desc: CaseDescriptor,
    diagnostics: LoadDiagnostics,
) -> tuple[CaseNode, dict[str, Any], list[Edge], list[Edge], dict, dict[str, str]]:
    """Searched form -> one `__rN` input per distinct `${...}` ref; `when:` rewritten.

    Returns `(node, wiring, data_edges, control_edges, param_ranges, param_labels)` — the last
    two locate each synth `__rN` at the `cases[N].when` it was first seen in and label it
    `cases[N].when` (not the synthetic `__rN`)."""
    params = _searched_params(desc, diagnostics)
    ref_to_local, local_case_index = params.ref_to_local, params.local_case_index

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
        handle_targets.append((ELSE_HANDLE, desc.else_))

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
    with no allocated local is left untouched. `rewrite_condition_refs` parses the value
    like `parse_condition`, so extraction in `_searched_params` and rewrite see the SAME
    leaves."""
    return rewrite_condition_refs(when, lambda path: ref_to_local.get(path))
