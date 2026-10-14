"""Graph-stage: every branch of a typed `case` must be able to produce its `output:` type.

A `case` that declares `output: T` gets one synth result node that checks the taken branch
against T at runtime (`binding.type`). This load pass rejects the branches that could never
pass that check, located at the branch itself rather than at the result node:

- a value branch (lifted to a synth `__value_<n>` node; its authored value is kept in
  `CaseDescriptor.value_branches`) with no `${...}` span anywhere is checked exactly with
  the write-boundary check, after the same preparation the lifted code applies (a date
  becomes its ISO text, a span-free string is `$$`-unescaped). A `null` into a
  non-Optional T, and a YAML non-string scalar (`true`, `3`) where T would take the text,
  get a message saying how to fix it;
- a value branch with spans is checked position by position: a span string's type is
  inferred over the producer and flow-input contracts as the value node sees them (with
  its union narrowing) and rejected only when disjoint from the expected type; a mapping
  needs a record (its keys known and complete), an opaque `object`, or a union member that
  takes it; a list needs a list type, checked by element; a span-free part is checked
  exactly. A value whose span plainly reads another branch target of the same case is
  rejected, since that branch never runs when this one does; a read that tolerates the
  branch's absence (inside a coalesce or under a `:-` default) loads;
- a node or `call(...)` branch is rejected when its producer contract is proven and
  disjoint from T (`type_relation`); a narrower, partial, or unproven one loads and is
  left to the runtime check;
- a branch that routes to another `case`: an untyped inner case has each of its branches
  checked against the outer T; a typed inner case is rejected when its own type is
  disjoint from the outer T.

Runs in `_assemble` right after `compute_narrowing` and before `check_typed_inputs`, so a
disjoint branch is reported here and not at the result node's typed input. A branch error
locates at its `then:`/`else:` range, falling back to the case node's range.

Imports flow down only: `compile.validation` (the type relation), `typesys`, `expr`,
`compose.helpers` (load diagnostics, YAML quoting, output-contract inference),
`compose.parse.file` (descriptors), `compose.desugar.case_values` (the value walk and
literal preparation the lifter uses), `compose.desugar.inline_calls` (the synth call
prefix), and `compose.validate.narrowing`. Only the loader pipeline imports this back.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, replace
from typing import Any, Mapping, Optional

from yian.compile.validation import type_relation
from yian.compose.desugar.case_values import ValuePath, prepared_literal, value_spans
from yian.compose.desugar.inline_calls import _SYNTH_PREFIX
from yian.compose.helpers.errors import LoadDiagnostics, LoadError
from yian.compose.helpers.output_contracts import resolve_output_binding_contract
from yian.compose.helpers.typed_bindings import yaml_quoted
from yian.compose.helpers.yaml_words import former_boolean_fix
from yian.compose.parse.file import CallDescriptor, CaseDescriptor
from yian.compose.validate.narrowing import Narrowing
from yian.diagnostics import SourceRange
from yian.expr import ExpressionError, expr_refs, scan_template
from yian.expr.template import Span
from yian.nodes.contracts import OutputContract, ValueContract, binding_source
from yian.typesys import TypeCheckError, build_value_as, canonical_type_label
from yian.typesys.values import Type, ValueKind

# Expression nodes whose operands may be absent: a coalesce takes the first present one,
# and a `:-` default supplies a fallback.
_ABSENCE_TOLERANT = frozenset({"coalesce", "default_expr"})


@dataclass(frozen=True)
class _Branch:
    """One `then:`/`else:` of a case, after desugar.

    Attributes:
        case_id (`str`): The case that owns the branch.
        label (`str`): `"then:"` or `"else:"`.
        target (`str`): The branch target: a node id, a synth value or call id.
        range (`Optional[SourceRange]`): The branch's range, else the case's range.
    """

    case_id: str
    label: str
    target: str
    range: Optional[SourceRange]

    @property
    def host(self) -> str:
        """The message prefix naming the branch, such as `"case node 'r' then:"`."""
        return f"case node {self.case_id!r} {self.label}"


def check_case_values(
    descriptors: Mapping[str, Any],
    case_types: Mapping[str, ValueContract],
    producers: Mapping[str, OutputContract],
    flow_input_contracts: Mapping[str, ValueContract],
    *,
    narrowing: Narrowing,
    diagnostics: Optional[LoadDiagnostics] = None,
) -> None:
    """Reject every branch of a typed `case` that can never produce the case's type.

    Args:
        descriptors (`Mapping[str, Any]`): The flow's desugared descriptors, keyed by node
            id; branch targets are node ids.
        case_types (`Mapping[str, ValueContract]`): The contract of each typed case's
            `output:`, keyed by case id.
        producers (`Mapping[str, OutputContract]`): Built node output contracts, keyed by
            node id.
        flow_input_contracts (`Mapping[str, ValueContract]`): Flow input contracts, keyed
            by input name.
        narrowing (`Narrowing`): The union narrowing each value node reads against.
        diagnostics (`Optional[LoadDiagnostics]`, *optional*, defaults to `None`): Source
            ownership for located errors.

    Raises:
        `LoadError`: A literal value does not fit the case's type; a value with spans has
            a part that can never fit it, or plainly reads another branch of its case; a
            node, call, or typed inner case branch is disjoint from it.
        `ValueError`: A value node does not bind a span its lifted value holds.
    """
    diagnostics = diagnostics or LoadDiagnostics()
    checker = _Checker(
        descriptors=descriptors,
        case_types=case_types,
        producers=producers,
        flow_inputs=flow_input_contracts,
        narrowing=narrowing,
        diagnostics=diagnostics,
    )
    for case_id, contract in case_types.items():
        if isinstance(descriptors.get(case_id), CaseDescriptor):
            checker.check_case(case_id, contract.type, declaring=case_id, seen={case_id})


def _branches(case_id: str, desc: CaseDescriptor) -> list[_Branch]:
    """Return the case's `then:` branches in order, then its `else:` when present."""
    branches = []
    for index, case in enumerate(desc.cases):
        if isinstance(case, dict) and case.get("then") is not None:
            rng = desc.then_ranges[index] if index < len(desc.then_ranges) else None
            branches.append(_Branch(case_id, "then:", case["then"], rng or desc.range))
    if desc.else_ is not None:
        rng = desc.field_ranges.get(("field", "else")) or desc.range
        branches.append(_Branch(case_id, "else:", desc.else_, rng))
    return branches


@dataclass(frozen=True)
class _Checker:
    """The flow-wide inputs of one `check_case_values` run."""

    descriptors: Mapping[str, Any]
    case_types: Mapping[str, ValueContract]
    producers: Mapping[str, OutputContract]
    flow_inputs: Mapping[str, ValueContract]
    narrowing: Narrowing
    diagnostics: LoadDiagnostics

    def check_case(
        self, case_id: str, expected: Type, *, declaring: str, seen: set[str]
    ) -> None:
        """Check each branch of `case_id` against `expected`, declared by `declaring`.

        Args:
            case_id (`str`): The case whose branches are checked.
            expected (`Type`): The type each branch must be able to produce.
            declaring (`str`): The typed case whose `output:` is `expected`.
            seen (`set[str]`): Cases already on this path, so a case cycle ends.

        Raises:
            `LoadError`: See `check_case_values`.
        """
        desc = self.descriptors[case_id]
        declared = (
            "the case declares"
            if case_id == declaring
            else f"case {declaring!r} declares"
        )
        for branch in _branches(case_id, desc):
            target = branch.target
            if target in desc.value_branches:
                self._check_value(branch, desc, expected)
                continue
            inner = self.descriptors.get(target)
            if isinstance(inner, CaseDescriptor):
                inner_type = self.case_types.get(target)
                if inner_type is None:
                    if target not in seen:
                        self.check_case(
                            target, expected, declaring=declaring, seen=seen | {target}
                        )
                elif type_relation(inner_type.type, expected) == "disjoint":
                    raise self._error(
                        branch,
                        f"case {target!r} produces "
                        f"{canonical_type_label(inner_type.type)}, but {declared} "
                        f"output: {canonical_type_label(expected)}",
                    )
                continue
            contract = self.producers.get(target)
            if type(contract) is not ValueContract:
                continue  # partial or unproven: the result node checks it at runtime
            if type_relation(contract.type, expected) == "disjoint":
                raise self._error(
                    branch,
                    f"{_target_name(target, inner)} produces "
                    f"{canonical_type_label(contract.type)}, but {declared} "
                    f"output: {canonical_type_label(expected)}",
                )

    def _check_value(self, branch: _Branch, desc: CaseDescriptor, expected: Type) -> None:
        """Check one value branch against `expected`.

        Args:
            branch (`_Branch`): The branch; its target is the synth value node id.
            desc (`CaseDescriptor`): The case, for the authored value, its other branch
                targets, and the authored `output:` text.
            expected (`Type`): The case's type.

        Raises:
            `LoadError`: The value does not fit `expected`, or plainly reads another
                branch of the case.
        """
        value = desc.value_branches[branch.target]
        sources = self._span_sources(branch.target, value)
        if sources:
            self._check_sibling_reads(branch, desc, sources)
            producers, inputs = self.narrowing.for_node(branch.target)
            message = _shape_error(value, expected, (), sources, producers, inputs)
            if message is not None:
                raise self._error(branch, message)
            return
        try:
            build_value_as(expected, prepared_literal(value))
        except TypeCheckError as exc:
            if value is None:
                raise self._error(
                    branch,
                    "null needs an Optional output; the case declares "
                    f"{canonical_type_label(expected)} (write "
                    f"`output: Optional[{_authored_type(desc, expected)}]`)",
                ) from None
            message = (
                f"the value {_shown(value)} is not a valid {canonical_type_label(expected)}"
            )
            if exc.path:
                message += f" ({exc})"
            message += self._quote_hint(branch, value, expected)
            raise self._error(branch, message) from None

    def _span_sources(self, synth_id: str, value: Any) -> dict[ValuePath, tuple[str, Any]]:
        """Map each span string of a value to its text as authored and as desugared.

        The desugared text (with `${case.output}` and inline calls rewritten) is the value
        node's input that the lifter bound the string to.

        Args:
            synth_id (`str`): The value node id.
            value (`Any`): The authored value.

        Returns:
            `dict[ValuePath, tuple[str, Any]]`: Path -> (authored text, desugared binding
            source).

        Raises:
            `ValueError`: The value node does not bind a span the lifter names.
        """
        inputs = getattr(self.descriptors.get(synth_id), "inputs", None) or {}
        sources: dict[ValuePath, tuple[str, Any]] = {}
        for span in value_spans(value):
            if span.name not in inputs:
                raise ValueError(
                    f"value node {synth_id!r} has no input {span.name!r} for the span "
                    f"{span.text!r}"
                )
            sources[span.path] = (span.text, binding_source(inputs[span.name]))
        return sources

    def _check_sibling_reads(
        self,
        branch: _Branch,
        desc: CaseDescriptor,
        sources: dict[ValuePath, tuple[str, Any]],
    ) -> None:
        """Reject a span that plainly reads another branch target of the same case.

        A read inside a coalesce or under a `:-` default loads: it tolerates that branch
        not running.

        Args:
            branch (`_Branch`): The value branch.
            desc (`CaseDescriptor`): The case that owns it.
            sources (`dict[ValuePath, tuple[str, Any]]`): The value's span strings by
                path, as authored and as desugared.

        Raises:
            `LoadError`: A span reads a sibling branch target outside a coalesce or a
                `:-` default.
        """
        siblings = {
            other.target
            for other in _branches(branch.case_id, desc)
            if other.target != branch.target
        }
        for authored, _ in sources.values():
            try:
                segments = scan_template(authored)
            except ExpressionError:
                continue  # malformed spans are rejected when the value is lifted
            for segment in segments:
                if not isinstance(segment, Span):
                    continue
                for ref in _plain_reads(segment.tree):
                    head = ref.split(".", 1)[0]
                    if head in siblings:
                        raise self._error(
                            branch,
                            f"the value reads `{head}`, another branch of "
                            f"{branch.case_id!r}, which never runs when this branch does",
                        )

    def _quote_hint(self, branch: _Branch, value: Any, expected: Type) -> str:
        """Return a hint to rewrite a value that does not fit, else `""`.

        A YAML 1.1 boolean word where a boolean fits, at any depth, gets the
        `true`/`false` to write; a YAML non-string scalar whose text would fit gets
        the quoted form.

        Args:
            branch (`_Branch`): The branch the value sits on.
            value (`Any`): The authored value.
            expected (`Type`): The case's type.

        Returns:
            `str`: The hint, starting with `"; "`, or `""`.
        """
        fix = former_boolean_fix(value, expected)
        if fix is not None:
            return f"; {fix[1]}"
        if not isinstance(value, (bool, int, float)):
            return ""
        text = self._source_text(branch.range) or _shown(value)
        try:
            build_value_as(expected, text)
        except TypeCheckError:
            return ""
        hint = f"; quote it to keep it as text: `{branch.label} {yaml_quoted(text)}`"
        if isinstance(value, bool):
            hint += " (YAML reads unquoted true/false as booleans)"
        return hint

    def _source_text(self, rng: Optional[SourceRange]) -> Optional[str]:
        """Return the authored text a one-line range covers, or `None` when unknown."""
        text = self.diagnostics.text
        if text is None or rng is None or rng.start_line != rng.end_line:
            return None
        lines = text.splitlines()
        if not 1 <= rng.start_line <= len(lines):
            return None
        return lines[rng.start_line - 1][rng.start_col - 1 : rng.end_col - 1].strip() or None

    def _error(self, branch: _Branch, message: str) -> LoadError:
        """Build the load error for `branch`, located at its range."""
        return self.diagnostics.error(f"{branch.host} {message}", range=branch.range)


def _shape_error(
    value: Any,
    expected: Type,
    path: ValuePath,
    sources: dict[ValuePath, tuple[str, Any]],
    producers: Mapping[str, OutputContract],
    flow_inputs: Mapping[str, ValueContract],
) -> Optional[str]:
    """Return why a value with spans can never fit `expected`, else `None`.

    Args:
        value (`Any`): The authored value, or a part of it.
        expected (`Type`): The type expected at `path`.
        path (`ValuePath`): The record fields and list indices from the value's root.
        sources (`dict[ValuePath, tuple[str, Any]]`): The value's span strings by path, as
            authored and as desugared.
        producers (`Mapping[str, OutputContract]`): Producer contracts as the value node
            sees them.
        flow_inputs (`Mapping[str, ValueContract]`): Flow input contracts as the value
            node sees them.

    Returns:
        `Optional[str]`: The error message without the branch prefix, or `None` when the
        value may fit (a narrower or unproven span is left to the runtime check).
    """
    if not any(span_path[: len(path)] == path for span_path in sources):
        try:
            build_value_as(expected, prepared_literal(value))
        except TypeCheckError as exc:
            message = _mismatch(path, f"is {_shown(value)}", expected)
            return message + f" ({exc})" if exc.path else message
        return None
    if isinstance(value, str):
        inferred = resolve_output_binding_contract(
            sources[path][1], dict(producers), dict(flow_inputs)
        )
        if type(inferred) is not ValueContract:
            return None
        if type_relation(inferred.type, expected) != "disjoint":
            return None
        return _mismatch(path, f"is {canonical_type_label(inferred.type)}", expected)
    # The value itself is not null, so only the non-null part of the type can take it.
    base = replace(expected, nullable=False) if expected.nullable else expected
    if isinstance(value, dict):
        if base.variants is not None:
            if any(
                _shape_error(value, member, path, sources, producers, flow_inputs) is None
                for member in base.variants
            ):
                return None
            return _mismatch(path, "fits no member", expected)
        if base.fields is not None:
            missing = sorted((base.required or frozenset()) - value.keys())
            if missing:
                return _mismatch(path, f"is missing required field {missing[0]!r}", expected)
            unknown = [key for key in value if key not in base.fields]
            if unknown:
                return _mismatch(path, f"has unknown field {unknown[0]!r}", expected)
            for key, item in value.items():
                message = _shape_error(
                    item, base.fields[key], path + (key,), sources, producers, flow_inputs
                )
                if message is not None:
                    return message
            return None
        if base.kind is ValueKind.OBJECT:
            return None
        return _mismatch(path, "is a record", expected)
    if isinstance(value, list):
        if not base.kind.is_list():
            return _mismatch(path, "is a list", expected)
        element = base.element
        if element is None and base.kind.element_kind is not None:
            element = Type.scalar(base.kind.element_kind)
        if element is None:
            return None
        for index, item in enumerate(value):
            message = _shape_error(
                item, element, path + (index,), sources, producers, flow_inputs
            )
            if message is not None:
                return message
    return None


def _mismatch(path: ValuePath, found: str, expected: Type) -> str:
    """Say that the value at `path` does not fit the type expected there.

    Args:
        path (`ValuePath`): Where the part sits; `()` for the whole value.
        found (`str`): What the part is, such as `"is int"` or `"is a record"`.
        expected (`Type`): The type expected at `path`.

    Returns:
        `str`: `"declared T, but the value <found>"` for the whole value, else
        `"the value at <path> <found>, but the output expects T there"`.
    """
    label = canonical_type_label(expected)
    if not path:
        return f"declared {label}, but the value {found}"
    return f"the value at {_path_text(path)} {found}, but the output expects {label} there"


def _plain_reads(tree: Any) -> list[str]:
    """Return the refs a parsed span reads outside every absence-tolerant operand.

    Args:
        tree (`Any`): A parsed span (a lark `Tree`, or a `Token` for a lone atom).

    Returns:
        `list[str]`: The refs of `tree` minus those inside a coalesce or a `:-` default.
    """
    tolerated: Counter[str] = Counter()
    for subtree in _tolerates_absence(tree):
        tolerated.update(expr_refs(subtree))
    return list((Counter(expr_refs(tree)) - tolerated).elements())


def _tolerates_absence(tree: Any) -> list[Any]:
    """Return the outermost coalesce and `:-` default subtrees of a parsed span.

    A ref inside one of them may read a node that never ran: the coalesce takes the
    next operand, and the default supplies its fallback.

    Args:
        tree (`Any`): A parsed span (a lark `Tree`, or a `Token` for a lone atom).

    Returns:
        `list[Any]`: The outermost absence-tolerant subtrees, in source order.
    """
    data = getattr(tree, "data", None)
    if data is None:
        return []  # a lone token
    if data in _ABSENCE_TOLERANT:
        return [tree]
    found: list[Any] = []
    for child in tree.children:
        found.extend(_tolerates_absence(child))
    return found


def _path_text(path: ValuePath) -> str:
    """Render a value path as `.reply.items[0]`."""
    return "".join(f"[{part}]" if isinstance(part, int) else f".{part}" for part in path)


def _target_name(target: str, desc: Any) -> str:
    """Name a node or synth call branch target for a message."""
    if isinstance(desc, CallDescriptor) and target.startswith(_SYNTH_PREFIX):
        return f"the call to `{desc.call}`"
    return f"`{target}`"


def _authored_type(desc: CaseDescriptor, expected: Type) -> str:
    """Return the case's `output:` as written when it is one string, else its label."""
    if isinstance(desc.outputs, str):
        return desc.outputs.strip()
    return canonical_type_label(expected)


def _shown(value: Any) -> str:
    """Render a value as JSON (`true`, `null`, `"text"`), else its `repr`."""
    try:
        return json.dumps(prepared_literal(value), ensure_ascii=False)
    except (TypeError, ValueError):
        return repr(value)
