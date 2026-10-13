"""Graph-stage: every branch of a typed `case` must be able to produce its `output:` type.

A `case` that declares `output: T` gets one synth result node that checks the taken branch
against T at runtime (`binding.type`). This load pass rejects the branches that could never
pass that check, located at the branch itself rather than at the result node:

- a value branch (lifted to a synth `__value_<n>` node; its authored value is kept in
  `CaseDescriptor.value_branches`) with no `${...}` span anywhere is checked exactly with
  the write-boundary check, after the same preparation the lifted code applies (a date
  becomes its ISO text, a span-free string is `$$`-unescaped). A `null` into a
  non-Optional T, and a YAML non-string scalar (`yes`, `3`) where T would take the text,
  get a message saying how to fix it;
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
`compose.helpers` (load diagnostics, YAML quoting), `compose.parse.file` (descriptors), and
`compose.validate.narrowing`. Only the loader pipeline imports this back.
"""

from __future__ import annotations

import datetime
import json
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from yian.compile.validation import type_relation
from yian.compose.helpers.errors import LoadDiagnostics
from yian.compose.helpers.typed_bindings import yaml_quoted
from yian.compose.parse.file import CallDescriptor, CaseDescriptor
from yian.compose.validate.narrowing import Narrowing
from yian.diagnostics import SourceRange
from yian.expr import eval_binding
from yian.nodes.contracts import OutputContract, ValueContract
from yian.typesys import TypeCheckError, build_value_as, canonical_type_label
from yian.typesys.values import Type


@dataclass(frozen=True)
class _Branch:
    """One `then:`/`else:` of a case, after desugar.

    Attributes:
        case_id (`str`): The case that owns the branch.
        label (`str`): `"then:"` or `"else:"`.
        target (`Any`): The branch target: a node id, a synth value or call id.
        range (`Optional[SourceRange]`): The branch's range, else the case's range.
    """

    case_id: str
    label: str
    target: Any
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
    narrowing: Optional[Narrowing] = None,
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
        narrowing (`Optional[Narrowing]`, *optional*, defaults to `None`): The union
            narrowing each node reads against; `None` reads every node against the
            unchanged maps.
        diagnostics (`Optional[LoadDiagnostics]`, *optional*, defaults to `None`): Source
            ownership for located errors.

    Raises:
        `LoadError`: A literal value does not fit the case's type; a node, call, or typed
            inner case branch is disjoint from it.
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
    narrowing: Optional[Narrowing]
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
                self._check_value(branch, desc.value_branches[target], expected, desc)
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

    def _check_value(
        self, branch: _Branch, value: Any, expected: Type, desc: CaseDescriptor
    ) -> None:
        """Check one value branch's authored value against `expected`.

        Args:
            branch (`_Branch`): The branch the value sits on.
            value (`Any`): The authored value.
            expected (`Type`): The case's type.
            desc (`CaseDescriptor`): The case, for the authored `output:` text.

        Raises:
            `LoadError`: The value does not fit `expected`.
        """
        if _has_span(value):
            return
        try:
            build_value_as(expected, _prepared(value))
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

    def _quote_hint(self, branch: _Branch, value: Any, expected: Type) -> str:
        """Return a hint to quote a YAML non-string scalar whose text would fit, else `""`.

        Args:
            branch (`_Branch`): The branch the value sits on.
            value (`Any`): The authored value.
            expected (`Type`): The case's type.

        Returns:
            `str`: The hint, starting with `"; "`, or `""`.
        """
        if not isinstance(value, (bool, int, float)):
            return ""
        text = self._source_text(branch.range) or _shown(value)
        try:
            build_value_as(expected, text)
        except TypeCheckError:
            return ""
        hint = f"; quote it to keep it as text: `{branch.label} {yaml_quoted(text)}`"
        if isinstance(value, bool):
            hint += " (YAML reads unquoted yes/no/on/off/true/false as booleans)"
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

    def _error(self, branch: _Branch, message: str):
        """Build the load error for `branch`, located at its range."""
        return self.diagnostics.error(f"{branch.host} {message}", range=branch.range)


def _target_name(target: str, desc: Any) -> str:
    """Name a node or synth call branch target for a message."""
    if isinstance(desc, CallDescriptor) and target.startswith("__call_"):
        return f"the call to `{desc.call}`"
    return f"`{target}`"


def _authored_type(desc: CaseDescriptor, expected: Type) -> str:
    """Return the case's `output:` as written when it is one string, else its label."""
    if isinstance(desc.outputs, str):
        return desc.outputs.strip()
    return canonical_type_label(expected)


def _has_span(value: Any) -> bool:
    """Whether any string in `value` (recursively) holds a `${` span."""
    if isinstance(value, str):
        return "${" in value
    if isinstance(value, dict):
        return any(_has_span(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_span(item) for item in value)
    return False


def _prepared(value: Any) -> Any:
    """Return a span-free value as its lifted code produces it.

    A date becomes its ISO text and a string is `$$`-unescaped, as the binder does.
    """
    if isinstance(value, str):
        return eval_binding(value, lambda path: None)
    if isinstance(value, (datetime.date, datetime.datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _prepared(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_prepared(item) for item in value]
    return value


def _shown(value: Any) -> str:
    """Render a value as JSON (`true`, `null`, `"text"`), else its `repr`."""
    try:
        return json.dumps(_prepared(value), ensure_ascii=False)
    except (TypeError, ValueError):
        return repr(value)
