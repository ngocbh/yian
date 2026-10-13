"""Graph-stage: `case` then:/else: value branches -> synth CODE nodes.

A `case` branch is a sibling node, a `call(...)`, or a value. The YAML scalar style tells
them apart, so every branch is classified first:

- an unquoted (plain, or source-free) string is a node name unless it is a `call(...)`
  (left for `desugar_case_call_targets`), holds ` = ` (a load error: the case's `output:`
  gives the type, so `then: int = ${x}` is a mistake; quoting keeps the text), or holds a
  `${...}` span (a template value). A bare word that names no node is a load error that
  suggests the nearest node name and the quoted form;
- a quoted string (`"..."`, `'...'`, or a block `|`/`>`) is text, so it is a value; a
  quoted `call(...)` holding a span is a load error, since quoting a call makes it text;
- a non-string scalar (`null`, bool, number, date), a mapping, or a list is a value.

A value branch needs the case's `output:` type. Each value lifts to one synth CODE node
`__value_<n>` whose inline code returns the value, and the branch is rewritten to that id
(recorded in `CaseDescriptor.value_branches`). Each string holding `${` becomes one input
binding `vN`, so the binder evaluates it exactly like a node input; everything else is a
Python literal in the generated code, so a quoted text never reaches the inline-call
desugar. A span that reads the case itself, or a span-bearing `call(...)` nested inside a
mapping or list, is a load error.

Runs in `_assemble` after the typed-binding splits and BEFORE `desugar_inline_calls`, so
the synth node's input bindings are walked by the inline-call and `${case.output}` passes
like any authored CODE input. Authored node ids may not use the reserved `__value_` and
`__case_` prefixes.

Located parse metadata: a synth value node and its inputs locate at the `then:`/`else:` it
was lifted from (the case's positional `then_ranges[N]`, or its `("field", "else")` range),
falling back to the case node's own range.

Imports flow down/peer only: `compose.parse.file` (descriptors), `compose.desugar.inline_calls`
(the reserved `__call_` prefix), `compose.helpers` (errors, YAML quoting), `expr`, `diagnostics`.
Only the loader imports this back.
"""

from __future__ import annotations

import datetime
import difflib
import itertools
import math
import re
from dataclasses import dataclass, replace
from typing import Any, Optional

from yian.compose.desugar.inline_calls import _SYNTH_PREFIX
from yian.compose.helpers.errors import LoadDiagnostics
from yian.compose.helpers.typed_bindings import yaml_quoted
from yian.compose.parse.file import CaseDescriptor, CodeDescriptor
from yian.diagnostics import SourceRange
from yian.expr import ExpressionError, eval_binding, expr_refs_of

VALUE_PREFIX = "__value_"
CASE_PREFIX = "__case_"
_RESERVED_PREFIXES = (VALUE_PREFIX, CASE_PREFIX)
_QUOTED_STYLES = frozenset({"single", "double", "literal", "folded"})
_CALL_HEAD = re.compile(r"call\s*\(")
_CLOSE_MATCH_CUTOFF = 0.6


def desugar_case_values(
    descriptors: dict,
    *,
    diagnostics: LoadDiagnostics | None = None,
) -> dict:
    """Classify every `case` branch and lift each value branch to a synth CODE node.

    Args:
        descriptors (`dict`): The flow's parsed descriptors, keyed by node id.
        diagnostics (`LoadDiagnostics | None`, *optional*, defaults to `None`): Source
            ownership for located errors; a fresh range-only owner when `None`.

    Returns:
        `dict`: A new descriptor map: each case with its value branches rewritten to synth
        ids, followed by one `CodeDescriptor` per value (`__value_0`, `__value_1`, ...).

    Raises:
        `LoadError`: An authored node id uses a reserved prefix; a bare word names no node;
            an unquoted branch holds ` = `; a quoted `call(...)` holds a span; a value
            branch sits on a case without `output:`; a value holds a key that is not a
            string, an unsupported scalar, a nested span-bearing `call(...)`, or a span
            that reads the case itself.

    Example:
        A `then: {a: 1, b: ${x.output}}` branch on a case with `output:` becomes a CODE node
        `__value_0` with `inputs == {"v0": "${x.output}"}` and code
        `return {'a': 1, 'b': inputs['v0']}`; the branch is rewritten to `__value_0`.
    """
    diagnostics = diagnostics or LoadDiagnostics()
    for nid, desc in descriptors.items():
        for prefix in _RESERVED_PREFIXES:
            if nid.startswith(prefix):
                raise diagnostics.error(
                    f"node id {nid!r} uses the reserved prefix {prefix!r}",
                    range=getattr(desc, "range", None),
                )

    counter = itertools.count()
    suggestable = [
        nid
        for nid in descriptors
        if not nid.startswith((_SYNTH_PREFIX, *_RESERVED_PREFIXES))
    ]
    synth: dict[str, CodeDescriptor] = {}
    new_descriptors: dict = {}
    for nid, desc in descriptors.items():
        if not isinstance(desc, CaseDescriptor):
            new_descriptors[nid] = desc
            continue
        value_branches = dict(desc.value_branches)

        new_cases = list(desc.cases)
        for index, case in enumerate(desc.cases):
            if not isinstance(case, dict) or "then" not in case:
                continue
            rng = _nth(desc.then_ranges, index) or desc.range
            target = _branch_target(
                case["then"],
                style=_nth(desc.then_styles, index),
                desc=desc,
                label="then:",
                node_name=f"{nid} then[{index}]",
                rng=rng,
                descriptors=descriptors,
                suggestable=suggestable,
                counter=counter,
                synth=synth,
                value_branches=value_branches,
                diagnostics=diagnostics,
            )
            if target is not case["then"]:
                new_cases[index] = {**case, "then": target}
        new_else = desc.else_
        if desc.has_else:
            new_else = _branch_target(
                desc.else_,
                style=desc.else_style,
                desc=desc,
                label="else:",
                node_name=f"{nid} else",
                rng=desc.field_ranges.get(("field", "else")) or desc.range,
                descriptors=descriptors,
                suggestable=suggestable,
                counter=counter,
                synth=synth,
                value_branches=value_branches,
                diagnostics=diagnostics,
            )
        # Every rewritten branch records its value, so a grown map means a rewrite.
        if len(value_branches) > len(desc.value_branches):
            desc = replace(
                desc, cases=new_cases, else_=new_else, value_branches=value_branches
            )
        new_descriptors[nid] = desc
    new_descriptors.update(synth)
    return new_descriptors


def _nth(seq: list, index: int) -> Any:
    """Return the `index`-th item of a positional list, or None when it is absent."""
    return seq[index] if 0 <= index < len(seq) else None


def _branch_target(
    value: Any,
    *,
    style: Optional[str],
    desc: CaseDescriptor,
    label: str,
    node_name: str,
    rng: Optional[SourceRange],
    descriptors: dict,
    suggestable: list[str],
    counter: itertools.count,
    synth: dict,
    value_branches: dict,
    diagnostics: LoadDiagnostics,
) -> Any:
    """Return the branch's target: the value unchanged, or the id of its synth value node.

    Args:
        value (`Any`): The authored `then:`/`else:` value.
        style (`Optional[str]`): Its YAML scalar style; `None` (unknown) reads as plain.
        desc (`CaseDescriptor`): The case that owns the branch.
        label (`str`): `"then:"` or `"else:"`, for messages.
        node_name (`str`): Display name for a synth value node.
        rng (`Optional[SourceRange]`): The branch's source range.
        descriptors (`dict`): The flow's descriptors; a bare word must name one.
        suggestable (`list[str]`): Authored node ids offered as close matches.
        counter (`itertools.count`): The synth-id counter of this pass.
        synth (`dict`): Collects synth value nodes by id.
        value_branches (`dict`): Collects synth id -> authored value for `desc`.
        diagnostics (`LoadDiagnostics`): Source ownership for located errors.

    Returns:
        `Any`: The (possibly rewritten) branch target.

    Raises:
        `LoadError`: See `desugar_case_values`.
    """
    host = f"case node {desc.id!r} {label}"
    if not _is_value(
        value,
        style=style,
        host=host,
        label=label,
        descriptors=descriptors,
        suggestable=suggestable,
        rng=rng,
        diagnostics=diagnostics,
    ):
        return value
    if desc.outputs is None:
        raise diagnostics.error(
            f"{host} a value branch needs the case's type; add `output: <Type>` to "
            f"{desc.id!r}",
            range=rng,
        )
    synth_id = f"{VALUE_PREFIX}{next(counter)}"
    node = value_node(
        synth_id,
        value,
        node_name=node_name,
        range=rng,
        diagnostics=diagnostics,
        host=host,
        label=label,
    )
    for source in node.inputs.values():
        if _reads_case(source, desc.id):
            raise diagnostics.error(f"{host} the value reads {desc.id!r} itself", range=rng)
    synth[synth_id] = node
    value_branches[synth_id] = value
    return synth_id


def _is_value(
    value: Any,
    *,
    style: Optional[str],
    host: str,
    label: str,
    descriptors: dict,
    suggestable: list[str],
    rng: Optional[SourceRange],
    diagnostics: LoadDiagnostics,
) -> bool:
    """Classify one branch: True for a value, False for a node name or a `call(...)`.

    Args:
        value (`Any`): The authored `then:`/`else:` value.
        style (`Optional[str]`): Its YAML scalar style; `None` (unknown) reads as plain.
        host (`str`): Message prefix naming the branch, such as `"case node 'r' then:"`.
        label (`str`): `"then:"` or `"else:"`, for the quoted form a message suggests.
        descriptors (`dict`): The flow's descriptors; a bare word must name one.
        suggestable (`list[str]`): Authored node ids offered as close matches.
        rng (`Optional[SourceRange]`): The branch's source range.
        diagnostics (`LoadDiagnostics`): Source ownership for located errors.

    Returns:
        `bool`: Whether the branch is a value to lift.

    Raises:
        `LoadError`: An unquoted branch holds ` = ` or names no node; a quoted
            `call(...)` holds a span.
    """
    if not isinstance(value, str):
        return True
    stripped = value.strip()
    # Matches every whole-value directive the call recognizer accepts, and the rejected
    # call shapes (a coalesce of calls, trailing text) it declines.
    is_call = _CALL_HEAD.match(stripped) is not None
    if style in _QUOTED_STYLES:
        if is_call and "${" in value:
            raise diagnostics.error(
                f"{host} a quoted `call(...)` is text, so it can't hold `${{...}}` here; "
                "drop the quotes to call the flow, or reword the text",
                range=rng,
            )
        return True
    if is_call:
        # Lifted (or rejected) by `desugar_case_call_targets`.
        return False
    if " = " in value or value.endswith(" ="):
        raise diagnostics.error(
            f"{host} `{value}` is not a branch; the case's `output:` gives the type. "
            f"If you meant the text, quote it: `{label} {yaml_quoted(value)}`",
            range=rng,
        )
    if "${" in value:
        return True
    if value not in descriptors:
        matches = difflib.get_close_matches(
            value, suggestable, n=1, cutoff=_CLOSE_MATCH_CUTOFF
        )
        suggestion = f" Did you mean `{matches[0]}`?" if matches else ""
        raise diagnostics.error(
            f"{host} `{value}` names no node.{suggestion} If you meant the text "
            f'"{value}", quote it: `{label} {yaml_quoted(value)}`',
            range=rng,
        )
    return False


def value_node(
    synth_id: str,
    value: Any,
    *,
    node_name: str,
    range: Optional[SourceRange],
    diagnostics: LoadDiagnostics,
    host: str = "case value",
    label: str = "then:",
) -> CodeDescriptor:
    """Build the synth CODE node that produces one branch value.

    Args:
        synth_id (`str`): The node id (`__value_<n>`).
        value (`Any`): The authored value: a scalar, mapping, or list, possibly holding
            `${...}` spans in its strings.
        node_name (`str`): Display name, such as `"route then[0]"`.
        range (`Optional[SourceRange]`): The branch's source range; the node and each of its
            inputs locate here.
        diagnostics (`LoadDiagnostics`): Source ownership for located errors.
        host (`str`, *optional*, defaults to `"case value"`): Message prefix naming the
            branch.
        label (`str`, *optional*, defaults to `"then:"`): The branch key, `"then:"` or
            `"else:"`, for the fix a message suggests.

    Returns:
        `CodeDescriptor`: An untyped inline CODE node. Each string holding `${` is one
        input `v0`, `v1`, ... (walk order); the code `return <expr>` builds the value with
        every other part written as a Python literal.

    Raises:
        `LoadError`: A mapping key is not a string, a scalar has no literal form, or a
            span-bearing `call(...)` is nested inside a mapping or list.

    Example:
        `value_node("__value_0", {"t": "${input.t}", "done": True}, ...)` has
        `inputs == {"v0": "${input.t}"}` and code `return {'t': inputs['v0'], 'done': True}`.
    """
    inputs: dict[str, str] = {}
    site = _Site(host=host, label=label, range=range, diagnostics=diagnostics)
    expr = _literal(value, inputs, nested=False, site=site)
    return CodeDescriptor(
        id=synth_id,
        node_name=node_name,
        code=f"return {expr}",
        inputs=inputs,
        outputs=None,
        range=range,
        field_ranges={("input", name): range for name in inputs} if range else {},
    )


@dataclass(frozen=True)
class _Site:
    """Where a value sits, for located errors while it is written as Python."""

    host: str
    label: str
    range: Optional[SourceRange]
    diagnostics: LoadDiagnostics


def _literal(value: Any, inputs: dict[str, str], *, nested: bool, site: _Site) -> str:
    """Return Python source that builds `value`, adding each span-bearing string to `inputs`.

    Args:
        value (`Any`): A branch value or a part of one.
        inputs (`dict[str, str]`): Collects input name -> template string, in walk order.
        nested (`bool`): Whether `value` sits inside a mapping or list.
        site (`_Site`): Where the branch sits, for located errors.

    Returns:
        `str`: A Python expression that evaluates to the value.

    Raises:
        `LoadError`: A mapping key is not a string, a scalar has no literal form, or a
            nested span-bearing string is a `call(...)`.
    """
    if isinstance(value, str):
        if "${" in value:
            if nested and _CALL_HEAD.match(value.strip()):
                raise site.diagnostics.error(
                    f"{site.host} a `call(...)` nested in a value would run outside the "
                    f"branch; write the call as the branch (`{site.label} call(...)`) or in "
                    "a node",
                    range=site.range,
                )
            name = f"v{len(inputs)}"
            inputs[name] = value
            return f"inputs[{name!r}]"
        # The binder unescapes `$$` in a span-free string; the literal must match it.
        return repr(eval_binding(value, lambda path: None))
    if value is None or isinstance(value, (bool, int)):
        return repr(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "float('nan')"
        if math.isinf(value):
            return "float('inf')" if value > 0 else "float('-inf')"
        return repr(value)
    if isinstance(value, (datetime.date, datetime.datetime)):
        return repr(value.isoformat())
    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            if not isinstance(key, str):
                raise site.diagnostics.error(
                    f"{site.host} record keys must be strings (got {key!r})",
                    range=site.range,
                )
            parts.append(f"{key!r}: {_literal(item, inputs, nested=True, site=site)}")
        return "{" + ", ".join(parts) + "}"
    if isinstance(value, list):
        items = [_literal(item, inputs, nested=True, site=site) for item in value]
        return "[" + ", ".join(items) + "]"
    raise site.diagnostics.error(
        f"{site.host} a branch value can't hold a {type(value).__name__}",
        range=site.range,
    )


def _reads_case(source: str, case_id: str) -> bool:
    """Return True when a template string reads `${<case_id>.output...}`."""
    try:
        refs = expr_refs_of(source)
    except ExpressionError:
        return False  # malformed spans are reported by reference validation
    for ref in refs:
        parts = ref.split(".")
        if len(parts) >= 2 and parts[0] == case_id and parts[1] == "output":
            return True
    return False
