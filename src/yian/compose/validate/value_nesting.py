"""Reject a binding, case branch or flow output written out deeper than a value may nest.

Runs on the authored values, before any pass that walks them recursively (the
tagged-binding read, the case-branch check, the desugars), so a value built deep
through YAML aliases is a located load error rather than a `RecursionError`. Each
binding is checked as the run-time binding check counts it. MAP's `over:` is checked
when the node is parsed, so it is skipped here. A case branch that holds a value is
always a load error that suggests a value entry; one past the limit reports the limit
instead, since that entry couldn't load either. Flow outputs follow the tagged split:
an untagged mapping is a record whose fields sit one level in, and a tagged section or
any other value is the single output.
"""

from __future__ import annotations

from typing import Any, Callable

from yian.compose.desugar.value_entries import ValueEntryDescriptor
from yian.compose.helpers.errors import LoadDiagnostics
from yian.compose.helpers.value_nesting import check_written_value
from yian.compose.nodes import BINDING_FIELDS
from yian.compose.nodes.case import CaseDescriptor
from yian.diagnostics import SourceRange

# The descriptor field that names one binding per key.
_INPUTS_FIELD = "inputs"


def check_written_values(
    descriptors: dict[str, Any],
    outputs_section: Any,
    *,
    outputs_tagged: bool,
    output_range: Callable[[Any], SourceRange | None],
    outputs_range: SourceRange | None,
    diagnostics: LoadDiagnostics,
) -> None:
    """Reject a binding, case branch or flow output whose written-out value nests too deep.

    Args:
        descriptors (`dict[str, Any]`): Authored node descriptors keyed by node id.
        outputs_section (`Any`): The authored `output:` section, tags not yet split.
        outputs_tagged (`bool`): Whether the whole section carries a tag, which makes
            it the single output even when it is a mapping.
        output_range (`Callable[[Any], SourceRange | None]`): Location of a record
            output, by name.
        outputs_range (`SourceRange | None`): Location of the whole `output:` section.
        diagnostics (`LoadDiagnostics`): Factory for located load errors.

    Raises:
        `LoadError`: Raised at the offending binding, branch or output for the first
            value past the limit, nodes in authored order and then outputs.
    """
    for node_id, desc in descriptors.items():
        if isinstance(desc, ValueEntryDescriptor):
            continue
        if isinstance(desc, CaseDescriptor):
            for slot, value, site in _case_value_branches(desc):
                check_written_value(
                    value, f"node {node_id!r} {slot}", site or desc.range, diagnostics
                )
            continue
        if _INPUTS_FIELD not in BINDING_FIELDS.get(desc.kind, ()):
            continue
        inputs = getattr(desc, _INPUTS_FIELD, None)
        if not isinstance(inputs, dict):
            continue
        for key, value in inputs.items():
            check_written_value(
                value,
                f"node {node_id!r} input {str(key)!r}",
                desc.field_ranges.get(("input", key))
                or desc.field_ranges.get(("field", _INPUTS_FIELD))
                or desc.range,
                diagnostics,
            )
    if outputs_tagged or not isinstance(outputs_section, dict):
        check_written_value(outputs_section, "flow output", outputs_range, diagnostics)
        return
    for key, value in outputs_section.items():
        check_written_value(
            value,
            f"flow output {str(key)!r}",
            output_range(key),
            diagnostics,
            enclosing_levels=1,
        )


def _case_value_branches(
    desc: CaseDescriptor,
) -> list[tuple[str, Any, SourceRange | None]]:
    """List a case's authored `then:` and `else:` branches with their slots and ranges.

    Args:
        desc (`CaseDescriptor`): The authored case.

    Returns:
        `list[tuple[str, Any, SourceRange | None]]`: Each branch's slot as messages
            name it (`"then[0] value"`, `"else value"`), its authored value, and its
            range, `then:` branches in order and then `else:`.
    """
    branches = [
        (
            f"then[{index}] value",
            case["then"],
            desc.then_ranges[index] if index < len(desc.then_ranges) else None,
        )
        for index, case in enumerate(desc.cases)
        if isinstance(case, dict) and "then" in case
    ]
    if desc.has_else:
        branches.append(
            ("else value", desc.else_, desc.field_ranges.get(("field", "else")))
        )
    return branches
