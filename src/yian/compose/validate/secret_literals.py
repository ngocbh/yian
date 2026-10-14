"""Reject a literal value under a secret name in node bindings and flow outputs.

Runs on the authored values, before typed bindings are split or anything is desugared,
because those later passes may echo a value in their own errors. In a binding position a
whole-value `call(...)` directive also counts as a binding, and a typed binding
(`Type = value`) is checked on its right side.
"""

from __future__ import annotations

from typing import Any, Callable

from yian.compose.desugar.inline_calls import desugar_call_directives
from yian.compose.helpers.errors import LoadDiagnostics
from yian.compose.helpers.secret_literals import (
    is_binding_text,
    secret_literal_message,
    secret_literal_path,
)
from yian.compose.helpers.typed_bindings import (
    CALL_HEAD,
    QUOTED_STYLES,
    read_typed_value,
    typed_binding_parts,
)
from yian.compose.nodes import BINDING_FIELDS
from yian.compose.source import SourceSyntaxError
from yian.diagnostics import SourceRange
from yian.expr import ExpressionError
from yian.secret_names import is_secret_name

# The descriptor field whose bindings may be typed, as `_split_typed_inputs` splits them.
_TYPED_FIELD = "inputs"


def check_authored_bindings(
    descriptors: dict[str, Any],
    outputs_section: Any,
    *,
    binding_style_at: Callable[[str, str, Any], str | None],
    output_style_at: Callable[[Any], str | None],
    output_range: Callable[[Any], SourceRange | None],
    diagnostics: LoadDiagnostics,
) -> None:
    """Reject a literal value under a secret name in any node binding or flow output.

    Args:
        descriptors (`dict[str, Any]`): Authored node descriptors keyed by node id.
        outputs_section (`Any`): The authored `output:` section; only a mapping has
            named outputs to check.
        binding_style_at (`Callable[[str, str, Any], str | None]`): YAML scalar style
            of a node binding, by node id, field and key (the field again for a
            scalar field).
        output_style_at (`Callable[[Any], str | None]`): YAML scalar style of a flow
            output, by name.
        output_range (`Callable[[Any], SourceRange | None]`): Location of a flow
            output, by name.
        diagnostics (`LoadDiagnostics`): Factory for located load errors.

    Raises:
        `LoadError`: Raised at the offending binding or output for the first literal
            value under a secret name, in authored order.
    """
    for node_id, desc in descriptors.items():
        owner = f"node {node_id!r}"
        for field in BINDING_FIELDS.get(desc.kind, ()):
            current = getattr(desc, field, None)
            if isinstance(current, dict):
                for key, value in current.items():
                    path = _binding_path(
                        key,
                        value,
                        binding_style_at(node_id, field, key),
                        typed=field == _TYPED_FIELD,
                    )
                    if path is None:
                        continue
                    is_input = field == _TYPED_FIELD
                    raise diagnostics.error(
                        secret_literal_message(
                            f"{'input' if is_input else field} {str(key)!r}",
                            str(key),
                            path,
                            owner=owner,
                        ),
                        range=(desc.field_ranges.get(("input", key)) if is_input else None)
                        or desc.field_ranges.get(("field", field))
                        or desc.range,
                    )
            elif current is not None:
                path = _binding_path(
                    field, current, binding_style_at(node_id, field, field), typed=False
                )
                if path is not None:
                    raise diagnostics.error(
                        secret_literal_message(
                            f"field {field!r}", field, path, owner=owner
                        ),
                        range=desc.field_ranges.get(("field", field)) or desc.range,
                    )
    if not isinstance(outputs_section, dict):
        return
    for key, value in outputs_section.items():
        path = _binding_path(key, value, output_style_at(key), typed=True)
        if path is not None:
            raise diagnostics.error(
                secret_literal_message(f"flow output {str(key)!r}", str(key), path),
                range=output_range(key),
            )


def _binding_path(
    name: Any, value: Any, style: str | None, *, typed: bool
) -> tuple[str | int, ...] | None:
    """Locate a literal under a secret name in one binding, as `secret_literal_path`.

    Args:
        name (`Any`): The binding's name.
        value (`Any`): The authored binding value.
        style (`str | None`): The value's YAML scalar style, `None` when unknown.
        typed (`bool`): Whether this position accepts a typed binding.

    Returns:
        `tuple[str | int, ...] | None`: The offending position, or `None`.
    """
    parts = typed_binding_parts(value, style) if typed else None
    if parts is None:
        return secret_literal_path(
            name, value, is_binding=lambda v: _is_binding(v, style)
        )
    try:
        right = read_typed_value(parts[1])
    except SourceSyntaxError:
        # Under another name the typed split reports the malformed right side.
        return () if isinstance(name, str) and is_secret_name(name) else None
    # The right side of a typed binding is never quoted text: the split unquotes it.
    return secret_literal_path(name, right, is_binding=lambda v: _is_binding(v, None))


def _is_binding(value: Any, style: str | None) -> bool:
    """Return whether a binding-position value is a `${...}` or a `call(...)` directive.

    Args:
        value (`Any`): The authored value.
        style (`str | None`): Its YAML scalar style; a quoted `call(...)` is text.

    Returns:
        `bool`: `True` for template text with a span, or an unquoted whole-value
            `call(...)` directive that parses.
    """
    if is_binding_text(value):
        return True
    if not isinstance(value, str) or style in QUOTED_STYLES:
        return False
    if CALL_HEAD.match(value.strip()) is None:
        return False
    try:
        desugar_call_directives(value, lambda: "_")
    except ExpressionError:
        return False
    return True
