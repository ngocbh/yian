"""kind=map — mapped function application (`List.map`); a `CallDescriptor` subclass.

The mapped variant of callable application. A `map` is `kind: map` + `over: ${list}` — iteration with
`${item}` in element scope, built as a `MapNode`. `parallel:` is retained but currently inert;
run-level workers control concurrency.
`MapDescriptor` subclasses `CallDescriptor` (in `call.py`), adding only the map fields
(`over`/`parallel`) via `allowed`/`required`; it inherits the normalized `inputs` carrier,
the resolve-and-bake `build` (which builds a `MapNode` because the descriptor is a
`MapDescriptor`), and every field (`over`/`parallel` live on the shared base class). Its
`parse` wraps the template parse and rejects an `over:` literal that is not a list, at the
value: a reference or a `call(...)` directive is typed later, from the graph. Registered into
`KINDS`/`BINDING_FIELDS` at import. The inherited `_finish` retains concrete ranges on the
Compose-owned `BuildResult`; the runtime node receives only the descriptor's opaque origin.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from yian.compose.helpers.directives import CALL_HEAD
from yian.compose.helpers.errors import LoadDiagnostics
from yian.compose.helpers.tags import QUOTED_STYLES, has_span
from yian.compose.helpers.value_nesting import check_written_value
from yian.compose.nodes.base import reject_container_spans, value_type_error
from yian.compose.nodes.call import CallDescriptor
from yian.compose.source import SourceCursor
from yian.diagnostics import SourceRange

_OVER_FIX = "write a list, or a reference such as `over: ${input.items}`"


def _reject_non_list_over(
    node_id: str,
    value: Any,
    style: Optional[str],
    range: Optional[SourceRange],
    diagnostics: LoadDiagnostics,
) -> None:
    """Reject an `over:` literal that is not a list.

    Text holding a `${...}` span is a reference and passes, as is an unquoted plain scalar
    starting with `call(`: a directive the loader lifts into a node after this check. A
    quoted or block scalar is text, as in any binding.

    Args:
        node_id (`str`): The node id, for the message.
        value (`Any`): The authored `over:` value.
        style (`str | None`): The value's YAML scalar style; `None` reads as plain.
        range (`SourceRange | None`): Where the error points.
        diagnostics (`LoadDiagnostics`): Factory for located load errors.

    Raises:
        `LoadError`: Raised when `value` is neither a list nor a reference.
    """
    if isinstance(value, list):
        return
    if isinstance(value, str):
        if has_span(value):
            return
        if style not in QUOTED_STYLES and CALL_HEAD.match(value.strip()):
            return
        problem = f"`over:` must be a list, but YAML read it as text; {_OVER_FIX}"
    else:
        problem = value_type_error("over", "a list", value, _OVER_FIX)
    raise diagnostics.error(f"node {node_id!r} (kind=map): {problem}", range=range)


@dataclass(frozen=True, kw_only=True)
class MapDescriptor(CallDescriptor):
    """kind=map — `List.map`: apply a callable over each element of an `over: ${list}` source.

    A `CallDescriptor` with additional authored fields: `over` (the iteration source) is required
    alongside `call`; `parallel` is accepted as inert authoring metadata. It declares NO new
    fields (`over`/`parallel` already live on `CallDescriptor`); it only widens
    `allowed`/`required` and sets `kind="map"`, which is what makes `CallDescriptor.build`
    construct a `MapNode` (it branches on `isinstance(self, MapDescriptor)`). Wraps the
    template `parse` to check `over:` against its scalar style, and inherits
    `binding_fields=("inputs","over")` from `CallDescriptor`.
    """

    kind = "map"
    allowed = CallDescriptor.allow("over", "parallel")
    required = CallDescriptor.require("over")

    @classmethod
    def _validate_fields(
        cls,
        node_id: str,
        kwargs: dict[str, Any],
        ranges: dict[tuple[str, str], SourceRange],
        own_range: Optional[SourceRange],
        diagnostics: LoadDiagnostics,
    ) -> None:
        """Reject `over:` nested past the limit or holding `${...}` in a list or mapping."""
        site = ranges.get(("field", "over")) or own_range
        # Before the span walk, which recurses through the value.
        check_written_value(
            kwargs.get("over"), f"node {node_id!r} field 'over'", site, diagnostics
        )
        reject_container_spans(
            cls.kind,
            node_id,
            "over",
            kwargs.get("over"),
            site,
            diagnostics,
        )
        super()._validate_fields(node_id, kwargs, ranges, own_range, diagnostics)

    @classmethod
    def parse(cls, node_id: str, cursor: SourceCursor) -> "MapDescriptor":
        """Run the template `parse`, then reject an `over:` literal that is not a list.

        The check runs here rather than in `_validate_fields` because whether `call(...)`
        text is a directive depends on the value's scalar style, which only the cursor knows.

        Args:
            node_id (`str`): The node id.
            cursor (`SourceCursor`): The node body.

        Returns:
            `MapDescriptor`: The parsed descriptor.

        Raises:
            `LoadError`: Raised for any field the template rejects, or an `over:` that is
                neither a list nor a reference.
        """
        desc = super().parse(node_id, cursor)
        _reject_non_list_over(
            node_id,
            desc.over,
            cursor.child("over").scalar_style,
            desc.field_ranges.get(("field", "over")) or desc.range,
            LoadDiagnostics.from_cursor(cursor),
        )
        return desc
