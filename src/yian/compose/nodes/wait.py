"""kind=wait — the blocking absolute-deadline node's descriptor + build handle.

One of the two-axis `compose/nodes/<kind>.py` modules: it owns the `wait` node's frozen
`WaitDescriptor`, inheriting the template `parse` from `NodeDescriptor` (which records each
field's `SourceRange`), and overriding `build` (which constructs the runtime `WaitNode` + the
reserved `until` wiring). `_finish` stamps only the descriptor's opaque origin onto that node
and retains concrete ranges on the Compose-owned `BuildResult`. Its `_validate_fields` rejects
an `until:` literal that is not a date or datetime, at the value: a reference is typed later,
from the graph. Registered into `KINDS`/`BINDING_FIELDS` at import so the shared dispatch
routes here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Optional

from yian.compose.helpers.errors import LoadDiagnostics
from yian.compose.helpers.tags import has_span
from yian.compose.helpers.value_nesting import check_written_value
from yian.compose.nodes.base import (
    BuildContext,
    BuildResult,
    NodeDescriptor,
    reject_container_spans,
    value_type_error,
)
from yian.diagnostics import SourceRange
from yian.nodes.wait import WaitNode
from yian.typesys.temporal import parse_date_or_datetime

_UNTIL_FIX = (
    "write one such as `until: 2026-05-21`, or a reference such as `until: ${input.as_of}`"
)


def _reject_non_date_until(
    node_id: str,
    value: Any,
    range: Optional[SourceRange],
    diagnostics: LoadDiagnostics,
) -> None:
    """Reject an `until:` literal that is not a date or datetime.

    A YAML date or datetime passes, as does text that parses as one or holds a `${...}`
    span. A deadline whose UTC conversion overflows at the calendar's ends (such as
    `'0001-01-01T00:00:00+05:00'`) parses here and is left to fail at run time.

    Args:
        node_id (`str`): The node id, for the message.
        value (`Any`): The authored `until:` value.
        range (`SourceRange | None`): Where the error points.
        diagnostics (`LoadDiagnostics`): Factory for located load errors.

    Raises:
        `LoadError`: Raised when `value` is neither a date, a datetime, date text, nor a
            reference.
    """
    # A datetime is a date, so this passes both.
    if isinstance(value, date):
        return
    if isinstance(value, str):
        if has_span(value):
            return
        try:
            parse_date_or_datetime(value)
        except ValueError:
            pass
        else:
            return
        problem = (
            "`until:` must be a date or datetime, but YAML read it as text that is not a "
            f"date or datetime; {_UNTIL_FIX}"
        )
    else:
        problem = value_type_error("until", "a date or datetime", value, _UNTIL_FIX)
    raise diagnostics.error(f"node {node_id!r} (kind=wait): {problem}", range=range)


@dataclass(frozen=True, kw_only=True)
class WaitDescriptor(NodeDescriptor):
    """kind=wait — block until an absolute date/datetime; produces ``None``.

    Inherits the common descriptor fields (`id`/`node_name`/`depends_on`/`runs_after`/`env` +
    the Compose-only `range`/`field_ranges` parse metadata) and the template `parse` from `NodeDescriptor` (no
    custom parse hook — a wait authors no `asserts`, so the default asserts check is a no-op).
    Only `until` is declared here; a wait never populates `inputs`/`outputs`.

    Class attrs: `kind="wait"`; `allowed` adds only `until` to the commons; `required` is
    `{"until"}`; `binding_fields=()` — a WAIT's `until` source rides the reserved wiring key,
    not the kind-blind inline-call walk, so there is no author sink-binding surface.
    """

    kind = "wait"
    allowed = NodeDescriptor.allow("until")
    required = NodeDescriptor.require("until")
    binding_fields = ()

    until: str

    @classmethod
    def _validate_fields(
        cls,
        node_id: str,
        kwargs: dict[str, Any],
        ranges: dict[tuple[str, str], SourceRange],
        own_range: Optional[SourceRange],
        diagnostics: LoadDiagnostics,
    ) -> None:
        """Reject an `until:` value that is not a date, a datetime, or a reference.

        A value nested past the limit, or a list or mapping holding `${...}`, gets its own
        message first; then the base checks run.
        """
        until = kwargs.get("until")
        site = ranges.get(("field", "until")) or own_range
        # Before the span walk, which recurses through the value.
        check_written_value(until, f"node {node_id!r} field 'until'", site, diagnostics)
        reject_container_spans(cls.kind, node_id, "until", until, site, diagnostics)
        _reject_non_date_until(node_id, until, site, diagnostics)
        super()._validate_fields(node_id, kwargs, ranges, own_range, diagnostics)

    def build(self, ctx: BuildContext) -> BuildResult:
        """Construct the runtime `WaitNode` (+ wiring) for this `WaitDescriptor`.

        Builds the wait leaf: its required `until` SOURCE rides the reserved wiring key
        (`{"until": self.until}`), never a node-side param. `_finish` stamps the descriptor
        origin on the node and keeps its ranges on the Compose-owned `BuildResult`.
        `ctx.resolver` is unused (leaf kind), and the inline-CODE anchors on `ctx` are read only
        by the CODE build.
        """
        node = WaitNode(self.id, title=self.node_name)
        node.params = []
        return self._finish(node, {"until": self.until})
