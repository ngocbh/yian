"""kind=map — mapped function application (`List.map`); a `CallDescriptor` subclass (#70).

The second of the two REF/MAP kinds. A `map` is `kind: map` + `over: ${list}` — iteration with
`${item}` in element scope, built as a `MapNode` (`parallel:` overlaps the element runs).
`MapDescriptor` subclasses `CallDescriptor` (in `kinds/call.py`), adding only the map FIELD
SURFACE (`over`/`parallel` join call's `call`/`inputs`) via `allowed`/`required`; it inherits the
template `parse`, the resolve-and-bake `build` (which builds a `MapNode` because the descriptor is
a `MapDescriptor`), and every field (`over`/`parallel` live on the shared base class). Registered
into `KINDS`/`BINDING_FIELDS` at import.
"""

from __future__ import annotations

from dataclasses import dataclass

from agent_composer.compose.kinds.call import CallDescriptor


@dataclass(frozen=True, kw_only=True)
class MapDescriptor(CallDescriptor):
    """kind=map — `List.map`: apply a callable over each element of an `over: ${list}` source.

    A `CallDescriptor` with a wider authoring surface: `over` (the iteration source) is required
    alongside `call`, and `parallel` opts into overlapping element runs. It declares NO new fields
    (`over`/`parallel` already live on `CallDescriptor`); it only widens `allowed`/`required` and
    sets `kind="map"`, which is what makes `CallDescriptor.build` construct a `MapNode` (it
    branches on `isinstance(self, MapDescriptor)`). Inherits the template `parse` and
    `binding_fields=("inputs","over")` from `CallDescriptor`.
    """

    kind = "map"
    allowed = CallDescriptor.allow("over", "parallel")
    required = CallDescriptor.require("over")
