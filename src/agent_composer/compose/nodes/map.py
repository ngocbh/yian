"""kind=map — mapped function application (`List.map`); a `CallDescriptor` subclass.

The mapped variant of callable application. A `map` is `kind: map` + `over: ${list}` — iteration with
`${item}` in element scope, built as a `MapNode`. `parallel:` is retained but currently inert;
run-level workers control concurrency.
`MapDescriptor` subclasses `CallDescriptor` (in `call.py`), adding only the map fields
(`over`/`parallel`) via `allowed`/`required`; it inherits the normalized `inputs` carrier,
template `parse`, the resolve-and-bake `build` (which builds a `MapNode` because the descriptor is
a `MapDescriptor`), and every field (`over`/`parallel` live on the shared base class). Registered
into `KINDS`/`BINDING_FIELDS` at import. The inherited `_finish` retains concrete ranges on the
Compose-owned `BuildResult`; the runtime node receives only the descriptor's opaque origin.
"""

from __future__ import annotations

from dataclasses import dataclass

from agent_composer.compose.nodes.call import CallDescriptor


@dataclass(frozen=True, kw_only=True)
class MapDescriptor(CallDescriptor):
    """kind=map — `List.map`: apply a callable over each element of an `over: ${list}` source.

    A `CallDescriptor` with additional authored fields: `over` (the iteration source) is required
    alongside `call`; `parallel` is accepted as inert authoring metadata. It declares NO new
    fields (`over`/`parallel` already live on `CallDescriptor`); it only widens
    `allowed`/`required` and sets `kind="map"`, which is what makes `CallDescriptor.build`
    construct a `MapNode` (it branches on `isinstance(self, MapDescriptor)`). Inherits the
    template `parse` and `binding_fields=("inputs","over")` from `CallDescriptor`.
    """

    kind = "map"
    allowed = CallDescriptor.allow("over", "parallel")
    required = CallDescriptor.require("over")
