"""The per-kind axis of the Compose front end: one handle per author-written `kind:`.

The Compose front end is split along two axes: the *node-kind* axis lives here
(`compose/nodes/<kind>.py`), and the cross-node *graph-stage* axis lives in the
`compose/parse/`, `compose/desugar/`, `compose/edges/`, and `compose/validate/` phases.
Each kind module owns four things:

- its frozen **descriptor** dataclass (subclassing `NodeDescriptor`),
- **`parse(id, cursor)`** — a classmethod that reads one node body via a located
  `SourceCursor`, recording each field's `SourceRange` into the descriptor's
  `field_ranges`, and
- **`with_source_origins(cursor, *, site_path, catalog_builder)`** — an immutable
  attachment hook for kind-owned semantic source sites; the base implementation is a
  no-op, and
- **`build(ctx: BuildContext) -> BuildResult`** — a method that constructs the runtime `Node`
  (+ wiring, + any edges a `case` owns), stamping the descriptor's optional `.origin`
  onto it and retaining concrete ranges only on the Compose-owned build result. The
  assembly conductor, not a
  per-kind parser, assigns authored node-header origins and invokes the source hook
  before build. CODE alone also consumes the context's opaque Python filename and
  optional registered-origin presentation callable for load-time syntax errors; YAML
  objects never cross into the runtime node.

`parse`/`build` ride on the descriptor class itself, so `KINDS` maps the surface `kind:`
string straight to the descriptor class — no bundling wrapper. The shared dispatch sites
(`parse/nodes.py::_parse_node`, the loader's conductor loop) look the class up and
call `cls.parse(...)` / `desc.with_source_origins(...)` / `desc.build(ctx)`, so
adding/altering a kind is a local edit here —
the engine core never sees any of it (kind-blind invariant, `tests/engine/test_kind_census.py`).

`BINDING_FIELDS` is the sibling trait table: per kind, the descriptor fields that carry
`${...}` binding strings. `compose/nodes/bindings.py` walks it kind-blind.

Both tables are derived **below** from a single `_DESCRIPTORS` tuple — one entry per
author-written kind, keyed by each class's own `kind`/`binding_fields` ClassVars. Adding a
kind means adding its import and one entry to that tuple.

The AGENT, HUMAN_INPUT, and WAIT descriptors build ordinary live leaves: AGENT
owns its prompt/tool/model surface, HUMAN_INPUT owns its question and answer
contract, and WAIT owns only its absolute ``until`` wiring.

Importing the kind modules here cannot cycle: each takes its `build` helpers from `nodes.*`
(the engine node package, below `compose` in the layer ladder) and from `NodeDescriptor` in
`compose/nodes/base.py` (a leaf of this package). The one import each still defers is in
`parse`: the shared field constants from `parse/file.py`, which re-exports these descriptors, so a
top-level import there WOULD cycle. Deriving the tables below only reads each class's
already-defined ClassVars; it never touches `KINDS`.

Knows about: diagnostics, compose source/helpers/peer phases, compile.model, nodes,
             typesys, expr, events, and llm_clients.
Never imports: runtime, the package-root run boundary, CLI, or provider SDKs.
"""

from __future__ import annotations

from yian.compose.nodes import (
    agent,
    call,
    case,
    code,
    human_input,
    loop,
    model,
    tool,
    wait,
)
from yian.compose.nodes import map as map_kind  # `map` shadows the builtin

# One entry per author-written kind. `map`'s `MapDescriptor` subclasses `call`'s
# `CallDescriptor` (a map IS a call with an `over:` iteration source), inheriting its build.
# `start`/`end` are loader-synthesized boundaries, never author-written, so absent here.
_DESCRIPTORS = (
    agent.AgentDescriptor,
    code.CodeDescriptor,
    model.ModelDescriptor,
    tool.ToolDescriptor,
    wait.WaitDescriptor,
    human_input.HumanInputDescriptor,
    case.CaseDescriptor,
    call.CallDescriptor,
    map_kind.MapDescriptor,
    loop.LoopDescriptor,
)

# Surface `kind:` string -> its descriptor class; the shared dispatch (`parse/nodes.py`,
# the loader's conductor loop) looks up the class and calls its `parse`/`build`.
KINDS: dict[str, type] = {d.kind: d for d in _DESCRIPTORS}

# Surface `kind:` string -> the descriptor field names that carry `${...}` binding strings.
# Consumed by the kind-blind walk in `compose/nodes/bindings.py`.
BINDING_FIELDS: dict[str, tuple[str, ...]] = {d.kind: d.binding_fields for d in _DESCRIPTORS}
