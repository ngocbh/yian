"""The per-kind axis of the Compose front end: one handle per author-written `kind:`.

Issue #70 splits `compose/` along two axes: the *node-kind* axis lives here
(`compose/nodes/<kind>.py`), and the cross-node *graph-stage* axis lives in
`compose/stages/`. Each kind module owns the three things that used to be scattered
across `parser.py` (its descriptor + the `_KIND_SPECS` row) and `build.py` (its branch
of the `build_leaf_node` isinstance ladder):

- its frozen **descriptor** dataclass (subclassing `NodeDescriptor`),
- **`parse(id, reader)`** — a classmethod that reads one node body via a located
  `NodeReader`, recording each field's `SourceRange` into the descriptor's `field_ranges`, and
- **`build(*, registry, resolver)`** — a method that constructs the runtime `Node`
  (+ wiring), stamping `.range`/`.field_ranges` onto it.

`parse`/`build` ride on the descriptor class itself, so `KINDS` maps the surface `kind:`
string straight to the descriptor class — no bundling wrapper. The shared dispatch sites
(`stages/parse.py::_parse_node`, `build.build_leaf_node`, `loader.py`) look the class up and
call `cls.parse(...)` / `desc.build(...)`, so adding/altering a kind is a local edit here —
the engine core never sees any of it (kind-blind invariant, `tests/engine/test_kind_census.py`).

`BINDING_FIELDS` is the sibling trait table: per kind, the descriptor fields that carry
`${...}` binding strings. `compose/nodes/bindings.py` walks it kind-blind (replacing the old
`map_binding_strings_in_descriptor` isinstance ladder).

Both tables are derived **below** from a single `_DESCRIPTORS` tuple — one entry per
author-written kind, keyed by each class's own `kind`/`binding_fields` ClassVars. Adding a
kind means adding its import and one entry to that tuple.

Importing the kind modules here cannot cycle: each takes its `build` helpers from `nodes.*`
(the engine node package, below `compose` in the layer ladder) and from `NodeDescriptor` in
`compose/nodes/base.py` (a leaf of this package). The one import each still defers is in
`parse`: the shared field constants from `parser.py`, which re-exports these descriptors, so a
top-level import there WOULD cycle. Deriving the tables below only reads each class's
already-defined ClassVars; it never touches `KINDS`.
"""

from __future__ import annotations

from agent_composer.compose.nodes import (
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
from agent_composer.compose.nodes import map as map_kind  # `map` shadows the builtin

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

# Surface `kind:` string -> its descriptor class; the shared dispatch (`stages/parse.py`,
# `build.build_leaf_node`, `loader.py`) looks up the class and calls its `parse`/`build`.
KINDS: dict[str, type] = {d.kind: d for d in _DESCRIPTORS}

# Surface `kind:` string -> the descriptor field names that carry `${...}` binding strings.
# Consumed by the kind-blind walk in `compose/nodes/bindings.py`.
BINDING_FIELDS: dict[str, tuple[str, ...]] = {d.kind: d.binding_fields for d in _DESCRIPTORS}
