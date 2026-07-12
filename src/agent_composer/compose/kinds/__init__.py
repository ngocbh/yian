"""The per-kind axis of the Compose front end: one handle per author-written `kind:`.

Issue #70 splits `compose/` along two axes: the *node-kind* axis lives here
(`compose/kinds/<kind>.py`), and the cross-node *graph-stage* axis lives in
`compose/stages/`. Each kind module owns the three things that used to be scattered
across `parser.py` (its descriptor + the `_KIND_SPECS` row) and `build.py` (its branch
of the `build_leaf_node` isinstance ladder):

- its frozen **descriptor** dataclass,
- **`parse(reader)`** — read one node body via a located `NodeReader`, recording each
  field's `SourceRange` into the descriptor's `field_ranges`, and
- **`build(desc, *, registry, resolver)`** — construct the runtime `Node` (+ wiring),
  stamping `.range`/`.field_ranges` onto it.

A `KindSpec` bundles those three; the `KINDS` table maps the surface `kind:` string to
its spec. The shared dispatch sites (`stages/parse.py::_parse_node`, `build.build_leaf_node`)
consult `KINDS` first, so adding/altering a kind is a local edit here — the engine core
never sees any of it (kind-blind invariant, `tests/engine/test_kind_census.py`).

`BINDING_FIELDS` is the sibling trait table: per kind, the descriptor fields that carry
`${...}` binding strings. `compose/bindings.py` walks it kind-blind (replacing the old
`map_binding_strings_in_descriptor` isinstance ladder).

Both tables are assembled **explicitly below** from each kind module's exports — one row
per author-written kind. Adding a kind means adding its import and one row to each table.
The `#70` rollout is complete — there is no `_KIND_SPECS` fallback.

Kind modules take their `build` helpers at module top level — from `kinds/helpers.py` (a leaf
of this package: it imports no `kinds/__init__` symbol) and from `nodes.*` (below `compose` in
the layer ladder) — so importing them here cannot cycle. The one import each still defers is in
`parse`: the shared field constants from `parser.py`, which re-exports these descriptors, so a
top-level import there WOULD cycle. Assembling the tables below therefore only reads each
module's already-defined descriptor + functions; it never touches `KINDS`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from agent_composer.compose.kinds import (
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
from agent_composer.compose.kinds import map as map_kind  # `map` shadows the builtin


@dataclass(frozen=True)
class KindSpec:
    """One author-written node kind's parse+build handle.

    - `descriptor_cls` — the frozen descriptor dataclass this kind parses into.
    - `parse` — `(reader: NodeReader) -> descriptor`; reads a located node body and
      records each field's `SourceRange` into the descriptor's `field_ranges`.
    - `build` — `(desc, *, registry, resolver) -> (node, wiring)`; constructs the runtime
      `Node` and its flow-owned wiring, stamping `.range`/`.field_ranges` onto the node.
      Leaf kinds ignore `resolver`; call/map/loop ignore `registry`.
    """

    descriptor_cls: type
    parse: Callable[..., object]
    build: Callable[..., tuple]


# Surface `kind:` string -> its `KindSpec`; the shared dispatch (`stages/parse.py`,
# `build.build_leaf_node`) consults this. `map` reuses `call`'s `CallDescriptor` + build
# (a map IS a call with an `over:` iteration source), with its own `over`-aware `parse`.
KINDS: dict[str, KindSpec] = {
    "agent": KindSpec(agent.AgentDescriptor, agent.AgentDescriptor.parse, agent.AgentDescriptor.build),
    "code": KindSpec(code.CodeDescriptor, code.CodeDescriptor.parse, code.CodeDescriptor.build),
    "model": KindSpec(model.ModelDescriptor, model.ModelDescriptor.parse, model.ModelDescriptor.build),
    "tool": KindSpec(tool.ToolDescriptor, tool.ToolDescriptor.parse, tool.ToolDescriptor.build),
    "wait": KindSpec(wait.WaitDescriptor, wait.WaitDescriptor.parse, wait.WaitDescriptor.build),
    "human_input": KindSpec(human_input.HumanInputDescriptor, human_input.parse, human_input.build),
    "case": KindSpec(case.CaseDescriptor, case.parse, case.build),
    "call": KindSpec(call.CallDescriptor, call.parse, call.build),
    "map": KindSpec(call.CallDescriptor, map_kind.parse, call.build),
    "loop": KindSpec(loop.LoopDescriptor, loop.parse, loop.build),
}

# Surface `kind:` string -> the descriptor field names that carry `${...}` binding strings.
# Consumed by the kind-blind walk in `compose/bindings.py`.
BINDING_FIELDS: dict[str, tuple[str, ...]] = {
    "agent": agent.AgentDescriptor.binding_fields,
    "code": code.CodeDescriptor.binding_fields,
    "model": model.ModelDescriptor.binding_fields,
    "tool": tool.ToolDescriptor.binding_fields,
    "wait": wait.WaitDescriptor.binding_fields,
    "human_input": human_input.HUMAN_INPUT_BINDING_FIELDS,
    "case": case.CASE_BINDING_FIELDS,
    "call": call.CALL_BINDING_FIELDS,
    "map": map_kind.MAP_BINDING_FIELDS,
    "loop": loop.LOOP_BINDING_FIELDS,
}
