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
its spec. The shared dispatch sites (`parser._parse_node`, `build.build_leaf_node`)
consult `KINDS` first, so adding/altering a kind is a local edit here — the engine core
never sees any of it (kind-blind invariant, `tests/engine/test_kind_census.py`).

`BINDING_FIELDS` is the sibling trait table: per kind, the descriptor fields that carry
`${...}` binding strings. `compose/bindings.py` walks it kind-blind (replacing the old
`map_binding_strings_in_descriptor` isinstance ladder).

Both tables are empty at first; kinds register themselves as they are migrated
(#70 rollout, one kind at a time).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


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


# Surface `kind:` string -> its `KindSpec`. Populated per-kind during the #70 rollout;
# the shared dispatch consults this before the legacy `_KIND_SPECS` / isinstance ladder.
KINDS: dict[str, KindSpec] = {}

# Surface `kind:` string -> the descriptor field names that carry `${...}` binding strings.
# Consumed by the kind-blind walk in `compose/bindings.py`.
BINDING_FIELDS: dict[str, tuple[str, ...]] = {}


# Import each migrated kind module for its import-time `register()` side effect (populates
# KINDS/BINDING_FIELDS). Kept at the BOTTOM so the tables above already exist when a kind's
# `register()` reads them. Kind modules import parser/build helpers lazily, so no cycle.
from agent_composer.compose.kinds import agent as _agent  # noqa: E402,F401
from agent_composer.compose.kinds import code as _code  # noqa: E402,F401
from agent_composer.compose.kinds import model as _model  # noqa: E402,F401
from agent_composer.compose.kinds import tool as _tool  # noqa: E402,F401
from agent_composer.compose.kinds import wait as _wait  # noqa: E402,F401
from agent_composer.compose.kinds import human_input as _human_input  # noqa: E402,F401
