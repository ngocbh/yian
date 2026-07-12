"""The kind-blind binding-field walk — one place for "which fields are bindings".

A node descriptor carries `${...}` binding strings in kind-specific fields (an
agent/code/model's `inputs:`, a tool's `args:`, a mapped call's `inputs:`/`over:`).
Two passes need to rewrite those strings in place: inline-call desugar
(`compose.stages.inline_calls`) and case-output expansion (`compose.cases`). Both
used to reach for the same isinstance ladder (the old
`calls.map_binding_strings_in_descriptor`); this module replaces it with a
table-driven walk over `BINDING_FIELDS[kind]`, so adding or altering a kind's
binding surface is a local edit in `compose/kinds/<kind>.py` (its
`*_BINDING_FIELDS`) — not a new ladder arm here.

Layer: sits above `compose.kinds` (reads `KINDS`/`BINDING_FIELDS`) and is imported
by the `compose.stages.inline_calls` + `compose.cases` peers; it never imports either back.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Optional

from agent_composer.compose.kinds import BINDING_FIELDS, KINDS


def _descriptor_kind(desc: Any) -> Optional[str]:
    """Resolve a descriptor's surface `kind:` string, for the `BINDING_FIELDS` lookup.

    A `CallDescriptor` serves two kinds (`call`/`map`) so it carries its own `.kind`;
    every other descriptor class is 1:1 with a kind, resolved by reverse-lookup of
    `type(desc)` in `KINDS`. Returns `None` for an unregistered descriptor (walked as
    having no binding fields — a safe pass-through).

    A descriptor carrying only the `NodeDescriptor` base's empty-string `kind` ClassVar
    default (a kind not yet converted to declare its own `kind` class attr) counts as
    "no kind set" and falls through to the reverse-lookup — so this stays correct through
    the incremental kind-by-kind conversion, then collapses to `return desc.kind` once
    every kind sets its class attr (see the plan's Step 14)."""
    kind = getattr(desc, "kind", None)
    if kind:
        return kind
    for name, spec in KINDS.items():
        if spec.descriptor_cls is type(desc):
            return name
    return None


def map_binding_strings(
    desc: Any, mapper: Callable[[str], Any], *, binding_fields: dict = BINDING_FIELDS
) -> Any:
    """Apply `mapper` to each STRING binding in `desc`'s kind-declared binding fields.

    The fields are `binding_fields[kind]` (from each kind's `*_BINDING_FIELDS`): a dict
    field (e.g. `inputs`/`args`) has `mapper` applied to each string VALUE; a scalar field
    (e.g. a mapped call's `over`) has `mapper` applied directly when it is a string.
    Non-string values (literals / native YAML) are left as-is; a descriptor whose kind
    lists no binding fields (case/wait/loop/human_input) passes through unchanged. Returns
    a new frozen descriptor when anything changed, else `desc` itself. `mapper` may raise."""
    kind = _descriptor_kind(desc)
    fields = binding_fields.get(kind, ()) if kind is not None else ()
    if not fields:
        return desc

    updates: dict[str, Any] = {}
    for name in fields:
        current = getattr(desc, name)
        if isinstance(current, dict):
            updates[name] = {
                k: (mapper(v) if isinstance(v, str) else v) for k, v in current.items()
            }
        elif isinstance(current, str):
            updates[name] = mapper(current)
        # None / non-string scalar: nothing to rewrite.

    return replace(desc, **updates) if updates else desc
