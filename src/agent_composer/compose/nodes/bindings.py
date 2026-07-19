"""The kind-blind binding-field walk — one place for "which fields are bindings".

A node descriptor carries `${...}` binding strings in kind-specific fields (an
agent/code/model's `inputs:`, a tool's `inputs:`, a mapped call's `inputs:`/`over:`).
Two passes need to rewrite those strings in place: inline-call desugar
(`compose.desugar.inline_calls`) and case-output expansion
(`compose.desugar.case_outputs`). Both
used to reach for the same isinstance ladder (the old
`calls.map_binding_strings_in_descriptor`); this module replaces it with a
table-driven walk over `BINDING_FIELDS[kind]`, so adding or altering a kind's
binding surface is a local edit in `compose/nodes/<kind>.py` (its
`binding_fields`) — not a new ladder arm here.

Layer: sits inside `compose.nodes` (reads its sibling `BINDING_FIELDS` table) and is
imported by the `compose.desugar.inline_calls` + `compose.desugar.case_outputs` peers;
it never imports either back.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable

from agent_composer.compose.nodes import BINDING_FIELDS


def _descriptor_kind(desc: Any) -> str:
    """Resolve a descriptor's surface `kind:` string, for the `BINDING_FIELDS` lookup.

    Every descriptor class declares its own `kind` ClassVar (`CallDescriptor` serves
    `call` and its `MapDescriptor` subclass serves `map`), so this is just `desc.kind`."""
    return desc.kind


def map_binding_strings(
    desc: Any, mapper: Callable[[str, str], Any], *, binding_fields: dict = BINDING_FIELDS
) -> Any:
    """Apply `mapper(key, value)` to each STRING binding in `desc`'s kind-declared binding fields.

    The fields are `binding_fields[kind]` (from each kind's `binding_fields`): a dict field
    (e.g. `inputs`) applies `mapper(binding_name, value)` to each string VALUE; a scalar field
    (e.g. a mapped call's `over`) applies `mapper(field_name, value)` when it is a string. The
    `key` lets a caller locate what it rewrote (the inline-call desugar reads it to anchor a
    synth node at that binding's own source line); a caller that only needs the value wraps its
    function with `keyless`. Non-string values (literals / native YAML) are left as-is; a
    descriptor whose kind lists no binding fields (case/wait/loop/human_input) passes through
    unchanged. Returns a new frozen descriptor when anything changed, else `desc` itself.
    `mapper` may raise."""
    kind = _descriptor_kind(desc)
    fields = binding_fields.get(kind, ())
    if not fields:
        return desc

    updates: dict[str, Any] = {}
    for name in fields:
        current = getattr(desc, name)
        if isinstance(current, dict):
            updates[name] = {
                k: (mapper(k, v) if isinstance(v, str) else v) for k, v in current.items()
            }
        elif isinstance(current, str):
            updates[name] = mapper(name, current)
        # None / non-string scalar: nothing to rewrite.

    return replace(desc, **updates) if updates else desc


def keyless(fn: Callable[[str], Any]) -> Callable[[str, str], Any]:
    """Adapt a value-only binding mapper to `map_binding_strings`' `(key, value)` contract,
    dropping the key — for a caller that rewrites a binding purely from its value."""
    return lambda _key, value: fn(value)
