"""kind=case — the branch-routing node's descriptor + parse/build handle (#70).

One of the two-axis `compose/kinds/<kind>.py` modules: it owns the `case` node's frozen
`CaseDescriptor`. It inherits the template `parse` from `NodeDescriptor` (which records each
field's `SourceRange`), overriding only the `_normalize_reader` hook — case is the sole kind
whose surface carries a key YAML 1.1 coerces (`on:`->`True`), so it undoes that coercion before
the template's key-legality check runs. Registered into `KINDS`/`BINDING_FIELDS` at import so the
shared dispatch routes here.

A `case` is DESUGARED (into a `CaseNode` + call targets) by the graph-stage pass in
`compose/cases.py`; it never becomes a leaf runtime node. So `build` here is unreachable
in the normal load path (the loader desugars case before `build_leaf_node`) and exists only
to satisfy the dispatch contract — it raises the same "case is desugared" guard the old
`build_leaf_node` fall-through raised.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from agent_composer.compose.helpers.errors import LoadError
from agent_composer.compose.helpers.reader import NodeReader
from agent_composer.compose.kinds.base import NodeDescriptor

# YAML 1.1 (PyYAML `safe_load`) coerces the bare key `on:` to the boolean `True` (likewise
# `off`/`yes`/`no`). The case-node surface key `on:` is the only such key in any node body, so
# this back-map + `_normalize_keys` (below) undo the coercion. Case-local: no other kind body
# carries a YAML-bool-coerced key.
_YAML_BOOL_KEYS = {True: "on", False: "off"}


def _normalize_keys(body: dict) -> dict:
    """Undo YAML 1.1 boolean-key coercion for the surface key `on:`."""
    if not any(k in _YAML_BOOL_KEYS for k in body):
        return body
    return {_YAML_BOOL_KEYS.get(k, k): v for k, v in body.items()}


@dataclass(frozen=True, kw_only=True)
class CaseDescriptor(NodeDescriptor):
    """kind=case — branch routing (searched `when:` or `on:`-value form).

    A `case` routes only — it carries NO `inputs:` (the desugar maps it onto the
    inputs-bearing `CaseNode`). `else_` is the `else:` fallback.

    Inherits the common node fields (`id`/`node_name`/`depends_on`/`runs_after`/`env` + the
    `range`/`field_ranges` parse metadata) and the template `parse` from `NodeDescriptor`.
    It overrides ONLY the `_normalize_reader` hook (to undo the `on:`->`True` YAML coercion
    before key legality runs); its field legality is otherwise the default allowed/required key
    check plus the default asserts-list check. Only the case-specific fields are declared here.

    Class attrs: `kind="case"`; `allowed` adds `cases`/`else`/`on` to the commons; `required`
    is `{"cases"}`; `binding_fields=()` — a `case` carries CONDITIONS (`on:` + each `when:`),
    not `${...}` sink bindings, so the kind-blind binding walk in `bindings.py` touches no field.
    `keyword_remap` is case-owned (the base's is empty): `else` is a Python keyword, so the
    template `parse` reads the surface `else:` into the `else_` descriptor field.

    `asserts` stays a declared field (the desugar path may set it) but is deliberately NOT in
    `allowed` — a `case` never authors `asserts`, so the surface key stays rejected.
    """

    kind = "case"
    allowed = NodeDescriptor.allow("cases", "else", "on")
    required = NodeDescriptor.require("cases")
    binding_fields = ()
    # Case owns this remap (the base's is empty): `else` is a Python keyword, so the template
    # `parse` reads the surface `else:` into the `else_` descriptor field.
    keyword_remap = {"else": "else_"}

    cases: list[dict[str, Any]] = field(default_factory=list)
    else_: Optional[str] = None
    on: Optional[str] = None
    asserts: list[str] = field(default_factory=list)

    @classmethod
    def _normalize_reader(cls, reader: NodeReader) -> None:
        """Undo YAML 1.1's `on:`->`True` key coercion before the template's key check.

        Overrides the base no-op hook: the template `parse` calls this first, before any key
        is inspected. Rewrites `reader.value` in place via `_normalize_keys`, so spanless
        presence checks (`_pair` reads `reader.value`) find `on`; span-aware range lookups
        still resolve because `NodeReader` matches the RAW key text (`"on"`)."""
        reader.value = _normalize_keys(reader.value)

    def build(self, *, registry=None, resolver=None, **_loader_hints):
        """A `case` is not a leaf node — it is desugared (see the module docstring).

        Unreachable in the normal load path (the loader desugars case before dispatching to a
        kind's `build`). Present only to satisfy the dispatch contract; raises the same guard
        the old `build_leaf_node` fall-through raised, so a stray call fails loud.
        """
        raise LoadError(
            f"node {self.id!r}: kind case is not a leaf node "
            f"(case nodes are desugared; ref/map nodes expand at runtime)"
        )
