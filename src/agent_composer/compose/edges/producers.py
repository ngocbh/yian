"""Shared producer helpers: the node ids a `${...}` reference reads from.

The three functions here map a binding/condition value to the producer node ids it
depends on — the primitive the edges phase (`edges/data.py`) and the later boundary
synthesis (`build/boundary.py`, via `synthesize_boundary_graph`) both build on. A ref
head of `<id>.output[.…]` produces from node `<id>`; an `input.X` ref produces from
the synthesized `START_ID`; `system.X`/`item` produce nothing.
"""

from __future__ import annotations

from typing import Any, Optional

from agent_composer.compile.model import START_ID
from agent_composer.expr import ExpressionError, condition_refs, expr_refs_of


def _ref_producer(path: str) -> Optional[str]:
    """The producer node id of a `${...}` reference head (singular):
    - `<id>.output[.…]`: produce from node `<id>`.
    - `input.<key>`: produce from the synthesized START_ID, so an `${input.X}`
      reader gets a `START_ID->reader` data edge.
    - `system.<key>` is run-global and produces NO edge; `item` is MAP-body-local.
    Legacy plural heads (`outputs`/`inputs`) are rejected at parse time."""
    parts = path.split(".")
    if len(parts) >= 2 and parts[0] == "input":
        return START_ID
    if len(parts) >= 2 and parts[1] == "output":
        return parts[0]
    return None


def _binding_producers(value: Any) -> list[str]:
    """The producer node ids a binding/expression value reads (in source order).

    `value` may be a `${...}` binding string, a `when:`/`on:` expression template,
    or a non-string literal (no refs). Coalesce alternatives flatten to several
    producers — the caller assigns them ALL the one sink `input_group`. An `inputs.X` ref
    produces from START_ID (the input-reader data edge); `system.X` produces nothing.
    """
    if not isinstance(value, str):
        return []
    try:
        refs = expr_refs_of(value)
    except ExpressionError:
        return []  # malformed refs surface in the ref-wiring pass, located
    producers: list[str] = []
    for ref in refs:
        producer = _ref_producer(ref)
        if producer is not None:
            producers.append(producer)
    return producers


def _condition_producers(value: Any) -> list[str]:
    """The producer node ids a CONDITION expression reads (in source order).

    A `case` `when:`/`on:` is a whole boolean/value expression, not a template — it is
    written in any of the three condition spellings (bare `a.output > 5`, mixed
    `${a.output} > 5`, whole-span `${a.output > 5}`), so refs are extracted via
    `condition_refs` (which parses the whole expression) rather than `expr_refs_of`
    (which only sees `${...}` spans and would miss the bare spelling). `system.X`
    produces nothing; an `input.X` ref produces from START_ID.
    """
    if not isinstance(value, str):
        return []
    try:
        refs = condition_refs(value)
    except ExpressionError:
        return []  # malformed refs surface in the ref-wiring pass, located
    producers: list[str] = []
    for ref in refs:
        producer = _ref_producer(ref)
        if producer is not None:
            producers.append(producer)
    return producers
