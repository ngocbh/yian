"""Parse-free boundary range helpers for the Compose front end.

This module projects synthesized boundary-input ranges from an existing source
cursor. It never parses or calls `parse_nodes`/`parse_file`, keeping the helper
layer below `parse`. Assertion origins and temporary ranges are deliberately
absent: the build conductor derives both from one retained item cursor.
"""

from __future__ import annotations

from collections.abc import Mapping

from agent_composer.compose.source import SourceAccessError, SourceCursor
from agent_composer.diagnostics import SourceRange


def _boundary_field_ranges(
    root: SourceCursor,
) -> dict[tuple[str, str], SourceRange]:
    """Return Compose-owned ranges for authored flow input declarations.

    Keys: each top-level `input:`/`inputs:` declaration name, tagged
    `("input", name)` -> its declaration value `SourceRange`. Assembly uses the
    map while compiling flow parameters and discards it before semantic graph
    construction. Flow-level `asserts:` are not here: assembly derives their
    ranges and origins from the retained assertion cursor. Empty for a
    spanless/in-memory flow. This projection never raises.
    """
    try:
        value = root._value_view()
        if not isinstance(value, Mapping):
            return {}
        out: dict[tuple[str, str], SourceRange] = {}
        for key in ("input", "inputs"):
            if isinstance(value.get(key), Mapping):
                out.update(
                    {
                        ("input", name): rng
                        for name, rng in root.mapping_value_ranges(key).items()
                    }
                )
        return out
    except SourceAccessError:
        return {}
