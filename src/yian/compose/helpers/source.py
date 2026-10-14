"""Parse-free source helpers for the Compose front end.

This module projects synthesized boundary-input ranges from an existing source
cursor and slices authored text out of a document. It never parses or calls
`parse_nodes`/`parse_file`, keeping the helper layer below `parse`. Assertion
origins and temporary ranges are deliberately absent: the build conductor derives
both from one retained item cursor.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from yian.compose.source import SourceAccessError, SourceCursor, SourceDocument
from yian.diagnostics import SourceRange


def is_extension_key(key: Any) -> bool:
    """Return whether a top-level key is an `x-*` extension key, which a load ignores.

    Args:
        key (`Any`): A top-level key as YAML constructed it.

    Returns:
        `bool`: True when the key's text starts with `x-`.
    """
    return str(key).startswith("x-")


def one_line_text(
    document: SourceDocument | None, rng: SourceRange | None
) -> str | None:
    """Return the authored text a one-line range covers.

    Args:
        document (`SourceDocument | None`): The document `rng` points into.
        rng (`SourceRange | None`): The range to slice.

    Returns:
        `str | None`: The text as written, or `None` when there is no document or
            range, or the range spans more than one line.
    """
    if document is None or rng is None or rng.start_line != rng.end_line:
        return None
    line = document.text.splitlines()[rng.start_line - 1]
    return line[rng.start_col - 1 : rng.end_col - 1]


def _boundary_field_ranges(
    root: SourceCursor,
) -> dict[tuple[str, str], SourceRange]:
    """Return Compose-owned ranges for authored flow input declarations.

    Keys: each top-level `input:` declaration name, tagged
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
