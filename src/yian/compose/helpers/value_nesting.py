"""Reject a written-out value that nests deeper than a yian value may, as a load error.

A value written in a flow file is checked at load with the counting the run-time check
applies to it (`typesys.values.reject_deep_nesting`), so it fails before any recursive
load walker reads it, even when YAML aliases build its depth while the text stays
shallow. `${...}` text counts as a scalar; the run-time check still covers computed
values.
"""

from __future__ import annotations

from typing import Any, Mapping

from yian.compose.helpers.errors import LoadDiagnostics
from yian.compose.source import SourceCursor
from yian.diagnostics import SourceRange
from yian.typesys.values import TypeCheckError, reject_deep_nesting


def check_written_value(
    value: Any,
    where: str,
    range: SourceRange | None,
    diagnostics: LoadDiagnostics,
    *,
    enclosing_levels: int = 0,
) -> None:
    """Reject `value` if it nests deeper than `current().max_value_nesting` levels.

    Args:
        value (`Any`): The authored value.
        where (`str`): Names the site for the message, e.g. `"node 'use' input 'topic'"`.
        range (`SourceRange | None`): Where the error points.
        diagnostics (`LoadDiagnostics`): Factory for located load errors.
        enclosing_levels (`int`, *optional*, defaults to `0`): Levels the value sits
            in at run time, as for `reject_deep_nesting`.

    Raises:
        `LoadError`: Raised at `range` with the run-time message when a container sits
            past the limit.
    """
    try:
        reject_deep_nesting(value, where, enclosing_levels=enclosing_levels)
    except TypeCheckError as exc:
        raise diagnostics.error(str(exc), range=range) from exc


def check_env_values(
    env: Mapping[str, Any],
    prefix: str,
    env_cursor: SourceCursor | None,
    diagnostics: LoadDiagnostics,
) -> None:
    """Reject an `env:` value that nests deeper than `current().max_value_nesting` levels.

    Args:
        env (`Mapping[str, Any]`): The authored `env:` mapping.
        prefix (`str`): Starts the message before `env '<key>'`, e.g. `"flow "`.
        env_cursor (`SourceCursor | None`): Cursor over the `env:` mapping, for ranges.
        diagnostics (`LoadDiagnostics`): Factory for located load errors.

    Raises:
        `LoadError`: Raised at the entry's value (else at `env:`) when a value sits past
            the limit.
    """
    for key, value in env.items():
        range = None
        if env_cursor is not None:
            range = env_cursor.range_of(key) or env_cursor.range
        check_written_value(value, f"{prefix}env {key!r}", range, diagnostics)
