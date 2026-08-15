"""Loud, locatable errors for the Compose loader."""

from typing import Optional

from agent_composer.events import SourceRange


class LoadError(ValueError):
    """A Compose flow cannot be loaded (bad shape, type, or structure).

    `.line` carries the source `.yaml` line when known (filled in by later slices
    that track positions); it is None for errors raised away from a parsed node.

    `.lines` carries *all* source lines relevant to one error, for errors that span
    more than one place — e.g. a cycle, which implicates every node in the loop. A
    renderer (the CLI) spans its source window across them and highlights each; `.line`
    stays the single primary anchor (the title, back-compat). None when the error is
    a single point (then only `.line` applies). When `.lines` is given and `.line` is
    not, `.line` defaults to the smallest of `.lines`.

    `.notes` carries extra explanatory lines printed *below* the message/source frame —
    a "why" legend for an error the source alone doesn't make obvious. A cycle uses it to
    spell out the dependency edges that close the loop (e.g. "a depends on b (a.input.x)").
    None when there's nothing to add.

    `.range` carries the precise 1-based `SourceRange` the error points at, when known,
    filled in by a `SourceCursor`. A renderer prefers `.range` when set and falls back
    to `.line`. None for errors raised away from a parsed node. When a
    `range` is given and no explicit `line`/`lines`, `.line` defaults to the range's start
    line, so legacy line-based consumers keep working as callers move to `range=`.
    """

    def __init__(
        self,
        message: str,
        line: Optional[int] = None,
        lines: Optional[list[int]] = None,
        notes: Optional[list[str]] = None,
        range: Optional[SourceRange] = None,
    ):
        super().__init__(message)
        self.lines = lines
        if line is not None:
            self.line = line
        elif lines:
            self.line = min(lines)
        elif range is not None:
            self.line = range.start_line
        else:
            self.line = None
        self.notes = notes
        self.range = range
