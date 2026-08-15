"""Source-neutral values for locating and presenting diagnostics.

This dependency-free leaf separates semantic code from any authoring front end.
Engine values carry opaque ``OriginRef`` instances; a front end implements
``OriginResolver`` to turn those references into concrete presentation frames.
"""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class SourceRange:
    """A concrete 1-based source span: [start, end) with an exclusive end column.

    Coordinates are 1-based (line 1, col 1 = first char); `end_col` is one past the
    last covered column (half-open) so a zero-width caret is start == end. Produced by
    the compose loader from PyYAML marks and by the funnel that resolves a symbolic
    `SourceSpan` key to the node's recorded range.

    Because the end is EXCLUSIVE, `end_line` is not necessarily a line this span covers —
    use `covered_lines` to render it rather than reading `end_line` directly.
    """

    start_line: int
    start_col: int
    end_line: int
    end_col: int

    def covered_lines(self, source_text: str) -> list[int]:
        """The 1-based lines this span actually covers, in order — what a renderer highlights.

        The end is EXCLUSIVE, so `end_line` is covered only if the span reaches real text on it.
        A YAML block construct (a block mapping, or a `|-` scalar) ends at the START OF THE NEXT
        TOKEN, which sits on a later line whose text before `end_col` is nothing but indentation —
        that line holds none of this span and is dropped. Naively taking `end_line` is what makes
        an error on one node's `code:` highlight the *next* node's header.

        Args:
            source_text (`str`): the full flow YAML this range was resolved against — needed to
                tell "the end stops just past real text" from "the end is only indentation".

        Returns:
            `list[int]`: the covered lines, always non-empty and starting at `start_line` (a
            zero-width caret, `start == end`, still marks its own line).
        """
        lines = source_text.splitlines()
        last = self.end_line
        # Text on `end_line` before the exclusive end. Out-of-bounds (a range past EOF) reads as
        # blank and drops the line — defensive, not expected.
        prefix = lines[last - 1][: self.end_col - 1] if 0 < last <= len(lines) else ""
        if not prefix.strip():
            last -= 1
        return list(range(self.start_line, max(last, self.start_line) + 1))


@dataclass(frozen=True)
class OriginRef:
    """An opaque source identity interpreted only by its producing front end.

    Attributes:
        key (`str`): A non-empty key unique within one returned authored flow.
    """

    key: str

    def __post_init__(self) -> None:
        """Reject keys that cannot serve as stable opaque identities."""
        if type(self.key) is not str or not self.key:
            raise ValueError("origin key must be a non-empty string")


@dataclass(frozen=True)
class DiagnosticFrame:
    """A resolved source frame ready for a format-neutral diagnostic renderer.

    Attributes:
        label (`str`): Human-facing source name, such as a path or notebook cell.
        text (`str`): Complete source text that ``range`` indexes.
        range (`SourceRange`): Concrete primary location within ``text``.
        note (`str | None`): Optional context attached to this frame.
    """

    label: str
    text: str
    range: SourceRange
    note: str | None = None


class OriginResolver(Protocol):
    """Authoring-format-neutral seam from opaque origins to source frames."""

    def resolve(self, origin: OriginRef) -> DiagnosticFrame | None:
        """Resolve ``origin``, or return ``None`` when no frame is available."""
        ...


__all__ = ["DiagnosticFrame", "OriginRef", "OriginResolver", "SourceRange"]
