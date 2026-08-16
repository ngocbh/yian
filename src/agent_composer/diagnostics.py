"""Source-neutral values for locating and presenting diagnostics.

This dependency-free leaf separates semantic code from any authoring front end.
Engine values carry opaque ``OriginRef`` instances; a front end implements
``OriginResolver`` to turn those references into concrete presentation frames.
"""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Iterable, Mapping, Protocol


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


@dataclass(frozen=True, slots=True)
class _ResolutionTarget:
    """One exact delegation target retained by a resolver table.

    Attributes:
        resolver (`OriginResolver`): Authority for the target's local namespace.
        origin (`OriginRef`): Exact local identity passed to that authority.
    """

    resolver: OriginResolver
    origin: OriginRef


@dataclass(frozen=True, slots=True, eq=False)
class ResolverTable:
    """Immutable collision-safe composition of opaque origin namespaces.

    Each public origin routes to exactly one resolver and one origin in that
    resolver's local namespace. Keys stay opaque: the table performs no prefix
    parsing and never falls through to another resolver.

    Args:
        _targets (`Mapping[OriginRef, _ResolutionTarget]`, *optional*):
            Exact public-origin routes. Callers normally use `from_entries()`;
            the mapping is defensively copied and made read-only.
    """

    _targets: Mapping[OriginRef, _ResolutionTarget] = field(
        default_factory=dict,
        repr=False,
    )

    def __post_init__(self) -> None:
        """Validate and freeze a defensive copy of every exact route."""
        if not isinstance(self._targets, Mapping):
            raise TypeError("resolver table targets must be a mapping")
        copied: dict[OriginRef, _ResolutionTarget] = {}
        for origin, target in self._targets.items():
            if not isinstance(origin, OriginRef):
                raise TypeError("resolver table origin must be an OriginRef")
            if not isinstance(target, _ResolutionTarget):
                raise TypeError("resolver table target must be a resolution target")
            copied[origin] = target
        object.__setattr__(self, "_targets", MappingProxyType(copied))

    @classmethod
    def from_entries(
        cls,
        entries: Iterable[tuple[OriginRef, OriginResolver, OriginRef]],
    ) -> "ResolverTable":
        """Build a table from exact new-to-delegate/old routes.

        Args:
            entries (`Iterable[tuple[OriginRef, OriginResolver, OriginRef]]`):
                Triples of public origin, delegate resolver, and the exact origin
                understood by that delegate.

        Returns:
            `ResolverTable`: An immutable defensive snapshot of the routes.

        Raises:
            TypeError: Raised when an entry contains a malformed origin or delegate.
            ValueError: Raised when the same public origin occurs more than once.
        """
        targets: dict[OriginRef, _ResolutionTarget] = {}
        for new_origin, delegate, old_origin in entries:
            if not isinstance(new_origin, OriginRef):
                raise TypeError("resolver table new origin must be an OriginRef")
            if not callable(getattr(delegate, "resolve", None)):
                raise TypeError("resolver table delegate must implement resolve")
            if not isinstance(old_origin, OriginRef):
                raise TypeError("resolver table old origin must be an OriginRef")
            if new_origin in targets:
                raise ValueError(
                    f"duplicate resolver table origin {new_origin.key!r}"
                )
            targets[new_origin] = _ResolutionTarget(delegate, old_origin)
        return cls(targets)

    @classmethod
    def merge(cls, *tables: "ResolverTable") -> "ResolverTable":
        """Combine disjoint tables without adding resolver precedence.

        Args:
            *tables (`ResolverTable`): Tables whose public origins must be disjoint.

        Returns:
            `ResolverTable`: One immutable table containing every route.

        Raises:
            TypeError: Raised when an argument is not a `ResolverTable`.
            ValueError: Raised when two tables contain the same public origin.
        """
        entries: list[tuple[OriginRef, OriginResolver, OriginRef]] = []
        for table in tables:
            if not isinstance(table, ResolverTable):
                raise TypeError("resolver table merge requires ResolverTable values")
            entries.extend(
                (origin, target.resolver, target.origin)
                for origin, target in table._targets.items()
            )
        return cls.from_entries(entries)

    @property
    def origins(self) -> frozenset[OriginRef]:
        """Return the exact public origins routed by this table."""
        return frozenset(self._targets)

    def __len__(self) -> int:
        """Return the number of exact public-origin routes."""
        return len(self._targets)

    def __contains__(self, origin: object) -> bool:
        """Return whether one exact `OriginRef` has a route."""
        return origin in self._targets

    def resolve(self, origin: OriginRef) -> DiagnosticFrame | None:
        """Resolve one public origin through its exact delegate route.

        Args:
            origin (`OriginRef`): Public opaque identity to look up.

        Returns:
            `DiagnosticFrame | None`: The delegate result, or `None` when no route
            exists or the selected delegate cannot resolve its local origin.

        Raises:
            TypeError: Raised when `origin` is not an `OriginRef`.
        """
        if not isinstance(origin, OriginRef):
            raise TypeError("resolver table resolve requires an OriginRef")
        target = self._targets.get(origin)
        if target is None:
            return None
        return target.resolver.resolve(target.origin)

    def __deepcopy__(self, memo: dict[int, object]) -> "ResolverTable":
        """Return this immutable table without copying delegate resolvers.

        Args:
            memo (`dict[int, object]`): Standard deepcopy identity memo; unused.

        Returns:
            `ResolverTable`: This same immutable table.
        """
        return self


__all__ = [
    "DiagnosticFrame",
    "OriginRef",
    "OriginResolver",
    "ResolverTable",
    "SourceRange",
]
