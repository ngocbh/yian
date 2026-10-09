"""Source-neutral values for semantic failures and diagnostic presentation.

This dependency-free leaf separates semantic code from any authoring front end.
``Failure`` carries stable meaning plus opaque ``OriginRef`` instances; a front
end implements ``OriginResolver`` to turn those references into concrete
presentation frames. ``ResolverTable`` composes those namespaces through exact
immutable routes.
"""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Iterable, Literal, Mapping, Protocol, TypeAlias


FailureCode: TypeAlias = Literal[
    "binding.evaluate",
    "contract.input",
    "assertion.failed",
    "node.execute",
    "contract.node_result",
    "contract.output",
    "contract.route",
    "contract.growth",
    "graph.terminal_unreachable",
]
"""Closed semantic categories for expected flow-owned failures."""


FAILURE_CODES: Final[frozenset[str]] = frozenset(
    {
        "binding.evaluate",
        "contract.input",
        "assertion.failed",
        "node.execute",
        "contract.node_result",
        "contract.output",
        "contract.route",
        "contract.growth",
        "graph.terminal_unreachable",
    }
)
"""Runtime census of every accepted :class:`Failure` code."""


@dataclass(frozen=True)
class SourceRange:
    """A concrete 1-based source span: [start, end) with an exclusive end column.

    Coordinates are 1-based (line 1, col 1 = first char); `end_col` is one past the
    last covered column (half-open) so a zero-width caret is start == end. Produced and
    resolved by an authoring front end such as the Compose loader.

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
        line_count = len(lines)
        if not 1 <= self.start_line <= line_count:
            return [self.start_line]

        if self.end_line > line_count:
            # A stale range may extend beyond the retained source. The available
            # tail is still real context, but nonexistent line numbers are not.
            last = line_count
        else:
            last = self.end_line
            prefix = (
                lines[last - 1][: self.end_col - 1]
                if 0 < last <= line_count
                else ""
            )
            if not prefix.strip():
                last -= 1
        last = min(last, line_count)
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


@dataclass(frozen=True, slots=True)
class Failure:
    """One immutable source-neutral account of an expected flow failure.

    Attributes:
        code (`FailureCode`): Stable semantic category from ``FAILURE_CODES``.
        message (`str`): Human-readable failure detail; it may be empty.
        node_id (`str | None`): Runtime execution identity, or ``None`` when the
            failure is not attributable to one executing node.
        trace (`tuple[OriginRef, ...]`): Authored identities ordered outermost to
            innermost; the final item is the most precise available operation.
        cause_type (`str | None`): Python exception class name retained only for
            debugging, never for semantic classification.
        traceback (`str | None`): Optional formatted Python traceback retained as
            opaque debugging text.
    """

    code: FailureCode
    message: str
    node_id: str | None
    trace: tuple[OriginRef, ...] = ()
    cause_type: str | None = None
    traceback: str | None = None

    def __post_init__(self) -> None:
        """Reject values that cannot enter the canonical failure pipeline."""
        if type(self.code) is not str:
            raise TypeError("failure code must be a string")
        if self.code not in FAILURE_CODES:
            raise ValueError(f"unknown failure code {self.code!r}")
        if type(self.message) is not str:
            raise TypeError("failure message must be a string")
        if self.node_id is not None and type(self.node_id) is not str:
            raise TypeError("failure node_id must be a string or None")
        if type(self.trace) is not tuple:
            raise TypeError("failure trace must be a tuple")
        for origin in self.trace:
            if not isinstance(origin, OriginRef):
                raise TypeError("failure trace item must be an OriginRef")
            # A subclass can bypass OriginRef.__post_init__; replay the canonical
            # constructor without erasing the exact authored identity object.
            OriginRef(origin.key)
        if self.cause_type is not None and type(self.cause_type) is not str:
            raise TypeError("failure cause_type must be a string or None")
        if self.traceback is not None and type(self.traceback) is not str:
            raise TypeError("failure traceback must be a string or None")


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


@dataclass(frozen=True, slots=True)
class Diagnostic:
    """One immutable presentation-ready diagnostic.

    Attributes:
        code (`str`): Stable semantic or load-time category.
        message (`str`): Human-readable failure detail.
        frames (`tuple[DiagnosticFrame, ...]`): Resolved source context ordered
            outermost to innermost.
        notes (`tuple[str, ...]`): Presentation context not tied to one frame.
        node_id (`str | None`): Optional runtime execution identity.
        cause_type (`str | None`): Optional exception class name for debugging.
        traceback (`str | None`): Optional opaque Python traceback text.
    """

    code: str
    message: str
    frames: tuple[DiagnosticFrame, ...] = ()
    notes: tuple[str, ...] = ()
    node_id: str | None = None
    cause_type: str | None = None
    traceback: str | None = None


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


class ResolverTable:
    """Immutable collision-safe composition of opaque origin namespaces.

    Each public `OriginRef.key` routes to exactly one resolver and one origin in
    that resolver's local namespace. Keys stay opaque: the table performs no
    prefix parsing and never falls through to another resolver. Delegate
    exceptions propagate; delegate results must honor the `OriginResolver`
    return contract.

    Construct an empty table directly, or use `from_entries()` for exact routes.
    Raw target mappings are deliberately not part of the public constructor.
    """

    __slots__ = ("_targets",)

    _targets: Mapping[str, _ResolutionTarget]

    def __init__(self) -> None:
        """Create an empty immutable resolver table."""
        object.__setattr__(self, "_targets", MappingProxyType({}))

    def __setattr__(self, name: str, value: object) -> None:
        """Reject mutation after construction.

        Args:
            name (`str`): Attribute name a caller attempted to assign.
            value (`object`): Replacement value a caller attempted to assign.

        Raises:
            AttributeError: Always raised because the route table is immutable.
        """
        raise AttributeError("ResolverTable is immutable")

    def __delattr__(self, name: str) -> None:
        """Reject route deletion after construction.

        Args:
            name (`str`): Attribute name a caller attempted to delete.

        Raises:
            AttributeError: Always raised because the route table is immutable.
        """
        raise AttributeError("ResolverTable is immutable")

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
            ValueError:
                Raised when an origin subclass carries an invalid key or the same
                public key occurs more than once.
        """
        targets: dict[str, _ResolutionTarget] = {}
        for entry in entries:
            if not isinstance(entry, tuple) or len(entry) != 3:
                raise TypeError("resolver table entry must be a three-item tuple")
            new_origin, delegate, old_origin = entry
            if not isinstance(new_origin, OriginRef):
                raise TypeError("resolver table new origin must be an OriginRef")
            new_key = OriginRef(new_origin.key).key
            if not callable(getattr(delegate, "resolve", None)):
                raise TypeError("resolver table delegate must implement resolve")
            if not isinstance(old_origin, OriginRef):
                raise TypeError("resolver table old origin must be an OriginRef")
            # Subclasses can override OriginRef.__post_init__; replay the canonical
            # value constructor without discarding the exact delegate-local object.
            OriginRef(old_origin.key)
            if new_key in targets:
                raise ValueError(
                    f"duplicate resolver table origin {new_key!r}"
                )
            targets[new_key] = _ResolutionTarget(delegate, old_origin)
        table = cls()
        object.__setattr__(table, "_targets", MappingProxyType(dict(targets)))
        return table

    @classmethod
    def merge(cls, *tables: "ResolverTable") -> "ResolverTable":
        """Combine disjoint tables without adding resolver precedence.

        Args:
            *tables (`ResolverTable`): Tables whose public origins must be disjoint.

        Returns:
            `ResolverTable`: One immutable table containing every route.

        Raises:
            TypeError: Raised when an argument is not a `ResolverTable`.
            ValueError: Raised when two tables contain the same public origin key.
        """
        entries: list[tuple[OriginRef, OriginResolver, OriginRef]] = []
        for table in tables:
            if not isinstance(table, ResolverTable):
                raise TypeError("resolver table merge requires ResolverTable values")
            entries.extend(
                (OriginRef(key), target.resolver, target.origin)
                for key, target in table._targets.items()
            )
        return cls.from_entries(entries)

    @property
    def origins(self) -> frozenset[OriginRef]:
        """Return the exact public origins routed by this table.

        Returns:
            `frozenset[OriginRef]`: Canonical refs for every routed opaque key.
        """
        return frozenset(OriginRef(key) for key in self._targets)

    def __len__(self) -> int:
        """Return the number of exact public-origin routes."""
        return len(self._targets)

    def __contains__(self, origin: object) -> bool:
        """Return whether one exact `OriginRef` key has a route.

        Args:
            origin (`object`): Candidate identity; non-`OriginRef` values are absent.

        Returns:
            `bool`: `True` exactly when the candidate's opaque key is routed.
        """
        return (
            isinstance(origin, OriginRef)
            and type(origin.key) is str
            and bool(origin.key)
            and origin.key in self._targets
        )

    def resolve(self, origin: OriginRef) -> DiagnosticFrame | None:
        """Resolve one public origin through its exact delegate route.

        Args:
            origin (`OriginRef`): Public opaque identity to look up.

        Returns:
            `DiagnosticFrame | None`: The delegate result, or `None` when no route
            exists or the selected delegate cannot resolve its local origin.

        Raises:
            TypeError:
                Raised when `origin` is not an `OriginRef` or the selected
                delegate returns neither a `DiagnosticFrame` nor `None`.
            ValueError: Raised when an origin subclass carries an invalid key.
        """
        if not isinstance(origin, OriginRef):
            raise TypeError("resolver table resolve requires an OriginRef")
        origin_key = OriginRef(origin.key).key
        target = self._targets.get(origin_key)
        if target is None:
            return None
        frame = target.resolver.resolve(target.origin)
        if frame is not None and not isinstance(frame, DiagnosticFrame):
            raise TypeError(
                "resolver table delegate must return a DiagnosticFrame or None"
            )
        return frame

    def __copy__(self) -> "ResolverTable":
        """Return this immutable table without reconstructing its routes.

        Returns:
            `ResolverTable`: This same immutable table.
        """
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> "ResolverTable":
        """Return this immutable table without copying delegate resolvers.

        Args:
            memo (`dict[int, object]`): Standard deepcopy identity memo; unused.

        Returns:
            `ResolverTable`: This same immutable table.
        """
        return self


__all__ = [
    "Diagnostic",
    "DiagnosticFrame",
    "FAILURE_CODES",
    "Failure",
    "FailureCode",
    "OriginRef",
    "OriginResolver",
    "ResolverTable",
    "SourceRange",
]
