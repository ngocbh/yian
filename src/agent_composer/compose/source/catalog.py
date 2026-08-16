"""Occurrence-backed origin resolution for the Compose front end.

A source catalog is the format-specific authority behind source-neutral
``OriginRef`` values.  It retains a physical ``SourceDocument`` occurrence and
replays that occurrence only when a diagnostic needs a concrete frame.  No
source text is parsed or copied during resolution.

The mutable builder is private to Compose construction.  ``freeze`` takes an
immutable snapshot so later registrations cannot change an already returned
catalog.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, TypeAlias

from agent_composer.diagnostics import DiagnosticFrame, OriginRef

from agent_composer.compose.source.common import OccurrencePath, SourceAccessError
from agent_composer.compose.source.cursor import SourceCursor
from agent_composer.compose.source.document import SourceDocument


_SourceSitePath: TypeAlias = tuple[str | int, ...]
"""A deterministic semantic path local to one Compose catalog."""

_Anchor: TypeAlias = Literal["value", "key"]


@dataclass(frozen=True, slots=True)
class _ConstructedLine:
    """One zero-based line selector within a constructed scalar value."""

    index: int


_SiteSelector: TypeAlias = _Anchor | _ConstructedLine

_SCOPED_ORIGIN_PREFIX = "@scope"
"""Domain marker that keeps rebased keys outside document-local site keys."""


def _escape_segment(value: str) -> str:
    """Escape one string segment using JSON Pointer's collision-free rules.

    Args:
        value (`str`): Exact semantic-path segment.

    Returns:
        `str`: Segment with ``~`` and ``/`` escaped unambiguously.
    """
    return value.replace("~", "~0").replace("/", "~1")


def _encode_site_path(path: _SourceSitePath) -> str:
    """Encode exact string/integer segments as one opaque non-empty origin key.

    Args:
        path (`tuple[str | int, ...]`):
            Non-empty semantic site path. Exact strings and integers are
            accepted; booleans and subclasses are rejected.

    Returns:
        `str`:
            Deterministic typed JSON-Pointer-style key.

    Raises:
        `TypeError`:
            Raised when ``path`` or one of its segments has the wrong type.
        `ValueError`:
            Raised when ``path`` is empty.
    """
    if type(path) is not tuple:
        raise TypeError("source site path must be a tuple")
    if not path:
        raise ValueError("source site path cannot be empty")

    encoded: list[str] = []
    for segment in path:
        if type(segment) is str:
            encoded.append("s:" + _escape_segment(segment))
        elif type(segment) is int:
            encoded.append("i:" + str(segment))
        else:
            raise TypeError(
                "source site path segments must be exact str or int values, "
                f"got {type(segment).__name__}"
            )
    return "/" + "/".join(encoded)


def _scope_origin(alias: str, origin: OriginRef) -> OriginRef:
    """Return one Compose origin rebased under a lexical ``uses:`` alias.

    The previous opaque key is encoded as one string segment; it is never parsed.
    The reserved prefix cannot overlap document-local keys, which always begin
    with ``/``. Applying this function repeatedly therefore composes namespaces
    without losing the boundary between aliases or nested keys.

    Args:
        alias (`str`):
            Exact non-empty authored ``uses:`` alias that owns the child view.
        origin (`OriginRef`):
            Child identity to scope. Its complete key remains opaque.

    Returns:
        `OriginRef`:
            Deterministic alias-scoped identity.

    Raises:
        TypeError:
            Raised when ``alias`` is not an exact string or ``origin`` is not an
            `OriginRef`.
        ValueError:
            Raised when ``alias`` or the origin key is empty or otherwise invalid.
    """
    if type(alias) is not str:
        raise TypeError("origin scope alias must be an exact string")
    if not alias:
        raise ValueError("origin scope alias cannot be empty")
    if not isinstance(origin, OriginRef):
        raise TypeError("origin scope requires an OriginRef")
    old_key = OriginRef(origin.key).key
    scoped_path = _encode_site_path(("uses", alias, "origin", old_key))
    return OriginRef(_SCOPED_ORIGIN_PREFIX + scoped_path)


@dataclass(frozen=True, slots=True)
class _SourceSite:
    """One physical YAML occurrence and the range selected within it.

    Attributes:
        document (`SourceDocument`):
            Retained single-parse document that owns the occurrence.
        occurrence_path (`OccurrencePath`):
            Finite physical path replayed from the document root.
        selector (`Literal["value", "key"] | _ConstructedLine`):
            Whole occurrence anchor or semantic constructed-line selector.
    """

    document: SourceDocument
    occurrence_path: OccurrencePath
    selector: _SiteSelector


@dataclass(frozen=True, slots=True)
class SourceCatalog:
    """Immutable Compose resolver from opaque origins to diagnostic frames.

    The catalog owns no semantic program state. Its keys are local source
    identities, and its values retain physical document occurrences. Unknown
    origins resolve to ``None`` so a caller can compose resolvers later.

    Args:
        _sites (`Mapping[str, _SourceSite]`, *optional*):
            Origin key to physical source site. The mapping is defensively
            copied and made read-only.
    """

    _sites: Mapping[str, _SourceSite] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        """Validate and freeze a defensive copy of the source-site mapping."""
        copied: dict[str, _SourceSite] = {}
        for key, site in self._sites.items():
            if type(key) is not str or not key:
                raise ValueError("source catalog keys must be non-empty strings")
            if type(site) is not _SourceSite:
                raise TypeError("source catalog values must be source sites")
            copied[key] = site
        object.__setattr__(self, "_sites", MappingProxyType(copied))

    def resolve(self, origin: OriginRef) -> DiagnosticFrame | None:
        """Resolve one origin by replaying its retained physical occurrence.

        Args:
            origin (`OriginRef`):
                Opaque local identity minted by this catalog's builder. A scoped
                child identity must first route back to its local identity through
                a `ResolverTable`.

        Returns:
            `DiagnosticFrame | None`:
                Concrete source frame, or ``None`` when ``origin`` is unknown
                or its retained anchor no longer has a range.

        Raises:
            `TypeError`:
                Raised when ``origin`` is not an `OriginRef`.
            `SourceAccessError`:
                Raised if the retained occurrence path cannot be replayed.
        """
        if not isinstance(origin, OriginRef):
            raise TypeError("source catalog resolve requires an OriginRef")
        site = self._sites.get(origin.key)
        if site is None:
            return None

        cursor = site.document._resolve_occurrence(site.occurrence_path)
        if site.selector == "value":
            source_range = cursor.range
        elif site.selector == "key":
            source_range = cursor.key_range
        else:
            scalar_source = cursor.scalar_source
            line_ranges = (
                scalar_source.constructed_line_ranges
                if scalar_source is not None
                else None
            )
            source_range = (
                line_ranges[site.selector.index]
                if line_ranges is not None
                and site.selector.index < len(line_ranges)
                else None
            )
        if source_range is None:
            return None
        return DiagnosticFrame(
            label=site.document.label,
            text=site.document.text,
            range=source_range,
        )

    def __contains__(self, origin: object) -> bool:
        """Return whether one exact opaque origin belongs to this catalog.

        Membership reads only the immutable key table. It does not replay the
        retained source occurrence or resolve a diagnostic frame.

        Args:
            origin (`object`):
                Candidate identity; non-`OriginRef` and malformed values are absent.

        Returns:
            `bool`:
                `True` exactly when the candidate key was registered in this
                catalog snapshot.
        """
        return (
            isinstance(origin, OriginRef)
            and type(origin.key) is str
            and bool(origin.key)
            and origin.key in self._sites
        )

    def __deepcopy__(self, memo: dict[int, object]) -> SourceCatalog:
        """Return this immutable catalog without copying retained source text.

        Args:
            memo (`dict[int, object]`): Standard deepcopy identity memo; unused.

        Returns:
            `SourceCatalog`: This same immutable catalog.
        """
        return self


class _SourceCatalogBuilder:
    """Mutable registration boundary that freezes immutable catalog snapshots."""

    __slots__ = ("_sites",)

    def __init__(self) -> None:
        """Create an empty local source-site registry."""
        self._sites: dict[str, _SourceSite] = {}

    def register(
        self,
        site_path: _SourceSitePath,
        cursor: SourceCursor,
        *,
        anchor: _Anchor = "value",
    ) -> OriginRef | None:
        """Register one physical cursor under a deterministic semantic path.

        A first registration with a source-free cursor or missing selected
        anchor returns ``None`` and does not reserve the semantic path. An
        already registered encoded path is always an error, even when the later
        cursor is unlocated or points at the same occurrence.

        Args:
            site_path (`tuple[str | int, ...]`):
                Non-empty local semantic path used only to mint the opaque key.
            cursor (`SourceCursor`):
                Physical, projected, or virtual source occurrence. Projected
                cursors retain their explicitly supplied physical backing.
            anchor (`Literal["value", "key"]`, *optional*, defaults to `"value"`):
                Range selected when the catalog resolves the returned origin.
        Returns:
            `OriginRef | None`:
                Registered opaque identity, or ``None`` when no concrete
                selected range exists.

        Raises:
            `TypeError`:
                Raised for an invalid cursor, path, or anchor type.
            `ValueError`:
                Raised for an empty path.
            `SourceAccessError`:
                Raised when ``site_path`` is already registered.
        """
        key = _encode_site_path(site_path)
        if not isinstance(cursor, SourceCursor):
            raise TypeError("source catalog registration requires a SourceCursor")
        if type(anchor) is not str or anchor not in ("value", "key"):
            raise TypeError("source catalog anchor must be 'value' or 'key'")
        document = cursor.document
        source_range = cursor.range if anchor == "value" else cursor.key_range
        if key in self._sites:
            raise SourceAccessError(
                f"duplicate source catalog site {site_path!r}",
                source_range,
            )
        if document is None or source_range is None:
            return None

        self._sites[key] = _SourceSite(
            document=document,
            occurrence_path=cursor.path,
            selector=anchor,
        )
        return OriginRef(key)

    def register_constructed_line(
        self,
        site_path: _SourceSitePath,
        cursor: SourceCursor,
        index: int,
    ) -> OriginRef | None:
        """Register one exact constructed literal line by semantic index.

        Unchanged literal block scalars retain exact line ranges. Other scalar
        styles, virtual cursors, and value-changing projections return ``None``
        without reserving the path so the caller can register a whole-value
        fallback. A successful site retains the semantic index rather than an
        absolute coordinate.

        Args:
            site_path (`tuple[str | int, ...]`):
                Non-empty local semantic path used only to mint the opaque key.
            cursor (`SourceCursor`):
                Candidate scalar occurrence.
            index (`int`):
                Exact nonnegative zero-based constructed-line index.

        Returns:
            `OriginRef | None`:
                Registered opaque identity, or ``None`` when exact scalar-line
                provenance is unavailable.

        Raises:
            `TypeError`:
                Raised for an invalid cursor, path, or index type.
            `ValueError`:
                Raised for an empty path or negative/out-of-range exact index.
            `SourceAccessError`:
                Raised when ``site_path`` is already registered.
        """

        key = _encode_site_path(site_path)
        if not isinstance(cursor, SourceCursor):
            raise TypeError("source catalog registration requires a SourceCursor")
        if type(index) is not int:
            raise TypeError("constructed line index must be an exact integer")
        if index < 0:
            raise SourceAccessError(
                "constructed line index must be non-negative",
                cursor.range,
            )
        if key in self._sites:
            raise SourceAccessError(
                f"duplicate source catalog site {site_path!r}",
                cursor.range,
            )

        scalar_source = cursor.scalar_source
        line_ranges = (
            scalar_source.constructed_line_ranges
            if scalar_source is not None
            else None
        )
        if line_ranges is None:
            return None
        if index >= len(line_ranges):
            raise SourceAccessError(
                f"constructed line index {index} is out of range",
                scalar_source.range,
            )

        document = cursor.document
        if document is None:
            return None
        self._sites[key] = _SourceSite(
            document=document,
            occurrence_path=cursor.path,
            selector=_ConstructedLine(index),
        )
        return OriginRef(key)

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Roll back registrations made by a failed construction attempt.

        Callable compilation is success-cached and may be retried after an
        error. A failed attempt must therefore leave the catalog at the same
        insertion boundary as the cache. Nested transactions are savepoints;
        the builder is single-threaded during Compose loading.

        Yields:
            `None`: Control to the construction attempt.
        """
        checkpoint = len(self._sites)
        try:
            yield
        except BaseException:
            while len(self._sites) > checkpoint:
                self._sites.popitem()
            raise

    def freeze(self) -> SourceCatalog:
        """Return an immutable snapshot of all successful registrations.

        Returns:
            `SourceCatalog`: Snapshot isolated from later builder changes.
        """
        return SourceCatalog(self._sites)


__all__ = ["SourceCatalog"]
