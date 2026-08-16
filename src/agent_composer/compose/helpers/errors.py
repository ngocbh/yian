"""Canonical, source-owned diagnostics for Compose load failures."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, TYPE_CHECKING

from agent_composer.diagnostics import Diagnostic, DiagnosticFrame, SourceRange

if TYPE_CHECKING:
    from agent_composer.compose.source import SourceCursor, SourceDocument


@dataclass(frozen=True, slots=True)
class LoadDiagnostics:
    """Build immutable load diagnostics owned by one Compose source document.

    The context holds only the presentation data already retained by a parsed
    document. A source-free context still constructs valid unlocated diagnostics,
    which keeps direct Python composition independent of YAML.

    Args:
        label (`str | None`, *optional*, defaults to `None`):
            Human-facing source name. `None` creates an unlocated context and
            requires `text` to be `None` too.
        text (`str | None`, *optional*, defaults to `None`):
            Complete source text indexed by ranges. `None` creates an unlocated
            context and requires `label` to be `None` too.
    """

    label: str | None = None
    text: str | None = None

    def __post_init__(self) -> None:
        """Reject partial or malformed source ownership."""
        if (self.label is None) != (self.text is None):
            raise ValueError("load diagnostic source requires both label and text")
        if self.label is not None and type(self.label) is not str:
            raise TypeError("load diagnostic label must be a string")
        if self.text is not None and type(self.text) is not str:
            raise TypeError("load diagnostic text must be a string")

    @classmethod
    def from_document(
        cls,
        document: SourceDocument | None,
    ) -> LoadDiagnostics:
        """Build a source authority from an existing parsed document.

        Args:
            document (`SourceDocument | None`):
                Parsed document, or `None` for source-free composition.

        Returns:
            `LoadDiagnostics`:
                Context owning the document's label and text, or an unlocated
                context when `document` is `None`.
        """
        if document is None:
            return cls()
        return cls(label=document.label, text=document.text)

    @classmethod
    def from_cursor(
        cls,
        cursor: SourceCursor | None,
    ) -> LoadDiagnostics:
        """Build a source authority from a physical or virtual cursor.

        Args:
            cursor (`SourceCursor | None`):
                Cursor whose retained document owns later ranges. A virtual or
                absent cursor creates a source-free context.

        Returns:
            `LoadDiagnostics`:
                Context for the cursor's document, or an unlocated context.
        """
        document = cursor.document if cursor is not None else None
        return cls.from_document(document)

    def frame(
        self,
        range: SourceRange | None,
        *,
        note: str | None = None,
    ) -> DiagnosticFrame | None:
        """Resolve one concrete range against this context's owned source.

        Args:
            range (`SourceRange | None`):
                Concrete span in this context's text, or `None` for no frame.
            note (`str | None`, *optional*, defaults to `None`):
                Context printed next to the frame.

        Returns:
            `DiagnosticFrame | None`:
                Owned frame, or `None` when either the range or source is absent.
        """
        if range is None or self.label is None or self.text is None:
            return None
        if not isinstance(range, SourceRange):
            raise TypeError("load diagnostic range must be a SourceRange or None")
        if note is not None and type(note) is not str:
            raise TypeError("load diagnostic frame note must be a string or None")
        return DiagnosticFrame(self.label, self.text, range, note)

    def diagnostic(
        self,
        message: str,
        *,
        range: SourceRange | None = None,
        frames: Iterable[DiagnosticFrame] = (),
        notes: Iterable[str] = (),
    ) -> Diagnostic:
        """Construct one canonical Compose load diagnostic.

        Args:
            message (`str`): Human-readable failure detail.
            range (`SourceRange | None`, *optional*, defaults to `None`):
                Primary location in this context's source.
            frames (`Iterable[DiagnosticFrame]`, *optional*, defaults to `()`):
                Explicit ordered frames. Mutually exclusive with `range`.
            notes (`Iterable[str]`, *optional*, defaults to `()`):
                Explanatory context not owned by one frame.

        Returns:
            `Diagnostic`: Frozen presentation-ready load diagnostic.

        Raises:
            TypeError: Raised for malformed messages, frames, or notes.
            ValueError: Raised when `range` and explicit frames are both supplied.
        """
        if type(message) is not str:
            raise TypeError("load diagnostic message must be a string")
        owned_frames = tuple(frames)
        if range is not None and owned_frames:
            raise ValueError("load diagnostic accepts a range or frames, not both")
        for frame in owned_frames:
            if not isinstance(frame, DiagnosticFrame):
                raise TypeError("load diagnostic frames must be DiagnosticFrame values")
        if range is not None:
            frame = self.frame(range)
            owned_frames = (frame,) if frame is not None else ()
        owned_notes = tuple(notes)
        if any(type(note) is not str for note in owned_notes):
            raise TypeError("load diagnostic notes must be strings")
        return Diagnostic(
            code="compose.load",
            message=message,
            frames=owned_frames,
            notes=owned_notes,
        )

    def error(
        self,
        message: str,
        *,
        range: SourceRange | None = None,
        frames: Iterable[DiagnosticFrame] = (),
        notes: Iterable[str] = (),
    ) -> LoadError:
        """Construct the one exception type crossing the Compose boundary.

        Args:
            message (`str`): Human-readable failure detail.
            range (`SourceRange | None`, *optional*, defaults to `None`):
                Primary location in this context's source.
            frames (`Iterable[DiagnosticFrame]`, *optional*, defaults to `()`):
                Explicit ordered frames. Mutually exclusive with `range`.
            notes (`Iterable[str]`, *optional*, defaults to `()`):
                Explanatory context not owned by one frame.

        Returns:
            `LoadError`: Exception carrying the canonical diagnostic.
        """
        return LoadError(
            self.diagnostic(
                message,
                range=range,
                frames=frames,
                notes=notes,
            )
        )

    def prepend(
        self,
        error: LoadError,
        *,
        range: SourceRange | None,
        message: str | None = None,
        note: str | None = None,
    ) -> LoadError:
        """Return a fresh error with one outer source context frame.

        Args:
            error (`LoadError`): Existing inner failure; it is never mutated.
            range (`SourceRange | None`): Outer location in this context's source.
            message (`str | None`, *optional*, defaults to `None`):
                Replacement message. `None` preserves the inner message.
            note (`str | None`, *optional*, defaults to `None`):
                Context attached to the prepended frame.

        Returns:
            `LoadError`: Fresh exception with outer-to-inner frames.
        """
        if not isinstance(error, LoadError):
            raise TypeError("load diagnostic context requires a LoadError")
        outer = self.frame(range, note=note)
        frames = error.diagnostic.frames
        if outer is not None:
            frames = (outer, *frames)
        diagnostic = replace(
            error.diagnostic,
            message=(error.diagnostic.message if message is None else message),
            frames=frames,
        )
        return LoadError(diagnostic)


class LoadError(ValueError):
    """A Compose flow cannot be loaded, expressed as one canonical diagnostic."""

    def __init__(self, diagnostic: Diagnostic):
        """Retain the immutable diagnostic as the exception's only payload."""
        if not isinstance(diagnostic, Diagnostic):
            raise TypeError("LoadError requires a Diagnostic")
        self.diagnostic = diagnostic
        super().__init__(diagnostic.message)
