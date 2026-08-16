"""Canonical diagnostic construction at the Compose load boundary."""

from agent_composer.compose.helpers.errors import LoadDiagnostics, LoadError
from agent_composer.compose.source import SourceDocument
from agent_composer.diagnostics import Diagnostic, DiagnosticFrame


def test_load_error_preserves_one_canonical_diagnostic() -> None:
    diagnostic = Diagnostic(code="compose.load", message="broken")

    error = LoadError(diagnostic)

    assert error.diagnostic is diagnostic
    assert str(error) == "broken"


def test_document_factory_owns_the_frame_source() -> None:
    document = SourceDocument.parse("value: broken\n", "child.yaml")
    value = document.root.child("value")

    error = LoadDiagnostics.from_document(document).error(
        "bad value",
        range=value.range,
        notes=("expected an integer",),
    )

    assert error.diagnostic == Diagnostic(
        code="compose.load",
        message="bad value",
        frames=(
            DiagnosticFrame(
                label="child.yaml",
                text=document.text,
                range=value.range,
            ),
        ),
        notes=("expected an integer",),
    )


def test_prepend_adds_outer_context_without_mutating_inner_error() -> None:
    parent = SourceDocument.parse("uses:\n  child: child\n", "parent.yaml")
    child = SourceDocument.parse("value: broken\n", "child.yaml")
    child_value = child.root.child("value")
    inner = LoadDiagnostics.from_document(child).error(
        "bad child",
        range=child_value.range,
        notes=("inner note",),
    )
    original = inner.diagnostic

    wrapped = LoadDiagnostics.from_document(parent).prepend(
        inner,
        range=parent.root.child("uses").child("child").range,
        message="uses alias 'child': bad child",
        note="imported here",
    )

    assert inner.diagnostic is original
    assert wrapped is not inner
    assert wrapped.diagnostic.message == "uses alias 'child': bad child"
    assert [frame.label for frame in wrapped.diagnostic.frames] == [
        "parent.yaml",
        "child.yaml",
    ]
    assert wrapped.diagnostic.frames[0].note == "imported here"
    assert wrapped.diagnostic.notes == ("inner note",)


def test_source_free_factory_keeps_the_diagnostic_unlocated() -> None:
    document = SourceDocument.parse("value: broken\n", "unused.yaml")

    error = LoadDiagnostics().error(
        "source-free failure",
        range=document.root.range,
    )

    assert error.diagnostic.frames == ()
