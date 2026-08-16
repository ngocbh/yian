"""Contracts for resolving semantic failures at the Compose boundary."""

from __future__ import annotations

from typing import cast

import pytest

from agent_composer.compile.model import CompiledFlow
from agent_composer.compose import ComposedFlow, load_flow
from agent_composer.compose.helpers.source import SourceFrame
from agent_composer.compose.source import SourceDocument
from agent_composer.diagnostics import (
    Diagnostic,
    DiagnosticFrame,
    Failure,
    OriginRef,
    OriginResolver,
    SourceRange,
)


_UNAVAILABLE_NOTE = ("Source presentation was unavailable.",)


def _composed_flow(
    resolver: OriginResolver | None,
    *,
    compiled: object | None = None,
    source: object | None = None,
) -> ComposedFlow:
    """Construct a minimal Compose boundary around a test resolver."""
    return ComposedFlow(
        compiled=cast(CompiledFlow, compiled if compiled is not None else object()),
        source=cast(SourceFrame, source),
        origin_resolver=resolver,
    )


def _failure(*origins: OriginRef) -> Failure:
    """Return one fully populated semantic failure for copy assertions."""
    return Failure(
        code="contract.input",
        message="expected an integer",
        node_id="outer/inner",
        trace=origins,
        cause_type="TypeCheckError",
        traceback="Traceback: original failure",
    )


def test_diagnose_copies_semantics_and_resolves_ordered_duplicate_frames() -> None:
    """Every trace occurrence becomes one frame without deduplication or reordering."""
    outer = OriginRef("outer")
    inner = OriginRef("inner")
    outer_frame = DiagnosticFrame(
        label="root.yaml",
        text="outer:\n",
        range=SourceRange(1, 1, 1, 6),
        note="called child",
    )
    inner_frame = DiagnosticFrame(
        label="child.yaml",
        text="inner:\n",
        range=SourceRange(1, 1, 1, 6),
    )

    class Resolver:
        def __init__(self) -> None:
            self.calls: list[OriginRef] = []

        def resolve(self, origin: OriginRef) -> DiagnosticFrame | None:
            self.calls.append(origin)
            return outer_frame if origin == outer else inner_frame

    resolver = Resolver()
    failure = _failure(outer, inner, inner)

    diagnostic = _composed_flow(resolver).diagnose(failure)

    assert resolver.calls == [outer, inner, inner]
    assert diagnostic == Diagnostic(
        code=failure.code,
        message=failure.message,
        frames=(outer_frame, inner_frame, inner_frame),
        notes=(),
        node_id=failure.node_id,
        cause_type=failure.cause_type,
        traceback=failure.traceback,
    )


def test_diagnose_empty_trace_is_unlocated_without_touching_resolver() -> None:
    """A source-free failure needs no resolver and no presentation warning."""

    class UntouchedResolver:
        def resolve(self, origin: OriginRef) -> DiagnosticFrame | None:
            raise AssertionError(f"resolver unexpectedly received {origin!r}")

    failure = _failure()

    diagnostic = _composed_flow(UntouchedResolver()).diagnose(failure)

    assert diagnostic == Diagnostic(
        code=failure.code,
        message=failure.message,
        node_id=failure.node_id,
        cause_type=failure.cause_type,
        traceback=failure.traceback,
    )


def test_diagnose_missing_resolver_degrades_nonempty_trace_atomically() -> None:
    """A traced failure remains intact when its source authority is absent."""
    failure = _failure(OriginRef("missing"))

    diagnostic = _composed_flow(None).diagnose(failure)

    assert diagnostic == Diagnostic(
        code=failure.code,
        message=failure.message,
        notes=_UNAVAILABLE_NOTE,
        node_id=failure.node_id,
        cause_type=failure.cause_type,
        traceback=failure.traceback,
    )


def test_diagnose_unresolved_later_origin_discards_partial_frame_prefix() -> None:
    """A trace is presented only when every origin resolves."""
    first = OriginRef("first")
    missing = OriginRef("missing")
    first_frame = DiagnosticFrame(
        label="flow.yaml",
        text="first:\n",
        range=SourceRange(1, 1, 1, 6),
    )

    class Resolver:
        def __init__(self) -> None:
            self.calls: list[OriginRef] = []

        def resolve(self, origin: OriginRef) -> DiagnosticFrame | None:
            self.calls.append(origin)
            return first_frame if origin == first else None

    resolver = Resolver()

    diagnostic = _composed_flow(resolver).diagnose(_failure(first, missing))

    assert resolver.calls == [first, missing]
    assert diagnostic.frames == ()
    assert diagnostic.notes == _UNAVAILABLE_NOTE


def test_diagnose_resolver_exception_does_not_mask_semantic_failure() -> None:
    """Ordinary presentation exceptions do not escape or leak their details."""
    failure = _failure(OriginRef("broken"))

    class Resolver:
        def resolve(self, origin: OriginRef) -> DiagnosticFrame | None:
            raise RuntimeError(f"secret resolver detail for {origin.key}")

    diagnostic = _composed_flow(Resolver()).diagnose(failure)

    assert diagnostic.message == failure.message
    assert diagnostic.frames == ()
    assert diagnostic.notes == _UNAVAILABLE_NOTE
    assert "secret" not in " ".join(diagnostic.notes)


def test_diagnose_malformed_resolver_result_degrades_unlocated() -> None:
    """A resolver result outside its protocol cannot enter presentation data."""

    class Resolver:
        def resolve(self, origin: OriginRef) -> DiagnosticFrame | None:
            return cast(DiagnosticFrame, object())

    diagnostic = _composed_flow(Resolver()).diagnose(_failure(OriginRef("bad")))

    assert diagnostic.frames == ()
    assert diagnostic.notes == _UNAVAILABLE_NOTE


def test_diagnose_reads_only_failure_and_resolver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Diagnosis never inspects compiled/source objects or reparses YAML."""

    class ForbiddenSurface:
        def __getattribute__(self, name: str) -> object:
            raise AssertionError(f"diagnose inspected forbidden surface {name!r}")

    def reject_parse(*args: object, **kwargs: object) -> SourceDocument:
        raise AssertionError("diagnose reparsed YAML")

    monkeypatch.setattr(SourceDocument, "parse", reject_parse)
    origin = OriginRef("only")
    frame = DiagnosticFrame(
        label="memory",
        text="value\n",
        range=SourceRange(1, 1, 1, 6),
    )

    class ResolveOnly:
        def resolve(self, candidate: OriginRef) -> DiagnosticFrame | None:
            assert candidate is origin
            return frame

    diagnostic = _composed_flow(
        ResolveOnly(),
        compiled=ForbiddenSurface(),
        source=ForbiddenSurface(),
    ).diagnose(_failure(origin))

    assert diagnostic.frames == (frame,)
    assert diagnostic.notes == ()


def test_diagnose_rejects_non_failure_input() -> None:
    """The total presentation guarantee starts only after Failure validation."""
    with pytest.raises(TypeError, match="diagnose requires a Failure"):
        _composed_flow(None).diagnose(cast(Failure, object()))


def test_load_flow_origin_resolves_through_installed_authority() -> None:
    """A real authored node origin resolves through the finalized loader table."""
    source = """\
id: located
name: located
nodes:
  work:
    kind: code
    code: |
      return 1
    output: int
output: ${work.output}
"""
    loaded = load_flow(source)
    origin = loaded.compiled.nodes["work"].origin
    assert isinstance(origin, OriginRef)

    diagnostic = loaded.diagnose(_failure(origin))

    assert diagnostic.frames == (
        DiagnosticFrame(
            label="<flow>",
            text=source,
            range=SourceRange(4, 3, 4, 7),
        ),
    )
    assert diagnostic.notes == ()
