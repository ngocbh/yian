"""Contracts for resolving semantic failures at the Compose boundary."""

from __future__ import annotations

from typing import cast

import pytest

from agent_composer.compile.model import CompiledFlow
from agent_composer.compose import ComposedFlow, load_flow
from agent_composer.compose.source import SourceDocument
from agent_composer.diagnostics import (
    Diagnostic,
    DiagnosticFrame,
    Failure,
    OriginRef,
    OriginResolver,
    SourceRange,
)
from agent_composer.run import run_flow


_UNAVAILABLE_NOTE = ("Source presentation was unavailable.",)


def _composed_flow(
    resolver: OriginResolver | None,
    *,
    compiled: object | None = None,
) -> ComposedFlow:
    """Construct a minimal Compose boundary around a test resolver."""
    return ComposedFlow(
        compiled=cast(CompiledFlow, compiled if compiled is not None else object()),
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


def test_diagnose_malformed_base_frame_field_degrades_unlocated() -> None:
    """A nominal frame cannot carry an invalid range into the renderer."""
    malformed = DiagnosticFrame(
        label="flow.yaml",
        text="value\n",
        range=cast(SourceRange, object()),
    )

    class Resolver:
        def resolve(self, origin: OriginRef) -> DiagnosticFrame | None:
            return malformed

    diagnostic = _composed_flow(Resolver()).diagnose(_failure(OriginRef("bad")))

    assert diagnostic.frames == ()
    assert diagnostic.notes == _UNAVAILABLE_NOTE


def test_diagnose_throwing_frame_subclass_degrades_unlocated() -> None:
    """Ordinary field-access faults cannot mask the semantic failure."""

    class ThrowingFrame(DiagnosticFrame):
        def __getattribute__(self, name: str) -> object:
            if name == "text":
                raise RuntimeError("secret frame detail")
            return super().__getattribute__(name)

    frame = ThrowingFrame(
        label="flow.yaml",
        text="value\n",
        range=SourceRange(1, 1, 1, 6),
    )

    class Resolver:
        def resolve(self, origin: OriginRef) -> DiagnosticFrame | None:
            return frame

    diagnostic = _composed_flow(Resolver()).diagnose(_failure(OriginRef("bad")))

    assert diagnostic.frames == ()
    assert diagnostic.notes == _UNAVAILABLE_NOTE
    assert "secret" not in " ".join(diagnostic.notes)


def test_diagnose_valid_frame_subclasses_are_canonicalized() -> None:
    """Presentation snapshots contain exact base values without subclass behavior."""

    class DerivedRange(SourceRange):
        pass

    class DerivedFrame(DiagnosticFrame):
        pass

    derived = DerivedFrame(
        label="flow.yaml",
        text="value\n",
        range=DerivedRange(1, 1, 1, 6),
        note="context",
    )

    class Resolver:
        def resolve(self, origin: OriginRef) -> DiagnosticFrame | None:
            return derived

    diagnostic = _composed_flow(Resolver()).diagnose(_failure(OriginRef("valid")))

    assert diagnostic.frames == (
        DiagnosticFrame(
            label="flow.yaml",
            text="value\n",
            range=SourceRange(1, 1, 1, 6),
            note="context",
        ),
    )
    assert type(diagnostic.frames[0]) is DiagnosticFrame
    assert type(diagnostic.frames[0].range) is SourceRange


@pytest.mark.parametrize(
    "coordinates",
    [
        (True, 1, 1, 1),
        (1, True, 1, 1),
        (1, 1, True, 1),
        (1, 1, 1, True),
        (0, 1, 1, 1),
        (1, 0, 1, 1),
        (1, 1, 0, 1),
        (1, 1, 1, 0),
        (2, 1, 1, 1),
        (2, 3, 2, 2),
    ],
)
def test_diagnose_invalid_frame_coordinates_discard_partial_prefix(
    coordinates: tuple[int, int, int, int],
) -> None:
    """Every coordinate is exact and positive, and the half-open end follows start."""
    first = OriginRef("first")
    invalid = OriginRef("invalid")
    good_frame = DiagnosticFrame(
        label="outer.yaml",
        text="outer\n",
        range=SourceRange(1, 1, 1, 6),
    )
    invalid_frame = DiagnosticFrame(
        label="inner.yaml",
        text="inner\n",
        range=SourceRange(*coordinates),
    )

    class Resolver:
        def resolve(self, origin: OriginRef) -> DiagnosticFrame | None:
            return good_frame if origin == first else invalid_frame

    diagnostic = _composed_flow(Resolver()).diagnose(_failure(first, invalid))

    assert diagnostic.frames == ()
    assert diagnostic.notes == _UNAVAILABLE_NOTE


def test_diagnose_frame_control_exception_propagates() -> None:
    """Presentation fallback catches ordinary faults, not process control."""

    class InterruptingFrame(DiagnosticFrame):
        def __getattribute__(self, name: str) -> object:
            if name == "label":
                raise KeyboardInterrupt
            return super().__getattribute__(name)

    frame = InterruptingFrame(
        label="flow.yaml",
        text="value\n",
        range=SourceRange(1, 1, 1, 6),
    )

    class Resolver:
        def resolve(self, origin: OriginRef) -> DiagnosticFrame | None:
            return frame

    with pytest.raises(KeyboardInterrupt):
        _composed_flow(Resolver()).diagnose(_failure(OriginRef("interrupt")))


def test_diagnose_reads_only_failure_and_resolver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Diagnosis never inspects the compiled graph or reparses YAML."""

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


_INLINE_FAILURES = {
    "map": """\
id: inline-map-failure
name: inline_map_failure
nodes:
  outer:
    kind: map
    over: [item]
    input:
      value: ${item}
    nodes:
      fail:
        kind: code
        code: |
          raise RuntimeError("boom")
          return inputs["value"]
        input:
          value: ${input.value}
        output: str
    output: ${fail.output}
output: ${outer.output}
""",
    "loop": """\
id: inline-loop-failure
name: inline_loop_failure
nodes:
  outer:
    kind: loop
    times: 1
    input:
      n: 0
    nodes:
      fail:
        kind: code
        code: |
          raise RuntimeError("boom")
          return {"n": inputs["n"]}
        input:
          n: ${input.n}
        output:
          n: int
    output: ${fail.output}
output: ${outer.output}
""",
}


@pytest.mark.parametrize("kind", ("map", "loop"))
def test_inline_composite_failure_resolves_catalog_frames(kind: str) -> None:
    """Inline MAP/LOOP diagnostics need no child source snapshot."""
    label = f"inline-{kind}.yaml"
    loaded = load_flow(_INLINE_FAILURES[kind], source_label=label)

    result = run_flow(loaded, {})

    assert result.status == "failed"
    assert result.failure is not None
    assert result.failure.traceback is not None
    assert label in result.failure.traceback
    diagnostic = loaded.diagnose(result.failure)
    assert len(diagnostic.frames) == 2
    assert [frame.label for frame in diagnostic.frames] == [label, label]
    assert "outer:" in diagnostic.frames[0].text
    assert 'raise RuntimeError("boom")' in diagnostic.frames[1].text


_EXTERNAL_FAILING_CHILD = """\
id: failing-child
name: failing_child
input:
  n: int
nodes:
  fail:
    kind: code
    code: |
      raise RuntimeError("boom")
      return {"n": inputs["n"]}
    input:
      n: ${input.n}
    output:
      n: int
output: ${fail.output}
"""

_EXTERNAL_PARENTS = {
    "map": """\
id: external-map-failure
name: external_map_failure
uses:
  child: failing-child
nodes:
  outer:
    kind: map
    call: child
    over: [0]
    input:
      n: ${item}
output: ${outer.output}
""",
    "loop": """\
id: external-loop-failure
name: external_loop_failure
uses:
  child: failing-child
nodes:
  outer:
    kind: loop
    call: child
    times: 1
    input:
      n: 0
output: ${outer.output}
""",
}


@pytest.mark.parametrize("kind", ("map", "loop"))
def test_external_composite_failure_resolves_each_document(kind: str) -> None:
    """External MAP/LOOP diagnostics resolve parent and child catalogs in order."""
    child = load_flow(_EXTERNAL_FAILING_CHILD, source_label="child.yaml")
    parent_label = f"external-{kind}.yaml"
    loaded = load_flow(
        _EXTERNAL_PARENTS[kind],
        child_resolver=lambda _flow_id, _version: child,
        source_label=parent_label,
    )

    result = run_flow(loaded, {})

    assert result.status == "failed"
    assert result.failure is not None
    assert result.failure.traceback is not None
    assert "child.yaml" in result.failure.traceback
    diagnostic = loaded.diagnose(result.failure)
    assert len(diagnostic.frames) == 2
    assert [frame.label for frame in diagnostic.frames] == [
        parent_label,
        "child.yaml",
    ]
    assert diagnostic.frames[0].text == _EXTERNAL_PARENTS[kind]
    assert diagnostic.frames[1].text == _EXTERNAL_FAILING_CHILD
