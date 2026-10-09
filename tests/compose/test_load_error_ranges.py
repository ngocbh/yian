"""Source-backed Compose load failures retain authoritative ``SourceRange`` values."""

from pathlib import Path

import pytest

from yian.compose import LoadError, load_flow
from yian.compose.helpers.errors import LoadDiagnostics
from yian.compose.source import SourceDocument


_ERRORS = Path(__file__).resolve().parents[1] / "seeds" / "errors"


@pytest.mark.parametrize(
    ("filename", "section"),
    [
        ("e01-undeclared-ref.yaml", "output"),
        ("e16-bad-typedef-name.yaml", "typedefs"),
    ],
)
def test_root_section_failures_keep_the_authored_range(filename: str, section: str) -> None:
    """Section-level errors use the retained document instead of a line-only map."""
    text = (_ERRORS / filename).read_text()
    root = SourceDocument.parse(text).root
    expected = (
        root.child(section).range
        if section == "output"
        else root.key_range_of(section)
    )

    with pytest.raises(LoadError) as caught:
        load_flow(text)

    assert caught.value.diagnostic.frames[-1].range == expected


def test_definition_output_failure_keeps_its_absolute_value_range() -> None:
    """A def output is located in the owning document, not at the parent call node."""
    text = """id: def-output-range
name: def_output_range
defs:
  child:
    nodes:
      emit:
        kind: code
        code: |
          return 1
        output: int
    output: ${missing.output}
nodes:
  run:
    kind: call
    call: child
output: ${run.output}
"""
    document = SourceDocument.parse(text)
    expected = document.root.child("defs").child("child").child("output").range

    with pytest.raises(LoadError, match="defs entry 'child'") as caught:
        load_flow(text)

    frames = caught.value.diagnostic.frames
    assert [frame.range for frame in frames] == [
        document.root.child("defs").child("child").range,
        expected,
    ]


def test_mapped_root_output_keeps_the_retained_range() -> None:
    """A mapped output retains its complete authored value-block range."""
    text = """id: mapped-output-range
name: mapped_output_range
nodes:
  emit:
    kind: code
    code: |
      return 1
    output: int
output:
  result: ${missing.output}
"""
    document = SourceDocument.parse(text)
    expected = document.root.child("output").range

    with pytest.raises(LoadError, match="unresolved references") as caught:
        load_flow(text)

    assert caught.value.diagnostic.frames[-1].range == expected


@pytest.mark.parametrize(
    "definition",
    [
        """    nodes:
      invoke:
        kind: call
        call: external
    output: ${invoke.output}
""",
        """    kind: call
    call: external
""",
    ],
)
def test_definition_wrapper_prepends_context_without_replacing_inner_location(
    definition: str,
) -> None:
    """A def wrapper retains both its outer body and the precise inner uses site."""
    text = f"""id: def-inner-line
name: def_inner_line
uses:
  external: child.yaml
defs:
  local:
{definition}nodes:
  invoke:
    kind: call
    call: local
output: ${{invoke.output}}
"""

    document = SourceDocument.parse(text)
    expected = [
        document.root.child("defs").child("local").range,
        document.root.child("uses").child("external").range,
    ]

    def fail_resolver(_path: str, _version: str | None = None):
        raise LoadDiagnostics().error("external failure")

    with pytest.raises(LoadError, match="defs entry 'local'") as caught:
        load_flow(text, child_resolver=fail_resolver)

    assert [frame.range for frame in caught.value.diagnostic.frames] == expected


def test_external_resolver_error_location_is_owned_by_each_source_document() -> None:
    """Locating a shared resolver exception never mutates or reuses its first site."""
    shared_error = LoadDiagnostics().error("external failure")
    original_diagnostic = shared_error.diagnostic

    def fail_resolver(_path: str, _version: str | None = None):
        raise shared_error

    sources = [
        """id: first-parent
name: first_parent
uses:
  external: child.yaml
nodes:
  invoke:
    kind: call
    call: external
output: ${invoke.output}
""",
        """id: second-parent
name: second_parent

# Keep the same alias at a different authored site.
uses:
  external: child.yaml
nodes:
  invoke:
    kind: call
    call: external
output: ${invoke.output}
""",
    ]

    seen_ranges = []
    for text in sources:
        expected = (
            SourceDocument.parse(text)
            .root.child("uses")
            .child("external")
            .range
        )
        with pytest.raises(LoadError, match="external failure") as caught:
            load_flow(text, child_resolver=fail_resolver)
        assert caught.value is not shared_error
        frames = caught.value.diagnostic.frames
        assert len(frames) == 1
        assert frames[0].range == expected
        seen_ranges.append(frames[0].range)

    assert seen_ranges[0] != seen_ranges[1]
    assert shared_error.diagnostic is original_diagnostic
    assert shared_error.diagnostic.frames == ()
