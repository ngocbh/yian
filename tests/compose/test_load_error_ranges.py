"""Source-backed Compose load failures retain authoritative ``SourceRange`` values."""

from pathlib import Path

import pytest

from agent_composer.compose import LoadError, load_flow
from agent_composer.compose.source import SourceDocument


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

    assert caught.value.range == expected
    assert caught.value.line == expected.start_line


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

    assert caught.value.range == expected
    assert caught.value.line == expected.start_line


def test_mapped_root_output_legacy_line_derives_from_retained_range() -> None:
    """A mapped output never pairs its key line with its value-block range."""
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

    assert caught.value.range == expected
    assert caught.value.line == expected.start_line


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
def test_definition_wrapper_does_not_mix_an_inner_line_with_its_fallback_range(
    definition: str,
) -> None:
    """A def wrapper adds its coarse range only when the inner error is unlocated."""
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

    def fail_resolver(_path: str, _version: str | None = None):
        raise LoadError("external failure", line=999)

    with pytest.raises(LoadError, match="defs entry 'local'") as caught:
        load_flow(text, child_resolver=fail_resolver)

    assert caught.value.line == 999
    assert caught.value.range is None


def test_external_resolver_error_location_is_owned_by_each_source_document() -> None:
    """Locating a shared resolver exception never mutates or reuses its first site."""
    shared_error = LoadError("external failure")

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
        assert caught.value.range == expected
        assert caught.value.line == expected.start_line
        seen_ranges.append(caught.value.range)

    assert seen_ranges[0] != seen_ranges[1]
    assert shared_error.range is None
    assert shared_error.line is None
