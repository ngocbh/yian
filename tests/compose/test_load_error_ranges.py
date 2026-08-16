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
