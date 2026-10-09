"""Tests for the format-neutral diagnostic CLI renderer."""

from __future__ import annotations

import re

from yian.cli.run import err_console, render_diagnostic
from yian.diagnostics import Diagnostic, DiagnosticFrame, SourceRange

_SOURCE = """\
id: f
name: f
nodes:
  echo:
    kind: code
    input:
      topic: ${input.topic}
    output: str
    code: tests.seeds.fns:echo_value
output:
  topic: ${echo.output}
"""


def _marked(out: str) -> list[int]:
    """The line numbers the frame actually HIGHLIGHTED (`❱`), not merely displayed.

    The box renders a window of surrounding context, so asserting that some text appears
    proves nothing about the highlight — only the marker does.
    """
    return [int(n) for n in re.findall(r"❱\s+(\d+)", out)]


def test_render_diagnostic_boxes_one_frame() -> None:
    """One resolved frame supplies its own source, label, and exact mark."""
    diag = Diagnostic(
        code="compose.load",
        message="bad thing happened",
        frames=(
            DiagnosticFrame(
                label="flow.yaml",
                text=_SOURCE,
                range=SourceRange(
                    start_line=5,
                    start_col=5,
                    end_line=5,
                    end_col=15,
                ),
            ),
        ),
    )
    with err_console.capture() as cap:
        render_diagnostic(diag)
    out = cap.get()
    assert "bad thing happened" in out
    assert "flow.yaml:5" in out
    assert "kind: code" in out  # line 5 is inside the boxed source window
    assert _marked(out) == [5]  # ...and it is the line actually highlighted


def test_render_diagnostic_block_range_stops_before_the_next_field() -> None:
    # `input:`'s value is a block MAPPING (line 7), so PyYAML ends it at the start of the next
    # token — `output:` on line 8, column 5. The end is EXCLUSIVE and line 8 holds only
    # indentation before column 5, so the span covers line 7 alone. Reading `end_line` directly
    # would highlight `output: str`, a different field.
    diag = Diagnostic(
        code="compose.load",
        message="bad ref",
        frames=(
            DiagnosticFrame(
                label="flow.yaml",
                text=_SOURCE,
                range=SourceRange(
                    start_line=7,
                    start_col=7,
                    end_line=8,
                    end_col=5,
                ),
            ),
        ),
    )
    with err_console.capture() as cap:
        render_diagnostic(diag)
    assert _marked(cap.get()) == [7]


def test_render_diagnostic_multi_line_block_keeps_its_content_lines() -> None:
    # The complement — a span genuinely covering several lines still marks them all. `nodes:`
    # opens on 3 and its block runs through 9, ending at 10:1 (`output:` at column 1). Line 10
    # is excluded; 3-9 are not.
    diag = Diagnostic(
        code="compose.load",
        message="bad node",
        frames=(
            DiagnosticFrame(
                label="flow.yaml",
                text=_SOURCE,
                range=SourceRange(
                    start_line=4,
                    start_col=3,
                    end_line=10,
                    end_col=1,
                ),
            ),
        ),
    )
    with err_console.capture() as cap:
        render_diagnostic(diag)
    assert _marked(cap.get()) == [4, 5, 6, 7, 8, 9]


def test_render_diagnostic_renders_ordered_frames_and_context_notes() -> None:
    """Resolved frames retain tuple order and both note channels stay visible."""
    child_text = "nodes:\n  leaf:\n    kind: code\n"
    diag = Diagnostic(
        code="node.execute",
        message="leaf failed",
        frames=(
            DiagnosticFrame(
                label="root.yaml",
                text=_SOURCE,
                range=SourceRange(4, 3, 4, 8),
                note="while invoking child",
            ),
            DiagnosticFrame(
                label="child.yaml",
                text=child_text,
                range=SourceRange(2, 3, 2, 8),
                note="inside leaf",
            ),
        ),
        notes=("check the child input",),
    )
    with err_console.capture() as cap:
        render_diagnostic(diag)
    out = cap.get()
    assert "source locations:" in out
    assert out.index("root.yaml:4") < out.index("child.yaml:2") < out.index("leaf failed")
    assert "while invoking child" in out
    assert "inside leaf" in out
    assert "check the child input" in out
    assert _marked(out) == [4, 2]


def test_render_diagnostic_unlocated_message_needs_no_source_context() -> None:
    """A source-free diagnostic preserves its message without inventing a label."""
    diag = Diagnostic(code="node.execute", message="unchanged failure message")

    with err_console.capture() as cap:
        render_diagnostic(diag)

    assert cap.get().strip() == "unchanged failure message"
