"""Tests for `render_diagnostic` — the single Step-8 CLI error renderer.

`render_diagnostic` funnels every compile/run error through one path: it boxes the
`Diagnostic.primary` `SourceRange` in the flow source and prints the message, with any
`notes` (the outer call-traceback frames of a nested `call`/`map` failure, innermost last)
stacked above under a "call traceback" header. These tests capture `err_console` output and
assert the boxed line + message (single-range case) and the stacked header + notes + primary
(notes case).
"""

from __future__ import annotations

import re

from agent_composer.cli.run import err_console, render_diagnostic
from agent_composer.events import Diagnostic, SourceRange

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


def test_render_diagnostic_boxes_range():
    # A Diagnostic whose primary points at line 5 (`kind: code`) boxes that line and prints
    # the message.
    diag = Diagnostic(
        severity="error",
        message="bad thing happened",
        primary=SourceRange(start_line=5, start_col=5, end_line=5, end_col=15),
    )
    with err_console.capture() as cap:
        render_diagnostic(diag, _SOURCE, title="flow.yaml")
    out = cap.get()
    assert "bad thing happened" in out
    assert "kind: code" in out  # line 5 is inside the boxed source window
    assert _marked(out) == [5]  # ...and it is the line actually highlighted


def test_render_diagnostic_block_range_stops_before_the_next_field():
    # `input:`'s value is a block MAPPING (line 7), so PyYAML ends it at the start of the next
    # token — `output:` on line 8, column 5. The end is EXCLUSIVE and line 8 holds only
    # indentation before column 5, so the span covers line 7 alone. Reading `end_line` directly
    # would highlight `output: str`, a different field.
    diag = Diagnostic(
        severity="error",
        message="bad ref",
        primary=SourceRange(start_line=7, start_col=7, end_line=8, end_col=5),
    )
    with err_console.capture() as cap:
        render_diagnostic(diag, _SOURCE, title="flow.yaml")
    assert _marked(cap.get()) == [7]


def test_render_diagnostic_multi_line_block_keeps_its_content_lines():
    # The complement — a span genuinely covering several lines still marks them all. `nodes:`
    # opens on 3 and its block runs through 9, ending at 10:1 (`output:` at column 1). Line 10
    # is excluded; 3-9 are not.
    diag = Diagnostic(
        severity="error",
        message="bad node",
        primary=SourceRange(start_line=4, start_col=3, end_line=10, end_col=1),
    )
    with err_console.capture() as cap:
        render_diagnostic(diag, _SOURCE, title="flow.yaml")
    assert _marked(cap.get()) == [4, 5, 6, 7, 8, 9]


def test_render_diagnostic_renders_notes_as_stack():
    # Notes (outer call frames) render as a stacked "call traceback" header + the frames,
    # then the primary box + message below.
    diag = Diagnostic(
        severity="error",
        message="leaf failed",
        primary=SourceRange(start_line=9, start_col=5, end_line=9, end_col=40),
        notes=("in call gate (flow.yaml:4)", "in call inner (child.yaml:3)"),
    )
    with err_console.capture() as cap:
        render_diagnostic(diag, _SOURCE, title="flow.yaml")
    out = cap.get()
    assert "call traceback (most recent call last):" in out
    assert "in call gate (flow.yaml:4)" in out
    assert "in call inner (child.yaml:3)" in out
    assert "leaf failed" in out
