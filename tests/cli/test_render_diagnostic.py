"""Tests for `render_diagnostic` — the single Step-8 CLI error renderer.

`render_diagnostic` funnels every compile/run error through one path: it boxes the
`Diagnostic.primary` `SourceRange` in the flow source and prints the message, with any
`notes` (the outer call-traceback frames of a nested `call`/`map` failure, innermost last)
stacked above under a "call traceback" header. These tests capture `err_console` output and
assert the boxed line + message (single-range case) and the stacked header + notes + primary
(notes case).
"""

from __future__ import annotations

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
