"""The call-traceback stack (`_walk_call_frames`) reads its frame lines from `SourceRange`.

Step 8 re-plumbs `_walk_call_frames` off the retired `*_lines` maps and onto the range
metadata the span parser records: the top flow's frames come from the baked IR
(`node.range`/`node.field_ranges`), and each descended child's frames come from its
render-only `SourceFrame` (`node_ranges`/`field_ranges`). This test pins that contract —
the boxed frame line for a nested failure must equal the `SourceRange.start_line` the
child's `SourceFrame` carries, not a hard-coded lookup.
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path

from rich.console import Console

import agent_composer.cli.run as climod
from agent_composer.cli.run import _render_run_error
from agent_composer.compose.loader import load_flow
from agent_composer.run import RunResult, run_flow

_ERRORS = Path(__file__).resolve().parents[1] / "seeds" / "errors"

_CALL_INPUT_FAILURE = """id: nested-binding
name: nested_binding
defs:
  inner:
    input:
      maybe: Optional[str]
    nodes:
      leaf:
        kind: code
        input:
          okay: literal
          bad: ${input.maybe:?"missing nested value"}
        output: str
        code: tests.engine._compose_codefns:echo
    output: ${leaf.output}
nodes:
  invoke:
    kind: call
    call: inner
    input: {}
output: ${invoke.output}
"""

_MAP_INPUT_FAILURE = """id: nested-map-binding
name: nested_map_binding
defs:
  inner:
    input:
      maybe: Optional[str]
    nodes:
      leaf:
        kind: code
        input:
          okay: literal
          bad: ${input.maybe:?"missing mapped value"}
        output: str
        code: tests.engine._compose_codefns:echo
    output: ${leaf.output}
nodes:
  invoke:
    kind: map
    call: inner
    over: ${[1]}
    input:
      maybe: null
output: ${invoke.output}
"""


def _sink(monkeypatch) -> StringIO:
    buf = StringIO()
    monkeypatch.setattr(
        climod, "err_console", Console(file=buf, force_terminal=False, width=100)
    )
    return buf


def _render(source: str, monkeypatch) -> tuple[str, RunResult]:
    """Run and render one in-file nested failure against its retained source."""
    flow = Path("nested.yaml")
    loaded = load_flow(source, source_label=flow.name)
    result = run_flow(loaded, {})
    assert result.status == "failed"

    buf = _sink(monkeypatch)
    _render_run_error(result, flow, source, loaded=loaded)
    return buf.getvalue(), result


def test_nested_call_error_frames_at_child_range(monkeypatch):
    # e24 raises in `run/boom` — a CODE node inside a called `defs:inner` child. The stack
    # descends one level: frame 0 the top `run:` call node, frame 1 the failing `boom` node
    # boxed at its `code:` line in `defs:inner`. Both frame lines must come from the recorded
    # SourceRanges (the child's `SourceFrame`), which is the migration's contract.
    flow = _ERRORS / "e24-nested-code-raise.yaml"
    text = flow.read_text()
    loaded = load_flow(text, search_paths=[flow.parent])
    result = run_flow(loaded, {"topic": "X"})
    assert result.status == "failed"

    frame = loaded.compiled.nodes["run"].child_source
    boom_code_line = frame.field_ranges["boom"][("field", "code")].start_line

    buf = _sink(monkeypatch)
    _render_run_error(result, flow, text, loaded=loaded)
    out = buf.getvalue()

    # A stacked traceback with at least two frames, the deepest boxed at the child's
    # recorded `code:` range start line (not a top-level line).
    assert "call traceback (most recent call last):" in out
    assert f"defs:inner:{boom_code_line}" in out
    assert boom_code_line == 20


def test_nested_call_input_failure_uses_resolved_child_range(monkeypatch):
    out, result = _render(_CALL_INPUT_FAILURE, monkeypatch)

    assert result.failure is not None
    assert result.failure.node_id == "invoke/leaf"
    assert result.locator is not None
    assert (result.locator.kind, result.locator.key) == ("input", "bad")
    assert result.range is not None and result.range.start_line == 12
    assert "nested.yaml:17" in out
    assert "nested.yaml defs:inner:12" in out
    assert "nested.yaml defs:inner:14" not in out


def test_nested_map_input_failure_uses_resolved_child_range(monkeypatch):
    out, result = _render(_MAP_INPUT_FAILURE, monkeypatch)

    assert result.failure is not None
    assert result.failure.node_id == "invoke#0/leaf"
    assert result.locator is not None
    assert (result.locator.kind, result.locator.key) == ("input", "bad")
    assert result.range is not None and result.range.start_line == 12
    assert "nested.yaml:17" in out
    assert "nested.yaml defs:inner:12" in out
    assert "nested.yaml defs:inner:14" not in out
