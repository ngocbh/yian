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
from agent_composer.cli.run import _node_owns_resolved_range, _render_run_error
from agent_composer.compose.loader import load_flow
from agent_composer.diagnostics import SourceRange
from agent_composer.events import SourceSpan
from agent_composer.nodes.base import Node
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

_EXTERNAL_CHILD = """id: echo-child
name: echo_child
input:
  v: int
nodes:
  x:
    kind: code
    code: tests.engine._compose_codefns:double
    input:
      n: ${input.v}
    output: int
output: ${x.output}
"""

_CALLER_POST_ASSERT = """id: out-parent
name: out_parent
uses:
  echo-child: echo-child
nodes:
  callee:
    kind: call
    call: echo-child
    input:
      v: 60
    asserts:
      - ${output} < 100
output: ${callee.output}
"""

_NESTED_CALLER_POST_ASSERT = """id: nested-assert-parent
name: nested_assert_parent
defs:
  deep:
    input:
      v: int
    nodes:
      x:
        kind: code
        code: tests.engine._compose_codefns:double
        input:
          n: ${input.v}
        output: int
    output: ${x.output}
  middle:
    input:
      v: int
    nodes:
      via:
        kind: call
        call: deep
        input:
          v: ${input.v}
        asserts:
          - ${output} < 100
    output: ${via.output}
nodes:
  outer:
    kind: call
    call: middle
    input:
      v: 60
output: ${outer.output}
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


def test_assert_range_requires_an_aligned_assertion_site() -> None:
    """A coincident node-header range cannot own an unmatched assertion locator."""
    range_ = SourceRange(8, 3, 8, 9)
    node = Node("leaf")
    node.range = range_

    assert not _node_owns_resolved_range(
        node,
        SourceSpan("outer/leaf", "assert", "${output} > 0"),
        range_,
    )


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


def test_caller_assert_range_is_not_attached_to_external_child(monkeypatch):
    child = load_flow(_EXTERNAL_CHILD, source_label="echo_child.yaml")
    loaded = load_flow(
        _CALLER_POST_ASSERT,
        child_resolver=lambda _flow_id, _version: child,
        source_label="parent.yaml",
    )
    result = run_flow(loaded, {})

    assert result.status == "failed"
    assert result.failure is not None
    assert result.failure.node_id == "callee/__end__"
    assert result.locator is not None
    assert (result.locator.kind, result.locator.key) == (
        "assert",
        "${output} < 100",
    )
    assert result.range is not None

    buf = _sink(monkeypatch)
    _render_run_error(
        result,
        Path("parent.yaml"),
        _CALLER_POST_ASSERT,
        loaded=loaded,
    )
    out = buf.getvalue()

    assert f"parent.yaml:{result.range.start_line}" in out
    assert "echo_child.yaml:" not in out


def test_stale_child_range_falls_back_to_current_child_source(monkeypatch):
    old_loaded = load_flow(_CALL_INPUT_FAILURE, source_label="nested.yaml")
    result = run_flow(old_loaded, {})
    assert result.range is not None and result.range.start_line == 12

    shifted_source = "\n" + _CALL_INPUT_FAILURE
    shifted_loaded = load_flow(shifted_source, source_label="nested.yaml")
    buf = _sink(monkeypatch)
    _render_run_error(
        result,
        Path("nested.yaml"),
        shifted_source,
        loaded=shifted_loaded,
    )
    out = buf.getvalue()

    assert "nested.yaml:18" in out
    assert "nested.yaml defs:inner:15" in out
    assert "nested.yaml defs:inner:12" not in out


def test_nested_caller_assert_range_stays_with_owning_call_frame(monkeypatch):
    source = _NESTED_CALLER_POST_ASSERT
    loaded = load_flow(source, source_label="nested-assert.yaml")
    result = run_flow(loaded, {})

    assert result.status == "failed"
    assert result.failure is not None
    assert result.failure.node_id == "outer/via/__end__"
    assert result.range is not None
    middle = loaded.compiled.nodes["outer"].child
    assert_line = middle.nodes["via"].post_assert_ranges[0].start_line
    assert result.range.start_line == assert_line

    buf = _sink(monkeypatch)
    _render_run_error(
        result,
        Path("nested-assert.yaml"),
        source,
        loaded=loaded,
    )
    out = buf.getvalue()

    assert f"nested-assert.yaml defs:middle:{assert_line}" in out
    assert f"nested-assert.yaml defs:deep:{assert_line}" not in out
