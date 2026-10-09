"""Runtime CLI diagnostics preserve canonical nested source-frame ordering."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

from rich.console import Console

import yian.cli.run as climod
from yian.cli.run import _render_run_error
from yian.compose.loader import load_flow
from yian.run import RunResult, run_flow

_ERRORS = Path(__file__).resolve().parents[1] / "seeds" / "errors"

_NESTED_CALL_LOOP_MAXED = """
id: nested-call-loop-maxed
name: nested_call_loop_maxed
defs:
  keep:
    input:
      n: int
    nodes:
      step:
        kind: code
        code: 'return {"n": inputs["n"]}'
        input:
          n: ${input.n}
        output:
          n: int
    output: ${step.output}
  runaway:
    nodes:
      repeat:
        kind: loop
        call: keep
        input:
          n: 0
        while: "true"
        max: 1
    output: ${repeat.output}
nodes:
  outer:
    kind: call
    call: runaway
output: ${outer.output}
"""

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
    _render_run_error(result, loaded)
    return buf.getvalue(), result


def test_nested_call_error_frames_at_child_range(monkeypatch):
    # e24 raises in `run/boom` — canonical diagnosis resolves the call and its CODE
    # invocation directly from the ordered origin trace.
    flow = _ERRORS / "e24-nested-code-raise.yaml"
    text = flow.read_text()
    loaded = load_flow(text, search_paths=[flow.parent], source_label=flow.name)
    result = run_flow(loaded, {"topic": "X"})
    assert result.status == "failed"

    diagnostic = loaded.diagnose(result.failure)
    assert [frame.range.start_line for frame in diagnostic.frames] == [27, 20]

    buf = _sink(monkeypatch)
    _render_run_error(result, loaded)
    out = buf.getvalue()

    assert "source locations:" in out
    assert f"{flow.name}:27" in out
    assert f"{flow.name}:20" in out


def test_nested_call_input_failure_uses_resolved_child_range(monkeypatch):
    out, result = _render(_CALL_INPUT_FAILURE, monkeypatch)

    assert result.failure is not None
    assert result.failure.node_id == "invoke/leaf"
    assert "nested.yaml:17" in out
    assert "nested.yaml:12" in out
    assert "nested.yaml:14" not in out


def test_nested_map_input_failure_uses_resolved_child_range(monkeypatch):
    out, result = _render(_MAP_INPUT_FAILURE, monkeypatch)

    assert result.failure is not None
    assert result.failure.node_id == "invoke#0/leaf"
    assert "nested.yaml:17" in out
    assert "nested.yaml:12" in out
    assert "nested.yaml:14" not in out


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
    diagnostic = loaded.diagnose(result.failure)
    assert diagnostic.frames
    assert {frame.label for frame in diagnostic.frames} == {"parent.yaml"}
    assert_line = diagnostic.frames[-1].range.start_line

    buf = _sink(monkeypatch)
    _render_run_error(result, loaded)
    out = buf.getvalue()

    assert f"parent.yaml:{assert_line}" in out
    assert "echo_child.yaml:" not in out


def test_diagnosis_reresolves_origins_against_current_source(monkeypatch):
    """Rendering uses current source origins, not the old runtime coordinate."""
    old_loaded = load_flow(_CALL_INPUT_FAILURE, source_label="nested.yaml")
    result = run_flow(old_loaded, {})
    assert old_loaded.diagnose(result.failure).frames[-1].range.start_line == 12

    shifted_source = "\n" + _CALL_INPUT_FAILURE
    shifted_loaded = load_flow(shifted_source, source_label="nested.yaml")
    buf = _sink(monkeypatch)
    _render_run_error(result, shifted_loaded)
    out = buf.getvalue()

    assert "nested.yaml:18" in out
    assert "nested.yaml:13" in out
    assert "nested.yaml:12" not in out


def test_nested_caller_assert_range_stays_with_owning_call_frame(monkeypatch):
    source = _NESTED_CALLER_POST_ASSERT
    loaded = load_flow(source, source_label="nested-assert.yaml")
    result = run_flow(loaded, {})

    assert result.status == "failed"
    assert result.failure is not None
    assert result.failure.node_id == "outer/via/__end__"
    diagnostic = loaded.diagnose(result.failure)
    assert_line = diagnostic.frames[-1].range.start_line

    buf = _sink(monkeypatch)
    _render_run_error(result, loaded)
    out = buf.getvalue()

    assert [frame.range.start_line for frame in diagnostic.frames] == [28, 19, 25]
    assert f"nested-assert.yaml:{assert_line}" in out


def test_nested_loop_failure_renders_ordered_catalog_frames(monkeypatch):
    """A respawned LOOP id needs no clone parsing to locate its authored frames."""
    loaded = load_flow(_NESTED_CALL_LOOP_MAXED, source_label="nested-loop.yaml")
    result = run_flow(loaded, {})
    assert result.status == "failed"

    diagnostic = loaded.diagnose(result.failure)
    assert [frame.range.start_line for frame in diagnostic.frames] == [28, 19]

    buf = _sink(monkeypatch)
    _render_run_error(result, loaded)
    out = buf.getvalue()
    assert out.index("nested-loop.yaml:28") < out.index("nested-loop.yaml:19")
    assert "exceeded max (1)" in out
