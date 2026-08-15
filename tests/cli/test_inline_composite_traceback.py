"""Runtime tracebacks for failures inside authored inline composite bodies."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

from rich.console import Console

import agent_composer.cli.run as climod
from agent_composer.cli.run import _render_run_error
from agent_composer.compose.desugar.inline_bodies import INLINE_FLOW_PREFIX
from agent_composer.compose.loader import load_flow
from agent_composer.run import run_flow


def _sink(monkeypatch) -> StringIO:
    """Capture CLI diagnostics without terminal control sequences."""
    buf = StringIO()
    monkeypatch.setattr(
        climod, "err_console", Console(file=buf, force_terminal=False, width=100)
    )
    return buf


def test_inline_call_runtime_failure_uses_authored_friendly_frames(monkeypatch):
    source = """id: inline-composite-traceback
name: inline_composite_traceback
nodes:
  wrapper:
    kind: call
    nodes:
      boom:
        kind: code
        code: |
          raise RuntimeError("inline child boom")
          return "unreachable"
        output: str
    output: ${boom.output}
output: ${wrapper.output}
"""
    flow = Path("inline-composite.yaml")
    loaded = load_flow(source)
    result = run_flow(loaded, {})

    assert result.status == "failed"

    wrapper = loaded.compiled.nodes["wrapper"]
    child_source = wrapper.child_source
    wrapper_line = loaded.source.node_ranges["wrapper"].start_line
    code_line = child_source.field_ranges["boom"][("field", "code")].start_line

    buf = _sink(monkeypatch)
    _render_run_error(result, flow, source, loaded=loaded)
    output = buf.getvalue()

    outer_frame = f"{flow.name}:{wrapper_line}"
    inner_frame = f"inline body of node 'wrapper':{code_line}"
    assert "call traceback (most recent call last):" in output
    assert outer_frame in output
    assert inner_frame in output
    assert output.index(outer_frame) < output.index(inner_frame)
    assert "inline child boom" in output
    assert INLINE_FLOW_PREFIX not in output
    assert wrapper.flow_id not in output
