"""`yian run` compile-error rendering: a failed `load_flow` points at the `.yaml`, not the engine.

By default a `LoadError` is shown as a boxed source frame — a Rich panel titled `file:line`
with the `.yaml` source (line numbers, the offending line highlighted) and the message below,
like a Python traceback's code box — and the engine Python traceback is suppressed.
`--engine-trace` adds the traceback back for engine debugging.
"""

from io import StringIO

import typer
from rich.console import Console
from typer.testing import CliRunner

import yian.cli.run as climod
from yian.cli.run import _render_load_error
from yian.compose.helpers.errors import LoadDiagnostics
from yian.diagnostics import SourceRange


def _sink(monkeypatch) -> StringIO:
    """Redirect the module's stderr console to an in-memory non-terminal sink."""
    buf = StringIO()
    monkeypatch.setattr(climod, "err_console", Console(file=buf, force_terminal=False, width=100))
    return buf


# --- _render_load_error: boxed, located output ------------------------------- #
def test_render_located_error(monkeypatch):
    buf = _sink(monkeypatch)
    text = "id: f\nname: f\nnodes:\n  b:\n    kind: agent\n"
    error = LoadDiagnostics("flow.yaml", text).error(
        "node 'b' is broken",
        range=SourceRange(4, 3, 4, 5),
    )
    _render_load_error(error)
    out = buf.getvalue()
    assert "flow.yaml:4" in out          # panel title carries file:line
    assert "b:" in out                   # the source frame echoes the YAML
    assert "4" in out                    # line numbers are shown
    assert "node 'b' is broken" in out   # the message is printed


def test_render_unlocated_error_has_no_frame(monkeypatch):
    buf = _sink(monkeypatch)
    _render_load_error(LoadDiagnostics().error("global problem"))
    out = buf.getvalue()
    assert "flow.yaml" not in out
    assert "global problem" in out


def test_render_out_of_range_line_falls_back(monkeypatch):
    buf = _sink(monkeypatch)
    # A line past EOF must not crash; degrade to the message-only form.
    error = LoadDiagnostics("flow.yaml", "id: f\n").error(
        "weird",
        range=SourceRange(999, 1, 999, 2),
    )
    _render_load_error(error)
    assert "weird" in buf.getvalue()


def test_render_multi_line_error_spans_and_titles_all(monkeypatch):
    buf = _sink(monkeypatch)
    text = "\n".join(f"line {i}" for i in range(1, 21)) + "\n"
    diagnostics = LoadDiagnostics("flow.yaml", text)
    frames = tuple(
        diagnostics.frame(SourceRange(line, 1, line, 7))
        for line in (4, 12)
    )
    assert all(frame is not None for frame in frames)
    _render_load_error(diagnostics.error("a cycle", frames=frames))
    out = buf.getvalue()
    assert "source locations:" in out
    assert "flow.yaml:4" in out
    assert "flow.yaml:12" in out
    assert "line 4" in out           # window covers the first mark
    assert "line 12" in out


# --- CLI integration: a real broken flow ------------------------------------- #
_BROKEN = (
    "id: f\nname: f\nnodes:\n"
    "  a:\n    kind: agent\n    prompt: hi\n"
    "  b:\n    kind: agent\n    input:\n      brief: ${frame_typo.output}\n    prompt: use ${brief}\n"
    "output: ${b.output}\n"
)


def _app():
    app = typer.Typer()
    app.command()(climod.run)
    return app


def test_cli_compile_error_is_located_not_a_traceback(monkeypatch, tmp_path):
    buf = _sink(monkeypatch)
    f = tmp_path / "flow.yaml"
    f.write_text(_BROKEN)
    res = CliRunner().invoke(_app(), [str(f)])
    assert res.exit_code == 1
    out = buf.getvalue()
    assert "flow.yaml:" in out
    assert "frame_typo" in out                 # the offending ref is named
    assert "Traceback (most recent call last)" not in out  # no engine traceback by default


def test_cli_engine_trace_adds_traceback(monkeypatch, tmp_path):
    buf = _sink(monkeypatch)
    f = tmp_path / "flow.yaml"
    f.write_text(_BROKEN)
    res = CliRunner().invoke(_app(), [str(f), "--engine-trace"])
    assert res.exit_code == 1
    out = buf.getvalue()
    assert "flow.yaml:" in out                  # still located
    assert "Traceback" in out or "LoadError" in out  # plus the engine traceback


# A cycle a -> b -> a: each node implicates the other, so the error spans both lines.
_CYCLE = (
    "id: f\nname: f\nnodes:\n"
    "  a:\n    kind: agent\n    input:\n      x: ${b.output}\n    prompt: use ${x}\n"
    "  b:\n    kind: agent\n    input:\n      y: ${a.output}\n    prompt: use ${y}\n"
    "output: ${a.output}\n"
)


def test_cli_cycle_error_highlights_both_nodes(monkeypatch, tmp_path):
    buf = _sink(monkeypatch)
    f = tmp_path / "flow.yaml"
    f.write_text(_CYCLE)
    res = CliRunner().invoke(_app(), [str(f)])
    assert res.exit_code == 1
    out = buf.getvalue()
    # Node `a` is on line 4, node `b` on line 9; each is its own ordered source frame.
    assert "source locations:" in out
    assert "flow.yaml:4" in out
    assert "flow.yaml:9" in out
    assert "a:" in out and "b:" in out
    assert "cycle" in out
    # The "why" legend names each dependency edge that closes the loop.
    assert "a depends on b (a.input.x)" in out
    assert "b depends on a (b.input.y)" in out


def test_render_notes_printed_under_message(monkeypatch):
    buf = _sink(monkeypatch)
    diagnostics = LoadDiagnostics("flow.yaml", "id: f\nname: f\n")
    _render_load_error(
        diagnostics.error(
            "a cycle",
            range=SourceRange(1, 1, 1, 6),
            notes=("a depends on b (a.input.x)",),
        )
    )
    assert "a depends on b (a.input.x)" in buf.getvalue()

