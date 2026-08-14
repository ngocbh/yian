"""`ac run` per-node progress: ✓ on success, ✗ + error on failure, output when verbose.

The reporter is exercised through a non-terminal `Console` (a StringIO sink), which is the
same path the CLI takes off a TTY: no live spinner, just the final line per node. A thin
CliRunner test confirms the `--verbose` flag is wired and the run still succeeds.
"""

from io import StringIO

import typer
from rich.console import Console
from typer.testing import CliRunner

import agent_composer.cli.run as climod
from agent_composer.cli.run import _ProgressReporter
from agent_composer.compile.model import END_ID, START_ID
from agent_composer.run import RunResult
from agent_composer.events import NodeFailed, NodeRouted, NodeStarted, NodeSucceeded


def _sink_reporter(verbose: bool = False) -> tuple[_ProgressReporter, StringIO]:
    """A reporter writing to an in-memory non-terminal console (no spinner path)."""
    buf = StringIO()
    console = Console(file=buf, force_terminal=False, width=80)
    return _ProgressReporter(console, verbose=verbose), buf


def test_success_renders_check():
    reporter, buf = _sink_reporter()
    reporter.handle(NodeStarted("frame"))
    reporter.handle(NodeSucceeded("frame", output="ok"))
    out = buf.getvalue()
    assert "✓ frame" in out
    assert "✗" not in out


def test_failure_renders_cross_and_error():
    reporter, buf = _sink_reporter()
    reporter.handle(NodeStarted("verdict"))
    reporter.handle(NodeFailed("verdict", error="boom", error_type="ValueError"))
    out = buf.getvalue()
    assert "✗ verdict" in out
    assert "boom" in out


def test_routed_renders_check_with_handle():
    """A router (CASE) emits NodeRouted, not NodeSucceeded — the reporter must still
    clear the spinner and print a completion line naming the chosen handle."""
    reporter, buf = _sink_reporter()
    reporter.handle(NodeStarted("gate"))
    reporter.handle(NodeRouted("gate", handle="yes"))
    out = buf.getvalue()
    assert "✓ gate" in out
    assert "yes" in out
    assert "✗" not in out
    assert "gate" not in reporter._running


def test_verbose_prints_output():
    reporter, buf = _sink_reporter(verbose=True)
    reporter.handle(NodeStarted("frame"))
    reporter.handle(NodeSucceeded("frame", output="line one\nline two"))
    out = buf.getvalue()
    assert "✓ frame" in out
    assert "line one" in out
    assert "line two" in out


def test_non_verbose_omits_output():
    reporter, buf = _sink_reporter(verbose=False)
    reporter.handle(NodeStarted("frame"))
    reporter.handle(NodeSucceeded("frame", output="secret-payload"))
    assert "secret-payload" not in buf.getvalue()


def test_boundary_nodes_are_ignored():
    reporter, buf = _sink_reporter()
    for nid in (START_ID, END_ID):
        reporter.handle(NodeStarted(nid))
        reporter.handle(NodeSucceeded(nid, output="x"))
    assert buf.getvalue().strip() == ""


def test_namespaced_boundary_nodes_are_ignored():
    """A loop/map iteration's synthesized boundary nodes are namespaced
    (`polish#0/__start__`); they carry no authored meaning and must be skipped too,
    while a real body node in the same namespace (`polish#0/critic`) still renders."""
    reporter, buf = _sink_reporter()
    for nid in ("polish#0/__start__", "polish#0/__end__"):
        reporter.handle(NodeStarted(nid))
        reporter.handle(NodeSucceeded(nid, output="x"))
    reporter.handle(NodeStarted("polish#0/critic"))
    reporter.handle(NodeSucceeded("polish#0/critic", output="x"))
    out = buf.getvalue()
    assert "__start__" not in out
    assert "__end__" not in out
    assert "✓ polish#0/critic" in out


def test_expanded_clears_spinner_without_check():
    """A spawner (loop driver continue, call/map splice) emits NodeExpanded, not
    NodeSucceeded — it has no terminal of its own. The reporter must clear its spinner
    (remove it from `_running`) and NOT print a ✓/✗ line, else the driver clones would
    spin forever in the live region."""
    from agent_composer.events import NodeExpanded

    reporter, buf = _sink_reporter()
    reporter.handle(NodeStarted("polish~1"))
    assert "polish~1" in reporter._running
    reporter.handle(NodeExpanded("polish~1"))
    assert "polish~1" not in reporter._running
    out = buf.getvalue()
    assert "✓" not in out
    assert "✗" not in out


_FLOW = "id: f\nname: f\nnodes:\n  a: {kind: agent, prompt: hi}\noutput: ${a.output}\n"


def _app():
    app = typer.Typer()
    app.command()(climod.run)
    return app


def test_cli_accepts_verbose_flag(monkeypatch, tmp_path):
    """`--verbose` is a recognized flag and the run drives to completion."""

    def fake_run_flow(loaded, supplied, *, on_event=None, llm_config=None, **kw):
        if on_event is not None:
            on_event(NodeStarted("a"))
            on_event(NodeSucceeded("a", output="hello"))
        return RunResult(
            input={},
            status="succeeded",
            run_id="cli-test",
            log=None,
            output="hello",
        )

    monkeypatch.setattr(climod, "_run_flow_controlled", fake_run_flow)
    monkeypatch.setattr(climod, "_ensure_provider_keys", lambda *a, **k: None)
    f = tmp_path / "f.yaml"
    f.write_text(_FLOW)
    res = CliRunner().invoke(_app(), [str(f), "--verbose"])
    assert res.exit_code == 0


def test_cli_quiet_suppresses_events(monkeypatch, tmp_path):
    """`--quiet` passes on_event=None so no node events are streamed."""
    seen = {}

    def fake_run_flow(loaded, supplied, *, on_event=None, llm_config=None, **kw):
        seen["on_event"] = on_event
        return RunResult(
            input={},
            status="succeeded",
            run_id="cli-test",
            log=None,
            output="hello",
        )

    monkeypatch.setattr(climod, "_run_flow_controlled", fake_run_flow)
    monkeypatch.setattr(climod, "_ensure_provider_keys", lambda *a, **k: None)
    f = tmp_path / "f.yaml"
    f.write_text(_FLOW)
    res = CliRunner().invoke(_app(), [str(f), "--quiet"])
    assert res.exit_code == 0
    assert seen["on_event"] is None


def test_cli_num_workers_threads_through(monkeypatch, tmp_path):
    """`--num-workers N` is forwarded to run_flow; default is 0 (serial)."""
    seen = {}

    def fake_run_flow(loaded, supplied, *, on_event=None, llm_config=None, num_workers=0, **kw):
        seen["num_workers"] = num_workers
        return RunResult(
            input={},
            status="succeeded",
            run_id="cli-test",
            log=None,
            output="hello",
        )

    monkeypatch.setattr(climod, "_run_flow_controlled", fake_run_flow)
    monkeypatch.setattr(climod, "_ensure_provider_keys", lambda *a, **k: None)
    f = tmp_path / "f.yaml"
    f.write_text(_FLOW)

    assert CliRunner().invoke(_app(), [str(f)]).exit_code == 0
    assert seen["num_workers"] == 0
    assert CliRunner().invoke(_app(), [str(f), "--num-workers", "4"]).exit_code == 0
    assert seen["num_workers"] == 4
    assert CliRunner().invoke(_app(), [str(f), "-w", "2"]).exit_code == 0
    assert seen["num_workers"] == 2


def test_cli_rejects_negative_workers(monkeypatch, tmp_path):
    """`--num-workers` is clamped at the CLI boundary: a negative value is rejected."""
    monkeypatch.setattr(
        climod, "_run_flow_controlled",
        lambda *a, **k: RunResult(
            input={},
            status="succeeded",
            run_id="cli-test",
            log=None,
            output="x",
        ),
    )
    f = tmp_path / "f.yaml"
    f.write_text(_FLOW)
    assert CliRunner().invoke(_app(), [str(f), "--num-workers", "-1"]).exit_code != 0
