"""`ac run` interactive input prompt: a "Running flow" banner + per-input metadata labels.

At run start the CLI prints a `_flow_banner` (name/description/version); each prompt's
label carries the input's declared `type`, a required (`*`) / `optional` mark, and any
default (`_input_label`). `_prompt_missing` is exercised with a fake `questionary` so the
labels it builds are observable without a TTY. A runtime failure (`_render_run_error`)
boxes the `.yaml` at the node that raised, mirroring the compile-error frame.
"""

import sys
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

from rich.console import Console

import agent_composer.cli.run as climod
from agent_composer.cli.run import (
    _flow_banner,
    _input_label,
    _prompt_missing,
    _render_run_error,
)
from agent_composer.compose.loader import load_flow
from agent_composer.run import RunResult, resume_command, resume_flow, run_flow
from agent_composer.compose.types import read_flow_inputs

_SEEDS = Path(__file__).resolve().parents[2] / "tests" / "seeds"
_ERRORS = _SEEDS / "errors"


def _decls():
    return read_flow_inputs(
        {
            "topic": "str",
            "as_of": "Optional[date]",
            "window": "int = 30",
            "mode": 'Literal["fast", "slow"]',
        },
        {},
    )


# --- _flow_banner: the "Running flow" identity block ------------------------- #
def _plain(text) -> str:
    """Render a Rich Text to a plain (un-styled) string for substring assertions."""
    buf = StringIO()
    Console(file=buf, force_terminal=False, width=100).print(text)
    return buf.getvalue()


def test_flow_banner_name_and_description():
    out = _plain(_flow_banner("my_flow", "Does a useful thing."))
    assert "Running flow: my_flow" in out
    assert "Description: Does a useful thing." in out


def test_flow_banner_includes_version():
    out = _plain(_flow_banner("f", "d", version="v2"))
    assert "version: v2" in out


def test_flow_banner_none_when_no_metadata():
    assert _flow_banner(None, None, None) is None


# --- _input_label: name + type + required/optional/default -------------------- #
def test_input_label_required():
    topic = next(d for d in _decls() if d.name == "topic")
    assert _input_label(topic) == "topic (str) *"


def test_input_label_optional_no_default():
    as_of = next(d for d in _decls() if d.name == "as_of")
    assert _input_label(as_of) == "as_of (Optional[date]) [optional] e.g. 2026-05-21"


def test_input_label_optional_with_default():
    window = next(d for d in _decls() if d.name == "window")
    assert _input_label(window) == "window (int) [default: 30]"


def test_input_label_datetime_example():
    decls = read_flow_inputs({"at": "datetime"}, {})
    at = next(d for d in decls if d.name == "at")
    assert _input_label(at) == "at (datetime) * e.g. 2026-05-21T14:30"


def test_input_label_default_suppresses_example():
    # A date default already shows the accepted format, so no redundant `e.g.`.
    decls = read_flow_inputs({"as_of": "Optional[date] = 2026-01-01"}, {})
    as_of = next(d for d in decls if d.name == "as_of")
    assert _input_label(as_of) == "as_of (Optional[date]) [default: 2026-01-01]"


# --- _prompt_missing: header + labels via a fake questionary ------------------ #
class _FakeQuestionary:
    """Records the labels it is asked with; answers from `answers` (substr -> value)."""

    def __init__(self, answers):
        self.labels: list[str] = []
        self._answers = answers

    def Style(self, spec):  # questionary.Style(...) — styling is irrelevant to the labels
        return None

    def _resolve(self, label):
        self.labels.append(label)
        for key, val in self._answers.items():
            if key in label:
                return val
        return ""

    def text(self, label, default="", **kwargs):
        val = self._resolve(label)
        return SimpleNamespace(ask=lambda: val)

    def confirm(self, label, default=False, **kwargs):
        val = self._resolve(label)
        return SimpleNamespace(ask=lambda: val)

    def select(self, label, choices, **kwargs):
        val = self._resolve(label)
        return SimpleNamespace(ask=lambda: val)


def _sink(monkeypatch) -> StringIO:
    buf = StringIO()
    monkeypatch.setattr(climod, "err_console", Console(file=buf, force_terminal=False, width=100))
    return buf


def test_prompt_labels_carry_marks(monkeypatch):
    fake = _FakeQuestionary({"topic": "ACME", "mode": "fast"})
    monkeypatch.setitem(sys.modules, "questionary", fake)
    _sink(monkeypatch)
    _prompt_missing(_decls(), {})
    joined = " || ".join(fake.labels)
    assert "topic (str) *" in joined
    assert "as_of (Optional[date]) [optional]" in joined
    assert "window (int) [default: 30]" in joined


def test_prompt_skips_already_supplied(monkeypatch):
    # Everything supplied -> nothing to prompt -> no prompt issued, empty gathered.
    fake = _FakeQuestionary({})
    monkeypatch.setitem(sys.modules, "questionary", fake)
    _sink(monkeypatch)
    have = {"topic": "x", "as_of": "2026-01-01", "window": 5, "mode": "fast"}
    gathered = _prompt_missing(_decls(), have)
    assert gathered == {}
    assert fake.labels == []          # no prompt issued


# --- _render_run_error: a runtime failure boxes the .yaml at the PRECISE line ------ #
def test_run_error_boxes_failing_node(monkeypatch):
    # e07 omits `as_of` -> the `report` node's `:?` fires at bind time -> NodeFailed
    # carrying an input locator -> the box points at the `as_of:` binding, not the header.
    flow = _ERRORS / "e07-required-missing.yaml"
    text = flow.read_text()
    result = run_flow(load_flow(text, search_paths=[flow.parent]), {"topic": "X"})
    assert result.status == "failed"
    buf = _sink(monkeypatch)
    _render_run_error(result, flow, text)
    out = buf.getvalue()
    assert "e07-required-missing.yaml:23" in out   # the as_of binding, not the node header
    assert "report" in out                         # still inside the source window
    assert "as_of is required for the report" in out


def test_run_error_kind_fallback_for_code_raise(monkeypatch):
    # e20's code callable raises -> NodeFailed with no precise locator -> the chain falls
    # back to the node-kind best sub-line: a code node's `code:` field (not the header, not plain).
    flow = _ERRORS / "e20-code-raises.yaml"
    text = flow.read_text()
    result = run_flow(load_flow(text, search_paths=[flow.parent]), {"topic": "X"})
    assert result.status == "failed"
    buf = _sink(monkeypatch)
    _render_run_error(result, flow, text)
    out = buf.getvalue()
    assert "╭" in out and "code:" in out           # boxed at the `code:` field


def test_run_error_boxes_boundary_assert_line(monkeypatch):
    # e18's boundary assert fails before any node runs; in phase 2 the RunFailed carries a
    # flow-level assert locator -> the failure is now BOXED at the `asserts:` line (was plain).
    flow = _ERRORS / "e18-false-boundary-assert.yaml"
    text = flow.read_text()
    result = run_flow(load_flow(text, search_paths=[flow.parent]), {"window": -5})
    assert result.status == "failed"
    buf = _sink(monkeypatch)
    _render_run_error(result, flow, text)
    out = buf.getvalue()
    assert "e18-false-boundary-assert.yaml:15" in out   # the `${input.window} > 0` line
    assert "assert failed" in out


def test_run_error_boxes_code_wrong_type_output(monkeypatch):
    # e21's code node returns a value that fails its declared `output: int` -> the typed write
    # boundary rejects it. The node-less RunFailed carries a `field` locator -> BOXED at the
    # node's `output:` declaration (not a plain message, not the node header).
    flow = _ERRORS / "e21-code-wrong-type.yaml"
    text = flow.read_text()
    result = run_flow(load_flow(text, search_paths=[flow.parent]), {"topic": "X"})
    assert result.status == "failed"
    buf = _sink(monkeypatch)
    _render_run_error(result, flow, text)
    out = buf.getvalue()
    assert "e21-code-wrong-type.yaml:12" in out   # the `output: int` field line
    assert "╭" in out


def test_run_error_boxes_input_decl_line(monkeypatch):
    # e08's `window` can't coerce to int -> input-coercion failure carries an input_decl
    # locator -> boxed at the `window:` declaration line.
    flow = _ERRORS / "e08-input-type-mismatch.yaml"
    text = flow.read_text()
    result = run_flow(load_flow(text, search_paths=[flow.parent]), {"topic": "X", "window": "soon"})
    assert result.status == "failed"
    buf = _sink(monkeypatch)
    _render_run_error(result, flow, text)
    out = buf.getvalue()
    assert "e08-input-type-mismatch.yaml:20" in out     # the `window: int` decl line
    assert "╭" in out


def test_run_error_plain_when_unlocatable(monkeypatch):
    # A failure with no NodeFailed AND no locator (e.g. a terminal-skipped run) has nowhere
    # to point -> the plain `run <status>: <message>` line, no frame.
    flow = _ERRORS / "e18-false-boundary-assert.yaml"
    text = flow.read_text()
    result = RunResult(input={}, status="failed", error="boom", locator=None, events=[])
    buf = _sink(monkeypatch)
    _render_run_error(result, flow, text)
    out = buf.getvalue()
    assert "run failed:" in out
    assert "╭" not in out                          # no boxed frame


def test_run_error_boxes_namespaced_failure_at_owning_call(monkeypatch):
    # e24's code node raises INSIDE a called child, so its NodeFailed id is runtime-namespaced
    # (`run/boom`) — a line the parser never indexes. The renderer must fall back to the OWNING
    # top-level call node (`run`) and still box a real source frame, not degrade to a bare line.
    flow = _ERRORS / "e24-nested-code-raise.yaml"
    text = flow.read_text()
    result = run_flow(load_flow(text, search_paths=[flow.parent]), {"topic": "X"})
    assert result.status == "failed"
    buf = _sink(monkeypatch)
    _render_run_error(result, flow, text)
    out = buf.getvalue()
    assert "e24-nested-code-raise.yaml:27" in out   # the owning `run:` call node line
    assert "╭" in out                               # boxed, not a plain `run failed:` line


def test_run_error_boxes_loop_max_exceeded_at_loop_node(monkeypatch):
    # The loop runaway guard raises `LoopMaxExceeded` from a driver CLONE whose id carries a
    # `~<k>` suffix (`polish~5`) — a line the parser never indexes and which has no locator. The
    # renderer must strip the clone suffix, resolve to the authored `polish:` loop node, and box a
    # real source frame instead of degrading to a bare `run failed:` line.
    from agent_composer.events import NodeFailed

    flow = _SEEDS / "28-refine-loop.yaml"
    text = flow.read_text()
    nf = NodeFailed("polish~5", error="loop 'polish' exceeded max (5)",
                    error_type="LoopMaxExceeded", locator=None)
    result = RunResult(input={}, status="failed",
                       error="loop 'polish' exceeded max (5)", locator=None, events=[nf])
    buf = _sink(monkeypatch)
    _render_run_error(result, flow, text)
    out = buf.getvalue()
    assert "╭" in out                                # boxed, not a plain `run failed:` line
    assert "28-refine-loop.yaml:45" in out           # resolved to the authored `polish:` loop node
    assert "exceeded max (5)" in out


def test_run_error_traceback_only_under_engine_trace(monkeypatch):
    # The captured Python traceback (a code/tool/agent raise) is surfaced ONLY when
    # `engine_trace=True`; the default terse display omits it.
    flow = _ERRORS / "e24-nested-code-raise.yaml"
    text = flow.read_text()
    result = run_flow(load_flow(text, search_paths=[flow.parent]), {"topic": "X"})
    assert result.traceback                          # captured at the node-failure boundary

    buf = _sink(monkeypatch)
    _render_run_error(result, flow, text, engine_trace=False)
    assert "python traceback" not in buf.getvalue()

    buf = _sink(monkeypatch)
    _render_run_error(result, flow, text, engine_trace=True)
    out = buf.getvalue()
    assert "python traceback" in out
    assert "intentional CODE failure" in out         # the originating Python exception


# --- _render_run_error: a MULTI-FRAME call traceback into the called child(ren) ----- #
def test_run_error_multi_frame_stack_into_def(monkeypatch):
    # e24's code raises inside a called `defs:` child (`run/boom`). With the loaded IR the
    # renderer descends ONE level: a stack of two boxed frames — the top-level `run:` call
    # node in this file, then the failing `boom` node's `code:` line in `defs:inner`.
    flow = _ERRORS / "e24-nested-code-raise.yaml"
    text = flow.read_text()
    loaded = load_flow(text, search_paths=[flow.parent])
    result = run_flow(loaded, {"topic": "X"})
    assert result.status == "failed"
    buf = _sink(monkeypatch)
    _render_run_error(result, flow, text, loaded=loaded)
    out = buf.getvalue()
    assert "call traceback (most recent call last):" in out
    assert "e24-nested-code-raise.yaml:27" in out   # frame 0: the owning `run:` call node (header)
    assert "e24-nested-code-raise.yaml defs:inner:20" in out  # frame 1: filename-qualified def
    assert "intentional CODE failure" in out


def test_run_error_multi_frame_stack_into_external(monkeypatch):
    # e25's code raises inside an EXTERNAL `uses:` flow (`go/kaboom`). The descent crosses the
    # file boundary: frame 0 the `go:` call node in this file, frame 1 the failing node boxed
    # in lib_boom.yaml (the frame label is THAT external filename, not this seed's).
    flow = _ERRORS / "e25-external-raise.yaml"
    text = flow.read_text()
    loaded = load_flow(text, search_paths=[flow.parent])
    result = run_flow(loaded, {"topic": "X"})
    assert result.status == "failed"
    buf = _sink(monkeypatch)
    _render_run_error(result, flow, text, loaded=loaded)
    out = buf.getvalue()
    assert "call traceback (most recent call last):" in out
    assert "e25-external-raise.yaml:19" in out      # frame 0: the `go:` call node (header)
    assert "lib_boom.yaml:18" in out                # frame 1: the external file's `code:` line


def test_run_error_multi_frame_three_levels(monkeypatch):
    # e26 raises THREE levels deep (`outer/via/boom`): call -> def `middle` -> def `deep`.
    # The stack must box all three frames, most-recent-call-last.
    flow = _ERRORS / "e26-three-level-raise.yaml"
    text = flow.read_text()
    loaded = load_flow(text, search_paths=[flow.parent])
    result = run_flow(loaded, {"topic": "X"})
    assert result.status == "failed"
    buf = _sink(monkeypatch)
    _render_run_error(result, flow, text, loaded=loaded)
    out = buf.getvalue()
    assert "call traceback (most recent call last):" in out
    assert "e26-three-level-raise.yaml:37" in out   # frame 0: the top `outer:` call node (header)
    assert "e26-three-level-raise.yaml defs:middle:29" in out  # frame 1: `via:` call in `middle` (header)
    assert "e26-three-level-raise.yaml defs:deep:20" in out    # frame 2: `boom` code in `deep`


def test_run_error_multi_frame_on_nested_resume_failure(monkeypatch):
    # Seed 25 suspends on a HUMAN_INPUT inside a called `defs:` child (`gate/approve`). Resuming
    # with an INVALID answer fails the typed write boundary with a `field` locator. The traceback
    # descends into the def and points at the `output:` line (precise field locator), not the
    # node header — frame 0 the `gate:` call, frame 1 `defs:review` `output:`.
    flow = _SEEDS / "25-nested-suspension.yaml"
    text = flow.read_text()
    loaded = load_flow(text, search_paths=[flow.parent])
    paused = run_flow(loaded, {"action": "ship it"})
    assert paused.status == "paused"
    cmds = [resume_command(loaded, rs, "maybe") for rs in paused.pause_reasons]
    result = resume_flow(loaded, engine=paused.engine, commands=cmds)
    assert result.status == "failed"
    buf = _sink(monkeypatch)
    _render_run_error(result, flow, text, loaded=loaded)
    out = buf.getvalue()
    assert "call traceback (most recent call last):" in out
    assert "25-nested-suspension.yaml:41" in out    # frame 0: the `gate:` call node (header)
    assert "25-nested-suspension.yaml defs:review:31" in out  # frame 1: precise `output:` line
    assert "is not a member of variant" in out


def test_run_error_single_frame_when_loaded_absent(monkeypatch):
    # The multi-frame stack is gated on the loaded IR being available: without it the renderer
    # can't descend, so a namespaced failure falls back to the single owning-call frame (no
    # "call traceback" header). Guards that the new path is opt-in, not always-on.
    flow = _ERRORS / "e24-nested-code-raise.yaml"
    text = flow.read_text()
    result = run_flow(load_flow(text, search_paths=[flow.parent]), {"topic": "X"})
    assert result.status == "failed"
    buf = _sink(monkeypatch)
    _render_run_error(result, flow, text)            # no loaded= -> single frame
    out = buf.getvalue()
    assert "call traceback (most recent call last):" not in out
    assert "e24-nested-code-raise.yaml:27" in out    # the owning call node, single box


# --- _ensure_cwd_importable: a flow's `code:` refs to repo-local packages resolve ---- #
def test_ensure_cwd_importable_prepends_cwd(monkeypatch, tmp_path):
    # Without the repo root on sys.path a flow's `code: tests.seeds.fns:fn` fails to import;
    # the CLI prepends the cwd so such refs resolve as `python script.py` would. Run from a
    # dir NOT already on the path and assert it lands at sys.path[0], idempotently.
    import agent_composer.cli.run as cli

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", [p for p in sys.path if p != str(tmp_path)])
    cli._ensure_cwd_importable()
    assert sys.path[0] == str(tmp_path)
    cli._ensure_cwd_importable()                      # idempotent: no duplicate entry
    assert sys.path.count(str(tmp_path)) == 1


