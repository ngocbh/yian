"""`ac run` interactive input prompt: a "Running flow" banner + per-input metadata labels.

At run start the CLI prints a `_flow_banner` (name/description/version); each prompt's
label carries the input's declared `type`, a required (`*`) / `optional` mark, and any
default (`_input_label`). `_prompt_missing` is exercised with a fake `questionary` so the
labels it builds are observable without a TTY. A runtime failure (`_render_run_error`)
boxes the `.yaml` at the node that raised, mirroring the compile-error frame.
"""

import sys
from dataclasses import replace
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
from agent_composer.diagnostics import Diagnostic, Failure, OriginRef
from agent_composer.nodes.contracts import Parameter, ValueContract
from agent_composer.run import RunResult, run_flow
from agent_composer.tools import install_ask_user
from agent_composer.typesys import Type, ValueKind

_SEEDS = Path(__file__).resolve().parents[2] / "tests" / "seeds"
_ERRORS = _SEEDS / "errors"


def _parameter(
    name: str,
    typ: Type,
    *,
    required: bool = False,
    has_default: bool = False,
    default=None,
) -> Parameter:
    """Build one source-free canonical flow parameter for CLI presentation."""
    return Parameter(
        name=name,
        contract=ValueContract(typ),
        required=required,
        has_default=has_default,
        default=default,
        coerce_strings=True,
    )


def _decls() -> list[Parameter]:
    return [
        _parameter("topic", Type.scalar(ValueKind.STRING), required=True),
        _parameter("as_of", Type(kind=ValueKind.DATE, nullable=True)),
        _parameter(
            "window",
            Type.scalar(ValueKind.INTEGER),
            has_default=True,
            default=30,
        ),
        _parameter(
            "mode",
            Type(
                kind=ValueKind.STRING,
                tags=frozenset({"fast", "slow"}),
            ),
            required=True,
        ),
    ]


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
    at = _parameter("at", Type.scalar(ValueKind.DATETIME), required=True)
    assert _input_label(at) == "at (datetime) * e.g. 2026-05-21T14:30"


def test_input_label_default_suppresses_example():
    # A date default already shows the accepted format, so no redundant `e.g.`.
    as_of = _parameter(
        "as_of",
        Type(kind=ValueKind.DATE, nullable=True),
        has_default=True,
        default="2026-01-01",
    )
    assert _input_label(as_of) == "as_of (Optional[date]) [default: 2026-01-01]"


def test_input_label_derives_a_canonical_structural_type():
    profile = _parameter(
        "profile",
        Type(
            kind=ValueKind.OBJECT,
            fields={
                "z": Type.scalar(ValueKind.INTEGER),
                "a": Type(kind=ValueKind.STRING, nullable=True),
            },
            required=frozenset({"z"}),
        ),
        required=True,
    )

    assert _input_label(profile) == 'profile ({"a":Optional[str],"z":int}) *'


def test_input_label_displays_an_explicit_null_default():
    maybe = _parameter(
        "maybe",
        Type(kind=ValueKind.INTEGER, nullable=True),
        has_default=True,
        default=None,
    )

    assert _input_label(maybe) == "maybe (Optional[int]) [default: None]"


def test_input_label_supports_a_flat_list_contract():
    items = _parameter(
        "items",
        Type.scalar(ValueKind.LIST_STRING),
        required=True,
    )

    assert _input_label(items) == "items (list[str]) *"


# --- _prompt_missing: header + labels via a fake questionary ------------------ #
class _FakeQuestionary:
    """Records the labels it is asked with; answers from `answers` (substr -> value)."""

    def __init__(self, answers):
        self.labels: list[str] = []
        self.text_defaults: list[str] = []
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
        self.text_defaults.append(default)
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


def test_prompt_leaves_explicit_null_defaults_omitted_for_every_widget_kind(monkeypatch):
    fake = _FakeQuestionary({})
    monkeypatch.setitem(sys.modules, "questionary", fake)
    _sink(monkeypatch)
    parameters = [
        _parameter(
            "maybe",
            Type(kind=ValueKind.INTEGER, nullable=True),
            has_default=True,
            default=None,
        ),
        _parameter(
            "flag",
            Type(kind=ValueKind.BOOLEAN, nullable=True),
            has_default=True,
            default=None,
        ),
        _parameter(
            "mode",
            Type(
                kind=ValueKind.STRING,
                tags=frozenset({"fast", "slow"}),
                nullable=True,
            ),
            has_default=True,
            default=None,
        ),
    ]

    gathered = _prompt_missing(parameters, {})

    assert gathered == {}
    assert fake.text_defaults == ["", "", ""]


# --- _render_run_error: a runtime failure boxes the .yaml at the PRECISE line ------ #
def test_run_error_uses_loaded_canonical_diagnosis(monkeypatch):
    """The CLI renders the front end's Diagnostic without resolving source itself."""
    failure = Failure(
        "node.execute",
        "semantic failure",
        "work",
        trace=(OriginRef("site"),),
    )
    result = RunResult(
        input={},
        status="failed",
        run_id="cli-test",
        log=None,
        failure=failure,
    )
    seen = []

    class _Loaded:
        def diagnose(self, candidate):
            seen.append(candidate)
            return Diagnostic(
                code=candidate.code,
                message=candidate.message,
                notes=("canonical presentation",),
                node_id=candidate.node_id,
            )

    buf = _sink(monkeypatch)
    _render_run_error(result, _Loaded())

    assert seen == [failure]
    assert "semantic failure" in buf.getvalue()
    assert "canonical presentation" in buf.getvalue()


def test_run_error_boxes_failing_node(monkeypatch):
    # e07 omits `as_of` -> the canonical failure trace points at the failing
    # `:?` binding.
    flow = _ERRORS / "e07-required-missing.yaml"
    text = flow.read_text()
    loaded = load_flow(text, search_paths=[flow.parent], source_label=flow.name)
    result = run_flow(loaded, {"topic": "X"})
    assert result.status == "failed"
    buf = _sink(monkeypatch)
    _render_run_error(result, loaded)
    out = buf.getvalue()
    assert "e07-required-missing.yaml:23" in out   # the as_of binding, not the node header
    assert "report" in out                         # still inside the source window
    assert "as_of is required for the report" in out


def test_run_error_uses_reference_code_origin(monkeypatch):
    # A referenced CODE failure carries its authored `code:` scalar in the canonical
    # origin trace; no CLI kind fallback is involved.
    flow = _ERRORS / "e20-code-raises.yaml"
    text = flow.read_text()
    loaded = load_flow(text, search_paths=[flow.parent], source_label=flow.name)
    result = run_flow(loaded, {"topic": "X"})
    assert result.status == "failed"
    buf = _sink(monkeypatch)
    _render_run_error(result, loaded)
    out = buf.getvalue()
    assert "e20-code-raises.yaml:12" in out
    assert "╭" in out


def test_run_error_boxes_boundary_assert_line(monkeypatch):
    # e18's boundary assert fails before any node runs; its authored origin points
    # at the exact `asserts:` expression.
    flow = _ERRORS / "e18-false-boundary-assert.yaml"
    text = flow.read_text()
    loaded = load_flow(text, search_paths=[flow.parent], source_label=flow.name)
    result = run_flow(loaded, {"window": -5})
    assert result.status == "failed"
    buf = _sink(monkeypatch)
    _render_run_error(result, loaded)
    out = buf.getvalue()
    assert "e18-false-boundary-assert.yaml:15" in out   # the `${input.window} > 0` line
    assert "assert failed" in out


def test_run_error_boxes_code_wrong_type_output(monkeypatch):
    # e21's code node returns a value that fails its declared `output: int` -> the typed write
    # boundary rejects it and attributes the failure to the `output:` declaration.
    flow = _ERRORS / "e21-code-wrong-type.yaml"
    text = flow.read_text()
    loaded = load_flow(text, search_paths=[flow.parent], source_label=flow.name)
    result = run_flow(loaded, {"topic": "X"})
    assert result.status == "failed"
    buf = _sink(monkeypatch)
    _render_run_error(result, loaded)
    out = buf.getvalue()
    assert "e21-code-wrong-type.yaml:12" in out   # the `output: int` field line
    assert "╭" in out


def test_run_error_boxes_flow_input_line(monkeypatch):
    # e08's `window` cannot coerce to int, so its origin selects the declaration.
    flow = _ERRORS / "e08-input-type-mismatch.yaml"
    text = flow.read_text()
    loaded = load_flow(text, search_paths=[flow.parent], source_label=flow.name)
    result = run_flow(loaded, {"topic": "X", "window": "soon"})
    assert result.status == "failed"
    buf = _sink(monkeypatch)
    _render_run_error(result, loaded)
    out = buf.getvalue()
    assert "e08-input-type-mismatch.yaml:20" in out     # the `window: int` decl line
    assert "╭" in out


def test_run_error_plain_when_unlocatable(monkeypatch):
    # A failure with no authored origin remains a semantic message with no frame.
    flow = _ERRORS / "e18-false-boundary-assert.yaml"
    text = flow.read_text()
    result = RunResult(
        input={},
        status="failed",
        run_id="cli-test",
        log=None,
        failure=Failure("graph.terminal_unreachable", "boom", None),
        events=[],
    )
    loaded = load_flow(text, search_paths=[flow.parent], source_label=flow.name)
    buf = _sink(monkeypatch)
    _render_run_error(result, loaded)
    out = buf.getvalue()
    assert out.strip() == "boom"
    assert "╭" not in out                          # no boxed frame


def test_run_error_renders_namespaced_failure_origin_trace(monkeypatch):
    # The failure trace carries the owning CALL site despite the runtime-namespaced
    # node id (`run/boom`).
    flow = _ERRORS / "e24-nested-code-raise.yaml"
    text = flow.read_text()
    loaded = load_flow(text, search_paths=[flow.parent], source_label=flow.name)
    result = run_flow(loaded, {"topic": "X"})
    assert result.status == "failed"
    buf = _sink(monkeypatch)
    _render_run_error(result, loaded)
    out = buf.getvalue()
    assert "e24-nested-code-raise.yaml:27" in out   # the owning `run:` call node line
    assert "╭" in out                               # boxed, not a plain `run failed:` line


def test_run_error_boxes_loop_max_exceeded_at_loop_node(monkeypatch):
    # The LOOP origin remains stable even when the runtime driver id carries a
    # respawn suffix (`polish~5`).
    flow = _SEEDS / "28-refine-loop.yaml"
    text = flow.read_text()
    loaded = load_flow(text, source_label=flow.name)
    loop_origin = loaded.compiled.nodes["polish"].origin
    assert loop_origin is not None
    result = RunResult(
        input={},
        status="failed",
        run_id="cli-test",
        log=None,
        failure=Failure(
            "node.execute",
            "loop 'polish' exceeded max (5)",
            "polish~5",
            trace=(loop_origin,),
            cause_type="LoopMaxExceeded",
        ),
        events=[],
    )
    buf = _sink(monkeypatch)
    _render_run_error(result, loaded)
    out = buf.getvalue()
    assert "╭" in out                                # boxed, not a plain `run failed:` line
    assert "28-refine-loop.yaml:45" in out           # resolved to the authored `polish:` loop node
    assert "exceeded max (5)" in out


def test_run_error_traceback_only_under_engine_trace(monkeypatch):
    # The captured Python traceback (a code/tool/agent raise) is surfaced ONLY when
    # `engine_trace=True`; the default terse display omits it.
    flow = _ERRORS / "e24-nested-code-raise.yaml"
    text = flow.read_text()
    loaded = load_flow(text, search_paths=[flow.parent], source_label=flow.name)
    result = run_flow(loaded, {"topic": "X"})
    assert result.failure is not None
    assert result.failure.traceback  # captured at the node-failure boundary

    buf = _sink(monkeypatch)
    _render_run_error(result, loaded, engine_trace=False)
    assert "python traceback" not in buf.getvalue()

    buf = _sink(monkeypatch)
    _render_run_error(result, loaded, engine_trace=True)
    out = buf.getvalue()
    assert "python traceback" in out
    assert "intentional CODE failure" in out         # the originating Python exception


# --- _render_run_error: ordered canonical frames across child flows ----------------- #
def test_run_error_multi_frame_stack_into_def(monkeypatch):
    # e24's trace resolves the top-level CALL site followed by the failing CODE
    # origin inside `defs:inner`.
    flow = _ERRORS / "e24-nested-code-raise.yaml"
    text = flow.read_text()
    loaded = load_flow(text, search_paths=[flow.parent], source_label=flow.name)
    result = run_flow(loaded, {"topic": "X"})
    assert result.status == "failed"
    buf = _sink(monkeypatch)
    _render_run_error(result, loaded)
    out = buf.getvalue()
    assert "source locations:" in out
    assert "e24-nested-code-raise.yaml:27" in out   # frame 0: the owning `run:` call node (header)
    assert "e24-nested-code-raise.yaml:20" in out  # frame 1: the authored `code:` value
    assert "intentional CODE failure" in out


def test_run_error_multi_frame_stack_into_external(monkeypatch):
    # e25's ordered origins cross the file boundary: the outer CALL site is in
    # this file and the failing CODE origin is in lib_boom.yaml.
    flow = _ERRORS / "e25-external-raise.yaml"
    text = flow.read_text()
    loaded = load_flow(text, search_paths=[flow.parent], source_label=flow.name)
    result = run_flow(loaded, {"topic": "X"})
    assert result.status == "failed"
    buf = _sink(monkeypatch)
    _render_run_error(result, loaded)
    out = buf.getvalue()
    assert "source locations:" in out
    assert "e25-external-raise.yaml:19" in out      # frame 0: the `go:` call node (header)
    assert "lib_boom.yaml:18" in out                # frame 1: the external file's `code:` line


def test_run_error_multi_frame_three_levels(monkeypatch):
    # e26's origin trace resolves all three levels outermost-to-innermost.
    flow = _ERRORS / "e26-three-level-raise.yaml"
    text = flow.read_text()
    loaded = load_flow(text, search_paths=[flow.parent], source_label=flow.name)
    result = run_flow(loaded, {"topic": "X"})
    assert result.status == "failed"
    buf = _sink(monkeypatch)
    _render_run_error(result, loaded)
    out = buf.getvalue()
    assert "source locations:" in out
    assert "e26-three-level-raise.yaml:37" in out   # frame 0: the top `outer:` call node (header)
    assert "e26-three-level-raise.yaml:29" in out  # frame 1: `via:` call in `middle`
    assert "e26-three-level-raise.yaml:20" in out  # frame 2: `boom` code in `deep`


def test_run_error_multi_frame_on_nested_human_input_failure(monkeypatch):
    # Seed 25 blocks on a HUMAN_INPUT inside a called `defs:` child (`gate/approve`). A host
    # INVALID answer fails the typed write boundary. The origin trace points first
    # at the `gate:` CALL and then the precise child `output:` declaration.
    flow = _SEEDS / "25-nested-suspension.yaml"
    text = flow.read_text()
    loaded = load_flow(text, search_paths=[flow.parent], source_label=flow.name)
    with install_ask_user(lambda **request: "maybe"):
        result = run_flow(loaded, {"action": "ship it"})
    assert result.status == "failed"
    buf = _sink(monkeypatch)
    _render_run_error(result, loaded)
    out = buf.getvalue()
    assert "source locations:" in out
    assert "25-nested-suspension.yaml:41" in out    # frame 0: the `gate:` call node (header)
    assert "25-nested-suspension.yaml:31" in out  # frame 1: precise `output:` line
    assert "is not a member of variant" in out


def test_run_error_unavailable_presentation_stays_unlocated(monkeypatch):
    """A missing resolver cannot be bypassed with CLI graph inference."""
    flow = _ERRORS / "e24-nested-code-raise.yaml"
    text = flow.read_text()
    loaded = load_flow(text, search_paths=[flow.parent], source_label=flow.name)
    result = run_flow(loaded, {"topic": "X"})
    assert result.status == "failed"
    buf = _sink(monkeypatch)
    _render_run_error(result, replace(loaded, origin_resolver=None))
    out = buf.getvalue()
    assert "╭" not in out
    assert "Source presentation was unavailable." in out
    assert "intentional CODE failure" in out


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
