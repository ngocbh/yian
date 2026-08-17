"""Negative `errors/` fixtures + canonical `.yaml` source diagnostics.

The "loud + located" contract for the loader: a malformed flow raises a
`LoadError` whose diagnostic frames point at the offending `.yaml` source. The
three seed fixtures pin that contract end-to-end:

- **e01** — a dangling flow-output ref (`${scor.output}` typo) -> located at the
  `output:` line (the offending line).
- **e02** — a cyclic data graph (`a -> b -> a`) -> one frame for each actual cycle
  member, in authored source order; the message names both nodes.
- **e03** — a dotted-field miss on an ANONYMOUS producer record
  (`${score.output.confidence}` on `{rating, rationale}`) -> located at the
  `output:` line because anonymous records are field-checked.

The **exhaustiveness mechanism** is asserted on a CONSTRUCTED inline
fixture — a `case … on <Literal enum producer>` missing a tag with no `else:` -> loud.
This is the supported way to model tagged data (a discriminant record + `case … on
<field>`); the old `kind: match` + payload-union design (and its e04/e05 seeds) was dropped.
"""

from pathlib import Path

import pytest

from agent_composer.compose import LoadError, load_flow
from agent_composer.compose.helpers.errors import LoadDiagnostics
from agent_composer.compose.parse.nodes import parse_nodes
from agent_composer.compose.source import SourceDocument
from agent_composer.diagnostics import SourceRange
from agent_composer.run import run_flow

_ERRORS = Path(__file__).resolve().parents[2] / "tests" / "seeds" / "errors"


def _load_error(filename: str) -> LoadError:
    """Load an `errors/` fixture, asserting it raises `LoadError`; return the error."""
    with pytest.raises(LoadError) as exc_info:
        load_flow((_ERRORS / filename).read_text())
    return exc_info.value


def _error_ranges(error: LoadError) -> tuple[SourceRange, ...]:
    """Return the canonical outer-to-inner ranges carried by one load error."""
    return tuple(frame.range for frame in error.diagnostic.frames)


def _only_error_range(error: LoadError) -> SourceRange:
    """Return the sole source range expected for a single-site failure."""
    ranges = _error_ranges(error)
    assert len(ranges) == 1
    return ranges[0]


# --------------------------------------------------------------------------- #
# e01 — dangling flow-output ref, located at the `output:` line (offending line)
# --------------------------------------------------------------------------- #


def test_e01_dangling_ref_located():
    err = _load_error("e01-undeclared-ref.yaml")
    msg = str(err)
    assert "unresolved references" in msg
    assert "scor" in msg  # names the dangling ref
    # `output: ${scor.output}` is line 19 of the fixture (the offending line).
    assert _only_error_range(err).start_line == 19


# --------------------------------------------------------------------------- #
# e02 — cycle, located at every actual cycle member
# --------------------------------------------------------------------------- #


def test_e02_cycle_located():
    err = _load_error("e02-cycle.yaml")
    msg = str(err)
    assert "cycle" in msg
    assert "'a'" in msg and "'b'" in msg  # names the stuck nodes
    # Nodes `a` and `b` begin at lines 12 and 18, in authored source order.
    assert tuple(range_.start_line for range_ in _error_ranges(err)) == (12, 18)


# --------------------------------------------------------------------------- #
# e03 — dotted-field miss on an anonymous producer record, at the `output:` line
# --------------------------------------------------------------------------- #


def test_e03_unknown_field_located():
    err = _load_error("e03-unknown-field.yaml")
    msg = str(err)
    assert "unresolved references" in msg
    assert "confidence" in msg  # names the missing record field
    # `output: ${score.output.confidence}` is line 21 of the fixture.
    assert _only_error_range(err).start_line == 21


# --------------------------------------------------------------------------- #
# every negative fixture carries a source frame
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("filename", "line"),
    [
        ("e01-undeclared-ref.yaml", 19),
        ("e02-cycle.yaml", 12),
        ("e03-unknown-field.yaml", 21),
    ],
)
def test_negative_fixtures_are_located(filename, line):
    err = _load_error(filename)
    assert _error_ranges(err)[0].start_line == line


# --------------------------------------------------------------------------- #
# Exhaustiveness MECHANISM on a CONSTRUCTED Literal-enum fixture: a
# `case … on <Literal enum>` producer missing a tag with no `else:` -> loud. This is
# the discriminant-record routing that replaced the dropped `kind: match` design.
# --------------------------------------------------------------------------- #

# classify produces a Literal["pro","con","flat"] enum; route covers only pro/con
# with NO else: -> the `flat` tag is uncovered -> non-exhaustive.
_E04_NONEXHAUSTIVE = """
id: e04-constructed
name: e04_constructed

input:
  topic: str

nodes:
  classify:
    kind: agent
    input:
      topic: ${input.topic}
    output: Literal["pro", "con", "flat"]
    prompt: "Classify ${topic}."
  route:
    kind: case
    on: ${classify.output}
    cases:
      - when: pro
        then: pro_note
      - when: con
        then: con_note
  pro_note:
    kind: agent
    input:
      topic: ${input.topic}
    output: str
    prompt: "Pro note for ${topic}."
  con_note:
    kind: agent
    input:
      topic: ${input.topic}
    output: str
    prompt: "Con note for ${topic}."

output: ${pro_note.output | con_note.output}
"""

# the same flow, but with an `else:` that satisfies coverage (must NOT fire).
_E04_WITH_ELSE = _E04_NONEXHAUSTIVE.replace(
    "        then: con_note\n",
    "        then: con_note\n    else: flat_note\n",
).replace(
    "  pro_note:",
    "  flat_note:\n"
    "    kind: agent\n"
    "    inputs:\n"
    "      topic: ${input.topic}\n"
    "    output: str\n"
    '    prompt: "Flat note for ${topic}."\n'
    "  pro_note:",
)


def test_e04_exhaustiveness_mechanism_loud():
    with pytest.raises(LoadError) as exc_info:
        load_flow(_E04_NONEXHAUSTIVE)
    msg = str(exc_info.value)
    assert "non-exhaustive" in msg
    assert "flat" in msg  # names the uncovered enum tag


def test_e04_else_satisfies_coverage():
    # a present `else:` covers the remaining tag — the flow loads (no exhaustiveness error).
    loaded = load_flow(_E04_WITH_ELSE)
    assert loaded is not None


# --------------------------------------------------------------------------- #
# e06 — cross-flow type mismatch at a `call` boundary (types_compatible).
# A float-typed node output bound to a callable input declared `str` -> loud + located.
# --------------------------------------------------------------------------- #

_SEEDS = _ERRORS.parent  # tests/seeds (the child `research-one` = seed 03 lives here)


def _research_resolver():
    child = load_flow((_SEEDS / "03-research-one.yaml").read_text())

    def resolve(flow_id, version=None):
        if flow_id == "research-one":
            return child
        raise LoadDiagnostics().error(f"unknown child {flow_id!r}")

    return resolve


def test_e06_cross_flow_type_mismatch_is_loud():
    with pytest.raises(LoadError) as exc_info:
        load_flow(
            (_ERRORS / "e06-type-mismatch-ref.yaml").read_text(),
            child_resolver=_research_resolver(),
        )
    msg = str(exc_info.value)
    assert "topic" in msg                       # names the offending binding
    assert "str" in msg and "float" in msg       # child expects str, source is float


# --------------------------------------------------------------------------- #
# The broader negative gallery (e09–e23) — one fixture per engine failure mode.
# Compile (load_flow raises LoadError) + runtime (run_flow -> status="failed").
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("filename", "substr", "line"),
    [
        ("e09-unknown-top-level-key.yaml", "unknown top-level key 'bogus'", 14),
        # Per-kind parse anchors these at the precise offending field (not the node body):
        ("e10-unknown-node-kind.yaml", "unknown kind 'llm'", 8),  # the `kind:` value
        ("e11-field-not-allowed.yaml", "field 'prompt' is not allowed", 13),  # the stray field
        ("e12-missing-required-field.yaml", "missing required field 'code'", 8),  # the node body
        ("e13-bad-type-expression.yaml", "bad type expression 'list['", 11),
        ("e14-unknown-system-ambient.yaml", "bogus", 11),
        ("e15-prompt-undeclared-input.yaml", "is not a declared input", 12),
        ("e16-bad-typedef-name.yaml", "shadows", 4),
        ("e17-case-nonexhaustive.yaml", "non-exhaustive", 13),
        # inline CODE load-time gates — all located at the `code:` content / field line.
        ("e30-inline-syntax-error.yaml", "never closed", 15),   # exact CODE line origin
        ("e31-inline-no-return.yaml", "must return a value", 15),  # has-`return` gate
        ("e32-malformed-code-ref.yaml", "did you mean", 15),    # reject bucket
    ],
)
def test_compile_error_gallery(filename, substr, line):
    err = _load_error(filename)
    assert substr in str(err)
    if line is not None:
        assert _error_ranges(err)[0].start_line == line


@pytest.mark.parametrize(
    ("filename", "line"),
    [
        ("e30-inline-syntax-error.yaml", 15),
        ("e31-inline-no-return.yaml", 15),
        ("e32-malformed-code-ref.yaml", 15),
    ],
)
def test_inline_code_construction_errors_carry_ranges(filename, line):
    """Every CODE construction rejection exposes one precise source frame."""
    err = _load_error(filename)
    assert _only_error_range(err).start_line == line


@pytest.mark.parametrize(
    ("filename", "inputs", "substr"),
    [
        ("e18-false-boundary-assert.yaml", {"window": -5}, "assert failed"),
        ("e19-false-post-assert.yaml", {"topic": "X"}, "assert failed"),
        ("e20-code-raises.yaml", {"topic": "X"}, "intentional CODE failure"),
        ("e21-code-wrong-type.yaml", {"topic": "X"}, "int"),
        ("e22-required-missing.yaml", {}, "as_of is required"),
        ("e23-unknown-tool.yaml", {"topic": "X"}, "unknown tool"),
        # e07 — the AGENT-form `:?`: binding (where `:?` fires) runs before the model
        # is built, so an omitted required ref fails the run with no LLM creds needed.
        ("e07-required-missing.yaml", {"topic": "X"}, "as_of is required"),
        # e08 — input type enforcement at the flow boundary: a non-coercible string for an
        # `int` input fails BEFORE any node runs (no LLM creds needed).
        ("e08-input-type-mismatch.yaml", {"topic": "X", "window": "soon"}, "int"),
        # inline CODE runtime failures — a raise, a wrong-typed return (write boundary),
        # and a nested non-serializable return (the node's deep serialize-once check).
        ("e27-inline-code-raises.yaml", {"topic": "X"}, "intentional inline failure"),
        ("e28-inline-wrong-type.yaml", {"topic": "X"}, "int"),
        ("e29-inline-nonserializable.yaml", {"topic": "X"}, "non-serializable"),
    ],
)
def test_runtime_error_gallery(filename, inputs, substr):
    result = run_flow(load_flow((_ERRORS / filename).read_text()), inputs)
    assert result.status == "failed"
    assert result.failure is not None
    assert substr in result.failure.message


def test_flow_input_type_failure_origin_resolves_to_declaration():
    # e08: `window` is declared `int` on line 20; its contract origin resolves there after
    # execution without a parallel runtime coordinate.
    loaded = load_flow((_ERRORS / "e08-input-type-mismatch.yaml").read_text())
    result = run_flow(
        loaded,
        {"topic": "X", "window": "soon"},
    )
    assert result.status == "failed"
    assert loaded.diagnose(result.failure).frames[-1].range.start_line == 20


def test_invalid_authored_default_carries_input_range():
    source = """id: invalid-default
name: invalid_default
input:
  value: int = null
nodes:
  echo:
    kind: code
    code: tests.engine._compose_codefns:echo_x
    input:
      x: ${input.value}
    output: int
output: ${echo.output}
"""

    with pytest.raises(LoadError, match="default") as caught:
        load_flow(source)

    assert _only_error_range(caught.value).start_line == 4


def test_false_boundary_assert_origin_resolves_to_assertion():
    # e18: the flow-level `asserts:` item is on line 15; its occurrence origin resolves there.
    loaded = load_flow((_ERRORS / "e18-false-boundary-assert.yaml").read_text())
    result = run_flow(
        loaded,
        {"window": -5},
    )
    assert result.status == "failed"
    assert loaded.diagnose(result.failure).frames[-1].range.start_line == 15


def test_bad_input_ref_locates_the_offending_entry():
    # e14's `input:` block spans lines 10-12, but only `rid:` on line 11 is bad
    # (`topic:` on 10 is valid; 12 is `output: str`, a different field). The range must box line
    # 11 alone — the block range would highlight all three.
    #
    # NOTE this asserts on e14 deliberately: of every error seed, it is the ONLY node-input ref
    # (e01/e03 are flow-output refs; e15 is the prompt path), so it IS the
    # regression surface for per-binding location. Its bad entry is also the SECOND one, which
    # proves the lookup is genuinely per-entry rather than accidentally "first entry wins".
    # Reordering e14's inputs will break this — correctly.
    err = _load_error("e14-unknown-system-ambient.yaml")
    assert "bogus" in str(err)
    source_range = _only_error_range(err)
    assert (source_range.start_line, source_range.end_line) == (11, 11)


def test_bad_map_per_element_input_ref_locates_the_offending_entry():
    # e36's MAP per-element branch is not the same code as the plain-input branch — it
    # reads `nwiring`, allows the `${item}` scope, and skips the reserved `over` key — so it gets
    # its own pin. `extra:` is on line 33; the `input:` block spans 31-33.
    err = _load_error("e36-bad-map-input-ref.yaml")
    assert "unresolved references" in str(err)
    assert "map input 'extra'" in str(err)
    source_range = _only_error_range(err)
    assert (source_range.start_line, source_range.end_line) == (33, 33)


def test_map_over_ref_locates_the_over_line():
    # e35 has a dangling ref in MAP's reserved `over:` key on line 26. Its tuple-tagged range
    # must box `over:` itself, not the unrelated `input:` block on lines 27-28.
    err = _load_error("e35-bad-map-over-ref.yaml")
    assert "unresolved references" in str(err)
    assert "map over" in str(err)  # names the reserved key, not an input
    source_range = _only_error_range(err)
    assert (source_range.start_line, source_range.end_line) == (26, 26)


def test_duplicate_asserts_caret_at_first_occurrence():
    # e34 has identical failing post-asserts on lines 12 and 13. Position-owned origins must
    # locate the first failure at its first occurrence on line 12.
    loaded = load_flow((_ERRORS / "e34-dup-assert.yaml").read_text())
    result = run_flow(loaded, {})
    assert result.status == "failed"
    assert loaded.diagnose(result.failure).frames[-1].range.start_line == 12


def test_input_named_like_a_field_does_not_collide():
    # e33 has an `env:` field on L32 and an input named `env` on L35. Tuple-tagged keys retain
    # separate source ranges for those author constructs.
    text = (_ERRORS / "e33-input-named-like-field.yaml").read_text()
    document = SourceDocument.parse(text.replace("${nope.output}", '"lit"'))
    fr = parse_nodes(document.root.child("nodes"))["calc"].field_ranges
    assert fr[("field", "env")].start_line == 32  # the `env:` field's VALUE
    assert fr[("input", "env")].start_line == 35  # the input NAMED `env`
    assert fr[("field", "env")] != fr[("input", "env")]

    err = _load_error("e33-input-named-like-field.yaml")
    assert "unresolved references" in str(err) and "'env'" in str(err)
    source_range = _only_error_range(err)
    # Per-entry resolution lands on the offending `env:` input (L35), never the unrelated
    # `env:` field value on L32.
    assert source_range.start_line == 35
    assert source_range.start_line != 32


# --------------------------------------------------------------------------- #
# Source-location regressions. Source-location contracts remain positive pins; the
# structured-output case below preserves its separate runtime reproduction until that
# contract is addressed.
# --------------------------------------------------------------------------- #


def test_llm_config_error_carries_the_field_range():
    # e39: the descriptor already records the `llm_config:` value range. The build error must
    # retain it, so the first offending config entry on L27 is the primary anchor rather than
    # the node header on L24.
    err = _load_error("e39-llm-config-no-range.yaml")
    assert "llm_config" in str(err) and "provider" in str(err)  # the entry IS named
    assert _only_error_range(err).start_line == 27


def test_agent_controls_field_is_retired_and_located():
    # `controls:` was removed with AGENT continuations. The descriptor rejects the whole
    # retired field at its value range and directs authors to the ordinary `tools:` surface.
    err = _load_error("e40-agent-controls-no-range.yaml")
    assert "field 'controls' is not allowed" in str(err)
    assert _only_error_range(err).start_line == 14


def test_structured_output_origin_is_exact():
    # Canonical diagnosis selects the exact failing contract path without runtime coordinates.
    flow = _ERRORS / "e41-structured-output-no-range.yaml"
    loaded = load_flow(flow.read_text(), source_label=flow.name)
    result = run_flow(loaded, {})
    assert result.status == "failed"
    assert "does not match declared type int" in result.failure.message
    assert "score" not in result.failure.message  # the failing field is never named
    assert loaded.diagnose(result.failure).frames[-1].range.start_line == 13


def test_synth_node_locates_at_its_call_directive_line():
    # e37's inline `call(...)` on L30 desugars to `__call_0`. Directive-owned provenance locates
    # the bad `topic` binding at L30 rather than the host node's like-named input on L29.
    err = _load_error("e37-synth-node-inherits-host-ranges.yaml")
    assert "unresolved references" in str(err)
    assert "__call_0" in str(err) and "'topic'" in str(err)  # the SYNTH node's input, not the host's
    # The `call(...)` directive owns the failing binding.
    assert _only_error_range(err).start_line == 30


def test_case_when_ref_locates_at_when_line():
    # e38's dangling ref in `cases[1].when` (L36) hoists to synthetic input `__r1`. Predicate-owned
    # provenance keeps the caret and author-facing label on `cases[1].when`.
    err = _load_error("e38-case-when-ref-locates-node-header.yaml")
    assert "unresolved references" in str(err)
    assert "'__r1'" not in str(err)       # the synthetic name no longer leaks
    assert "cases[1].when" in str(err)     # ... it is named by the author-facing site instead
    # The offending `cases[1].when` owns the failure range.
    assert _only_error_range(err).start_line == 36
