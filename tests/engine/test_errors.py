"""Negative `errors/` fixtures + `.yaml`-line source-mapping.

The "loud + located" contract for the loader: a malformed flow raises a
`LoadError` whose `.line` points at the offending `.yaml` line. The three
seed fixtures pin it end-to-end (load the fixture by path, call `load_flow`, assert the
right class + a located `.line`):

- **e01** — a dangling flow-output ref (`${scor.output}` typo) -> located at the
  `outputs:` line (the offending line).
- **e02** — a cyclic data graph (`a -> b -> a`) -> located at the first stuck node's
  line (a cycle spans nodes, so the coarser first-member anchor is the documented
  choice; the message names every stuck node).
- **e03** — a dotted-field miss on an ANONYMOUS producer record
  (`${score.output.confidence}` on `{rating, rationale}`) -> located at the
  `outputs:` line (C-ANON-CHECK: anon records are field-checked).

The **exhaustiveness mechanism** is asserted on a CONSTRUCTED inline
fixture — a `case … on <Literal enum producer>` missing a tag with no `else:` -> loud.
This is the supported way to model tagged data (a discriminant record + `case … on
<field>`); the old `kind: match` + payload-union design (and its e04/e05 seeds) was dropped.
"""

from pathlib import Path

import pytest

from agent_composer.compose import LoadError, load_flow
from agent_composer.run import run_flow

_ERRORS = Path(__file__).resolve().parents[2] / "tests" / "seeds" / "errors"


def _load_error(filename: str) -> LoadError:
    """Load an `errors/` fixture, asserting it raises `LoadError`; return the error."""
    with pytest.raises(LoadError) as exc_info:
        load_flow((_ERRORS / filename).read_text())
    return exc_info.value


# --------------------------------------------------------------------------- #
# e01 — dangling flow-output ref, located at the `outputs:` line (offending line)
# --------------------------------------------------------------------------- #


def test_e01_dangling_ref_located():
    err = _load_error("e01-undeclared-ref.yaml")
    msg = str(err)
    assert "unresolved references" in msg
    assert "scor" in msg  # names the dangling ref
    # `outputs: ${scor.output}` is line 19 of the fixture (the offending line).
    assert err.line == 19


# --------------------------------------------------------------------------- #
# e02 — cycle, located at the first stuck node's line (coarse: a cycle spans nodes)
# --------------------------------------------------------------------------- #


def test_e02_cycle_located():
    err = _load_error("e02-cycle.yaml")
    msg = str(err)
    assert "cycle" in msg
    assert "'a'" in msg and "'b'" in msg  # names the stuck nodes
    # node `a` (the first stuck node) is at line 12 — the documented coarse anchor.
    assert err.line == 12


# --------------------------------------------------------------------------- #
# e03 — dotted-field miss on an anonymous producer record, at the `outputs:` line
# --------------------------------------------------------------------------- #


def test_e03_unknown_field_located():
    err = _load_error("e03-unknown-field.yaml")
    msg = str(err)
    assert "unresolved references" in msg
    assert "confidence" in msg  # names the missing record field
    # `outputs: ${score.output.confidence}` is line 21 of the fixture.
    assert err.line == 21


# --------------------------------------------------------------------------- #
# every negative fixture carries a non-None located `.line`
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
    assert err.line is not None
    assert err.line == line


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

_SEEDS = _ERRORS.parent  # calpha/seeds (the child `research-one` = seed 03 lives here)


def _research_resolver():
    child = load_flow((_SEEDS / "03-research-one.yaml").read_text())

    def resolve(flow_id, version=None):
        if flow_id == "research-one":
            return child
        raise LoadError(f"unknown child {flow_id!r}")

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
        ("e13-bad-type-expression.yaml", "bad type expression 'list['", 7),
        ("e14-unknown-system-ambient.yaml", "bogus", 7),
        ("e15-prompt-undeclared-input.yaml", "is not a declared input", 7),
        ("e16-bad-typedef-name.yaml", "shadows", 4),
        ("e17-case-nonexhaustive.yaml", "non-exhaustive", 13),
        # inline CODE load-time gates — all located at the `code:` content / field line.
        ("e30-inline-syntax-error.yaml", "never closed", 15),   # padded compile -> real line
        ("e31-inline-no-return.yaml", "must return a value", 15),  # has-`return` gate
        ("e32-malformed-code-ref.yaml", "did you mean", 15),    # reject bucket
    ],
)
def test_compile_error_gallery(filename, substr, line):
    err = _load_error(filename)
    assert substr in str(err)
    if line is not None:
        assert err.line == line  # located at the offending .yaml line


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
    assert result.status != "succeeded"          # a failed/aborted run, not a crash
    assert substr in (result.error or "")


def test_flow_input_type_failure_carries_range():
    # e08: `window` is declared `int` on line 20; a non-coercible run value fails the boundary
    # type check. The flow-level RunFailed/RunResult carries a concrete `range` boxing the
    # failing input decl line (resolved from the StartNode's per-input-decl field_ranges).
    result = run_flow(
        load_flow((_ERRORS / "e08-input-type-mismatch.yaml").read_text()),
        {"topic": "X", "window": "soon"},
    )
    assert result.status == "failed"
    assert result.range is not None and result.range.start_line == 20


def test_false_boundary_assert_carries_range():
    # e18: the flow-level `asserts:` item `input.window > 0` is on line 15; window <= 0 makes it
    # false before any node runs. The RunResult carries a concrete `range` boxing that assert
    # line (resolved from the StartNode's flow-assert field_ranges keyed by the expr string).
    result = run_flow(
        load_flow((_ERRORS / "e18-false-boundary-assert.yaml").read_text()),
        {"window": -5},
    )
    assert result.status == "failed"
    assert result.range is not None and result.range.start_line == 15


def test_duplicate_asserts_caret_at_first_occurrence():
    # e34 (#88 case 1): two IDENTICAL post-asserts on lines 12 and 13; both fail (output is 1, not
    # < 0). Asserts are located BY POSITION, so the caret points at the FIRST occurrence (line 12) —
    # the old text-keyed field_ranges collapsed identical items last-wins and would box line 13.
    result = run_flow(load_flow((_ERRORS / "e34-dup-assert.yaml").read_text()), {})
    assert result.status == "failed"
    assert result.locator is not None and result.locator.kind == "assert"
    assert result.range is not None and result.range.start_line == 12


# --------------------------------------------------------------------------- #
# Deferred negative fixtures (NOTED, not asserted yet):
#   e05 — bare-tags union  -> read_typedefs mechanism (already covered)
# e08 — input type enforcement at the flow boundary — now asserted in the runtime gallery above.
# --------------------------------------------------------------------------- #
