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

    assert caught.value.range is not None
    assert caught.value.range.start_line == 4


def test_false_boundary_assert_carries_range():
    # e18: the flow-level `asserts:` item `input.window > 0` is on line 15; window <= 0 makes it
    # false before any node runs. The RunResult carries a concrete `range` boxing that assert
    # line — resolved BY POSITION off the StartNode's `pre_assert_ranges` (asserts are not in
    # `field_ranges`: identical assert text is not a unique key, so they are located by index).
    result = run_flow(
        load_flow((_ERRORS / "e18-false-boundary-assert.yaml").read_text()),
        {"window": -5},
    )
    assert result.status == "failed"
    assert result.range is not None and result.range.start_line == 15


def test_bad_input_ref_locates_the_offending_entry():
    # e14 (#87): the node's `input:` block spans lines 10-12, but only `rid:` on line 11 is bad
    # (`topic:` on 10 is valid; 12 is `output: str`, a different field). The range must box line
    # 11 alone — the block range would highlight all three.
    #
    # NOTE this asserts on e14 deliberately: of every error seed, it is the ONLY node-input ref
    # (e01/e03 are flow-output refs, which carry no range; e15 is the prompt path), so it IS the
    # regression surface for per-binding location. Its bad entry is also the SECOND one, which
    # proves the lookup is genuinely per-entry rather than accidentally "first entry wins".
    # Reordering e14's inputs will break this — correctly.
    err = _load_error("e14-unknown-system-ambient.yaml")
    assert "bogus" in str(err)
    assert err.range is not None
    assert (err.range.start_line, err.range.end_line) == (11, 11)


def test_bad_map_per_element_input_ref_locates_the_offending_entry():
    # e36 (#87): the MAP per-element branch is not the same code as the plain-input branch — it
    # reads `nwiring`, allows the `${item}` scope, and skips the reserved `over` key — so it gets
    # its own pin. `extra:` is on line 33; the `input:` block spans 31-33.
    err = _load_error("e36-bad-map-input-ref.yaml")
    assert "unresolved references" in str(err)
    assert "map input 'extra'" in str(err)
    assert err.range is not None
    assert (err.range.start_line, err.range.end_line) == (33, 33)


def test_map_over_ref_locates_the_over_line():
    # e35 (#101): a dangling ref in a MAP's reserved `over:` key (line 26). The range must box
    # `over:` itself — NOT the `input:` block on 27-28, which is what the bare-string lookup
    # `field_ranges.get("over")` silently degraded to once the map became tuple-keyed (#98).
    err = _load_error("e35-bad-map-over-ref.yaml")
    assert "unresolved references" in str(err)
    assert "map over" in str(err)  # names the reserved key, not an input
    assert err.range is not None
    assert (err.range.start_line, err.range.end_line) == (26, 26)


def test_duplicate_asserts_caret_at_first_occurrence():
    # e34 (#88 case 1): two IDENTICAL post-asserts on lines 12 and 13; both fail (output is 1, not
    # < 0). Asserts are located BY POSITION, so the caret points at the FIRST occurrence (line 12) —
    # the old text-keyed field_ranges collapsed identical items last-wins and would box line 13.
    result = run_flow(load_flow((_ERRORS / "e34-dup-assert.yaml").read_text()), {})
    assert result.status == "failed"
    assert result.locator is not None and result.locator.kind == "assert"
    assert result.range is not None and result.range.start_line == 12


def test_input_named_like_a_field_does_not_collide():
    # e33 (#88 case 2) — the one seed PR #98's plan promised and never shipped. The node has an
    # `env:` FIELD (value L32) and an input literally named `env` (L35). Under the old flat
    # `dict[str, SourceRange]` both keyed the bare string "env" and the later stamp won, so a caret
    # could land on the unrelated construct. The (kind, key) tuple keeps them distinct.
    #
    # Asserts CORRECT behavior — the regression pin for the part of #88 that IS fixed.
    text = (_ERRORS / "e33-input-named-like-field.yaml").read_text()
    fr = load_flow(text.replace("${nope.output}", '"lit"')).compiled.nodes["calc"].field_ranges
    assert fr[("field", "env")].start_line == 32  # the `env:` field's VALUE
    assert fr[("input", "env")].start_line == 35  # the input NAMED `env`
    assert fr[("field", "env")] != fr[("input", "env")]  # the collision #88 reported

    err = _load_error("e33-input-named-like-field.yaml")
    assert "unresolved references" in str(err) and "'env'" in str(err)
    assert err.range is not None
    # per-entry resolution (#87, PR #104) lands ON the offending `env:` input (L35), never L32 —
    # a collision would have put the caret on the unrelated `env:` field value instead.
    assert err.range.start_line == 35
    assert err.range.start_line != 32


# --------------------------------------------------------------------------- #
# KNOWN-WRONG location, pinned so each bug is reproducible and its fix is loud (#88, reopened).
#
# These assert the CURRENT, INCORRECT diagnostic — either a caret on the wrong line (e37/e38) or
# no range at all (e39/e40/e41). They are reproductions, not specifications: each fixture's header
# states the expected diagnostic next to the current one. **When the underlying bug is fixed these
# tests SHOULD fail** — that is the signal, and the fixer updates them to the expected line. They
# are here because a bug nobody can run is a bug nobody can confirm.
# --------------------------------------------------------------------------- #


def test_llm_config_error_names_the_key_but_carries_no_range():
    # e39: the message says `provider` — the failure site KNOWS which entry is bad — and then
    # attaches range=None. The caret comes from the LEGACY LINE-MAP (the node header, L24) while the
    # offending line is L27.
    #
    # The point for #88: descending into `llm_config:` would stamp a range nothing asks for. The
    # missing half is the LOCATOR, not the stamp.
    # ⚠ Prerequisite for #91/L4c: this renders ONLY via the legacy line — retiring line-maps without
    # giving it a range= first would drop its frame entirely.
    err = _load_error("e39-llm-config-no-range.yaml")
    assert "llm_config" in str(err) and "provider" in str(err)  # the entry IS named
    assert err.range is None                                     # ...and no range is attached
    assert err.line == 24  # the node header, via the legacy line-map; expected 27 (the `provider:`)


def test_agent_controls_field_is_retired_and_located():
    # `controls:` was removed with AGENT continuations. The descriptor rejects the whole
    # retired field at its value range and directs authors to the ordinary `tools:` surface.
    err = _load_error("e40-agent-controls-no-range.yaml")
    assert "field 'controls' is not allowed" in str(err)
    assert err.range is not None and err.range.start_line == 14
    assert err.line == 14


def test_structured_output_failure_names_whole_field_and_range_is_none():
    # e41: TWO defects in one run.
    # (1) granularity — `score:` is bad, `claim:` is fine, but the locator names the WHOLE field and
    #     the message never says `score`. A stamp for ("outputs","score") would go unused.
    # (2) the funnel misses entirely -> range=None. engine.py:472 mints key="output" (SINGULAR);
    #     field_ranges is keyed "outputs" (PLURAL, post-_back_map_input_output), so
    #     eval_node.py:149's .get(("field","output")) misses on EVERY output-validation failure.
    #     The CLI hides it via _FIELD_KEY_ALIAS — but a LIBRARY CALLER gets range=None. That is
    #     #81's opening complaint, live.
    result = run_flow(load_flow((_ERRORS / "e41-structured-output-no-range.yaml").read_text()), {})
    assert result.status == "failed"
    assert "does not match declared type int" in str(result.error)
    assert "score" not in str(result.error)  # the failing field is never named
    assert result.locator is not None
    assert result.locator.kind == "field" and result.locator.key == "output"  # not `score`
    assert result.range is None  # the singular/plural drift; the CLI recovers via the alias

    # the range the funnel SHOULD have found is present under the plural key
    node = load_flow((_ERRORS / "e41-structured-output-no-range.yaml").read_text()).compiled.nodes["calc"]
    assert node.field_ranges.get(("field", "output")) is None      # what the funnel asks for
    assert node.field_ranges.get(("field", "outputs")) is not None  # what is actually stored


def test_synth_node_locates_at_its_call_directive_line():
    # e37 (#106): the inline `call(...)` on L30 desugars to `__call_0`. It USED to be handed the HOST
    # node's field_ranges verbatim, so its ("input","topic") pointed at the HOST's valid `topic:` on
    # L29 — a foreign map (the key is right, the map belongs to another node), which per-entry
    # resolution (PR #104) turned into a confidently-wrong caret on a valid line.
    #
    # FIXED (#106): the desugar hands the synth node its OWN provenance — the `call(...)` directive's
    # binding line — so the caret is L30 (the actual error), and the message names the synth node.
    err = _load_error("e37-synth-node-inherits-host-ranges.yaml")
    assert "unresolved references" in str(err)
    assert "__call_0" in str(err) and "'topic'" in str(err)  # the SYNTH node's input, not the host's
    assert err.range is not None
    assert err.range.start_line == 30  # the `call(...)` directive (was L29, the host's valid `topic:`)


def test_case_when_ref_locates_at_when_line():
    # e38 (#106): a dangling ref in `cases[1].when` (L36) hoists to a synthetic CASE input `__r1`.
    # It USED to locate at the `gate:` header and print the synthetic `__r1` (a name that appears
    # nowhere in the file). FIXED: the CASE desugar stamps `__r1`'s own `cases[1].when` range and a
    # `cases[1].when` label, so the caret is L36 and the message names the author's construct.
    err = _load_error("e38-case-when-ref-locates-node-header.yaml")
    assert "unresolved references" in str(err)
    assert "'__r1'" not in str(err)       # the synthetic name no longer leaks
    assert "cases[1].when" in str(err)     # ... it is named by the author-facing site instead
    assert err.range is not None
    assert err.range.start_line == 36  # the offending `cases[1].when` (was L31, the `gate:` header)


# --------------------------------------------------------------------------- #
# Deferred negative fixtures (NOTED, not asserted yet):
#   e05 — bare-tags union  -> read_typedefs mechanism (already covered)
# e08 — input type enforcement at the flow boundary — now asserted in the runtime gallery above.
# --------------------------------------------------------------------------- #
