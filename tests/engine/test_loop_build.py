"""build_loop_node + loader dispatch + the `'a -> 'a` type contract.

A `kind: loop` node bakes a `LoopNode` from a `LoopDescriptor`: it resolves+bakes the
body subflow (the `call:` child, a `defs:` callable here so no external resolver is
needed), stamps `output_contract`, and enforces the two halves of the type contract:

- FIELD-NAME-set (in `build_loop_node`): body output field NAMES == carried keys,
  body input NAMES subset of carried keys — a located `LoadError` on a name mismatch.
- FULL TYPE equality (in the loader post-build pass `check_loop_type_contract`):
  body output Type == carried record Type per field — a located `LoadError` on a
  same-names/different-TYPES mismatch (needs the assembled producers/flow-input types).
"""

import pytest

from agent_composer.compose.loader import load_flow
from agent_composer.compose.helpers.errors import LoadError
from agent_composer.nodes.base import NodeKind
from agent_composer.typesys.values import ValueKind


# A good loop: the body `bump` reads a subset of the carried record and returns the
# WHOLE next carried record ({n, exited}); `chat_loop` carries {n, exited}, loops
# while `not ${exited}`, capped at max: 5.
GOOD = """
id: loop-good
name: loop_good
defs:
  bump:
    input:
      n: int
      exited: bool
    nodes:
      step:
        kind: code
        code: tests.engine._compose_codefns:make_report
        input:
          n: ${input.n}
          exited: ${input.exited}
        output:
          n: int
          exited: bool
    output: ${step.output}
nodes:
  chat_loop:
    kind: loop
    call: bump
    input:
      n: 0
      exited: false
    while: not ${exited}
    max: 5
output: ${chat_loop.output}
"""


PARTIAL_OUTPUT_REFERENCE = """
id: loop-partial-reference
name: loop_partial_reference
defs:
  body:
    input:
      n: int
      note: str
    nodes: {}
    output:
      n: ${input.n}
      note: ${input.note + "!"}
nodes:
  loop:
    kind: loop
    call: body
    input:
      n: 1
      note: seed
    times: 1
  sink:
    kind: code
    code: tests.engine._compose_codefns:echo_x
    input:
      x: ${loop.output.missing}
    output: str
output: ${sink.output}
"""


def test_partial_loop_contract_rejects_unknown_downstream_field_at_load():
    """The closed partial name set participates in graph reference validation."""
    with pytest.raises(LoadError, match="unknown field 'missing'"):
        load_flow(PARTIAL_OUTPUT_REFERENCE)


# The body output OMITS `exited`, so its output field NAMES ({n}) != the carried keys
# ({n, exited}) -> caught in build_loop_node (the field-name-set half).
BAD_FIELDSET = """
id: loop-bad-fieldset
name: loop_bad_fieldset
defs:
  bump:
    input:
      n: int
      exited: bool
    nodes:
      step:
        kind: code
        code: tests.engine._compose_codefns:make_report
        input:
          n: ${input.n}
          exited: ${input.exited}
        output:
          n: int
    output: ${step.output}
nodes:
  chat_loop:
    kind: loop
    call: bump
    input:
      n: 0
      exited: false
    while: not ${exited}
    max: 5
output: ${chat_loop.output}
"""


# Same field NAMES ({n, exited}) but the body output declares `n: str` while the
# carried `n` is seeded from `${input.n0}` (an int flow input) -> a same-names/
# different-TYPES mismatch, caught in the loader post-build pass
# check_loop_type_contract. (The carried field must be seeded from a ref with a
# resolvable Type — a bare literal stays lenient/opaque in the type pass.)
BAD_TYPES = """
id: loop-bad-types
name: loop_bad_types
input:
  n0: int
defs:
  bump:
    input:
      n: int
      exited: bool
    nodes:
      step:
        kind: code
        code: tests.engine._compose_codefns:make_report
        input:
          n: ${input.n}
          exited: ${input.exited}
        output:
          n: str
          exited: bool
    output: ${step.output}
nodes:
  chat_loop:
    kind: loop
    call: bump
    input:
      n: ${input.n0}
      exited: false
    while: not ${exited}
    max: 5
output: ${chat_loop.output}
"""


# One opaque body field must not erase the independently proven type of a sibling.
# `note + "!"` is deliberately outside static inference, while direct `input.note`
# proves that the body returns `str` for carried `n` (which is seeded as `int`).
PARTIAL_OPAQUE_BAD_TYPES = """
id: loop-partial-opaque-bad-types
name: loop_partial_opaque_bad_types
input:
  n0: int
  note0: str
defs:
  bump:
    input:
      n: int
      note: str
    nodes: {}
    output:
      n: ${input.note}
      note: ${input.note + "!"}
nodes:
  chat_loop:
    kind: loop
    call: bump
    input:
      n: ${input.n0}
      note: ${input.note0}
    times: 1
output: ${chat_loop.output}
"""


NULLABLE_SEED_LOOP = """
id: loop-nullable-seed
name: loop_nullable_seed
defs:
  keep:
    input:
      maybe: Optional[str]
      done: bool
    nodes: {}
    output:
      maybe: ${input.maybe}
      done: ${input.done}
nodes:
  chat_loop:
    kind: loop
    call: keep
    input:
      maybe: ${null}
      done: false
    times: 1
output: ${chat_loop.output}
"""


# `until:` is a POST-check (do-while) predicate — same carried-record scope as `while:`,
# `max:` still required. `times: N` is a fixed count — no predicate, `max:` redundant.
UNTIL_GOOD = GOOD.replace("    while: not ${exited}\n", "    until: ${exited}\n")
TIMES_GOOD = GOOD.replace("    while: not ${exited}\n    max: 5\n", "    times: 3\n")
# Exactly one of while:/until:/times: — both-set and neither-set are illegal.
BOTH = GOOD.replace("    while: not ${exited}\n", "    while: not ${exited}\n    until: ${exited}\n")
NEITHER = GOOD.replace("    while: not ${exited}\n", "")
# `times:` with `max:` is redundant (times bounds the count) -> rejected, not lenient-ignored.
TIMES_WITH_MAX = GOOD.replace("    while: not ${exited}\n", "    times: 3\n")  # keeps max: 5
# A fixed count must permit at least one run: `times: 0` is nonsensical.
TIMES_BAD_COUNT = GOOD.replace("    while: not ${exited}\n    max: 5\n", "    times: 0\n")


def test_good_loop_bakes_loopnode():
    flow = load_flow(GOOD)
    node = flow.compiled.nodes["chat_loop"]
    assert node.kind == NodeKind.LOOP
    assert node.child is not None
    # The output contract carries the body codomain — prove the contract was baked,
    # not just that a child was resolved.
    assert node.output_contract is not None
    assert set(node.output_contract.type.fields) == {"n", "exited"}


def test_body_output_fieldset_must_equal_carried():
    with pytest.raises(LoadError) as e:
        load_flow(BAD_FIELDSET)
    assert "output" in str(e.value).lower()
    assert e.value.diagnostic.frames[-1].range.start_line is not None


# A loop whose body builds a carried `str` field from a MULTI-SEGMENT template binding
# (embedded literal text + spans). eval_template concatenates such a binding to a str, so
# output typing infers `str` for it — matching the carried `transcript` seeded from the
# str flow input `${input.opening}`. No CODE fold node is needed to stamp the type.
TEMPLATE_STR = """
id: loop-template-str
name: loop_template_str
input:
  opening: str
defs:
  turn:
    input:
      transcript: str
      exited: bool
    nodes:
      step:
        kind: code
        code: tests.engine._compose_codefns:make_report
        input:
          n: ${input.transcript}
          exited: ${input.exited}
        output:
          n: int
          exited: bool
    output:
      transcript: |-
        ${input.transcript}
        turn: ${step.output.n}
      exited: ${step.output.exited}
nodes:
  chat_loop:
    kind: loop
    call: turn
    input:
      transcript: ${input.opening}
      exited: false
    while: not ${exited}
    max: 5
output: ${chat_loop.output.transcript}
"""


def test_body_output_types_must_equal_carried():
    with pytest.raises(LoadError) as e:
        load_flow(BAD_TYPES)
    # pin the failure to the TYPE pass (check_loop_type_contract), not any LoadError.
    assert "carried field" in str(e.value)
    assert e.value.diagnostic.frames[-1].range.start_line is not None


def test_opaque_body_field_does_not_erase_proven_sibling_type_mismatch():
    """Unknown evidence is field-local during LOOP's compile-time type check."""
    with pytest.raises(LoadError, match="carried field 'n'"):
        load_flow(PARTIAL_OPAQUE_BAD_TYPES)


def test_loop_accepts_proven_null_for_nullable_carried_field():
    """A None seed is admissible to the body's nullable field contract."""
    loaded = load_flow(NULLABLE_SEED_LOOP)

    maybe = loaded.compiled.nodes["chat_loop"].output_contract.type.fields["maybe"]
    assert maybe.kind is ValueKind.STRING
    assert maybe.nullable is True


def test_multi_segment_template_output_types_as_str():
    # The body's `transcript` output is a multi-segment template -> inferred str, so the
    # `'a -> 'a` contract holds against the str-seeded carried field with no fold node.
    flow = load_flow(TEMPLATE_STR)
    node = flow.compiled.nodes["chat_loop"]
    assert node.kind == NodeKind.LOOP
    assert node.output_contract is not None
    assert node.output_contract.type.fields["transcript"].kind == ValueKind.STRING


# A runaway guard below 1 (`max: 0`) is nonsensical — the guard bounds the iteration
# count, so it must permit at least one body run. Rejected at build.
BAD_MAX = GOOD.replace("    max: 5\n", "    max: 0\n")


# A `while:` predicate that references a name NOT in the carried record ({n, exited}) — here a
# typo `exted` for `exited`. Record-scoped eval would resolve it falsy and spin to `max`, so it
# must be rejected loudly at build.
BAD_PREDICATE_NAME = GOOD.replace("    while: not ${exited}\n", "    while: not ${exted}\n")

# `max:` must be a plain integer. The descriptor is a bare dataclass (its `Optional[int]`
# annotation isn't enforced), so these non-int YAML types pass through to build_loop_node:
# a quoted string, a float, and a bool (an int subclass that would silently read as 0/1).
BAD_MAX_STR = GOOD.replace("    max: 5\n", '    max: "5"\n')
BAD_MAX_FLOAT = GOOD.replace("    max: 5\n", "    max: 2.5\n")
BAD_MAX_BOOL = GOOD.replace("    max: 5\n", "    max: true\n")


def test_until_loop_builds():
    flow = load_flow(UNTIL_GOOD)
    assert flow.compiled.nodes["chat_loop"].predicate_kind == "until"


def test_times_loop_builds():
    flow = load_flow(TIMES_GOOD)
    n = flow.compiled.nodes["chat_loop"]
    assert n.predicate_kind == "times" and n.times == 3 and n.max_iters == 3


def test_exactly_one_predicate_required():
    for bad in (BOTH, NEITHER):
        with pytest.raises(LoadError) as e:
            load_flow(bad)
        assert "exactly one" in str(e.value).lower()


def test_times_with_max_is_rejected():
    with pytest.raises(LoadError) as e:
        load_flow(TIMES_WITH_MAX)
    assert "redundant" in str(e.value).lower()


def test_times_count_must_be_positive_int():
    with pytest.raises(LoadError) as e:
        load_flow(TIMES_BAD_COUNT)
    assert "times" in str(e.value).lower()


def test_max_below_one_is_rejected():
    with pytest.raises(LoadError) as e:
        load_flow(BAD_MAX)
    assert "max" in str(e.value).lower()
    assert ">= 1" in str(e.value)


def test_while_predicate_unknown_name_is_rejected():
    with pytest.raises(LoadError) as e:
        load_flow(BAD_PREDICATE_NAME)
    assert "while" in str(e.value).lower()
    assert "exted" in str(e.value)
    assert e.value.diagnostic.frames[-1].range.start_line is not None


# The predicate's carried-name heads are extracted by PARSING the whole `while:` value
# (via `condition_refs`), so the three condition spellings must all build and all catch a
# dangling carried name — not just the mixed `not ${exited}` form GOOD uses.
WHILE_BARE = GOOD.replace("    while: not ${exited}\n", "    while: not exited\n")
WHILE_WHOLE = GOOD.replace("    while: not ${exited}\n", "    while: ${not exited}\n")
WHILE_BARE_BAD = GOOD.replace("    while: not ${exited}\n", "    while: not exted\n")
WHILE_WHOLE_BAD = GOOD.replace("    while: not ${exited}\n", "    while: ${not exted}\n")


@pytest.mark.parametrize("src", [WHILE_BARE, WHILE_WHOLE])
def test_while_predicate_all_spellings_build(src):
    flow = load_flow(src)
    assert flow.compiled.nodes["chat_loop"].predicate_kind == "while"


@pytest.mark.parametrize("src", [WHILE_BARE_BAD, WHILE_WHOLE_BAD])
def test_while_predicate_unknown_name_rejected_in_all_spellings(src):
    with pytest.raises(LoadError) as e:
        load_flow(src)
    assert "exted" in str(e.value)


@pytest.mark.parametrize("bad", [BAD_MAX_STR, BAD_MAX_FLOAT, BAD_MAX_BOOL])
def test_non_int_max_is_rejected(bad):
    with pytest.raises(LoadError) as e:
        load_flow(bad)
    assert "max" in str(e.value).lower()
    assert "integer" in str(e.value)
