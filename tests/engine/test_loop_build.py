"""`LoopDescriptor.build`, loader dispatch, and the `'a -> 'a` type contract.

A `kind: loop` node bakes a `LoopNode` from a `LoopDescriptor`: it resolves+bakes the
body subflow (the `call:` child, a `defs:` callable here so no external resolver is
needed), stamps `output_contract`, and enforces the two halves of the type contract:

- FIELD-NAME-set (in `LoopDescriptor.build`): body output field NAMES subset of carried
  keys (an omitted field carries over), body input NAMES subset of carried keys — a
  located `LoadError` on a name mismatch.
- FULL TYPE equality (in the loader post-build pass `check_loop_type_contract`):
  each returned body output Type == carried record Type per field — a located
  `LoadError` on a same-names/different-TYPES mismatch (needs the assembled
  producers/flow-input types).
"""

import pytest

from yian.compose.loader import load_flow
from yian.compose.helpers.errors import LoadError
from yian.nodes.base import NodeKind
from yian.nodes.contracts import PartialRecordContract, ValueContract
from yian.run import run_flow
from yian.typesys.values import ValueKind



def _swap(src: str, old: str, new: str) -> str:
    """Replace `old` in a fixture, failing if it is absent (a stale needle)."""
    assert old in src, old
    return src.replace(old, new)


# A good loop: the body `bump` reads a subset of the carried record and returns the
# WHOLE next carried record ({n, exited}); `chat_loop` carries {n, exited}, loops
# while `not exited`, capped at max: 5.
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
    while: not exited
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


# The body output OMITS `exited`, so it returns a subset ({n}) of the carried keys
# ({n, exited}); the omitted `exited` carries over from the previous iteration.
SUBSET_BODY = """
id: loop-subset-body
name: loop_subset_body
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
    while: not exited
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
    while: not exited
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
# `max:` optional. `times: N` is a fixed count — no predicate, `max:` redundant.
UNTIL_GOOD = _swap(GOOD, "    while: not exited\n", "    until: ${exited}\n")
TIMES_GOOD = _swap(GOOD, "    while: not exited\n    max: 5\n", "    times: 3\n")
# Exactly one of while:/until:/times: — both-set and neither-set are illegal.
BOTH = _swap(GOOD, "    while: not exited\n", "    while: not exited\n    until: ${exited}\n")
NEITHER = _swap(GOOD, "    while: not exited\n", "")
# `times:` with `max:` is redundant (times bounds the count) -> rejected, not lenient-ignored.
TIMES_WITH_MAX = _swap(GOOD, "    while: not exited\n", "    times: 3\n")  # keeps max: 5
# A fixed count must permit at least one run: `times: 0` is nonsensical.
TIMES_BAD_COUNT = _swap(GOOD, "    while: not exited\n    max: 5\n", "    times: 0\n")
# Without `max:` (or with `max: null`) a predicate loop has no iteration limit.
NO_MAX = _swap(GOOD, "    max: 5\n", "")
UNTIL_NO_MAX = _swap(UNTIL_GOOD, "    max: 5\n", "")
NULL_MAX = _swap(GOOD, "    max: 5\n", "    max: null\n")


@pytest.mark.parametrize(
    "src, kind",
    [(NO_MAX, "while"), (UNTIL_NO_MAX, "until"), (NULL_MAX, "while")],
    ids=["while-no-max", "until-no-max", "while-null-max"],
)
def test_predicate_loop_without_max_builds_unlimited(src, kind):
    node = load_flow(src).compiled.nodes["chat_loop"]
    assert node.predicate_kind == kind
    assert node.max_iters is None


def test_predicate_loop_with_max_keeps_its_limit():
    assert load_flow(GOOD).compiled.nodes["chat_loop"].max_iters == 5


def test_times_with_null_max_builds():
    node = load_flow(
        _swap(TIMES_GOOD, "    times: 3\n", "    times: 3\n    max: null\n")
    ).compiled.nodes["chat_loop"]
    assert node.times == 3 and node.max_iters == 3


def test_good_loop_bakes_loopnode():
    flow = load_flow(GOOD)
    node = flow.compiled.nodes["chat_loop"]
    assert node.kind == NodeKind.LOOP
    assert node.child is not None
    # The output contract carries the body codomain — prove the contract was baked,
    # not just that a child was resolved.
    assert node.output_contract is not None
    assert set(node.output_contract.type.fields) == {"n", "exited"}


def _field_names(contract):
    """The ordered field names of an exact or partial record contract."""
    if type(contract) is PartialRecordContract:
        return tuple(name for name, _ in contract.fields)
    return tuple(contract.type.fields)


def _field_contract(contract, name):
    """One field's proven contract (exact type or partial entry), else None."""
    if type(contract) is PartialRecordContract:
        return dict(contract.fields)[name]
    return ValueContract(contract.type.fields[name])


def test_body_may_return_a_subset_of_carried_fields():
    node = load_flow(SUBSET_BODY).compiled.nodes["chat_loop"]
    assert node.returned_fields == ("n",)
    assert _field_names(node.output_contract) == ("n", "exited")


OUTSIDE_CARRIED = """
id: loop-outside-carried
name: loop_outside_carried
defs:
  bump:
    input:
      n: int
    nodes: {}
    output:
      n: ${input.n}
      extra: ${input.n}
nodes:
  chat_loop:
    kind: loop
    call: bump
    input:
      n: 0
      exited: false
    while: not exited
    max: 5
output: ${chat_loop.output}
"""


def test_body_output_field_outside_carried_record_is_rejected():
    with pytest.raises(LoadError) as e:
        load_flow(OUTSIDE_CARRIED)
    assert "body output field(s) ['extra'] are not carried record fields" in str(e.value)
    assert e.value.diagnostic.frames[-1].range.start_line is not None


# A body with `output: {n: ...}` returns a one-field record, so it returns the carried `n`.
SINGLE_OUTPUT_BODY = """
id: loop-single-output
name: loop_single_output
defs:
  bump:
    input:
      n: int
    nodes: {}
    output:
      n: ${input.n}
nodes:
  chat_loop:
    kind: loop
    call: bump
    input:
      n: 0
      exited: false
    while: not exited
    max: 5
output: ${chat_loop.output}
"""


def test_one_field_body_record_builds():
    node = load_flow(SINGLE_OUTPUT_BODY).compiled.nodes["chat_loop"]
    assert node.returned_fields == ("n",)
    assert _field_names(node.output_contract) == ("n", "exited")


# The one declared output `state` holds a record `{n}`. The body returns the record
# `{state: {n}}`, so only `state` changes; `{n}` is not read as the returned carried fields.
SINGLE_OUTPUT_WRAPS_RECORD = """
id: loop-single-output-record
name: loop_single_output_record
defs:
  bump:
    input:
      n: int
    nodes:
      step:
        kind: code
        input:
          n: ${input.n}
        output:
          n: int
        code: |
          return {"n": inputs["n"] + 100}
    output:
      state: ${step.output}
nodes:
  chat_loop:
    kind: loop
    call: bump
    input:
      n: 0
      state: init
    times: 2
output: ${chat_loop.output}
"""


def test_one_field_body_record_returns_only_that_carried_field():
    loaded = load_flow(SINGLE_OUTPUT_WRAPS_RECORD)
    assert loaded.compiled.nodes["chat_loop"].returned_fields == ("state",)
    assert run_flow(loaded, {}).output == {"n": 0, "state": {"n": 100}}


# A body must return a record with at least one carried field: a value body over a scalar
# returns no record, and a record body `output: {}` returns no field.
REJECTED_BODY = """
id: loop-rejected-body
name: loop_rejected_body
defs:
  bump:
    input:
      n: int
    nodes:
      step:
        kind: code
        input:
          n: ${input.n}
        output: int
        code: |
          return inputs["n"] + 1
    output: <output>
nodes:
  chat_loop:
    kind: loop
    call: bump
    input:
      n: 0
    times: 2
output: ${chat_loop.output}
"""


@pytest.mark.parametrize(
    "output", ["${step.output}", "{}"], ids=["scalar-value", "empty-record"]
)
def test_body_returning_no_carried_field_is_rejected(output):
    with pytest.raises(LoadError) as e:
        load_flow(_swap(REJECTED_BODY, "<output>", output))
    assert (
        "loop node 'chat_loop': body callable 'bump' must return a record with at least one "
        "carried field (a subset of the carried record ['n']); write `output: {field: ...}` "
        "or a whole reference to a node with a record output"
    ) in str(e.value)


# A value body re-exports the referenced record `{n}`, so it returns the carried `n` even
# though the carried record also has a field named `result`.
VALUE_BODY_BESIDE_CARRIED_RESULT = """
id: loop-value-body-carried-result
name: loop_value_body_carried_result
defs:
  bump:
    input:
      n: int
    nodes:
      step:
        kind: code
        input:
          n: ${input.n}
        output:
          n: int
        code: |
          return {"n": inputs["n"] + 1}
    output: ${step.output}
nodes:
  chat_loop:
    kind: loop
    call: bump
    input:
      result: start
      n: 0
    times: 2
output: ${chat_loop.output}
"""


def test_value_body_returns_the_referenced_record_beside_a_carried_result():
    loaded = load_flow(VALUE_BODY_BESIDE_CARRIED_RESULT)
    assert loaded.compiled.nodes["chat_loop"].returned_fields == ("n",)
    assert run_flow(loaded, {}).output == {"result": "start", "n": 2}


def test_total_loop_keeps_body_contract_and_no_returned_fields():
    node = load_flow(GOOD).compiled.nodes["chat_loop"]
    assert node.returned_fields is None
    contract = node.output_contract
    assert type(contract) is ValueContract
    assert tuple(contract.type.fields) == ("n", "exited")
    assert contract.type.fields["n"].kind is ValueKind.INTEGER
    assert contract.type.fields["exited"].kind is ValueKind.BOOLEAN


# A chat-shaped loop: the body reads `person`/`talking_to` but returns only
# `{exited, reply}`; `person` and `talking_to` carry over and type from their seeds.
CHAT = """
id: loop-chat
name: loop_chat
typedefs:
  Person:
    name: str
    handle: str
input:
  person: Person
  talking_to: str
defs:
  turn:
    input:
      person: Person
      talking_to: str
      exited: bool
      reply: str
    nodes: {}
    output:
      exited: ${input.exited}
      reply: ${input.talking_to}
nodes:
  chat:
    kind: loop
    call: turn
    input:
      person: ${input.person}
      talking_to: ${input.talking_to}
      exited: false
      reply: ""
    while: not exited
    max: 3
  sink:
    kind: code
    code: tests.engine._compose_codefns:echo_x
    input:
      x: ${chat.output.person.name}
    output: str
output: ${sink.output}
"""


def test_merged_contract_types_omitted_field_from_its_seed():
    loaded = load_flow(CHAT)
    node = loaded.compiled.nodes["chat"]
    assert node.returned_fields == ("exited", "reply")
    contract = node.output_contract
    assert type(contract) is ValueContract
    assert tuple(contract.type.fields) == ("person", "talking_to", "exited", "reply")
    person = contract.type.fields["person"]
    assert person.kind is ValueKind.OBJECT
    assert set(person.fields) == {"name", "handle"}
    assert contract.type.fields["talking_to"].kind is ValueKind.STRING
    assert contract.type.fields["exited"].kind is ValueKind.BOOLEAN

    with pytest.raises(LoadError, match="unknown field 'missing'"):
        load_flow(
            _swap(CHAT, "${chat.output.person.name}", "${chat.output.person.missing}")
        )


KEEP_LITERAL = _swap(
    SUBSET_BODY, "      exited: false\n    while:", "      exited: false\n      keep: 3\n    while:"
)


def test_merged_contract_keeps_literal_seed_opaque():
    node = load_flow(KEEP_LITERAL).compiled.nodes["chat_loop"]
    assert node.returned_fields == ("n",)
    contract = node.output_contract
    assert type(contract) is PartialRecordContract
    assert _field_names(contract) == ("n", "exited", "keep")
    assert dict(contract.fields)["keep"] is None
    assert dict(contract.fields)["n"].type.kind is ValueKind.INTEGER


# `second` (declared first, so it builds `first` on demand) seeds `person` from
# `first`'s carried-over `person` and omits it too.
CHAINED = """
id: loop-chained
name: loop_chained
typedefs:
  Person:
    name: str
    handle: str
input:
  person: Person
  talking_to: str
defs:
  turn:
    input:
      person: Person
      talking_to: str
      exited: bool
      reply: str
    nodes: {}
    output:
      exited: ${input.exited}
      reply: ${input.talking_to}
nodes:
  second:
    kind: loop
    call: turn
    input:
      person: ${first.output.person}
      talking_to: ${first.output.talking_to}
      exited: false
      reply: ""
    times: 1
  first:
    kind: loop
    call: turn
    input:
      person: ${input.person}
      talking_to: ${input.talking_to}
      exited: false
      reply: ""
    times: 1
output: ${second.output.person.name}
"""


def test_loop_seeded_from_partial_loop_sees_merged_type():
    node = load_flow(CHAINED).compiled.nodes["second"]
    person = _field_contract(node.output_contract, "person")
    assert person is not None
    assert person.type.kind is ValueKind.OBJECT
    assert set(person.type.fields) == {"name", "handle"}


# `first` and `second` each omit `keep` and seed it from the other's output, so resolving
# either merged contract needs the other built first.
SEED_CYCLE = """
id: loop-seed-cycle
name: loop_seed_cycle
defs:
  bump:
    input:
      n: int
      exited: bool
    nodes:
      step:
        kind: code
        code: tests.engine._compose_codefns:loop_bump
        input:
          n: ${input.n}
          exited: ${input.exited}
        output:
          n: int
    output: ${step.output}
nodes:
  first:
    kind: loop
    call: bump
    input:
      n: 0
      exited: false
      keep: ${second.output.keep}
    times: 1
  second:
    kind: loop
    call: bump
    input:
      n: 0
      exited: false
      keep: ${first.output.keep}
    times: 1
output: ${second.output}
"""


def test_omitted_field_seed_cycle_is_located():
    with pytest.raises(LoadError) as caught:
        load_flow(SEED_CYCLE)
    message = str(caught.value)
    assert "dependency cycle" in message
    assert "first.input.keep -> second.input.keep" in message
    # The binding that closes the cycle carries the caret, not the coarse `input:` span.
    closing_line = SEED_CYCLE.splitlines().index("      keep: ${first.output.keep}") + 1
    rng = caught.value.diagnostic.frames[-1].range
    assert (rng.start_line, rng.end_line) == (closing_line, closing_line)


PARTIAL_BAD_TYPES = """
id: loop-partial-bad-types
name: loop_partial_bad_types
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
    output: ${step.output}
nodes:
  chat_loop:
    kind: loop
    call: bump
    input:
      n: ${input.n0}
      exited: false
    while: not exited
    max: 5
output: ${chat_loop.output}
"""


def test_returned_field_type_mismatch_still_rejected_in_partial_body():
    with pytest.raises(LoadError, match="carried field 'n'"):
        load_flow(PARTIAL_BAD_TYPES)


# `step` declares no `output:`, so `n` projects an untyped producer value.
PARTIAL_OPAQUE_RETURNED = """
id: loop-partial-opaque-returned
name: loop_partial_opaque_returned
input:
  note0: str
  k0: int
defs:
  bump:
    input:
      n: int
      note: str
    nodes:
      step:
        kind: code
        code: tests.engine._compose_codefns:loop_bump
        input:
          n: ${input.n}
          exited: false
    output:
      n: ${step.output.n}
      note: ${input.note}
nodes:
  chat_loop:
    kind: loop
    call: bump
    input:
      n: 1
      note: ${input.note0}
      keep: ${input.k0}
    times: 1
output: ${chat_loop.output}
"""


def test_partial_body_with_opaque_returned_field():
    node = load_flow(PARTIAL_OPAQUE_RETURNED).compiled.nodes["chat_loop"]
    assert node.returned_fields == ("n", "note")
    contract = node.output_contract
    assert type(contract) is PartialRecordContract
    fields = dict(contract.fields)
    assert _field_names(contract) == ("n", "note", "keep")
    assert fields["n"] is None
    assert fields["note"].type.kind is ValueKind.STRING
    assert fields["keep"].type.kind is ValueKind.INTEGER


INLINE_SUBSET = """
id: loop-inline-subset
name: loop_inline_subset
nodes:
  repeat:
    kind: loop
    times: 2
    input:
      n: 0
      exited: false
    nodes:
      bump:
        kind: code
        code: tests.engine._compose_codefns:loop_bump
        input:
          n: ${input.n}
          exited: ${input.exited}
        output:
          n: int
    output: ${bump.output}
output: ${repeat.output}
"""


def test_inline_loop_body_may_return_a_subset():
    node = load_flow(INLINE_SUBSET).compiled.nodes["repeat"]
    assert node.returned_fields == ("n",)
    assert _field_names(node.output_contract) == ("n", "exited")


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
    while: not exited
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
BAD_MAX = _swap(GOOD, "    max: 5\n", "    max: 0\n")


# A `while:` predicate that references a name NOT in the carried record ({n, exited}) — here a
# typo `exted` for `exited`. Record-scoped eval would resolve it falsy and spin the loop (forever
# without `max:`), so it must be rejected loudly at build.
BAD_PREDICATE_NAME = _swap(GOOD, "    while: not exited\n", "    while: not exted\n")

# `max:` must be a plain integer. The descriptor is a bare dataclass (its `Optional[int]`
# annotation isn't enforced), so these non-int YAML types reach `LoopDescriptor.build`:
# a quoted string, a float, and a bool (an int subclass that would silently read as 0/1).
BAD_MAX_STR = _swap(GOOD, "    max: 5\n", '    max: "5"\n')
BAD_MAX_FLOAT = _swap(GOOD, "    max: 5\n", "    max: 2.5\n")
BAD_MAX_BOOL = _swap(GOOD, "    max: 5\n", "    max: true\n")


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
# (via `condition_refs`), so both condition spellings must build and catch a
# dangling carried name. GOOD uses the bare form.
WHILE_BARE = GOOD
WHILE_WHOLE = _swap(GOOD, "    while: not exited\n", "    while: ${not exited}\n")
WHILE_BARE_BAD = _swap(GOOD, "    while: not exited\n", "    while: not exted\n")
WHILE_WHOLE_BAD = _swap(GOOD, "    while: not exited\n", "    while: ${not exted}\n")


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


# `until: done` reads only `done`, which the body never returns, and there is no `max:`,
# so the loop can never end.
NEVER_ENDS = """
id: loop-never-ends
name: loop_never_ends
defs:
  body:
    input:
      done: bool
      attempts: int
    nodes: {}
    output:
      attempts: ${input.attempts + 1}
nodes:
  wait:
    kind: loop
    call: body
    input:
      done: false
      attempts: 0
    until: done
output: ${wait.output}
"""


def test_loop_whose_condition_reads_only_unreturned_fields_is_a_load_error():
    with pytest.raises(LoadError) as exc:
        load_flow(NEVER_ENDS)
    msg = str(exc.value)
    assert "this loop can never end" in msg
    assert "`until: done`" in msg
    assert "reads `done`" in msg
    assert "return `done` from the body" in msg
    assert "add `max:`" in msg


def test_while_loop_whose_condition_reads_only_unreturned_fields_is_a_load_error():
    # SUBSET_BODY returns only `n`; `while: not exited` reads only `exited`.
    with pytest.raises(LoadError, match=r"this loop can never end: `while: not exited`"):
        load_flow(_swap(SUBSET_BODY, "    max: 5\n", ""))


def test_never_ending_error_names_every_unreturned_field_read():
    src = """
id: loop-never-ends-two
name: loop_never_ends_two
defs:
  body:
    input:
      a: bool
      b: bool
      n: int
    nodes: {}
    output:
      n: ${input.n + 1}
nodes:
  wait:
    kind: loop
    call: body
    input:
      a: false
      b: false
      n: 0
    until: a or b
output: ${wait.output}
"""
    with pytest.raises(LoadError) as exc:
        load_flow(src)
    msg = str(exc.value)
    assert "reads `a`, `b`" in msg
    assert "return `a` or `b` from the body" in msg


def test_inline_body_loop_whose_condition_never_changes_is_a_load_error():
    # INLINE_SUBSET's body returns only `n`; `until: exited` reads only `exited`.
    with pytest.raises(LoadError, match="this loop can never end: `until: exited`"):
        load_flow(_swap(INLINE_SUBSET, "    times: 2\n", "    until: exited\n"))


@pytest.mark.parametrize(
    "pred", ["done", "${done}", "${not done}", "not done"],
)
def test_never_ending_check_reads_every_condition_spelling(pred):
    src = _swap(NEVER_ENDS, "    until: done\n", f'    until: "{pred}"\n')
    assert src != NEVER_ENDS
    with pytest.raises(LoadError, match="reads `done`, which the body never updates"):
        load_flow(src)


def test_null_max_still_gets_the_never_ending_error():
    src = _swap(NEVER_ENDS, "    until: done\n", "    until: done\n    max: null\n")
    assert src != NEVER_ENDS
    with pytest.raises(LoadError, match="this loop can never end"):
        load_flow(src)


def test_dotted_condition_ref_names_the_carried_field():
    src = _swap(NEVER_ENDS, "      done: false\n", "      st: {ok: false}\n").replace(
        "      done: bool\n", "      st: Any\n"
    ).replace("    until: done\n", "    until: st.ok\n")
    with pytest.raises(LoadError, match="reads `st`.*return `st` from the body"):
        load_flow(src)


@pytest.mark.parametrize(
    "src",
    [
        # two drivers: only the driver-count error
        _swap(NEVER_ENDS, "    until: done\n", "    until: done\n    while: not done\n"),
        # times + while: only the driver-count error
        _swap(NEVER_ENDS, "    until: done\n", "    times: 2\n    while: not done\n"),
        # unknown name: only the unknown-name error
        _swap(NEVER_ENDS, "    until: done\n", "    until: done or nope\n"),
        # invalid max: only the max error
        _swap(NEVER_ENDS, "    until: done\n", "    until: done\n    max: 0\n"),
        # parse error: only the parse error (no heads)
        _swap(NEVER_ENDS, "    until: done\n", '    until: "done and"\n'),
    ],
    ids=["two-drivers", "times-and-while", "unknown-name", "bad-max", "parse-error"],
)
def test_never_ending_error_does_not_stack_on_other_loop_errors(src):
    with pytest.raises(LoadError) as exc:
        load_flow(src)
    assert "can never end" not in str(exc.value)


def test_never_changing_condition_with_max_loads():
    node = load_flow(
        _swap(NEVER_ENDS, "    until: done\n", "    until: done\n    max: 3\n")
    ).compiled.nodes["wait"]
    assert node.max_iters == 3
    assert node.returned_fields == ("attempts",)


def test_condition_reading_one_returned_field_loads():
    # `done` is never returned, but `attempts` is, so the condition can change.
    node = load_flow(
        _swap(NEVER_ENDS, "    until: done\n", "    until: done or attempts >= 3\n")
    ).compiled.nodes["wait"]
    assert node.max_iters is None


UNTYPED_BODY = _swap(NEVER_ENDS, 
    "    nodes: {}\n    output:\n      attempts: ${input.attempts + 1}\n",
    "    nodes:\n"
    "      step:\n"
    "        kind: code\n"
    "        input:\n"
    "          a: ${input.attempts}\n"
    "        code: |\n"
    "          return {\"attempts\": inputs[\"a\"] + 1}\n"
    "    output: ${step.output}\n",
)


def test_unknown_returned_fields_get_only_the_record_error():
    # A value body over an untyped CODE node: its returned field set is unknown at load,
    # so only the "must return a record" error applies.
    assert UNTYPED_BODY != NEVER_ENDS
    with pytest.raises(LoadError) as exc:
        load_flow(UNTYPED_BODY)
    msg = str(exc.value)
    assert "must return a record" in msg
    assert "can never end" not in msg


# A `while:`/`until:` predicate reads fields below a carried name. Those reads are checked
# against the carried record's type at load, like a CASE `when:`: a misspelled nested field
# would read as null at run time, so `while:` would run zero times and `until:` would never end.
NESTED_STATE = """
id: loop-nested
name: loop_nested
nodes:
  repeat:
    kind: loop
    PRED
    input:
      state: {count: 0}
    nodes:
      bump:
        kind: code
        input:
          s: ${input.state}
        output:
          state:
            count: int
        code: |
          return {"state": {"count": inputs["s"]["count"] + 1}}
    output: ${bump.output}
output: ${repeat.output}
"""


@pytest.mark.parametrize(
    "pred",
    ["while: state.count < 3", "until: state.count >= 3", "while: ${state.count} < 3"],
)
def test_predicate_nested_field_read_builds(pred):
    flow = load_flow(_swap(NESTED_STATE, "PRED", pred))
    assert run_flow(flow, {}).output == {"state": {"count": 3}}


@pytest.mark.parametrize(
    "pred",
    ["while: state.cuont < 3", "until: state.cuont >= 3", "while: ${state.cuont < 3}"],
)
def test_predicate_unknown_nested_field_is_rejected(pred):
    with pytest.raises(LoadError) as e:
        load_flow(_swap(NESTED_STATE, "PRED", pred))
    message = str(e.value)
    assert f"`{pred.split(':', 1)[0]}:`" in message
    assert "reads unknown field 'cuont' on record" in message
    assert e.value.diagnostic.frames[-1].range.start_line is not None


def test_predicate_field_read_on_scalar_is_rejected():
    src = _swap(GOOD, "    while: not exited\n", "    while: n.value < 3\n")
    with pytest.raises(LoadError) as e:
        load_flow(src)
    assert "reads field 'value' on type int, which has no fields" in str(e.value)


# An omitted carried field keeps its seed's type, so a predicate read below it is checked
# against that seed contract; a literal seed stays opaque and is not checked.
OMITTED_SEED = """
id: loop-omitted
name: loop_omitted
input:
  cfg: {limit: int}
nodes:
  repeat:
    kind: loop
    PRED
    input:
      n: 0
      cfg: SEED
    nodes:
      inc:
        kind: code
        input:
          n: ${input.n}
        output: int
        code: |
          return inputs["n"] + 1
    output:
      n: ${inc.output}
output: ${repeat.output}
"""


def test_predicate_unknown_field_on_omitted_typed_seed_is_rejected():
    src = _swap(OMITTED_SEED, "PRED", "while: n < cfg.limt").replace("SEED", "${input.cfg}")
    with pytest.raises(LoadError) as e:
        load_flow(src)
    assert "reads unknown field 'limt' on record" in str(e.value)


def test_predicate_field_on_omitted_typed_seed_builds():
    src = _swap(OMITTED_SEED, "PRED", "while: n < cfg.limit").replace("SEED", "${input.cfg}")
    result = run_flow(load_flow(src), {"cfg": {"limit": 2}})
    assert result.output == {"n": 2, "cfg": {"limit": 2}}


def test_predicate_field_on_omitted_literal_seed_is_not_checked():
    src = _swap(OMITTED_SEED, "PRED", "while: n < cfg.other\n    max: 3").replace(
        "SEED", "{limit: 2}"
    )
    load_flow(src)


# A predicate reads a union's member-only field after a top-level `isinstance` conjunct on
# the same value: `and` short-circuits, so the read only runs on that member, as in a CASE.
UNION_STATE = """
id: loop-union
name: loop_union
typedefs:
  Done: {answer: str}
  More: {hint: str}
  Step: Union[Done, More]
nodes:
  ask:
    kind: loop
    PRED
    input:
      n: 0
      r: Step = {"hint":"start"}
    nodes:
      step:
        kind: code
        input:
          n: ${input.n}
        output:
          n: int
          r: Step
        code: |
          n = inputs["n"] + 1
          return {"n": n, "r": {"answer": "ok"} if n >= 3 else {"hint": "again"}}
    output: ${step.output}
output: ${ask.output}
"""


@pytest.mark.parametrize(
    "pred",
    [
        "until: isinstance(r, Done) and r.answer == 'ok'",
        "until: ${isinstance(r, Done) and r.answer == 'ok'}",
        "until: isinstance(${r}, Done) and n > 0 and ${r.answer} == 'ok'",
    ],
)
def test_predicate_member_field_read_guarded_by_isinstance_builds(pred):
    flow = load_flow(_swap(UNION_STATE, "PRED", pred))
    result = run_flow(flow, {})
    assert result.status == "succeeded", result.failure
    assert result.output == {"n": 3, "r": {"answer": "ok"}}


@pytest.mark.parametrize(
    "pred",
    [
        "until: r.answer == 'ok'",
        # Only a top-level `and` part guards: under `or` or `not` the check may not hold.
        "until: n > 5 or isinstance(r, Done) and r.answer == 'ok'",
        "while: not (isinstance(r, Done) and r.answer == 'ok')",
        # The check must come before the read.
        "until: r.answer == 'ok' and isinstance(r, Done)",
    ],
)
def test_predicate_unguarded_member_field_read_is_rejected(pred):
    with pytest.raises(LoadError) as e:
        load_flow(_swap(UNION_STATE, "PRED", pred))
    message = str(e.value)
    assert "reads field 'answer', which only 1 of 2 union members have" in message
    assert "check the member first with `isinstance(r, <Type>) and ...`" in message
    assert "case" not in message


def test_predicate_read_of_a_field_the_checked_member_lacks_is_rejected():
    pred = "until: isinstance(r, More) and r.answer == 'ok'"
    with pytest.raises(LoadError, match="reads unknown field 'answer' on record"):
        load_flow(_swap(UNION_STATE, "PRED", pred))


def test_predicate_tagged_member_field_error_points_at_isinstance():
    src = (
        _swap(UNION_STATE, 
            "  Done: {answer: str}\n  More: {hint: str}\n",
            "  Done: {kind: 'Literal[done]', answer: str}\n"
            "  More: {kind: 'Literal[more]', hint: str}\n",
        )
        .replace('{"hint":"start"}', '{"kind":"more","hint":"start"}')
        .replace('{"answer": "ok"}', '{"kind": "done", "answer": "ok"}')
        .replace('{"hint": "again"}', '{"kind": "more", "hint": "again"}')
    )
    with pytest.raises(LoadError) as e:
        load_flow(src.replace("PRED", "until: r.answer == 'ok'"))
    message = str(e.value)
    assert "reads field 'answer', which only union members with kind done have" in message
    assert "check the member first with `isinstance(r, <Type>) and ...`" in message
    assert "case on" not in message
    flow = load_flow(src.replace("PRED", "until: isinstance(r, Done) and r.answer == 'ok'"))
    assert run_flow(flow, {}).output == {"n": 3, "r": {"kind": "done", "answer": "ok"}}


def test_predicate_isinstance_check_on_an_object_field_checks_reads_on_the_type():
    src = _swap(UNION_STATE, 'r: Step = {"hint":"start"}', 'r: object = {"hint":"start"}').replace(
        "          r: Step\n", "          r: object\n"
    )
    with pytest.raises(LoadError, match="reads unknown field 'answr' on record"):
        load_flow(src.replace("PRED", "until: isinstance(r, Done) and r.answr == 'ok'"))
    flow = load_flow(src.replace("PRED", "until: isinstance(r, Done) and r.answer == 'ok'"))
    assert run_flow(flow, {}).output == {"n": 3, "r": {"answer": "ok"}}


def test_predicate_isinstance_check_with_an_unknown_type_reports_the_type():
    # A typo in the checked type is reported as that typo, not as an unguarded read.
    pred = "until: isinstance(r, Dnoe) and r.answer == 'ok'"
    with pytest.raises(LoadError) as e:
        load_flow(_swap(UNION_STATE, "PRED", pred))
    message = str(e.value)
    assert "unknown type 'Dnoe'" in message
    assert "union members" not in message
