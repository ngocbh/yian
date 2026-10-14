"""Typed LOOP seeds: `name: !Type value` on a loop's `input:` types its carried record.

A loop's `input:` is the carried record's turn-0 seed, so a typed seed is checked when the
driver binds it, and a field the body omits carries the declared type into the loop's merged
output contract.
"""

from __future__ import annotations

from yian.compose.loader import load_flow
from yian.nodes.contracts import Binding, ValueContract
from yian.run import run_flow
from yian.typesys import TypeRegistry, type_for
from yian.typesys.values import Type, ValueKind


TYPED_COUNTER = """
id: typed-counter
name: typed_counter
nodes:
  loop:
    kind: loop
    times: 3
    input:
      rounds: !int 0
    nodes:
      bump:
        kind: code
        input:
          n: ${input.rounds}
        output: int
        code: |
          return inputs["n"] + 1
    output:
      rounds: ${bump.output}
output: ${loop.output}
"""


def test_typed_literal_seed_binds_its_value_and_runs():
    """A typed literal seed carries its declared contract and seeds the first turn."""
    loaded = load_flow(TYPED_COUNTER)
    binding = loaded.compiled.wiring["loop"]["rounds"]
    assert type(binding) is Binding
    assert binding.source == 0
    assert binding.contract.type == type_for("int", TypeRegistry())

    result = run_flow(loaded, {})
    assert result.status == "succeeded", result.failure
    assert result.output == {"rounds": 3}


# `p` is an untyped CODE node, so the `label` seed's inferred contract is opaque; only its
# declared type can prove what the omitted field carries. The body is a named `defs:` flow
# so its own `n: int` parameter, not the seed declarations, types what it receives.
PARTIAL_TYPED_SEED = """
id: partial-typed-seed
name: partial_typed_seed
defs:
  bump:
    input:
      n: !int
    nodes:
      step:
        kind: code
        input:
          n: ${input.n}
        output: int
        code: |
          return inputs["n"] + 1
    output:
      n: ${step.output}
nodes:
  p:
    kind: code
    code: |
      return "seed"
  loop:
    kind: loop
    times: 2
    input:
      n: 0
      label: !Optional(str) ${p.output}
    call: bump
output: ${loop.output}
"""


def test_omitted_typed_seed_carries_its_declared_type():
    """A field the body omits keeps its declared seed type in the loop's output contract."""
    loaded = load_flow(PARTIAL_TYPED_SEED)
    loop = loaded.compiled.nodes["loop"]
    assert loop.returned_fields == ("n",)
    contract = loop.output_contract
    registry = TypeRegistry()
    assert type(contract) is ValueContract
    assert contract.type == Type(
        kind=ValueKind.OBJECT,
        fields={
            "label": type_for("Optional(str)", registry),
            "n": type_for("int", registry),
        },
        required=frozenset({"label", "n"}),
    )

    result = run_flow(loaded, {})
    assert result.status == "succeeded", result.failure
    assert result.output == {"n": 2, "label": "seed"}


MISFIT_SEED = """
id: misfit-seed
name: misfit_seed
defs:
  bump:
    input:
      n: !int
    nodes:
      step:
        kind: code
        input:
          n: ${input.n}
        output: int
        code: |
          return inputs["n"] + 1
    output:
      n: ${step.output}
nodes:
  p:
    kind: code
    code: |
      return "x"
  loop:
    kind: loop
    times: 2
    input:
      n: !int ${p.output}
    call: bump
output: ${loop.output}
"""


def test_typed_seed_rejects_a_value_that_does_not_fit():
    """A seed value outside its declared type fails the loop on that binding."""
    result = run_flow(load_flow(MISFIT_SEED), {})
    assert result.status == "failed"
    assert result.failure.node_id == "loop"
    assert result.failure.code == "binding.type"
    assert "binding 'n'" in result.failure.message
