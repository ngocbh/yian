"""Downstream checks see a typed binding's declared type.

A binding tagged with a type (`name: !Type value`) on a CALL, MAP, or LOOP node declares the type its
child receives, so the cross-flow checks compare that declared type with the child's
parameters instead of the type inferred from the value. The binder checks the declared type
at runtime, which is what lets a typed binding narrow a nullable source.
"""

from __future__ import annotations

import pytest

from yian.compile.model import START_ID
from yian.compose.helpers.errors import LoadError
from yian.compose.loader import load_flow
from yian.nodes.contracts import ValueContract
from yian.run import run_flow
from yian.typesys import TypeRegistry, type_for


_STRICT_CHILD = """
defs:
  child:
    input:
      x: !int
    nodes:
      twice:
        kind: code
        input:
          x: ${input.x}
        output: int
        code: |
          return inputs["x"] * 2
    output: ${twice.output}
"""


def _call_flow(binding: str, *, producer: str) -> str:
    """Return a flow that calls the strict `child` with `x` bound as `binding`.

    Args:
        binding (`str`): The value of the CALL node's `x:` binding.
        producer (`str`): The YAML of the parent producer nodes, indented under `nodes:`.
    """
    return f"""
id: downstream
name: downstream
{_STRICT_CHILD}nodes:
{producer}  run:
    kind: call
    call: child
    input:
      x: {binding}
output: ${{run.output}}
"""


def _maybe(value: str) -> str:
    """Return a `maybe` CODE producer declaring `Optional(int)` that returns `value`.

    Args:
        value (`str`): The Python expression the producer returns.
    """
    return (
        "  maybe:\n"
        "    kind: code\n"
        "    output: Optional(int)\n"
        "    code: |\n"
        f"      return {value}\n"
    )


def _int_producer() -> str:
    """Return an `i` CODE producer declaring `int`."""
    return (
        "  i:\n"
        "    kind: code\n"
        "    output: int\n"
        "    code: |\n"
        "      return 3\n"
    )


# --- CALL ------------------------------------------------------------------ #


def test_untyped_nullable_source_into_strict_child_fails_to_load():
    """An untyped binding keeps the cross-flow Optional rule."""
    with pytest.raises(LoadError, match="binding 'x'"):
        load_flow(_call_flow("${maybe.output}", producer=_maybe("3")))


def test_typed_binding_narrows_a_nullable_source_into_a_strict_child():
    """A declared `int` loads against a strict child and is checked when the run binds it."""
    loaded = load_flow(_call_flow("!int ${maybe.output}", producer=_maybe("3")))

    result = run_flow(loaded, {})

    assert result.status == "succeeded", result.failure
    assert result.output == 6


def test_typed_binding_rejects_a_null_at_runtime():
    """The narrowed binding fails on the CALL node when the source returns null."""
    loaded = load_flow(_call_flow("!int ${maybe.output}", producer=_maybe("None")))

    result = run_flow(loaded, {})

    assert result.status == "failed"
    assert result.failure.code == "binding.type"
    assert result.failure.node_id == "run"


def test_typed_nullable_binding_into_strict_child_fails_to_load():
    """A declared `Optional(int)` cannot fill a strict `int` child parameter."""
    with pytest.raises(LoadError, match="binding 'x'"):
        load_flow(_call_flow("!Optional(int) ${maybe.output}", producer=_maybe("3")))


def test_declared_float_does_not_fit_a_strict_int_child():
    """The child sees the declared `float`, not the inferred `int` of the source."""
    with pytest.raises(LoadError, match="binding 'x'.*'float'"):
        load_flow(_call_flow("!float ${i.output}", producer=_int_producer()))


def test_map_binding_uses_the_declared_type():
    """A MAP binding's declared type is compared with the child's parameter too."""
    text = f"""
id: downstream-map
name: downstream_map
{_STRICT_CHILD}nodes:
{_int_producer()}  each:
    kind: map
    over: [1, 2]
    call: child
    input:
      x: !float ${{i.output}}
output: ${{each.output}}
"""
    with pytest.raises(LoadError, match="binding 'x'.*'float'"):
        load_flow(text)


# --- LOOP ------------------------------------------------------------------ #


def _loop_flow(seed: str, *, producer: str) -> str:
    """Return a flow whose LOOP seeds `n` as `seed` into a named body returning `n: int`.

    Args:
        seed (`str`): The value of the LOOP node's `n:` seed.
        producer (`str`): The YAML of the parent producer nodes, indented under `nodes:`.
    """
    return f"""
id: downstream-loop
name: downstream_loop
defs:
  bump:
    input:
      n: !int
    nodes:
      step:
        kind: code
        input:
          n: ${{input.n}}
        output: int
        code: |
          return inputs["n"] + 1
    output:
      n: ${{step.output}}
nodes:
{producer}  loop:
    kind: loop
    times: 2
    input:
      n: {seed}
    call: bump
output: ${{loop.output}}
"""


def _opaque(value: str) -> str:
    """Return an `opaque` CODE producer with no declared output that returns `value`.

    Args:
        value (`str`): The Python expression the producer returns.
    """
    return (
        "  opaque:\n"
        "    kind: code\n"
        "    code: |\n"
        f"      return {value}\n"
    )


def test_declared_seed_type_is_compared_with_the_returned_field():
    """A seed declared `str` cannot carry a field the body returns as `int`.

    The producer is untyped, so only the declaration gives the seed a type: the same
    seed written without one loads.
    """
    load_flow(_loop_flow("${opaque.output}", producer=_opaque("1")))

    with pytest.raises(LoadError, match="carried field 'n'.*carried source is 'str'"):
        load_flow(_loop_flow("!str ${opaque.output}", producer=_opaque("1")))


def test_declared_seed_type_types_an_opaque_producer():
    """A seed declared `int` over an untyped producer matches an `int` body field."""
    loaded = load_flow(_loop_flow("!int ${opaque.output}", producer=_opaque("1")))

    result = run_flow(loaded, {})

    assert result.status == "succeeded", result.failure
    assert result.output == {"n": 3}


def test_declared_seed_type_is_checked_on_the_loop_node():
    """A declared `int` seed rejects a `"1"` on the LOOP node itself.

    Untyped, the text reaches the body's own `int` parameter, which does not convert
    it across the LOOP edge, so the body's start fails.
    """
    untyped = run_flow(load_flow(_loop_flow("${opaque.output}", producer=_opaque('"1"'))), {})
    assert untyped.status == "failed"
    assert untyped.failure.code == "contract.input"
    assert untyped.failure.node_id == "loop#0/__start__"

    loaded = load_flow(_loop_flow("!int ${opaque.output}", producer=_opaque('"1"')))

    result = run_flow(loaded, {})

    assert result.status == "failed"
    assert result.failure.code == "binding.type"
    assert result.failure.node_id == "loop"


# --- inline bodies --------------------------------------------------------- #


def _start_params(compiled, node_id: str) -> dict:
    """Return the START parameters of the child flow baked onto `node_id`, keyed by name.

    Args:
        compiled: The parent's compiled flow.
        node_id (`str`): The CALL, MAP, or LOOP node carrying the child.
    """
    child = compiled.nodes[node_id].child
    return {param.name: param for param in child.nodes[START_ID].params}


def test_typed_inline_binding_skips_inference_of_an_opaque_expression():
    """A declared type types the inline parameter even when its value cannot be inferred."""
    loaded = load_flow(
        """
id: inline-typed
name: inline_typed
nodes:
  a:
    kind: code
    code: |
      return 4
  wrap:
    kind: call
    input:
      n: !int ${a.output + 1}
    nodes:
      twice:
        kind: code
        input:
          n: ${input.n}
        output: int
        code: |
          return inputs["n"] * 2
    output: ${twice.output}
output: ${wrap.output}
"""
    )

    param = _start_params(loaded.compiled, "wrap")["n"]
    assert param.contract == ValueContract(type_for("int", TypeRegistry()))
    assert param.required is True

    result = run_flow(loaded, {})
    assert result.status == "succeeded", result.failure
    assert result.output == 10


def test_typed_inline_binding_types_an_empty_list():
    """An empty list literal takes its element type from the declaration."""
    loaded = load_flow(
        """
id: inline-empty
name: inline_empty
nodes:
  wrap:
    kind: call
    input:
      items: !list(str) []
    nodes:
      count:
        kind: code
        input:
          items: ${input.items}
        output: int
        code: |
          return len(inputs["items"])
    output: ${count.output}
output: ${wrap.output}
"""
    )

    param = _start_params(loaded.compiled, "wrap")["items"]
    assert param.contract.type == type_for("list(str)", TypeRegistry())

    result = run_flow(loaded, {})
    assert result.status == "succeeded", result.failure
    assert result.output == 0


def test_typed_inline_loop_seed_types_the_body_parameter():
    """A `float` seed lets an inline body carry a float past its int literal seed."""
    loaded = load_flow(
        """
id: inline-float-loop
name: inline_float_loop
nodes:
  loop:
    kind: loop
    times: 2
    input:
      r: !float 0
    nodes:
      half:
        kind: code
        input:
          r: ${input.r}
        output: float
        code: |
          return inputs["r"] + 0.5
    output:
      r: ${half.output}
output: ${loop.output}
"""
    )

    param = _start_params(loaded.compiled, "loop")["r"]
    assert param.contract.type == type_for("float", TypeRegistry())

    result = run_flow(loaded, {})
    assert result.status == "succeeded", result.failure
    assert result.output == {"r": 1.0}


def test_typed_inline_loop_seed_types_an_opaque_producer():
    """An `Optional(str)` seed over an untyped producer loads and carries its value."""
    loaded = load_flow(
        """
id: inline-optional-seed
name: inline_optional_seed
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
output: ${loop.output}
"""
    )

    param = _start_params(loaded.compiled, "loop")["label"]
    assert param.contract.type == type_for("Optional(str)", TypeRegistry())
    assert param.required is False

    result = run_flow(loaded, {})
    assert result.status == "succeeded", result.failure
    assert result.output == {"n": 2, "label": "seed"}


def _inline_map_flow(binding: str) -> str:
    """Return a flow whose inline MAP body shouts each item of an untyped producer.

    Args:
        binding (`str`): The value of the MAP node's `t:` binding.
    """
    return f"""
id: inline-map-item
name: inline_map_item
nodes:
{_opaque('["a", "b"]')}  each:
    kind: map
    over: ${{opaque.output}}
    input:
      t: {binding}
    nodes:
      shout:
        kind: code
        input:
          t: ${{input.t}}
        output: str
        code: |
          return inputs["t"].upper()
    output: ${{shout.output}}
output: ${{each.output}}
"""


def test_untyped_inline_map_item_over_an_opaque_list_fails_to_load():
    """An untyped `${item}` over an untyped producer has no element type to infer."""
    with pytest.raises(LoadError, match="cannot infer type for inline binding"):
        load_flow(_inline_map_flow("${item}"))


def test_typed_inline_map_item_binding_runs():
    """A typed `${item}` binding types an inline MAP body over an untyped producer."""
    loaded = load_flow(_inline_map_flow("!str ${item}"))

    param = _start_params(loaded.compiled, "each")["t"]
    assert param.contract.type == type_for("str", TypeRegistry())

    result = run_flow(loaded, {})
    assert result.status == "succeeded", result.failure
    assert result.output == ["A", "B"]
