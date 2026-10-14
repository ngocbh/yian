"""One-sided numeric ranges in whole flows: load-time checks and runtime checks.

`int(0..)` means 0 or more and `int(..100)` at most 100. Flow inputs, node input
tags, bindings, typedefs, cross-flow calls, case `when:` values, and AGENT outputs
treat an open end as unbounded on that side, with the same rules as two-sided ranges.
"""

from __future__ import annotations

import pytest

import yian.llm_clients as llm_clients
from yian import LoadError, load_flow, run_flow


def _load_error(text: str) -> str:
    """Load `text`, expecting a load error, and return its message."""
    with pytest.raises(LoadError) as info:
        load_flow(text)
    return str(info.value)


def _failure(text: str, inputs: dict | None = None):
    """Load and run `text`, expecting a failed run, and return its failure."""
    result = run_flow(load_flow(text), inputs or {})
    assert result.status == "failed"
    assert result.failure is not None
    return result.failure


def _output(text: str, inputs: dict | None = None):
    """Load and run `text`, expecting success, and return the flow output."""
    result = run_flow(load_flow(text), inputs or {})
    assert result.status == "succeeded", result.failure
    return result.output


# --- flow inputs ------------------------------------------------------------ #


def _input_flow(m_default: str = "50") -> str:
    """Return a flow with one-sided flow inputs that a code node echoes."""
    return f"""
id: open_inputs
name: open_inputs
input:
  n: !int(0..)
  m: !int(..100) {m_default}
  f: !float(0.0..) 0.0
nodes:
  echo:
    kind: code
    input:
      n: ${{input.n}}
      m: ${{input.m}}
      f: ${{input.f}}
    code: |
      return [inputs["n"], inputs["m"], inputs["f"]]
    output: list(float)
output: ${{echo.output}}
"""


def test_flow_input_inside_an_open_range_runs():
    assert _output(_input_flow(), {"n": 3}) == [3, 50, 0.0]


def test_flow_input_below_an_open_range_fails_the_input_contract():
    failure = _failure(_input_flow(), {"n": -1})

    assert failure.code == "contract.input"
    assert "-1 is below the inclusive int minimum 0" in failure.message


def test_flow_input_default_above_an_open_range_is_a_load_error():
    message = _load_error(_input_flow("101"))

    assert "'m': `101` is not a valid int(..100)" in message


# --- node input tags and bindings ------------------------------------------- #


def _tagged_literal_flow(value: str, tag: str = "int(0..)") -> str:
    """Return a flow whose code node input is the tagged literal `!<tag> value`."""
    return f"""
id: tagged_literal
name: tagged_literal
nodes:
  use:
    kind: code
    input:
      x: !{tag} {value}
    code: |
      return inputs["x"]
    output: int
output: ${{use.output}}
"""


def test_tagged_literal_outside_an_open_range_is_a_load_error():
    message = _load_error(_tagged_literal_flow("-1"))

    assert "'x': `-1` is not a valid int(0..)" in message


def test_tagged_literal_inside_an_open_range_runs():
    assert _output(_tagged_literal_flow("5")) == 5


def _binding_flow(producer_output: str, tag: str) -> str:
    """Return a flow where `use` binds `x: !<tag> ${a.output}` from a producer `a`.

    The producer returns the flow input `v`, typed by `producer_output`.
    """
    return f"""
id: tagged_binding
name: tagged_binding
input:
  v: !int
nodes:
  a:
    kind: code
    input:
      v: ${{input.v}}
    code: |
      return inputs["v"]
    output: {producer_output}
  use:
    kind: code
    input:
      x: !{tag} ${{a.output}}
    code: |
      return inputs["x"]
    output: int
output: ${{use.output}}
"""


def test_tagged_binding_checks_an_open_range_at_runtime():
    text = _binding_flow("int", "int(0..)")

    failure = _failure(text, {"v": -1})

    assert failure.code == "binding.type"
    assert "binding 'x': -1 is below the inclusive int minimum 0" in failure.message


def test_tagged_binding_narrows_an_open_range_at_runtime():
    text = _binding_flow("int(0..)", "int(0..100)")

    failure = _failure(text, {"v": 500})

    assert failure.code == "binding.type"
    assert "500 is outside inclusive int bounds (0, 100)" in failure.message
    assert _output(text, {"v": 5}) == 5


def test_tagged_binding_disjoint_from_an_open_range_is_a_load_error():
    message = _load_error(_binding_flow("int(..-1)", "int(0..)"))

    assert "input 'x': declared int(0..), but the value is int(..-1)" in message


# --- typedefs ---------------------------------------------------------------- #


def _typedef_flow(qty: int) -> str:
    """Return a flow whose code node returns an `Order` record with `qty`.

    The price is 1.0 plus the `counts` items and `maybe`, so tests can see those inputs.
    """
    return f"""
id: open_typedefs
name: open_typedefs
typedefs:
  Count: int(0..)
  Order:
    qty: int(1..)
    price: float(..100.0)
input:
  counts: !list(Count) []
  maybe: !Optional(Count) null
nodes:
  order:
    kind: code
    input:
      counts: ${{input.counts}}
      maybe: ${{input.maybe}}
    code: |
      extra = sum(inputs["counts"]) + (inputs["maybe"] or 0)
      return {{"qty": {qty}, "price": 1.0 + extra}}
    output: Order
output: ${{order.output}}
"""


def test_record_field_below_an_open_range_fails_the_output_contract():
    failure = _failure(_typedef_flow(0))

    assert failure.code == "contract.output"
    assert "0 is below the inclusive int minimum 1" in failure.message


def test_record_with_open_range_fields_runs():
    assert _output(_typedef_flow(2)) == {"qty": 2, "price": 1.0}


def test_list_and_optional_of_an_open_range_alias_accept_their_values():
    output = _output(_typedef_flow(2), {"counts": [0, 5], "maybe": 4})

    assert output == {"qty": 2, "price": 10.0}


def test_list_of_an_open_range_alias_rejects_a_negative_item():
    failure = _failure(_typedef_flow(2), {"counts": [0, -5]})

    assert failure.code == "contract.input"
    assert "-5 is below the inclusive int minimum 0" in failure.message


# --- cross-flow call ---------------------------------------------------------- #


def _call_flow(source_output: str, child_type: str, binding: str = "${a.output}") -> str:
    """Return a flow that calls a child taking `t: !<child_type>` from a producer `a`."""
    return f"""
id: open_call
name: open_call
input:
  v: !int
defs:
  child:
    input:
      t: !{child_type}
    nodes:
      echo:
        kind: code
        input:
          t: ${{input.t}}
        code: |
          return inputs["t"]
        output: int
    output: ${{echo.output}}
nodes:
  a:
    kind: code
    input:
      v: ${{input.v}}
    code: |
      return inputs["v"]
    output: {source_output}
  c:
    kind: call
    call: child
    input:
      t: {binding}
output: ${{c.output}}
"""


def test_call_with_a_contained_open_range_source_runs():
    assert _output(_call_flow("int(5..)", "int(0..)"), {"v": 7}) == 7


def test_call_with_an_unbounded_source_into_an_open_range_is_a_load_error():
    message = _load_error(_call_flow("int", "int(0..)"))

    assert "child expects 'int(0..)', source is 'int'" in message


def test_call_with_a_wider_open_range_source_is_a_load_error():
    message = _load_error(_call_flow("int(0..)", "int(0..100)"))

    assert "child expects 'int(0..100)', source is 'int(0..)'" in message


def test_call_with_a_tagged_wider_source_is_checked_at_runtime():
    text = _call_flow("int(0..)", "int(0..100)", binding="!int(0..100) ${a.output}")

    assert _output(text, {"v": 5}) == 5
    failure = _failure(text, {"v": 500})
    assert failure.code == "binding.type"
    assert "500 is outside inclusive int bounds (0, 100)" in failure.message


# --- case `when:` membership ------------------------------------------------- #


def _case_flow(when: str) -> str:
    """Return a flow whose case routes `on:` an `int(0..)` input with `when: <when>`."""
    return f"""
id: open_case
name: open_case
input:
  n: !int(0..)
nodes:
  route:
    kind: case
    on: ${{input.n}}
    cases:
      - when: {when}
        then: hit
    else: miss
  hit:
    kind: code
    code: |
      return "hit"
    output: str
  miss:
    kind: code
    code: |
      return "miss"
    output: str
output: ${{route.output}}
"""


def test_case_when_outside_an_open_range_is_a_load_error():
    message = _load_error(_case_flow("-1"))

    assert "`when: -1` is not a value of `on: ${input.n}` (`int(0..)`)" in message
    assert "-1 is below the inclusive int minimum 0" in message


def test_case_when_inside_an_open_range_runs():
    assert _output(_case_flow("3"), {"n": 3}) == "hit"


# --- `int(..)` ---------------------------------------------------------------- #


def test_no_ends_in_a_typedef_is_a_load_error():
    text = """
id: no_ends
name: no_ends
typedefs:
  Whole: int(..)
nodes:
  use:
    kind: code
    code: |
      return 1
    output: Whole
output: ${use.output}
"""
    assert "write int" in _load_error(text)


def test_no_ends_in_a_tag_is_a_load_error():
    assert "write int" in _load_error(_tagged_literal_flow("1", tag="int(..)"))


# --- AGENT structured output -------------------------------------------------- #


_AGENT_FLOW = """
id: open_agent
name: open_agent
typedefs:
  Order:
    qty: int(1..)
    price: float(..100.0)
nodes:
  writer:
    kind: agent
    prompt: write an order
    retries: 0
    llm_config:
      provider: openai
      model: gpt-4o
    output: Order
output: ${writer.output}
"""


def _stub_agent_reply(monkeypatch, reply: dict) -> None:
    """Make every AGENT model validate `reply` against its structured-output schema."""

    class _Model:
        """Return `reply` through the generated output schema."""

        def with_structured_output(self, schema):
            class _Bound:
                """Validate the fixed reply."""

                def invoke(self, _messages):
                    return schema.model_validate(reply)

            return _Bound()

    monkeypatch.setattr(llm_clients, "model_from_config", lambda _config: _Model())


def test_agent_reply_inside_open_ranges_runs(monkeypatch):
    _stub_agent_reply(monkeypatch, {"qty": 3, "price": 9.5})

    assert _output(_AGENT_FLOW) == {"qty": 3, "price": 9.5}


def test_agent_reply_below_an_open_range_fails_structured_output(monkeypatch):
    _stub_agent_reply(monkeypatch, {"qty": 0, "price": 9.5})

    failure = _failure(_AGENT_FLOW)

    assert failure.code == "node.execute"
    assert failure.cause_type == "ValidationError"
    assert "qty" in failure.message
    assert "greater than or equal to 1" in failure.message
