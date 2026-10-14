"""Arithmetic or rendering that Python can't complete fails the run, not the caller."""

import sys

import pytest

from yian import load_flow, run_flow

# Python 3.10.0-3.10.6 has no int-to-str digit limit, so a huge int renders there.
needs_digit_limit = pytest.mark.skipif(
    not hasattr(sys, "get_int_max_str_digits"), reason="no int-to-str digit limit"
)


def _flow(decl: str, value: str) -> str:
    """Return a flow whose CODE node `n` binds `v: <value>` over flow inputs `decl`."""
    return f"""
id: arith
name: arith
input:
  {decl}
nodes:
  n:
    kind: code
    input:
      v: {value}
    output: str
    code: return "ok"
output: ${{n.output}}
"""


def _failed(decl: str, value: str, inputs: dict):
    """Run the flow and return its failure, asserting it failed at `n`'s binding."""
    result = run_flow(load_flow(_flow(decl, value)), inputs)
    assert result.status == "failed"
    assert result.failure.code == "binding.evaluate"
    assert result.failure.node_id == "n"
    return result.failure


def test_a_float_power_overflow_fails_the_run():
    failure = _failed("n: int", "${2.0 ** input.n}", {"n": 5000})
    assert "cannot compute" in failure.message and "**" in failure.message


def test_an_int_too_large_for_float_division_fails_the_run():
    _failed("n: int", "${input.n / 3}", {"n": 10**400})


@needs_digit_limit
def test_a_type_error_on_a_huge_int_fails_the_run():
    failure = _failed("n: int", '${10 ** input.n + "a"}', {"n": 5000})
    assert "type error" in failure.message


def test_a_missing_format_key_fails_the_run():
    failure = _failed("r: object", '${"%(k)s" % input.r}', {"r": {"j": 1}})
    assert "cannot compute" in failure.message


def test_an_overflow_in_a_flow_assert_fails_the_run():
    flow = load_flow("""
id: arith
name: arith
input:
  n: int
asserts:
  - 2.0 ** input.n > 0
nodes:
  n:
    kind: code
    output: str
    code: return "ok"
output: ${n.output}
""")
    result = run_flow(flow, {"n": 5000})
    assert result.status == "failed"
    assert result.failure.code == "assertion.failed"
