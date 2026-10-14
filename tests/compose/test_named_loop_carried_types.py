"""A named LOOP body's returned fields must fit the inputs the body declares.

A `call:` body's declared input is the parameter each returned field is bound to on the
next iteration. A returned field that can never fit it is a load error; over a typed seed,
a nullable returned field whose non-null type fits is left to the runtime input check.
"""

from __future__ import annotations

import re

import pytest

from yian.compose.helpers.errors import LoadError
from yian.compose.loader import load_flow
from yian.run import run_flow


def _named_loop(
    body_in: str, body_out: str, ret: str, seed: str, *, driver: str = "times: 2"
) -> str:
    """Return a flow whose LOOP `repeat` calls the `defs:` body `step`, carrying `n`.

    Args:
        body_in (`str`): The type `step` declares for its input `n`.
        body_out (`str`): The type the body's CODE node declares for its output `n`.
        ret (`str`): The Python expression the body's CODE node returns as `n`.
        seed (`str`): The LOOP's `n:` seed binding, e.g. `${input.z}`.
        driver (`str`, *optional*, defaults to `"times: 2"`): The LOOP's driver lines.
    """
    return f"""
id: named
name: named
input:
  z: !int
defs:
  step:
    input:
      n: !{body_in}
    nodes:
      bump:
        kind: code
        input:
          n: ${{input.n}}
        output:
          n: {body_out}
        code: |
          return {{"n": {ret}}}
    output: ${{bump.output}}
nodes:
  repeat:
    kind: loop
    {driver}
    call: step
    input:
      n: {seed}
output: ${{repeat.output}}
"""


_PAGINATE = 'None if inputs["n"] >= 2 else inputs["n"] + 1'
_WHILE = "while: ${n != null}\n    max: 10"


def _nullable_message(seed: str) -> str:
    """Return the load error for a body returning `Optional(int)` into `n: int`."""
    return (
        "loop node 'repeat': the body 'step' returns 'n' as Optional(int), but takes 'n' "
        "as int; make the input Optional(int), or tag the seed to leave the check to "
        f"runtime: `n: !int {seed}`"
    )


@pytest.mark.parametrize("seed", ["${input.z}", "0", "${input.z :- 0}"])
def test_untyped_seed_with_an_optional_returned_field_is_rejected(seed: str) -> None:
    with pytest.raises(LoadError, match=re.escape(_nullable_message(seed))):
        load_flow(_named_loop("int", "Optional(int)", "None", seed))


def test_typed_seed_leaves_a_returned_null_to_the_input_check() -> None:
    flow = load_flow(_named_loop("int", "Optional(int)", "None", "!int ${input.z}"))
    result = run_flow(flow, {"z": 0})
    assert result.status == "failed"
    assert result.failure.code == "contract.input"
    assert result.failure.message == "input 'n': None does not match declared type int"


def test_typed_seed_with_an_optional_returned_field_runs() -> None:
    flow = load_flow(
        _named_loop("int", "Optional(int)", 'inputs["n"] + 1', "!int ${input.z}")
    )
    result = run_flow(flow, {"z": 0})
    assert result.status == "succeeded", result.failure
    assert result.output == {"n": 2}


def test_while_loop_ending_on_null_over_an_untyped_seed_is_rejected() -> None:
    with pytest.raises(LoadError, match=re.escape(_nullable_message("${input.z}"))):
        load_flow(
            _named_loop("int", "Optional(int)", _PAGINATE, "${input.z}", driver=_WHILE)
        )


def test_while_loop_ending_on_null_over_a_typed_seed_runs() -> None:
    flow = load_flow(
        _named_loop("int", "Optional(int)", _PAGINATE, "!int ${input.z}", driver=_WHILE)
    )
    result = run_flow(flow, {"z": 0})
    assert result.status == "succeeded", result.failure
    assert result.output == {"n": None}


def test_while_loop_ending_on_null_into_an_optional_input_runs() -> None:
    flow = load_flow(
        _named_loop("Optional(int)", "Optional(int)", _PAGINATE, "${input.z}", driver=_WHILE)
    )
    result = run_flow(flow, {"z": 0})
    assert result.status == "succeeded", result.failure
    assert result.output == {"n": None}


@pytest.mark.parametrize("seed", ["${input.z}", "!int ${input.z}"])
def test_float_returned_into_an_int_input_is_rejected(seed: str) -> None:
    message = (
        "loop node 'repeat': the body 'step' returns 'n' as float, but takes 'n' as int; "
        "change the body's input or output type so they agree"
    )
    with pytest.raises(LoadError, match=re.escape(message)):
        load_flow(_named_loop("int", "float", "1.5", seed))


def test_any_returned_into_an_int_input_is_rejected() -> None:
    message = (
        "loop node 'repeat': the body 'step' returns 'n' as Any, but takes 'n' as int; "
        "narrow it in the body with isinstance in a case"
    )
    with pytest.raises(LoadError, match=re.escape(message)):
        load_flow(_named_loop("int", "Any", "1", "${input.z}"))


def test_optional_any_returned_into_an_any_input_over_an_untyped_seed_is_rejected() -> None:
    with pytest.raises(LoadError, match=re.escape("make the input Optional(Any)")):
        load_flow(_named_loop("Any", "Optional(Any)", "None", "${input.z}"))


def test_optional_any_returned_into_an_any_input_over_a_typed_seed_loads() -> None:
    load_flow(_named_loop("Any", "Optional(Any)", "None", "!Any ${input.z}"))


def test_optional_returned_into_an_optional_input_runs() -> None:
    flow = load_flow(_named_loop("Optional(int)", "Optional(int)", "None", "${input.z}"))
    result = run_flow(flow, {"z": 0})
    assert result.status == "succeeded", result.failure
    assert result.output == {"n": None}


def test_null_seed_into_an_optional_input_runs() -> None:
    flow = load_flow(_named_loop("Optional(int)", "Optional(int)", "None", "${null}"))
    result = run_flow(flow, {"z": 0})
    assert result.status == "succeeded", result.failure
    assert result.output == {"n": None}


def test_int_returned_into_a_float_input_loads() -> None:
    load_flow(_named_loop("float", "int", "1", "${input.z}"))


def _record_loop(seed: str) -> str:
    """Return a flow whose named body takes `n: P` and returns it as `Optional(P)`."""
    return "typedefs:\n  P:\n    x: int\n" + _named_loop(
        "P", "Optional(P)", 'inputs["n"]', seed
    ).replace("  z: !int\n", "  p: !P\n")


def test_untyped_seed_with_an_optional_typedef_field_is_rejected() -> None:
    with pytest.raises(LoadError) as excinfo:
        load_flow(_record_loop("${input.p}"))
    message = str(excinfo.value)
    assert "make the input Optional(P)" in message
    assert "`n: !P ${input.p}`" in message


def test_typed_seed_with_an_optional_typedef_field_loads() -> None:
    load_flow(_record_loop("!P ${input.p}"))


def test_a_returned_field_the_body_does_not_take_loads() -> None:
    load_flow(
        """
id: extra
name: extra
input:
  z: !int
defs:
  step:
    input:
      n: !int
    nodes:
      bump:
        kind: code
        input:
          n: ${input.n}
        output:
          n: int
          m: int
        code: |
          return {"n": inputs["n"] + 1, "m": 0}
    output: ${bump.output}
nodes:
  repeat:
    kind: loop
    times: 2
    call: step
    input:
      n: ${input.z}
      m: 0
output: ${repeat.output}
"""
    )


def test_a_partial_loop_runs() -> None:
    flow = load_flow(
        """
id: partial
name: partial
input:
  z: !int
defs:
  step:
    input:
      n: !int
      m: !int
    nodes:
      bump:
        kind: code
        input:
          n: ${input.n}
        output:
          n: int
        code: |
          return {"n": inputs["n"] + 1}
    output: ${bump.output}
nodes:
  repeat:
    kind: loop
    times: 2
    call: step
    input:
      n: ${input.z}
      m: 5
output: ${repeat.output}
"""
    )
    result = run_flow(flow, {"z": 0})
    assert result.status == "succeeded", result.failure
    assert result.output == {"n": 2, "m": 5}


def test_a_loop_inside_a_called_body_is_rejected() -> None:
    text = """
id: nested
name: nested
input:
  z: !int
defs:
  step:
    input:
      n: !int
    nodes:
      bump:
        kind: code
        input:
          n: ${input.n}
        output:
          n: Optional(int)
        code: |
          return {"n": None}
    output: ${bump.output}
  outer:
    input:
      z: !int
    nodes:
      repeat:
        kind: loop
        times: 2
        call: step
        input:
          n: ${input.z}
    output: ${repeat.output}
nodes:
  run:
    kind: call
    call: outer
    input:
      z: ${input.z}
output: ${run.output}
"""
    with pytest.raises(LoadError) as excinfo:
        load_flow(text)
    message = str(excinfo.value)
    assert "defs entry 'outer': " in message
    assert _nullable_message("${input.z}") in message


def test_a_seed_that_misses_the_input_reports_the_binding() -> None:
    with pytest.raises(
        LoadError, match=re.escape("binding 'n' — child expects 'int'")
    ) as excinfo:
        load_flow(
            _named_loop("int", "Optional(int)", "None", "!Optional(int) ${input.z}")
        )
    assert "the body 'step' returns" not in str(excinfo.value)


def test_the_error_points_at_the_seed() -> None:
    text = _named_loop("int", "Optional(int)", "None", "${input.z}")
    with pytest.raises(LoadError) as excinfo:
        load_flow(text)
    line = text.splitlines().index("      n: ${input.z}") + 1
    frames = excinfo.value.diagnostic.frames
    assert len(frames) == 1
    assert frames[0].range.start_line == line
