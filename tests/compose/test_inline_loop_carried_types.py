"""An inline LOOP body's returned fields must fit the parameters generated from its seeds.

An inline body becomes a child flow whose parameters are typed from the seeds, and each
returned field is bound to its parameter again on the next iteration. A returned field
that can never fit that parameter is a load error, so loading accepts only flows that can
run.
"""

from __future__ import annotations

import pytest

from yian.compose.helpers.errors import LoadError
from yian.compose.loader import load_flow
from yian.run import run_flow


def _seen_loop(seed: str) -> str:
    """Return a flow whose inline LOOP body returns `seen` as `Optional[list[str]]`.

    Args:
        seed (`str`): The LOOP's `seen:` seed binding, e.g. `${null}`.
    """
    return f"""
id: seen
name: seen
nodes:
  repeat:
    kind: loop
    while: "n < 2"
    max: 5
    input:
      seen: {seed}
      n: 0
    nodes:
      bump:
        kind: code
        input:
          n: ${{input.n}}
        output:
          seen: Optional[list[str]]
          n: int
        code: |
          return {{"seen": ["a", "b"][: inputs["n"] + 1], "n": inputs["n"] + 1}}
    output: ${{bump.output}}
output: ${{repeat.output}}
"""


def _number_loop(body_output: str, body_value: str) -> str:
    """Return a flow whose inline LOOP carries `n`, seeded from an `int` flow input.

    Args:
        body_output (`str`): The body CODE node's declared output type.
        body_value (`str`): The Python expression the body node returns as `n`.
    """
    return f"""
id: number
name: number
input:
  z: int
nodes:
  repeat:
    kind: loop
    times: 2
    input:
      n: ${{input.z}}
    nodes:
      bump:
        kind: code
        input:
          n: ${{input.n}}
        output:
          n: {body_output}
        code: |
          return {{"n": {body_value}}}
    output: ${{bump.output}}
output: ${{repeat.output}}
"""


def test_untyped_null_seed_with_a_nullable_list_body_is_rejected_with_a_hint() -> None:
    with pytest.raises(LoadError) as excinfo:
        load_flow(_seen_loop("${null}"))
    message = str(excinfo.value)
    assert (
        "loop node 'repeat': the body returns 'seen' as Optional[list[str]], but the "
        "carried field is typed None from its seed; declare the seed's type: "
        "`seen: Optional[list[str]] = ${null}`"
    ) in message


def test_untyped_null_seed_error_points_at_the_seed() -> None:
    text = _seen_loop("${null}")
    with pytest.raises(LoadError) as excinfo:
        load_flow(text)
    line = text.splitlines().index("      seen: ${null}") + 1
    frames = excinfo.value.diagnostic.frames
    assert len(frames) == 1
    assert frames[0].range.start_line == line


def test_typed_null_seed_with_a_nullable_list_body_runs() -> None:
    flow = load_flow(_seen_loop("Optional[list[str]] = ${null}"))
    result = run_flow(flow, {})
    assert result.status == "succeeded"
    assert result.output == {"seen": ["a", "b"], "n": 2}


def test_int_seed_with_a_float_body_field_is_rejected() -> None:
    with pytest.raises(LoadError) as excinfo:
        load_flow(_number_loop("float", 'inputs["n"] + 1.5'))
    message = str(excinfo.value)
    assert "the body returns 'n' as float, but the carried field is typed int" in message
    assert "`n: float = ${input.z}`" in message


def test_int_seed_with_an_int_body_field_loads_and_runs() -> None:
    flow = load_flow(_number_loop("int", 'inputs["n"] + 1'))
    result = run_flow(flow, {"z": 0})
    assert result.status == "succeeded"
    assert result.output == {"n": 2}


def test_a_typed_seed_names_its_declared_type() -> None:
    text = _number_loop("float", 'inputs["n"] + 1.5').replace(
        "n: ${input.z}", "n: int = ${input.z}"
    )
    with pytest.raises(LoadError) as excinfo:
        load_flow(text)
    message = str(excinfo.value)
    assert "the body returns 'n' as float, but the carried field is declared int" in message
    assert "`n: float = ${input.z}`" in message


def test_a_field_the_body_omits_is_not_checked() -> None:
    flow = load_flow(
        """
id: partial
name: partial
nodes:
  repeat:
    kind: loop
    times: 2
    input:
      note: ${null}
      n: 0
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
output: ${repeat.output}
"""
    )
    result = run_flow(flow, {})
    assert result.status == "succeeded"
    assert result.output == {"note": None, "n": 2}


def _note_loop(seed: str, body_output: str, body_value: str) -> str:
    """Return a flow whose inline LOOP body returns `note` from one CODE node.

    Args:
        seed (`str`): The LOOP's `note:` seed binding, e.g. `str = "start"`.
        body_output (`str`): The body CODE node's declared output type.
        body_value (`str`): The Python expression the body node returns.
    """
    return f"""
id: note
name: note
nodes:
  repeat:
    kind: loop
    times: 2
    input:
      note: {seed}
    nodes:
      mk:
        kind: code
        input:
          note: ${{input.note}}
        output: {body_output}
        code: |
          return {body_value}
    output:
      note: ${{mk.output}}
output: ${{repeat.output}}
"""


def test_typed_seed_with_an_optional_body_field_loads() -> None:
    flow = load_flow(_note_loop('str = "start"', "Optional[str]", 'inputs["note"] + "!"'))
    result = run_flow(flow, {})
    assert result.status == "succeeded", result.failure
    assert result.output == {"note": "start!!"}


def test_typed_seed_with_a_returned_null_fails_the_next_iteration() -> None:
    flow = load_flow(_note_loop('str = "start"', "Optional[str]", "None"))
    result = run_flow(flow, {})
    assert result.status == "failed"
    assert result.failure.code == "contract.input"


def test_untyped_seed_with_an_optional_body_field_is_rejected_with_a_hint() -> None:
    with pytest.raises(LoadError) as excinfo:
        load_flow(_note_loop('"start"', "Optional[str]", 'inputs["note"] + "!"'))
    message = str(excinfo.value)
    assert (
        "the body returns 'note' as Optional[str], but the carried field is typed str "
        "from its seed; declare the seed's type: `note: Optional[str] = \"start\"`"
    ) in message


def test_typed_str_seed_with_an_int_body_field_is_rejected() -> None:
    with pytest.raises(LoadError) as excinfo:
        load_flow(_note_loop('str = "start"', "int", "1"))
    message = str(excinfo.value)
    assert "'note'" in message and "int" in message and "str" in message


def test_literal_seed_with_a_str_body_field_is_rejected() -> None:
    with pytest.raises(LoadError) as excinfo:
        load_flow(_note_loop("Literal[a, b] = a", "str", '"a"'))
    message = str(excinfo.value)
    assert (
        "the body returns 'note' as str, but the carried field is declared "
        'Literal["a","b"]'
    ) in message
    assert '`note: str = "a"`' in message
