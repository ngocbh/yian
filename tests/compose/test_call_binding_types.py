"""CALL/MAP bindings from `${item}`, literals, and null escapes are type-checked at load."""

from __future__ import annotations

import pytest

from yian import LoadError, load_flow, run_flow

_CALL = """id: p
name: p
input:
  m: Optional[str]
  n: Optional[int]
  s: str
  xs: list[int]
  ds: list[str]
  rs: list[R]
  os: list[Optional[str]]
  d: Optional[date]
typedefs:
  R:
    a: int
defs:
  child:
    input:
      t: {child}
    nodes:
      n:
        kind: code
        input:
          t: ${{input.t}}
        output: str
        code: return repr(inputs["t"])
    output: ${{n.output}}
nodes:
  c:
    kind: {kind}
    {over}
    call: child
    input:
      t: {binding}
output: ${{c.output}}
"""

_INPUTS = {"s": "x", "xs": [1], "ds": ["2026-01-01"], "rs": [{"a": 1}], "os": ["x"]}


def _flow(child: str, binding: str, *, over: str | None = None) -> str:
    """Return a parent flow binding `t` of a `child` taking `t: <child>`."""
    return _CALL.format(
        child=child,
        binding=binding,
        kind="map" if over else "call",
        over=f"over: {over}" if over else "",
    )


@pytest.mark.parametrize(
    "child, binding, over, message",
    [
        ("str", "${item}", "${input.xs}", "child expects 'str', source is 'int'"),
        ("str", "${item.a}", "${input.rs}", "child expects 'str', source is 'int'"),
        ("date", "${item}", "${input.ds}", "child expects 'date', source is 'str'"),
        ("str", "${item}", "${input.os}", "child expects 'str', source is 'Optional[str]'"),
        ("str", "5", None, "child expects 'str', but the value 5 does not fit"),
        ("int", '"abc"', None, "child expects 'int', but the value \"abc\" does not fit"),
        ("int[1, 3]", "9", None, "child expects 'int[1,3]', but the value 9 does not fit"),
        ("str", "null", None, "child expects 'str', but the value null does not fit"),
        ("str", "${input.m:-5}", None, "child expects 'str', but the default 5 does not fit"),
        ("str", "${input.n:-\"x\"}", None, "child expects 'str', source is 'int'"),
        ("str", "${input.n:?missing}", None, "child expects 'str', source is 'int'"),
        ("str", "${input.m:-null}", None, "child expects 'str', but the default null does not fit"),
        ("date", '${input.m:-"2026-01-01"}', None, "child expects 'date', source is 'str'"),
        ("date", '${input.d:-"today"}', None, "child expects 'date', but the default \"today\" does not fit"),
        ("str", "${input.m:-input.n}", None, "child expects 'str', source is 'Optional[int]'"),
        ("str", "${item:-5}", "${input.os}", "child expects 'str', but the default 5 does not fit"),
    ],
)
def test_unprovable_fit_is_a_load_error(child, binding, over, message):
    with pytest.raises(LoadError) as exc:
        load_flow(_flow(child, binding, over=over))
    assert f"binding 't' — {message}" in str(exc.value)


@pytest.mark.parametrize(
    "child, binding, over, output",
    [
        ("int", "${item}", "${input.xs}", ["1"]),
        ("float", "${item}", "${input.xs}", ["1.0"]),
        ("int", "${item.a}", "${input.rs}", ["1"]),
        ("float", "5", None, "5.0"),
        ("int", '"3"', None, "3"),
        ("date", '${input.d:-"2026-01-01"}', None, "'2026-01-01'"),
        ("date", '"2026-01-01"', None, "'2026-01-01'"),
        ("int[1, 3]", "2", None, "2"),
        ("Optional[str]", "null", None, "None"),
        ("str", '${input.m:-"z"}', None, "'z'"),
        ("int", "${input.n:-5}", None, "5"),
        ("str", "${input.m:-input.s}", None, "'x'"),
        ("Optional[str]", "${input.m:-null}", None, "None"),
        ("str", "${item:-\"z\"}", "${input.os}", ["'x'"]),
    ],
)
def test_bindings_that_fit_load_and_run(child, binding, over, output):
    result = run_flow(load_flow(_flow(child, binding, over=over)), _INPUTS)
    assert result.status == "succeeded", result.failure
    assert result.output == output
