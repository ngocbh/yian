"""Running a CASE that routes on a null check: a set value takes the branch, null the `else:`.

The branch passes the value, narrowed to not null, to a def's strict input (a CALL) or as a
MAP's per-item argument; these runs show the narrowing matches what the value does at run.
"""

import pytest

from yian.compose import load_flow
from yian.run import run_flow

_ECHO = "tests.engine._compose_codefns:echo"

_PRODUCER_FLOW = f"""
id: p
name: p
input:
  s: !Optional(str)
  items: !list(int)
nodes:
  x:
    kind: code
    code: {_ECHO}
    input:
      topic: ${{input.s}}
    output: Optional(str)
  gate:
    kind: case
    cases:
      - when: x.output != null
        then: use
    else: skip
  USE
  skip:
    kind: code
    code: {_ECHO}
    input:
      topic: skipped
    output: str
output: ${{use.output :- skip.output}}
defs:
  take:
    input:
      v: !str
    nodes:
      d:
        kind: code
        code: {_ECHO}
        input:
          topic: ${{input.v}}
        output: str
    output: ${{d.output}}
  take_each:
    input:
      i: !int
      v: !str
    nodes:
      d:
        kind: code
        code: {_ECHO}
        input:
          topic: ${{input.v}}
        output: str
    output: ${{d.output}}
"""

_CALL = """use:
    kind: call
    call: take
    input:
      v: ${x.output}"""

_MAP = """use:
    kind: map
    call: take_each
    over: ${input.items}
    input:
      i: ${item}
      v: ${x.output}"""

_INPUT_FLOW = f"""
id: p
name: p
typedefs:
  Item:
    qty: int
input:
  item: !Optional(Item)
defs:
  take:
    input:
      item: !Item
    nodes:
      d:
        kind: code
        code: {_ECHO}
        input:
          topic: ${{input.item.qty}}
        output: int
    output: ${{d.output}}
nodes:
  gate:
    kind: case
    cases:
      - when: input.item != null
        then: use
    else: skip
  use:
    kind: call
    call: take
    input:
      item: ${{input.item}}
  skip:
    kind: code
    code: {_ECHO}
    input:
      topic: 0
    output: int
output: ${{use.output :- skip.output}}
"""


@pytest.mark.parametrize(
    ("use", "s", "expected"),
    [
        pytest.param(_CALL, "hi", "hi", id="call-set"),
        pytest.param(_CALL, None, "skipped", id="call-null"),
        pytest.param(_MAP, "hi", ["hi", "hi"], id="map-set"),
        pytest.param(_MAP, None, "skipped", id="map-null"),
    ],
)
def test_a_null_checked_producer_output_runs_down_the_branch_it_takes(use, s, expected):
    flow = load_flow(_PRODUCER_FLOW.replace("USE", use))
    out = run_flow(flow, {"s": s, "items": [1, 2]})
    assert out.status == "succeeded", out.failure
    assert out.output == expected


@pytest.mark.parametrize(("item", "expected"), [({"qty": 3}, 3), (None, 0)])
def test_a_null_checked_input_runs_down_the_branch_it_takes(item, expected):
    out = run_flow(load_flow(_INPUT_FLOW), {"item": item})
    assert out.status == "succeeded", out.failure
    assert out.output == expected
