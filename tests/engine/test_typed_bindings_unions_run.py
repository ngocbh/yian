"""Typed bindings that narrow a union, checked when a run binds the value."""

from __future__ import annotations

from yian.compose.loader import load_flow
from yian.run import run_flow

_HEAD = """
id: narrow
name: narrow
typedefs:
  Buy: {action: 'Literal[buy]', qty: int}
  Sell: {action: 'Literal[sell]', qty: int, price: float}
  Decision: Union[Buy, Sell]
input:
  decision: object
"""

_NODE_FLOW = _HEAD + """
nodes:
  decide:
    kind: code
    input:
      decision: ${input.decision}
    code: return inputs['decision']
    output: Decision
  place:
    kind: code
    input:
      buy: Buy = ${decide.output}
    code: return 'buy ' + str(inputs['buy']['qty'])
    output: str
output: ${place.output}
"""

_CALL_FLOW = _HEAD + """
defs:
  pick:
    input:
      decision: object
    nodes:
      decide:
        kind: code
        input:
          decision: ${input.decision}
        code: return inputs['decision']
        output: Decision
    output: Buy = ${decide.output}
nodes:
  picked:
    kind: call
    call: pick
    input:
      decision: ${input.decision}
  widen:
    kind: code
    input:
      d: Decision = ${picked.output}
    code: return inputs['d']['qty']
    output: int
output: ${widen.output}
"""

BUY = {"action": "buy", "qty": 3}
SELL = {"action": "sell", "qty": 2, "price": 1.5}


def test_a_member_value_passes_and_the_consumer_reads_its_fields():
    result = run_flow(load_flow(_NODE_FLOW), {"decision": BUY})

    assert result.status == "succeeded", result.failure
    assert result.output == "buy 3"


def test_another_member_fails_the_consumer_on_its_binding():
    result = run_flow(load_flow(_NODE_FLOW), {"decision": SELL})

    assert result.status == "failed"
    assert result.failure.code == "binding.type"
    assert result.failure.node_id == "place"
    assert "binding 'buy'" in result.failure.message


def test_a_narrowed_child_output_reaches_a_union_typed_parent_input():
    result = run_flow(load_flow(_CALL_FLOW), {"decision": BUY})

    assert result.status == "succeeded", result.failure
    assert result.output == 3


def test_another_member_fails_the_childs_narrowed_output():
    result = run_flow(load_flow(_CALL_FLOW), {"decision": SELL})

    assert result.status == "failed"
    assert result.failure.code == "binding.type"
    assert result.failure.node_id == "picked/__end__"
