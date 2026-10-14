"""Typed bindings over unions of records: explicit narrowing and union declarations."""

from __future__ import annotations

import pytest

from yian.compose.helpers.errors import LoadError
from yian.compose.loader import load_flow
from yian.typesys import canonical_type_label, type_for
from yian.typesys.types import read_typedefs

_TYPEDEFS = """
typedefs:
  Buy: {action: 'Literal(buy)', qty: int}
  Sell: {action: 'Literal(sell)', qty: int, price: float}
  Hold: {action: 'Literal(hold)', note: str}
  Decision: Union(Buy, Sell, Hold)
  BuyOrSell: Union(Buy, Sell)
"""
_REGISTRY = read_typedefs({
    "Buy": {"action": "Literal(buy)", "qty": "int"},
    "Sell": {"action": "Literal(sell)", "qty": "int", "price": "float"},
    "Hold": {"action": "Literal(hold)", "note": "str"},
    "Decision": "Union(Buy, Sell, Hold)",
    "BuyOrSell": "Union(Buy, Sell)",
})


def _label(text: str) -> str:
    """Return the label load errors give the type `text` names."""
    return canonical_type_label(type_for(text, _REGISTRY))


def _flow(nodes: str, output: str = "${decide.output.action}") -> str:
    """Return a flow whose `decide` node returns its input as a `Decision`, then `nodes`."""
    return (
        "id: unions\nname: unions\n"
        + _TYPEDEFS
        + "input:\n  decision: !object\n"
        + "nodes:\n"
        + "  decide:\n"
        + "    kind: code\n"
        + "    input:\n      decision: ${input.decision}\n"
        + "    code: return inputs['decision']\n"
        + "    output: Decision\n"
        + nodes
        + f"output: {output}\n"
    )


def _consumer(node_id: str, binding: str, **extra: str) -> str:
    """Return a CODE node with the one input line `binding` and extra top-level keys."""
    text = f"  {node_id}:\n    kind: code\n"
    for key, value in extra.items():
        text += f"    {key}: {value}\n"
    return text + f"    input:\n      {binding}\n    code: return 1\n"


def _route(sell_target: str = "on_sell", else_target: str = "on_hold") -> str:
    """Return an `on:` CASE over the decision's tag routing buy, sell, and the rest."""
    return (
        "  route:\n    kind: case\n    on: ${decide.output.action}\n    cases:\n"
        "      - when: buy\n        then: on_buy\n"
        f"      - when: sell\n        then: {sell_target}\n"
        f"    else: {else_target}\n"
    )


def _load_error(text: str) -> str:
    with pytest.raises(LoadError) as caught:
        load_flow(text)
    return str(caught.value)


def test_a_member_over_the_union_loads():
    loaded = load_flow(_flow(_consumer("use", "buy: !Buy ${decide.output}")))

    binding = loaded.compiled.wiring["use"]["buy"]
    assert binding.contract.type == type_for("Buy", _REGISTRY)


def test_a_declared_union_accepts_one_member():
    load_flow(_flow(
        _consumer("b", "buy: !Buy ${decide.output}")
        + _consumer("use", "d: !Decision ${b.output}", output="Buy")
    ))


def test_a_union_disjoint_from_the_declared_type_fails():
    message = _load_error(_flow(_consumer("use", "n: !int ${decide.output}")))

    assert "node 'use' input 'n': declared int, but the value is" in message


def test_a_member_over_another_branchs_member_fails_in_that_branch():
    message = _load_error(_flow(
        _route()
        + _consumer("on_buy", "buy: !Buy ${decide.output}")
        + _consumer("on_sell", "buy: !Buy ${decide.output}")
        + _consumer("on_hold", "note: !str ${decide.output.note}")
    ))

    assert message.endswith(
        "node 'on_sell' input 'buy': declared Buy, "
        f"but the value is {_label('Sell')}"
    )


def test_the_else_branch_reads_the_unlisted_members():
    message = _load_error(_flow(
        _route()
        + _consumer("on_buy", "buy: !Buy ${decide.output}")
        + _consumer("on_sell", "sell: !Sell ${decide.output}")
        + _consumer("on_hold", "trade: !BuyOrSell ${decide.output}")
    ))

    assert message.endswith(
        "node 'on_hold' input 'trade': declared BuyOrSell, "
        f"but the value is {_label('Hold')}"
    )


def test_each_branch_may_name_its_own_member():
    load_flow(_flow(
        _route()
        + _consumer("on_buy", "buy: !Buy ${decide.output}")
        + _consumer("on_sell", "sell: !Sell ${decide.output}")
        + _consumer("on_hold", "hold: !Hold ${decide.output}")
    ))


def test_a_flow_output_narrows_to_a_member_and_a_call_parent_sees_it():
    text = (
        "id: parent\nname: parent\n"
        + _TYPEDEFS
        + "defs:\n"
        + "  pick:\n"
        + "    input:\n      decision: !object\n"
        + "    nodes:\n"
        + "      decide:\n"
        + "        kind: code\n"
        + "        input:\n          decision: ${input.decision}\n"
        + "        code: return inputs['decision']\n"
        + "        output: Decision\n"
        + "    output: !Buy ${decide.output}\n"
        + "nodes:\n"
        + "  use:\n    kind: call\n    call: pick\n"
        + "    input:\n      decision: {action: buy, qty: 1}\n"
        + "  qty:\n    kind: code\n    input:\n      q: ${use.output.qty}\n"
        + "    code: return inputs['q']\n"
        + "output: ${qty.output}\n"
    )

    loaded = load_flow(text)

    assert loaded.compiled.nodes["use"].output_contract.type == type_for("Buy", _REGISTRY)


def test_an_isinstance_branch_narrows_the_typed_binding_too():
    message = _load_error(_flow(
        "  route:\n    kind: case\n    cases:\n"
        "      - when: isinstance(decide.output, Buy)\n        then: on_buy\n"
        "    else: other\n"
        + _consumer("on_buy", "sell: !Sell ${decide.output}")
        + _consumer("other", "sell: !Sell ${decide.output}")
    ))

    assert message.endswith(
        "node 'on_buy' input 'sell': declared Sell, "
        f"but the value is {_label('Buy')}"
    )
