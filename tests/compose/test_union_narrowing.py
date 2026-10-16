"""Union narrowing at load: a CASE branch reads the members its tag or isinstance check selects."""

import pytest

from yian.compose import LoadError, load_flow
from yian.run import run_flow

_TYPEDEFS = (
    "typedefs:\n"
    "  Limit: {type: 'Literal(limit)', price: float}\n"
    "  Market: {type: 'Literal(market)'}\n"
    "  Leg: Union(Limit, Market)\n"
    "  Buy: {action: 'Literal(buy)', qty: int, limit: float, leg: Leg}\n"
    "  Hold: {action: 'Literal(hold)', note: str}\n"
    "  Sell: {action: 'Literal(sell)', qty: int, price: float, note: str}\n"
    "  Decision: Union(Buy, Hold, Sell)\n"
    "  HoldOrSell: Union(Hold, Sell)\n"
    "  Wrap: {decision: Decision}\n"
    "  BuyOrSell: Union(Buy, Sell)\n"
    "  Text: {text: str}\n"
    "  Num: {num: int}\n"
    "  Reply: Union(Text, Num)\n"
    "  Mode: Literal(a, b)\n"
    "  Cancel: {action: 'Literal(cancel)', reason: str}\n"
    "  Rest: Union(Hold, Sell, Cancel)\n"
    "  Narrow: {action: 'Literal(buy)', qty: int, limit: float, leg: Limit}\n"
    "  SellOrNarrow: Union(Sell, Narrow)\n"
)

# Fields: `qty` (buy, sell), `limit` and `leg` (buy), `price` (sell), `note` (hold, sell).
_QTY_ERROR = (
    "reference ${d.output.qty} reads field 'qty', which only union members with action "
    "buy, sell have; read it inside a case on ${d.output.action}"
)
_LIMIT_ERROR = (
    "reference ${d.output.limit} reads field 'limit', which only union members with "
    "action buy have; read it inside a case on ${d.output.action}"
)

_DEFS = (
    "defs:\n"
    "  place:\n"
    "    input:\n      order: !Buy\n"
    "    nodes:\n"
    "      p:\n        kind: code\n        input:\n          o: ${input.order}\n"
    "        code: return 1\n"
    "    output: ${p.output}\n"
    "  place_rest:\n"
    "    input:\n      order: !HoldOrSell\n"
    "    nodes:\n"
    "      p:\n        kind: code\n        input:\n          o: ${input.order}\n"
    "        code: return 1\n"
    "    output: ${p.output}\n"
    "  place_sell_or_narrow:\n"
    "    input:\n      order: !SellOrNarrow\n"
    "    nodes:\n"
    "      p:\n        kind: code\n        input:\n          o: ${input.order}\n"
    "        code: return 1\n"
    "    output: ${p.output}\n"
    "  place_maybe_rest:\n"
    "    input:\n      order: !Optional(HoldOrSell)\n"
    "    nodes:\n"
    "      p:\n        kind: code\n        input:\n          o: ${input.order}\n"
    "        code: return 1\n"
    "    output: ${p.output}\n"
    "  outer:\n"
    "    input:\n      x: !int\n"
    "    nodes:\n"
    "      p:\n        kind: code\n        input:\n          x: ${input.x}\n"
    "        code: return 1\n"
    "    output: ${p.output}\n"
    "  inner:\n"
    "    input:\n      y: !int\n"
    "    nodes:\n"
    "      p:\n        kind: code\n        input:\n          y: ${input.y}\n"
    "        code: return 1\n"
    "    output: 1\n"
)


def _producer(node_id: str, output: str) -> str:
    """A CODE node typed `output` that reads the flow input `s`."""
    return (
        f"  {node_id}:\n"
        "    kind: code\n"
        "    input:\n      s: ${input.s}\n"
        f"    output: {output}\n"
        "    code: return None\n"
    )


def _code(node_id: str, inputs: dict[str, str] | None = None, **extra: str) -> str:
    """An untyped CODE node with `inputs` and extra top-level YAML keys."""
    text = f"  {node_id}:\n    kind: code\n"
    for key, value in extra.items():
        text += f"    {key}: {value}\n"
    if inputs:
        text += "    input:\n"
        for name, source in inputs.items():
            text += f"      {name}: {source}\n"
    return text + "    code: return 1\n"


def _case(
    node_id: str,
    on: str | None,
    cases: list[tuple[str, str]],
    else_: str | None = None,
) -> str:
    """A CASE node; `on=None` is the searched form."""
    text = f"  {node_id}:\n    kind: case\n"
    if on is not None:
        text += f"    on: {on}\n"
    text += "    cases:\n"
    for when, then in cases:
        text += f"      - when: {when}\n        then: {then}\n"
    if else_ is not None:
        text += f"    else: {else_}\n"
    return text


def _flow(
    *nodes: str,
    output: str = "${d.output.action}",
    d_type: str = "Decision",
    inputs: str = "  s: !str\n",
) -> str:
    """A flow with producer `d` typed `d_type`, then `nodes`, the defs, and `output`."""
    return (
        "id: p\nname: p\n"
        + _TYPEDEFS
        + "input:\n"
        + inputs
        + "nodes:\n"
        + _producer("d", d_type)
        + "".join(nodes)
        + f"output: {output}\n"
        + _DEFS
    )


_ON = "${d.output.action}"


def _load_error(text: str) -> str:
    with pytest.raises(LoadError) as caught:
        load_flow(text)
    return str(caught.value)


def _loads(text: str) -> None:
    load_flow(text)


# --- narrowed reads --- #


def test_a_buy_branch_reads_a_member_field():
    _loads(_flow(
        _case("route", _ON, [("buy", "b"), ("sell", "s"), ("hold", "h")]),
        _code("b", {"q": "${d.output.qty}", "l": "${d.output.limit}"}),
        _code("s", {"p": "${d.output.price}"}),
        _code("h", {"n": "${d.output.note}"}),
    ))


def test_a_hold_branch_reading_qty_reads_an_unknown_field_on_hold():
    msg = _load_error(_flow(
        _case("route", _ON, [("buy", "b"), ("hold", "h")], else_="b"),
        _code("b", {"q": "${d.output.qty}"}),
        _code("h", {"q": "${d.output.qty}"}),
    ))
    assert "node 'h' input 'q' from: reference ${d.output.qty} reads unknown field 'qty'" in msg
    assert "node 'b'" not in msg


# --- calls --- #


def test_a_call_target_under_buy_passes_a_buy_order():
    _loads(_flow(
        _case("route", _ON, [("buy", "call(place, order=${d.output})")], else_="e"),
        _code("e"),
    ))


def test_a_call_target_under_sell_fails_the_call_type_check():
    msg = _load_error(_flow(
        _case("route", _ON, [("sell", "call(place, order=${d.output})")], else_="e"),
        _code("e"),
    ))
    assert "argument 'order' — child expects" in msg
    assert """source is '{"action":Literal(sell)""" in msg


def test_an_unrouted_call_of_the_whole_union_still_fails():
    msg = _load_error(_flow(
        _code("c", {"r": "call(place, order=${d.output})"}),
    ))
    assert "argument 'order' — child expects" in msg


# --- chains --- #


def test_a_plain_reference_chain_is_narrowed():
    _loads(_flow(
        _case("route", _ON, [("buy", "b")], else_="e"),
        _code("b", {"o": "${d.output}"}),
        _code("e"),
        _code("c", {"y": "${b.output}", "l": "${d.output.limit}"}),
    ))


@pytest.mark.parametrize("join", ["${b.output | s.output}", "${b.output :- s.output}"])
def test_a_join_of_buy_and_sell_branches_narrows_to_both(join):
    branches = (
        _case("route", _ON, [("buy", "b"), ("sell", "s")], else_="e"),
        _code("b"),
        _code("s"),
        _code("e"),
    )
    _loads(_flow(*branches, _code("c", {"y": join, "q": "${d.output.qty}"})))
    msg = _load_error(_flow(*branches, _code("c", {"y": join, "l": "${d.output.limit}"})))
    assert (
        "node 'c' input 'l' from: reference ${d.output.limit} reads field 'limit', which only "
        "union members with action buy have"
    ) in msg


def test_depends_on_propagates_the_branch():
    _loads(_flow(
        _case("route", _ON, [("buy", "b")], else_="e"),
        _code("b"),
        _code("e"),
        _code("c", {"l": "${d.output.limit}"}, depends_on="[b]"),
    ))


def test_runs_after_does_not_propagate_the_branch():
    msg = _load_error(_flow(
        _case("route", _ON, [("buy", "b")], else_="e"),
        _code("b"),
        _code("e"),
        _code("c", {"l": "${d.output.limit}"}, runs_after="[b]"),
    ))
    assert f"node 'c' input 'l' from: {_LIMIT_ERROR}" in msg


def test_an_optional_group_does_not_propagate_the_branch():
    msg = _load_error(_flow(
        _case("route", _ON, [("buy", "b")], else_="e"),
        _code("b"),
        _code("e"),
        _code("c", {"y": "${b.output:-0}", "l": "${d.output.limit}"}),
    ))
    assert f"node 'c' input 'l' from: {_LIMIT_ERROR}" in msg


def test_a_nested_inline_call_in_a_branch_target_is_narrowed():
    # `inner` is lifted out of the routed `outer`, so it carries the same route.
    _loads(_flow(
        _case(
            "route",
            _ON,
            [("buy", "call(outer, x=call(inner, y=${d.output.qty}))")],
            else_="e",
        ),
        _code("e"),
    ))


def test_an_inline_call_in_a_routed_node_input_is_narrowed():
    # The call is lifted out of `b`, so it carries `b`'s route.
    _loads(_flow(
        _case("route", _ON, [("buy", "b")], else_="e"),
        _code("b", {"r": "call(place, order=${d.output})"}),
        _code("e"),
    ))


# --- routing --- #


def test_two_whens_to_one_node_give_a_two_member_union():
    branches = (_case("route", _ON, [("buy", "b"), ("sell", "b")], else_="e"), _code("e"))
    _loads(_flow(*branches, _code("b", {"q": "${d.output.qty}"})))
    msg = _load_error(_flow(*branches, _code("b", {"l": "${d.output.limit}"})))
    assert "reads field 'limit', which only union members with action buy have" in msg


def test_else_gives_the_remaining_member():
    branches = (_case("route", _ON, [("buy", "b"), ("sell", "s")], else_="e"),
                _code("b"), _code("s"))
    _loads(_flow(*branches, _code("e", {"n": "${d.output.note}"})))
    msg = _load_error(_flow(*branches, _code("e", {"q": "${d.output.price}"})))
    assert "reads unknown field 'price' on record" in msg


def test_else_gives_the_remaining_union():
    branches = (_case("route", _ON, [("buy", "b")], else_="e"), _code("b"))
    _loads(_flow(*branches, _code("e", {"n": "${d.output.note}"})))
    msg = _load_error(_flow(*branches, _code("e", {"p": "${d.output.price}"})))
    assert (
        "reference ${d.output.price} reads field 'price', which only union members with "
        "action sell have"
    ) in msg


def test_else_passes_the_remaining_union_to_a_call():
    _loads(_flow(
        _case("route", _ON, [("buy", "b")], else_="call(place_rest, order=${d.output})"),
        _code("b"),
    ))


def test_a_nullable_union_else_target_sees_an_optional_union():
    def flow(callee: str) -> str:
        return _flow(
            _case("route", _ON, [("buy", "b")], else_=f"call({callee}, order=${{d.output}})"),
            _code("b"),
            d_type="Optional(Decision)",
        )

    _loads(flow("place_maybe_rest"))
    msg = _load_error(flow("place_rest"))
    assert "argument 'order' — child expects" in msg and "Optional(" in msg


def test_a_nullable_union_narrows_to_a_non_nullable_member_under_when():
    _loads(_flow(
        _case("route", _ON, [("buy", "call(place, order=${d.output})")], else_="e"),
        _code("e"),
        d_type="Optional(Decision)",
    ))


# --- control edges --- #


def test_a_node_routed_by_two_narrowing_cases_on_different_bases_is_not_narrowed():
    msg = _load_error(_flow(
        _producer("d2", "Decision"),
        _case("route", _ON, [("buy", "b")], else_="e"),
        _case("route2", "${d2.output.action}", [("buy", "b")], else_="e"),
        _code("b", {"l": "${d.output.limit}"}),
        _code("e"),
    ))
    assert f"node 'b' input 'l' from: {_LIMIT_ERROR}" in msg


def test_a_node_routed_by_a_narrowing_case_and_a_searched_case_is_not_narrowed():
    msg = _load_error(_flow(
        _case("route", _ON, [("buy", "b")], else_="e"),
        _case("gate", None, [("input.s == 'x'", "b")], else_="e"),
        _code("b", {"l": "${d.output.limit}"}),
        _code("e"),
    ))
    assert f"node 'b' input 'l' from: {_LIMIT_ERROR}" in msg


# --- case of case --- #


def test_an_inner_searched_case_in_a_buy_branch_keeps_the_narrowing():
    _loads(_flow(
        _case("route", _ON, [("buy", "inner")], else_="e"),
        _case("inner", None, [("${d.output.limit > 5}", "t")], else_="t2"),
        _code("t", {"l": "${d.output.limit}"}),
        _code("t2", {"q": "${d.output.qty}"}),
        _code("e"),
    ))


def test_a_nested_on_case_over_a_union_field_narrows_outer_then_inner():
    _loads(_flow(
        _case("route", _ON, [("buy", "inner")], else_="e"),
        _case("inner", "${d.output.leg.type}", [("limit", "t"), ("market", "m")]),
        _code("t", {"p": "${d.output.leg.price}"}),
        _code("m"),
        _code("e"),
    ))


def test_a_nested_on_case_outside_the_outer_branch_is_a_rule_one_error():
    msg = _load_error(_flow(
        _case("inner", "${d.output.leg.type}", [("limit", "t"), ("market", "m")]),
        _code("t"),
        _code("m"),
    ))
    assert "node 'inner'" in msg and "reads field 'leg', which only union members" in msg


def test_an_inner_on_case_is_exhaustive_over_its_narrowed_tag():
    _loads(_flow(
        _case("route", _ON, [("buy", "inner")], else_="e"),
        _case("inner", _ON, [("buy", "t")]),
        _code("t", {"l": "${d.output.limit}"}),
        _code("e"),
    ))


def test_an_inner_on_case_over_a_flow_input_is_exhaustive_over_its_narrowed_tag():
    _loads(_flow(
        _case("route", "${input.d.action}", [("buy", "inner")], else_="e"),
        _case("inner", "${input.d.action}", [("buy", "t")]),
        _code("t", {"l": "${input.d.limit}"}),
        _code("e"),
        inputs="  s: !str\n  d: !Decision\n",
    ))


# --- other bases --- #


def test_a_flow_input_base_narrows():
    def flow(*extra: str) -> str:
        return _flow(
            _case("route", "${input.d.action}", [("buy", "b")], else_="e"),
            _code("b", {"l": "${input.d.limit}"}),
            _code("e", {"n": "${input.d.note}"}),
            *extra,
            inputs="  s: !str\n  d: !Decision\n",
        )

    _loads(flow())
    msg = _load_error(flow(_code("o", {"l": "${input.d.limit}"})))
    assert "node 'o' input 'l' from: reference ${input.d.limit} reads field 'limit'" in msg
    assert "node 'b'" not in msg


def test_a_path_base_narrows():
    _loads(_flow(
        _producer("r", "Wrap"),
        _case("route", "${r.output.decision.action}", [("buy", "b")], else_="e"),
        _code("b", {"l": "${r.output.decision.limit}"}),
        _code("e", {"n": "${r.output.decision.note}"}),
    ))


def test_a_path_base_through_an_optional_record_keeps_nullability_in_else():
    def flow(callee: str) -> str:
        return _flow(
            _producer("r", "Optional(Wrap)"),
            _case(
                "route",
                "${r.output.decision.action}",
                [("buy", "b")],
                else_=f"call({callee}, order=${{r.output.decision}})",
            ),
            _code("b", {"l": "${r.output.decision.limit}"}),
        )

    _loads(flow("place_maybe_rest"))
    msg = _load_error(flow("place_rest"))
    assert "argument 'order' — child expects" in msg and "Optional(" in msg


# --- not narrowed --- #


def test_on_over_a_coalesce_does_not_narrow():
    msg = _load_error(_flow(
        _producer("d2", "Decision"),
        _case("route", "${d.output.action | d2.output.action}", [("buy", "b")], else_="e"),
        _code("b", {"l": "${d.output.limit}"}),
        _code("e"),
    ))
    assert f"node 'b' input 'l' from: {_LIMIT_ERROR}" in msg


def test_on_over_embedded_text_does_not_narrow():
    msg = _load_error(_flow(
        _case("route", "'x${d.output.action}'", [("xbuy", "b")], else_="e"),
        _code("b", {"l": "${d.output.limit}"}),
        _code("e"),
    ))
    assert f"node 'b' input 'l' from: {_LIMIT_ERROR}" in msg


def test_nodes_outside_the_branch_and_the_flow_output_are_not_narrowed():
    msg = _load_error(_flow(
        _case("route", _ON, [("buy", "b")], else_="e"),
        _code("b", {"l": "${d.output.limit}"}),
        _code("e"),
        _code("o", {"l": "${d.output.limit}"}),
        output="${d.output.limit}",
    ))
    assert f"node 'o' input 'l' from: {_LIMIT_ERROR}" in msg
    assert f"flow output from: {_LIMIT_ERROR}" in msg
    assert "node 'b'" not in msg


# --- robustness --- #


def test_a_cycle_through_a_branch_reports_the_cycle():
    msg = _load_error(_flow(
        _case("route", _ON, [("buy", "b")], else_="e"),
        _code("b", {"y": "${c.output}", "l": "${d.output.limit}"}),
        _code("c", {"y": "${b.output}", "l": "${d.output.limit}"}),
        _code("e"),
    ))
    assert "flow has a cycle involving ['b', 'c']" in msg


def test_a_member_only_field_on_a_cycle_reports_the_cycle():
    msg = _load_error(_flow(
        _code("b", {"y": "${c.output}", "l": "${d.output.limit}"}),
        _code("c", {"y": "${b.output}"}),
    ))
    assert "flow has a cycle involving ['b', 'c']" in msg


def test_a_field_no_member_has_on_a_cycle_is_still_an_unknown_field():
    msg = _load_error(_flow(
        _code("b", {"y": "${c.output}", "z": "${d.output.zzz}"}),
        _code("c", {"y": "${b.output}"}),
    ))
    assert "reads unknown field 'zzz' on record" in msg


def test_an_unknown_depends_on_still_reports_the_unknown_node():
    msg = _load_error(_flow(
        _case("route", _ON, [("buy", "b")], else_="e"),
        _code("b", {"l": "${d.output.limit}"}, depends_on="[nope]"),
        _code("e"),
    ))
    assert "node 'b': `depends_on` references unknown node 'nope'" in msg


def test_an_inline_body_branch_target_cannot_infer_a_member_field():
    target = (
        "  b:\n"
        "    kind: call\n"
        "    input:\n      q: ${d.output.qty}\n"
        "    nodes:\n"
        "      inner:\n        kind: code\n        input:\n          q: ${input.q}\n"
        "        code: return 1\n"
        "    output: ${inner.output}\n"
    )
    msg = _load_error(_flow(_case("route", _ON, [("buy", "b")], else_="e"), target, _code("e")))
    assert f"cannot infer type for inline binding 'b.input.q': {_QTY_ERROR}" in msg


# --- empty narrowing sets --- #


def test_a_defensive_else_after_every_tag_loads_and_is_not_narrowed():
    branches = (
        _case("route", _ON, [("buy", "b"), ("sell", "b"), ("hold", "b")], else_="e"),
        _code("b"),
    )
    _loads(_flow(*branches, _code("e", {"a": "${d.output.action}"})))
    msg = _load_error(_flow(*branches, _code("e", {"l": "${d.output.limit}"})))
    assert f"node 'e' input 'l' from: {_LIMIT_ERROR}" in msg


def test_a_nullable_union_defensive_else_sees_none():
    msg = _load_error(_flow(
        _case(
            "route",
            _ON,
            [("buy", "b"), ("sell", "b"), ("hold", "b")],
            else_="call(place_rest, order=${d.output})",
        ),
        _code("b"),
        d_type="Optional(Decision)",
    ))
    assert "argument 'order' — child expects" in msg
    assert msg.endswith("source is 'None'")


def test_a_when_that_is_not_a_tag_is_a_located_error():
    text = _flow(_case("route", _ON, [("buyy", "b")], else_="e"), _code("b"), _code("e"))
    with pytest.raises(LoadError) as caught:
        load_flow(text)
    message = str(caught.value)
    assert message.startswith(
        "case node 'route': `when: buyy` is not a tag of `on: ${d.output.action}` ("
    )
    assert message.endswith("); use one of: buy, hold, sell")
    line = next(n for n, row in enumerate(text.splitlines(), 1) if "when: buyy" in row)
    assert caught.value.diagnostic.frames[-1].range.start_line == line


def test_a_contradictory_nested_case_target_is_not_narrowed():
    branches = (
        _case("route", _ON, [("buy", "inner")], else_="e"),
        _case("inner", _ON, [("buy", "t1"), ("sell", "t")]),
        _code("t1"),
        _code("e"),
    )
    _loads(_flow(*branches, _code("t", {"a": "${d.output.action}"})))
    msg = _load_error(_flow(*branches, _code("t", {"l": "${d.output.limit}"})))
    assert f"node 't' input 'l' from: {_LIMIT_ERROR}" in msg


def test_required_groups_from_buy_and_sell_branches_meet_empty():
    msg = _load_error(_flow(
        _case("route", _ON, [("buy", "b"), ("sell", "s")], else_="e"),
        _code("b"),
        _code("s"),
        _code("e"),
        _code("c", {"x": "${b.output}", "y": "${s.output}", "q": "${d.output.qty}"}),
    ))
    assert f"node 'c' input 'q' from: {_QTY_ERROR}" in msg


# --- exhaustiveness --- #


def test_a_non_exhaustive_on_over_a_literal_input_errors():
    msg = _load_error(_flow(
        _case("route", "${input.mode}", [("a", "b")]),
        _code("b"),
        inputs="  s: !str\n  mode: !Mode\n",
    ))
    assert "enum tag(s) ['b'] not covered" in msg


def test_a_non_exhaustive_on_over_a_union_node_output_still_errors():
    msg = _load_error(_flow(_case("route", _ON, [("buy", "b")]), _code("b")))
    assert "enum tag(s) ['hold', 'sell'] not covered" in msg


# --- isinstance in a searched CASE --- #


def _searched(*cases: tuple[str, str], else_: str | None = "e") -> str:
    """A searched CASE `route` over `cases`."""
    return _case("route", None, list(cases), else_=else_)


def test_a_target_under_isinstance_reads_a_member_field():
    _loads(_flow(
        _searched(("isinstance(d.output, Buy)", "b")),
        _code("b", {"q": "${d.output.qty}", "l": "${d.output.limit}"}),
        _code("e"),
    ))


@pytest.mark.parametrize(
    "when",
    [
        "isinstance(d.output, Buy) and input.s == 'x'",
        "input.s == 'x' and isinstance(d.output, Buy)",
        "${isinstance(d.output, Buy) and input.s == 'x'}",
    ],
)
def test_an_isinstance_and_conjunct_narrows(when):
    _loads(_flow(
        _searched((f'"{when}"', "b")),
        _code("b", {"l": "${d.output.limit}"}),
        _code("e"),
    ))


@pytest.mark.parametrize(
    "when",
    [
        "isinstance(d.output, Buy) or input.s == 'x'",
        "not (isinstance(d.output, Buy) and input.s == 'x')",
        "not isinstance(d.output, Hold)",
    ],
)
def test_isinstance_under_or_narrows_only_what_the_when_proves(when):
    # The check may not hold when `input.s` decides, and `Sell` passes `not Hold` too.
    msg = _load_error(_flow(
        _searched((f'"{when}"', "b")),
        _code("b", {"l": "${d.output.limit}"}),
        _code("e"),
    ))
    assert f"node 'b' input 'l' from: {_LIMIT_ERROR}" in msg


@pytest.mark.parametrize("when", ["d.output.action == 'buy'", "'buy' == d.output.action"])
def test_a_branch_sees_a_tag_comparison(when):
    _loads(_flow(
        _searched((f'"{when}"', "b")),
        _code("b", {"l": "${d.output.limit}"}),
        _code("e"),
    ))


@pytest.mark.parametrize(
    "when",
    [
        "not isinstance(d.output, Hold)",
        "isinstance(d.output, Buy) or isinstance(d.output, Sell)",
        "d.output.action != 'hold'",
    ],
)
def test_a_branch_reads_the_fields_the_members_its_when_proves_share(when):
    _loads(_flow(
        _searched((f'"{when}"', "b")),
        _code("b", {"q": "${d.output.qty}"}),
        _code("e"),
    ))
    msg = _load_error(_flow(
        _searched((f'"{when}"', "b")),
        _code("b"),
        _code("e", {"l": "${d.output.limit}"}),
    ))
    assert f"node 'e' input 'l' from: {_LIMIT_ERROR}" in msg


@pytest.mark.parametrize(
    "when",
    [
        "not isinstance(d.output, Buy) or d.output.limit > 5",
        "d.output.action == 'buy' and d.output.limit > 5",
        "d.output.action != 'buy' or d.output.limit > 5",
        "not (isinstance(d.output, Buy) and d.output.limit > 5)",
        "(isinstance(d.output, Buy) and d.output.limit > 5) or d.output.action == 'hold'",
        "isinstance(d.output, Rest) or d.output.limit > 5",
    ],
)
def test_a_read_in_a_when_is_narrowed_through_and_or_and_not(when):
    _loads(_flow(_searched((f'"{when}"', "b")), _code("b"), _code("e")))


@pytest.mark.parametrize(
    ("when", "narrowed"),
    [
        ("isinstance(d.output, Buy) or d.output.limit > 5", "Hold | Sell"),
        # `Buy` is no `Rest`, so only the shared `Hold` and `Sell` reach the read.
        ("not isinstance(d.output, Rest) or d.output.limit > 5", "Hold | Sell"),
    ],
)
def test_a_read_in_a_when_of_a_field_the_narrowed_members_lack_is_an_error(when, narrowed):
    msg = _load_error(_flow(_searched((f'"{when}"', "b")), _code("b"), _code("e")))
    assert (
        f"node 'route' cases[0].when from: reference ${{d.output.limit}} reads field 'limit', "
        f"which {narrowed} doesn't have; d.output is {narrowed} here, by the checks before "
        "this read"
    ) in msg


def test_a_misspelled_read_in_a_when_names_the_narrowed_type():
    msg = _load_error(_flow(
        _searched(('"isinstance(d.output, Buy) and d.output.limt > 5"', "b")),
        _code("b"),
        _code("e"),
    ))
    assert (
        "node 'route' cases[0].when from: reference ${d.output.limt} reads field 'limt', "
        "which Buy doesn't have; d.output is Buy here, by the checks before this read; "
        "did you mean 'limit'?"
    ) in msg


def test_a_branch_of_a_negated_partly_shared_check_keeps_the_overlapping_member():
    when = '"not isinstance(d.output, SellOrNarrow)"'
    msg = _load_error(_flow(
        _searched((when, "b")), _code("b", {"l": "${d.output.limit}"}), _code("e")
    ))
    assert (
        "node 'b' input 'l' from: reference ${d.output.limit} reads field 'limit', which only "
        "union members with action buy have"
    ) in msg
    msg = _load_error(_flow(
        _searched((when, "b")), _code("b", {"p": "${d.output.price}"}), _code("e")
    ))
    assert "node 'b' input 'p' from: reference ${d.output.price} reads unknown field 'price'" in msg


def test_a_read_after_a_check_the_routing_decides_is_checked_against_the_routing():
    def flow(when: str) -> str:
        return _flow(
            _case("route", _ON, [("buy", "inner")], else_="e"),
            _case("inner", None, [(f'"{when}"', "t")], else_="m"),
            _code("t"),
            _code("m"),
            _code("e"),
        )

    _loads(flow("isinstance(d.output, Buy) or d.output.limit > 5"))
    msg = _load_error(flow("isinstance(d.output, Decision) or d.output.nosuch > 5"))
    assert "reads unknown field 'nosuch'" in msg


# --- impossible checks in a searched `when:` --- #


@pytest.mark.parametrize(
    ("when", "message"),
    [
        (
            "isinstance(d.output, Buy) and d.output.action == 'hold'",
            '`d.output.action == "hold"` can never be true here: d.output is Buy, by the '
            "checks before it",
        ),
        (
            "isinstance(d.output, Buy) and d.output.action != 'buy' and d.output.limit > 0",
            '`d.output.action != "buy"` can never be true here: d.output is Buy, by the checks '
            'before it, so `d.output.action == "buy"` is always true',
        ),
        (
            "d.output.action == 'biy'",
            "`d.output.action == \"biy\"` compares d.output.action with 'biy', which is not one "
            "of its tags (buy, hold, sell); did you mean 'buy'?",
        ),
        (
            "isinstance(d.output, Text)",
            "`isinstance(d.output, Text)` can never be true: d.output is declared Decision, "
            "and no Decision value fits Text",
        ),
        (
            "not not isinstance(d.output, Text)",
            "`isinstance(d.output, Text)` can never be true: d.output is declared Decision, "
            "and no Decision value fits Text",
        ),
    ],
)
def test_a_when_check_that_can_never_be_true_is_a_located_error(when, message):
    text = _flow(
        _searched(("input.s == 'x'", "b"), (f'"{when}"', "b")),
        _code("b"),
        _code("e"),
    )
    with pytest.raises(LoadError) as caught:
        load_flow(text)
    error = str(caught.value)
    assert f"case node 'route': `cases[1].when`: {message}" in error
    # A read after the impossible check is not checked on top of it.
    assert "'limit'" not in error
    line = next(n for n, row in enumerate(text.splitlines(), 1) if when in row)
    assert caught.value.diagnostic.frames[-1].range.start_line == line


def test_a_when_check_the_routing_rules_out_is_an_error():
    msg = _load_error(_flow(
        _case("route", _ON, [("buy", "inner")], else_="e"),
        _case("inner", None, [("\"d.output.action == 'hold'\"", "t")], else_="m"),
        _code("t"),
        _code("m"),
        _code("e"),
    ))
    assert (
        "case node 'inner': `cases[0].when`: `d.output.action == \"hold\"` can never be true "
        "here: d.output is Buy, by the case branch that routes this node"
    ) in msg


def test_a_searched_when_with_a_tag_check_and_a_member_read_runs():
    text = (
        "id: p\nname: p\n"
        + _TYPEDEFS
        + "input:\n  s: !str\n"
        + "nodes:\n"
        "  d:\n"
        "    kind: code\n"
        "    input:\n      s: ${input.s}\n"
        "    output: Decision\n"
        "    code: |\n"
        "      if inputs['s'] == 'buy':\n"
        "          return {'action': 'buy', 'qty': 1, 'limit': 6.0,"
        " 'leg': {'type': 'market'}}\n"
        "      return {'action': 'sell', 'qty': 2, 'price': 1.5, 'note': 'n'}\n"
        + _searched(("\"d.output.action == 'buy' and d.output.limit > 5\"", "t"))
        + "  t:\n    kind: code\n    output: str\n    code: return 'branch'\n"
        + "  e:\n    kind: code\n    output: str\n    code: return 'else'\n"
        + "output: ${route.output}\n"
    )
    flow = load_flow(text)
    result = run_flow(flow, {"s": "buy"})
    assert result.status == "succeeded", result.failure
    assert result.output == "branch"
    result = run_flow(flow, {"s": "sell"})
    assert result.status == "succeeded", result.failure
    assert result.output == "else"


def test_overlapping_checks_narrow_each_branch_by_its_own_check():
    def flow(bs_read: str) -> str:
        return _flow(
            _searched(
                ("isinstance(d.output, Buy)", "b"),
                ("isinstance(d.output, BuyOrSell)", "bs"),
            ),
            _code("b", {"l": "${d.output.limit}"}),
            _code("bs", {"x": bs_read}),
            _code("e"),
        )

    _loads(flow("${d.output.qty}"))
    msg = _load_error(flow("${d.output.limit}"))
    assert (
        "node 'bs' input 'x' from: reference ${d.output.limit} reads field 'limit', which "
        "only union members with action buy have"
    ) in msg


def test_a_conjunct_after_isinstance_reads_the_narrowed_base():
    _loads(_flow(
        _searched(('"isinstance(d.output, Buy) and d.output.limit > 5"', "b")),
        _code("b"),
        _code("e"),
    ))


def test_a_conjunct_before_isinstance_is_a_rule_one_error():
    msg = _load_error(_flow(
        _searched(('"d.output.limit > 5 and isinstance(d.output, Buy)"', "b")),
        _code("b"),
        _code("e"),
    ))
    assert f"node 'route' cases[0].when from: {_LIMIT_ERROR}" in msg


def test_a_read_guarded_in_one_case_and_unguarded_in_another_is_an_error():
    msg = _load_error(_flow(
        _searched(
            ('"isinstance(d.output, Buy) and d.output.limit > 5"', "b"),
            ('"d.output.limit > 1"', "b"),
        ),
        _code("b"),
        _code("e"),
    ))
    # The guarded read in cases[0] is fine; the unguarded one in cases[1] is the error.
    assert f"node 'route' cases[1].when from: {_LIMIT_ERROR}" in msg


def test_an_unguarded_read_error_points_at_its_own_when():
    text = _flow(
        _searched(
            ('"isinstance(d.output, Buy) and d.output.limit > 5"', "b"),
            ("input.s == 'x'", "b"),
            ('"d.output.limit > 1"', "b"),
        ),
        _code("b"),
        _code("e"),
    )
    with pytest.raises(LoadError) as caught:
        load_flow(text)
    assert f"node 'route' cases[2].when from: {_LIMIT_ERROR}" in str(caught.value)
    line = next(n for n, row in enumerate(text.splitlines(), 1) if "d.output.limit > 1" in row)
    assert caught.value.diagnostic.frames[-1].range.start_line == line


def test_a_read_unguarded_first_points_at_the_first_when():
    msg = _load_error(_flow(
        _searched(
            ('"d.output.limit > 1"', "b"),
            ('"isinstance(d.output, Buy) and d.output.limit > 5"', "b"),
        ),
        _code("b"),
        _code("e"),
    ))
    assert f"node 'route' cases[0].when from: {_LIMIT_ERROR}" in msg


@pytest.mark.parametrize("type_name", ["object", "Decision"])
def test_isinstance_of_a_type_the_base_already_fits_does_not_narrow(type_name):
    msg = _load_error(_flow(
        _searched((f"isinstance(d.output, {type_name})", "b")),
        _code("b", {"q": "${d.output.qty}"}),
        _code("e"),
    ))
    assert f"node 'b' input 'q' from: {_QTY_ERROR}" in msg


def test_a_read_guarded_by_different_members_reads_their_shared_fields():
    def flow(field: str) -> str:
        return _flow(
            _searched(
                (f'"isinstance(d.output, Buy) and d.output.{field} > 5"', "b"),
                (f'"isinstance(d.output, Sell) and d.output.{field} > 1"', "s"),
            ),
            _code("b"),
            _code("s"),
            _code("e"),
        )

    _loads(flow("qty"))
    msg = _load_error(flow("limit"))
    assert "reads field 'limit', which only union members with action buy have" in msg


def test_isinstance_narrows_an_untagged_union_to_the_member():
    def flow(*extra: str) -> str:
        return _flow(
            _searched(("isinstance(d.output, Text)", "b")),
            _code("b", {"t": "${d.output.text}"}),
            _code("e"),
            *extra,
            d_type="Reply",
            output="${d.output}",
        )

    _loads(flow())
    msg = _load_error(flow(_code("o", {"t": "${d.output.text}"})))
    assert (
        "node 'o' input 't' from: reference ${d.output.text} reads field 'text', which only "
        "1 of 2 union members have"
    ) in msg
    assert "node 'b'" not in msg


def test_isinstance_narrows_an_opaque_object_to_the_checked_type():
    def flow(when: str) -> str:
        return _flow(
            _producer("x", "object"),
            _searched((when, "call(place, order=${x.output})")),
            _code("e"),
        )

    _loads(flow("isinstance(x.output, Buy)"))
    msg = _load_error(flow("isinstance(x.output, Sell)"))
    assert "argument 'order' — child expects" in msg
    assert """source is '{"action":Literal(sell)""" in msg


def test_isinstance_cases_on_unrelated_types_into_one_target_do_not_narrow():
    def flow(*whens: str) -> str:
        return _flow(
            _producer("x", "object"),
            _searched(*((when, "b") for when in whens)),
            "  b:\n    kind: call\n    call: place\n    input:\n      order: ${x.output}\n",
            _code("e"),
        )

    _loads(flow("isinstance(x.output, Buy)"))
    msg = _load_error(flow("isinstance(x.output, Buy)", "isinstance(x.output, Text)"))
    assert "call node 'b': binding 'order' — child expects" in msg
    assert "source is 'object'" in msg


def test_an_opaque_object_outside_isinstance_does_not_fit_a_record():
    msg = _load_error(_flow(
        _producer("x", "object"),
        _searched(("input.s == 'x'", "call(place, order=${x.output})")),
        _code("e"),
    ))
    assert "argument 'order' — child expects" in msg and "source is 'object'" in msg


def test_a_target_routed_by_two_isinstance_cases_gets_their_union():
    def flow(read: str) -> str:
        return _flow(
            _searched(("isinstance(d.output, Buy)", "b"), ("isinstance(d.output, Sell)", "b")),
            _code("b", {"x": read}),
            _code("e"),
        )

    _loads(flow("${d.output.qty}"))
    msg = _load_error(flow("${d.output.limit}"))
    assert "reads field 'limit', which only union members with action buy have" in msg


def test_a_target_routed_by_isinstance_and_a_plain_case_is_not_narrowed():
    msg = _load_error(_flow(
        _searched(("isinstance(d.output, Buy)", "b"), ("input.s == 'x'", "b")),
        _code("b", {"l": "${d.output.limit}"}),
        _code("e"),
    ))
    assert f"node 'b' input 'l' from: {_LIMIT_ERROR}" in msg


def test_a_searched_else_is_not_narrowed():
    msg = _load_error(_flow(
        _searched(("isinstance(d.output, Buy)", "b")),
        _code("b"),
        _code("e", {"n": "${d.output.note}"}),
    ))
    assert (
        "node 'e' input 'n' from: reference ${d.output.note} reads field 'note', which only "
        "union members with action hold, sell have"
    ) in msg


def test_an_isinstance_branch_narrows_a_nullable_base_to_the_member():
    _loads(_flow(
        _searched(("isinstance(d.output, Buy)", "call(place, order=${d.output})")),
        _code("e"),
        d_type="Optional(Decision)",
    ))


def test_isinstance_on_a_flow_input_narrows():
    _loads(_flow(
        _searched(("isinstance(input.d, Buy)", "b")),
        _code("b", {"l": "${input.d.limit}"}),
        _code("e"),
        inputs="  s: !str\n  d: !Decision\n",
    ))


def test_isinstance_inside_an_on_branch_composes():
    _loads(_flow(
        _case("route", _ON, [("buy", "inner")], else_="e"),
        _case("inner", None, [("isinstance(d.output.leg, Limit)", "t")], else_="m"),
        _code("t", {"p": "${d.output.leg.price}"}),
        _code("m"),
        _code("e"),
    ))


def test_an_unknown_isinstance_type_keeps_its_own_error():
    msg = _load_error(_flow(
        _searched(("isinstance(d.output, Nope)", "b")),
        _code("b"),
        _code("e"),
    ))
    assert "unknown type 'Nope'" in msg


# --- isinstance against a partly shared union --- #


def test_isinstance_of_a_partly_shared_union_reads_the_shared_fields():
    _loads(_flow(
        _searched(("isinstance(d.output, Rest)", "b")),
        _code("b", {"n": "${d.output.note}"}),
        _code("e"),
    ))
    msg = _load_error(_flow(
        _searched(("isinstance(d.output, Rest)", "b")),
        _code("b", {"n": "${d.output.reason}"}),
        _code("e"),
    ))
    assert (
        "node 'b' input 'n' from: reference ${d.output.reason} reads unknown field 'reason'"
        in msg
    )
    assert "cancel" not in msg


def test_a_partly_shared_union_branch_passes_the_shared_members_to_a_call():
    branch = _searched(("isinstance(d.output, Rest)", "call(place_rest, order=${d.output})"))
    _loads(_flow(branch, _code("e")))
    _loads(_flow(branch, _code("e"), d_type="Optional(Decision)"))


def test_a_partly_shared_union_check_lines_up_with_an_on_else():
    _loads(_flow(
        _case("route", _ON, [("buy", "b")], else_="inner"),
        _case("inner", None, [("isinstance(d.output, Rest)", "t")], else_="m"),
        _code("b"),
        _code("t", {"n": "${d.output.note}"}),
        _code("m"),
    ))


def test_a_conjunct_after_a_partly_shared_union_check_reads_the_shared_fields():
    _loads(_flow(
        _searched(('"isinstance(d.output, Rest) and d.output.note == \'x\'"', "b")),
        _code("b"),
        _code("e"),
    ))


def test_an_on_inside_a_partly_shared_union_branch_covers_the_shared_tags():
    _loads(_flow(
        _searched(("isinstance(d.output, Rest)", "inner")),
        _case("inner", _ON, [("hold", "h"), ("sell", "s")]),
        _code("h"),
        _code("s"),
        _code("e"),
    ))


def test_a_partly_shared_union_branch_runs():
    text = (
        "id: p\nname: p\n"
        + _TYPEDEFS
        + "input:\n  s: !str\n"
        + "nodes:\n"
        "  d:\n"
        "    kind: code\n"
        "    input:\n      s: ${input.s}\n"
        "    output: Decision\n"
        "    code: |\n"
        "      return {'action': 'sell', 'qty': 2, 'price': 1.5, 'note': 'n'}\n"
        + _searched(("isinstance(d.output, Rest)", "t"))
        + "  t:\n"
        "    kind: code\n"
        "    input:\n      n: ${d.output.note}\n"
        "    output: str\n"
        "    code: return inputs['n']\n"
        + _code("e")
        + "output: ${route.output}\n"
    )
    result = run_flow(load_flow(text), {"s": "x"})
    assert result.status == "succeeded", result.failure
    assert result.output == "n"


def test_a_member_that_only_overlaps_the_checked_union_keeps_the_checked_type():
    call = "call(place_sell_or_narrow, order=${d.output})"
    _loads(_flow(_searched(("isinstance(d.output, SellOrNarrow)", call)), _code("e")))
    msg = _load_error(_flow(
        _searched(("isinstance(d.output, SellOrNarrow)", "b")),
        _code("b", {"p": "${d.output.price}"}),
        _code("e"),
    ))
    assert "which only union members with action sell have" in msg
