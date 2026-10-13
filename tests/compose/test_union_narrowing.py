"""Union narrowing at load: an `on:` CASE branch reads the members its tag selects."""

import pytest

from yian.compose import LoadError, load_flow

_TYPEDEFS = (
    "typedefs:\n"
    "  Limit: {type: 'Literal[limit]', price: float}\n"
    "  Market: {type: 'Literal[market]'}\n"
    "  Leg: Union[Limit, Market]\n"
    "  Buy: {action: 'Literal[buy]', qty: int, limit: float, leg: Leg}\n"
    "  Hold: {action: 'Literal[hold]', note: str}\n"
    "  Sell: {action: 'Literal[sell]', qty: int, price: float, note: str}\n"
    "  Decision: Union[Buy, Hold, Sell]\n"
    "  HoldOrSell: Union[Hold, Sell]\n"
    "  Wrap: {decision: Decision}\n"
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
    "    input:\n      order: Buy\n"
    "    nodes:\n"
    "      p:\n        kind: code\n        input:\n          o: ${input.order}\n"
    "        code: return 1\n"
    "    output: ${p.output}\n"
    "  place_rest:\n"
    "    input:\n      order: HoldOrSell\n"
    "    nodes:\n"
    "      p:\n        kind: code\n        input:\n          o: ${input.order}\n"
    "        code: return 1\n"
    "    output: ${p.output}\n"
    "  place_maybe_rest:\n"
    "    input:\n      order: Optional[HoldOrSell]\n"
    "    nodes:\n"
    "      p:\n        kind: code\n        input:\n          o: ${input.order}\n"
    "        code: return 1\n"
    "    output: ${p.output}\n"
    "  outer:\n"
    "    input:\n      x: int\n"
    "    nodes:\n"
    "      p:\n        kind: code\n        input:\n          x: ${input.x}\n"
    "        code: return 1\n"
    "    output: ${p.output}\n"
    "  inner:\n"
    "    input:\n      y: int\n"
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
    inputs: str = "  s: str\n",
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


def test_a_hold_branch_reading_qty_is_a_rule_one_error():
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
    assert "binding 'order' — child expects" in msg
    assert """source is '{"action":Literal["sell"]""" in msg


def test_an_unrouted_call_of_the_whole_union_still_fails():
    msg = _load_error(_flow(
        _code("c", {"r": "call(place, order=${d.output})"}),
    ))
    assert "binding 'order' — child expects" in msg


# --- chains --- #


def test_a_plain_reference_chain_is_narrowed():
    _loads(_flow(
        _case("route", _ON, [("buy", "b")], else_="e"),
        _code("b", {"o": "${d.output}"}),
        _code("e"),
        _code("c", {"y": "${b.output}", "l": "${d.output.limit}"}),
    ))


@pytest.mark.parametrize("join", ["${b.output | s.output}", "${b.output:-${s.output}}"])
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


def test_a_nested_inline_call_in_a_branch_target_is_not_narrowed():
    msg = _load_error(_flow(
        _case(
            "route",
            _ON,
            [("buy", "call(outer, x=call(inner, y=${d.output.qty}))")],
            else_="e",
        ),
        _code("e"),
    ))
    # `inner` reads `qty` as a data producer of `outer`; only `outer` is routed.
    assert f"input 'y' from: {_QTY_ERROR}" in msg


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
            d_type="Optional[Decision]",
        )

    _loads(flow("place_maybe_rest"))
    msg = _load_error(flow("place_rest"))
    assert "binding 'order' — child expects" in msg and "Optional[" in msg


def test_a_nullable_union_narrows_to_a_non_nullable_member_under_when():
    _loads(_flow(
        _case("route", _ON, [("buy", "call(place, order=${d.output})")], else_="e"),
        _code("e"),
        d_type="Optional[Decision]",
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
        _case("gate", None, [("${input.s} == 'x'", "b")], else_="e"),
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


# --- other bases --- #


def test_a_flow_input_base_narrows():
    def flow(*extra: str) -> str:
        return _flow(
            _case("route", "${input.d.action}", [("buy", "b")], else_="e"),
            _code("b", {"l": "${input.d.limit}"}),
            _code("e", {"n": "${input.d.note}"}),
            *extra,
            inputs="  s: str\n  d: Decision\n",
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
            _producer("r", "Optional[Wrap]"),
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
    assert "binding 'order' — child expects" in msg and "Optional[" in msg


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


def test_a_when_that_is_not_a_tag_is_a_located_error():
    text = _flow(_case("route", _ON, [("buyy", "b")], else_="e"), _code("b"), _code("e"))
    with pytest.raises(LoadError) as caught:
        load_flow(text)
    assert (
        "when: buyy is not a tag of the union at ${d.output.action} (tags: buy, hold, sell)"
    ) in str(caught.value)
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


def test_a_non_exhaustive_on_over_a_literal_input_still_loads():
    _loads(_flow(
        _case("route", "${input.mode}", [("a", "b")]),
        _code("b"),
        inputs="  s: str\n  mode: Literal[a, b]\n",
    ))


def test_a_non_exhaustive_on_over_a_union_node_output_still_errors():
    msg = _load_error(_flow(_case("route", _ON, [("buy", "b")]), _code("b")))
    assert "enum tag(s) ['hold', 'sell'] not covered" in msg
