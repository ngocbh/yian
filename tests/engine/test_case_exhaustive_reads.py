"""A CASE whose result is read through `${case.output}` must be exhaustive.

`${case.output}` reads a coalesce over the CASE's branch targets, so a CASE read that way
(or one such a CASE routes to) must take a branch on every run: it has an `else:`, or it
is an `on:` form whose `on:` type is a non-nullable Literal or bool listed in full, under
the narrowing of the CASE branches that reach it. A routing-only CASE may match nothing
and run no branch. An unquoted bare word names a node, even a YAML 1.1 boolean word.
"""

import pytest

from yian.compose import LoadError, load_flow
from yian.events import NodeSkipped, NodeStarted
from yian.run import run_flow


def _line_of(text: str, needle: str) -> int:
    """Return the 1-based line number of the first line containing `needle`."""
    return next(
        number
        for number, line in enumerate(text.splitlines(), start=1)
        if needle in line
    )


def _error_line(error: LoadError) -> int:
    """Return the start line of the innermost frame of a load error."""
    frame = error.diagnostic.frames[-1]
    assert frame.range is not None
    return frame.range.start_line


def _started(result) -> set[str]:
    """Return the ids of every node that started in a run."""
    return {e.node_id for e in result.events if isinstance(e, NodeStarted)}


def _skipped(result) -> set[str]:
    """Return the ids of every node skipped in a run."""
    return {e.node_id for e in result.events if isinstance(e, NodeSkipped)}


_EXHAUSTIVE = "case node {case!r} produces a value, so it must be exhaustive: add an `else:`"


def _gate_flow(
    gate: str,
    *,
    inputs: str = "  x: !str\n",
    output: str = "${gate.output}",
    extra_nodes: str = "",
    typedefs: str = "",
) -> str:
    """Return a flow with a `gate` case (indented body), CODE nodes `pa`/`pb`, and `output`."""
    return (
        "id: f\n"
        "name: f\n"
        f"{typedefs}"
        "input:\n"
        f"{inputs}"
        "nodes:\n"
        "  gate:\n"
        "    kind: case\n"
        f"{gate}"
        "  pa:\n"
        "    kind: code\n"
        "    code: |\n"
        '      return "A"\n'
        "    output: str\n"
        "  pb:\n"
        "    kind: code\n"
        "    code: |\n"
        '      return "B"\n'
        "    output: str\n"
        f"{extra_nodes}"
        f"output: {output}\n"
    )


_CHOICE = "typedefs:\n  Choice: Literal(a, b)\n"


def _routes(*tags: str) -> str:
    """Return `cases:` routing each tag to the CODE node `p<tag>`."""
    return "    cases:\n" + "".join(
        f"      - when: {tag}\n        then: p{tag}\n" for tag in tags
    )


def test_read_case_over_a_str_without_else_is_rejected():
    text = _gate_flow("    on: ${input.x}\n" + _routes("a", "b"))
    with pytest.raises(LoadError) as caught:
        load_flow(text)
    assert (
        _EXHAUSTIVE.format(case="gate") + " (on: ${input.x} is str, not a Literal or bool)"
    ) in str(caught.value)
    assert _error_line(caught.value) == _line_of(text, "  gate:")


def test_read_searched_case_without_else_is_rejected():
    text = _gate_flow(
        "    cases:\n"
        "      - when: \"input.x == 'a'\"\n"
        "        then: pa\n"
    )
    with pytest.raises(LoadError) as caught:
        load_flow(text)
    assert (_EXHAUSTIVE.format(case="gate") + " (a searched case needs an else:)") in str(
        caught.value
    )
    assert _error_line(caught.value) == _line_of(text, "  gate:")


_PICK = (
    "  pick:\n"
    "    kind: code\n"
    "    input:\n"
    "      x: ${input.x}\n"
    "    code: |\n"
    '      return inputs["x"]\n'
    "    output: Literal(a, b)\n"
)


def test_read_case_over_a_literal_node_output_listing_every_tag_runs():
    text = _gate_flow("    on: ${pick.output}\n" + _routes("a", "b"), extra_nodes=_PICK)
    loaded = load_flow(text)
    assert run_flow(loaded, {"x": "a"}).output == "A"
    assert run_flow(loaded, {"x": "b"}).output == "B"


def test_read_case_over_a_literal_node_output_missing_a_tag_is_rejected():
    text = _gate_flow("    on: ${pick.output}\n" + _routes("a"), extra_nodes=_PICK)
    with pytest.raises(LoadError) as caught:
        load_flow(text)
    assert (
        _EXHAUSTIVE.format(case="gate") + " (enum tag(s) ['b'] not covered)"
    ) in str(caught.value)
    assert _error_line(caught.value) == _line_of(text, "  gate:")


def test_read_case_over_a_literal_input_listing_every_tag_runs():
    text = _gate_flow(
        "    on: ${input.choice}\n" + _routes("a", "b"),
        typedefs=_CHOICE,
        inputs="  choice: !Choice\n",
    )
    loaded = load_flow(text)
    assert run_flow(loaded, {"choice": "a"}).output == "A"
    assert run_flow(loaded, {"choice": "b"}).output == "B"


def test_read_case_over_a_literal_input_missing_a_tag_is_rejected():
    text = _gate_flow(
        "    on: ${input.choice}\n" + _routes("a"),
        typedefs=_CHOICE,
        inputs="  choice: !Choice\n",
    )
    with pytest.raises(LoadError) as caught:
        load_flow(text)
    assert (
        _EXHAUSTIVE.format(case="gate") + " (enum tag(s) ['b'] not covered)"
    ) in str(caught.value)


def test_read_case_over_an_optional_literal_input_is_rejected():
    text = _gate_flow(
        "    on: ${input.choice}\n" + _routes("a", "b"),
        typedefs=_CHOICE,
        inputs="  choice: !Optional(Choice)\n",
    )
    with pytest.raises(LoadError) as caught:
        load_flow(text)
    assert (
        _EXHAUSTIVE.format(case="gate") + " (${input.choice} may be null)"
    ) in str(caught.value)


def test_routing_only_case_over_a_str_without_else_still_skips_on_no_match():
    text = _gate_flow(
        "    on: ${input.x}\n" + _routes("a", "b"),
        output="${pa.output | pb.output:-'none'}",
    )
    loaded = load_flow(text)
    assert run_flow(loaded, {"x": "a"}).output == "A"
    result = run_flow(loaded, {"x": "c"})
    assert result.status == "succeeded", result.failure
    assert result.output == "none"
    assert {"pa", "pb"} <= _skipped(result)
    assert not {"pa", "pb"} & _started(result)


def test_routing_only_case_over_a_literal_input_missing_a_tag_is_a_load_error():
    text = _gate_flow(
        "    on: ${input.choice}\n" + _routes("a"),
        typedefs=_CHOICE,
        inputs="  choice: !Choice\n",
        output="${input.choice}",
    )
    with pytest.raises(LoadError) as caught:
        load_flow(text)
    assert "non-exhaustive — enum tag(s) ['b'] not covered" in str(caught.value)
    assert _error_line(caught.value) == _line_of(text, "  gate:")


def test_inner_case_of_a_value_producing_case_must_be_exhaustive():
    text = _gate_flow(
        "    on: ${input.x}\n"
        "    cases:\n"
        "      - when: a\n"
        "        then: pa\n"
        "    else: inner\n",
        extra_nodes=(
            "  inner:\n"
            "    kind: case\n"
            "    on: ${input.x}\n"
            "    cases:\n"
            "      - when: b\n"
            "        then: pb\n"
        ),
    )
    with pytest.raises(LoadError) as caught:
        load_flow(text)
    assert (
        _EXHAUSTIVE.format(case="inner") + " (on: ${input.x} is str, not a Literal or bool)"
    ) in str(caught.value)
    assert _error_line(caught.value) == _line_of(text, "  inner:")


def test_case_read_inside_text_must_be_exhaustive():
    text = _gate_flow(
        "    on: ${input.x}\n" + _routes("a", "b"), output="got ${gate.output}"
    )
    with pytest.raises(LoadError) as caught:
        load_flow(text)
    assert (
        _EXHAUSTIVE.format(case="gate") + " (on: ${input.x} is str, not a Literal or bool)"
    ) in str(caught.value)


def _narrowed_flow(inner_tag: str) -> str:
    """Return a flow whose `route` sends `buy` to `inner`, a read case on the same tag."""
    return (
        "id: f\n"
        "name: f\n"
        "typedefs:\n"
        "  Buy:\n"
        "    action: Literal(buy)\n"
        "    qty: int\n"
        "  Sell:\n"
        "    action: Literal(sell)\n"
        "    price: float\n"
        "input:\n"
        "  kind: !str\n"
        "nodes:\n"
        "  d:\n"
        "    kind: code\n"
        "    input:\n"
        "      kind: ${input.kind}\n"
        "    code: |\n"
        '      if inputs["kind"] == "buy":\n'
        '          return {"action": "buy", "qty": 3}\n'
        '      return {"action": "sell", "price": 1.5}\n'
        "    output: Union(Buy, Sell)\n"
        "  route:\n"
        "    kind: case\n"
        "    on: ${d.output.action}\n"
        "    cases:\n"
        "      - when: buy\n"
        "        then: inner\n"
        "    else: sold\n"
        "  inner:\n"
        "    kind: case\n"
        "    on: ${d.output.action}\n"
        "    cases:\n"
        f"      - when: {inner_tag}\n"
        "        then: bought\n"
        '  bought: !str "bought ${d.output.qty}"\n'
        "  sold: !str sold\n"
        "output: ${route.output}\n"
    )


def test_narrowed_read_case_counts_only_the_tags_that_reach_it():
    loaded = load_flow(_narrowed_flow("buy"))
    assert run_flow(loaded, {"kind": "buy"}).output == "bought 3"
    assert run_flow(loaded, {"kind": "sell"}).output == "sold"


def test_narrowed_read_case_missing_its_reachable_tag_is_rejected():
    text = _narrowed_flow("sell")
    with pytest.raises(LoadError) as caught:
        load_flow(text)
    assert _EXHAUSTIVE.format(case="inner") in str(caught.value)
    assert "enum tag(s) ['buy'] not covered" in str(caught.value)


def test_unquoted_yes_routes_to_a_node_named_yes():
    text = (
        "id: f\n"
        "name: f\n"
        "input:\n"
        "  cmd: !str\n"
        "nodes:\n"
        "  route:\n"
        "    kind: case\n"
        "    on: ${input.cmd}\n"
        "    cases:\n"
        "      - when: a\n"
        "        then: reply\n"
        "    else: yes\n"
        "  reply:\n"
        "    kind: code\n"
        "    code: tests.engine._compose_codefns:echo\n"
        "    input:\n"
        "      topic: replied\n"
        "    output: str\n"
        "  yes:\n"
        "    kind: code\n"
        "    code: tests.engine._compose_codefns:echo\n"
        "    input:\n"
        "      topic: agreed\n"
        "    output: str\n"
        "output: ${route.output}\n"
    )
    result = run_flow(load_flow(text), {"cmd": "b"})
    assert result.status == "succeeded", result.failure
    assert "yes" in _started(result)
    assert "reply" not in _started(result)
