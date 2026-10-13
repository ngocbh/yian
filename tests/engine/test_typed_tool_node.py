"""A TOOL node's optional `output:` contract is checked against every value the tool returns."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from yian.compose import load_flow
from yian.nodes.base import _HostStop
from yian.run import resume_flow, run_flow
from yian.tools import TOOL_REGISTRY


@pytest.fixture(autouse=True)
def _restore_tool_registry():
    """Keep process-global registrations isolated between tests."""
    original = dict(TOOL_REGISTRY)
    yield
    TOOL_REGISTRY.clear()
    TOOL_REGISTRY.update(original)


_INT_FLOW = """
id: count
name: count
nodes:
  place:
    kind: tool
    tool_id: place_order
    input:
      qty: 5
    output: int
output: ${place.output}
"""


def _returning(value):
    def place_order(qty):
        return value

    return place_order


def test_a_value_that_fits_the_declared_output_succeeds():
    result = run_flow(load_flow(_INT_FLOW), {}, tools={"place_order": _returning(3)})

    assert result.status == "succeeded", result.failure
    assert result.output == 3


def test_a_value_that_does_not_fit_fails_at_the_output_declaration():
    flow = load_flow(_INT_FLOW)
    result = run_flow(flow, {}, tools={"place_order": _returning("x")})

    assert result.status == "failed"
    assert result.failure.node_id == "place"
    assert result.failure.code == "contract.output"
    assert "does not match declared type int" in result.failure.message
    (frame,) = flow.diagnose(result.failure).frames
    authored = frame.text.splitlines()[frame.range.start_line - 1]
    assert authored.strip().startswith("output: int")


_MAP_FLOW = """
id: batch
name: batch
nodes:
  each:
    kind: map
    over: [1, 2]
    input:
      x: ${item}
    nodes:
      place:
        kind: tool
        tool_id: place_order
        input:
          x: ${input.x}
        output: int
    output: ${place.output}
output: ${each.output}
"""


def test_the_gate_runs_on_every_call_inside_a_map():
    def place_order(x):
        return "bad" if x == 2 else x

    result = run_flow(load_flow(_MAP_FLOW), {}, tools={"place_order": place_order})

    assert result.status == "failed"
    assert result.failure.node_id == "each#1/place"
    assert result.failure.code == "contract.output"
    assert "'bad' does not match declared type int" in result.failure.message


_ONE_NODE_FLOW = """
id: count
name: count
input:
  qty: int
kind: tool
tool_id: place_order
output: int
"""


def test_a_one_node_tool_flow_gates_its_output():
    flow = load_flow(_ONE_NODE_FLOW)

    ok = run_flow(flow, {"qty": 5}, tools={"place_order": _returning(4)})
    bad = run_flow(flow, {"qty": 5}, tools={"place_order": _returning("x")})

    assert ok.status == "succeeded", ok.failure
    assert ok.output == 4
    assert bad.status == "failed"
    assert bad.failure.node_id == "count"
    assert bad.failure.code == "contract.output"
    (frame,) = flow.diagnose(bad.failure).frames
    authored = frame.text.splitlines()[frame.range.start_line - 1]
    assert authored.strip().startswith("output: int")


_ORDERS_FLOW = """
id: orders
name: orders
typedefs:
  Filled: {type: 'Literal[filled]', qty: int}
  Interrupted: {type: 'Literal[interrupted]', by: str, partial: 'Optional[int]'}
  OrderResult: Union[Filled, Interrupted]
nodes:
  place:
    kind: tool
    tool_id: place_order
    input:
      qty: 5
    output: OrderResult
  route:
    kind: case
    on: ${place.output.type}
    cases:
      - when: filled
        then: on_filled
      - when: interrupted
        then: on_interrupted
  on_filled:
    kind: code
    input:
      qty: ${place.output.qty}
    code: 'return f"filled {inputs[''qty'']}"'
    output: str
  on_interrupted:
    kind: code
    input:
      by: ${place.output.by}
    code: 'return "stopped by " + inputs["by"]'
    output: str
output: ${on_filled.output | on_interrupted.output}
"""


@pytest.mark.parametrize("returned, expected", [
    ({"type": "interrupted", "by": "owner", "partial": 2}, "stopped by owner"),
    ({"type": "filled", "qty": 5}, "filled 5"),
])
def test_a_variant_result_routes_to_its_branch(returned, expected):
    result = run_flow(load_flow(_ORDERS_FLOW), {}, tools={"place_order": _returning(returned)})

    assert result.status == "succeeded", result.failure
    assert result.output == expected


def test_an_unknown_variant_fails_at_the_gate():
    result = run_flow(
        load_flow(_ORDERS_FLOW), {}, tools={"place_order": _returning({"type": "rejected"})}
    )

    assert result.status == "failed"
    assert result.failure.node_id == "place"
    assert result.failure.code == "contract.output"
    assert "'rejected' is not a tag of the union" in result.failure.message


def test_an_unexpected_tool_error_fails_the_node():
    def place_order(qty):
        raise RuntimeError("exchange down")

    result = run_flow(load_flow(_INT_FLOW), {}, tools={"place_order": place_order})

    assert result.status == "failed"
    assert result.failure.node_id == "place"
    assert "exchange down" in result.failure.message


def test_a_host_stop_leaves_the_run_resumable(tmp_path: Path):
    path = tmp_path / "run.jsonl"
    flow = load_flow(_INT_FLOW)

    def interrupt(qty):
        raise _HostStop()

    with pytest.raises(_HostStop):
        run_flow(flow, {}, log=path, tools={"place_order": interrupt})

    resumed = resume_flow(flow, log=path, tools={"place_order": _returning(7)})

    assert resumed.status == "succeeded", resumed.failure
    assert resumed.output == 7


# --- Run-start checks against the tool's declared types ---------------------------------

_calls: list[dict] = []


def place_order(qty: int, sym: str) -> str:
    _calls.append({"qty": qty, "sym": sym})
    return f"{qty} {sym}"


def place_order_default_sym(qty: int, sym: str = "AAPL") -> str:
    _calls.append({"qty": qty, "sym": sym})
    return f"{qty} {sym}"


@pytest.fixture(autouse=True)
def _reset_calls():
    _calls.clear()


def _one_tool_flow(input_block: str, *, output: str = "") -> str:
    return f"""
id: placing
name: placing
nodes:
  place:
    kind: tool
    tool_id: place_order
    input:
{input_block}
{output}
output: ${{place.output}}
"""


def test_an_input_the_tool_does_not_take_fails_at_run_start(tmp_path: Path):
    path = tmp_path / "run.jsonl"
    flow = load_flow(_one_tool_flow("      qtty: 5\n      sym: X"))

    with pytest.raises(ValueError, match=re.escape(
        "TOOL node 'place' input 'qtty': tool 'place_order' takes no input 'qtty'; "
        "it takes ['qty', 'sym']"
    )):
        run_flow(flow, {}, log=path, tools={"place_order": place_order})
    assert not path.exists()
    assert _calls == []


def test_a_missing_required_input_fails_at_run_start(tmp_path: Path):
    path = tmp_path / "run.jsonl"
    flow = load_flow(_one_tool_flow("      sym: X"))

    with pytest.raises(ValueError, match=re.escape(
        "TOOL node 'place' is missing input 'qty' that tool 'place_order' requires"
    )):
        run_flow(flow, {}, log=path, tools={"place_order": place_order})
    assert not path.exists()
    assert _calls == []


def test_an_omitted_optional_input_runs():
    flow = load_flow(_one_tool_flow("      qty: 5"))

    result = run_flow(flow, {}, tools={"place_order": place_order_default_sym})

    assert result.status == "succeeded", result.failure
    assert result.output == "5 AAPL"


_UNREACHED_BRANCH_FLOW = """
id: branching
name: branching
nodes:
  route:
    kind: case
    cases:
      - when: 'true'
        then: fine
    else: place
  fine:
    kind: code
    code: 'return "ok"'
    output: str
  place:
    kind: tool
    tool_id: place_order
    input:
      qtty: 5
      sym: X
output: ${fine.output | place.output}
"""

_DEFS_CHILD_FLOW = """
id: nested
name: nested
defs:
  inner:
    nodes:
      place:
        kind: tool
        tool_id: place_order
        input:
          qtty: 5
          sym: X
    output: ${place.output}
nodes:
  a:
    kind: call
    call: inner
output: ${a.output}
"""


@pytest.mark.parametrize(
    ("source", "node_path"),
    [(_UNREACHED_BRANCH_FLOW, "place"), (_DEFS_CHILD_FLOW, "a/place")],
    ids=["unreached-branch", "defs-child"],
)
def test_tool_nodes_that_may_never_run_are_checked(tmp_path: Path, source: str, node_path: str):
    path = tmp_path / "run.jsonl"

    with pytest.raises(ValueError, match=re.escape(
        f"TOOL node {node_path!r} input 'qtty': tool 'place_order' takes no input 'qtty'"
    )):
        run_flow(load_flow(source), {}, log=path, tools={"place_order": place_order})
    assert not path.exists()


def test_an_unknown_tool_id_still_fails_the_node():
    flow = load_flow(_one_tool_flow("      qty: 5"))

    result = run_flow(flow, {}, tools={"other": place_order})

    assert result.status == "failed"
    assert result.failure.node_id == "place"
    assert "unknown tool id" in result.failure.message
