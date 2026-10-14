"""A TOOL node's optional `output:` contract is checked against every value the tool returns."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal, Optional, Union

import pytest
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict
from typing_extensions import TypedDict

from yian.compose import load_flow
from yian.nodes.base import _HostStop
from yian.run import resume_flow, run_flow
from yian.tools import TOOL_REGISTRY, ToolSet
from yian.tools.types import tool_types


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
  qty: !int
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
  Filled: {type: 'Literal(filled)', qty: int}
  Interrupted: {type: 'Literal(interrupted)', by: str, partial: 'Optional(int)'}
  OrderResult: Union(Filled, Interrupted)
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


def _one_tool_flow(input_block: str) -> str:
    return f"""
id: placing
name: placing
nodes:
  place:
    kind: tool
    tool_id: place_order
    input:
{input_block}
output: ${{place.output}}
"""


def test_an_input_the_tool_does_not_take_fails_at_run_start(tmp_path: Path):
    path = tmp_path / "run.jsonl"
    flow = load_flow(_one_tool_flow("      qtty: 5\n      sym: X"))

    with pytest.raises(ValueError, match=re.escape(
        "TOOL node 'place' input 'qtty': tool 'place_order' takes no such input; "
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


def place_three(qty: int, sym: str, side: str) -> str:
    return f"{side} {qty} {sym}"


def test_every_missing_required_input_is_named(tmp_path: Path):
    flow = load_flow(_one_tool_flow("      qty: 5"))

    with pytest.raises(ValueError, match=re.escape(
        "TOOL node 'place' is missing inputs ['side', 'sym'] that tool 'place_order' requires"
    )):
        run_flow(flow, {}, tools={"place_order": place_three})


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

_MAP_BODY_FLOW = """
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
          qtty: ${input.x}
          sym: X
    output: ${place.output}
output: ${each.output}
"""

_LOOP_BODY_FLOW = """
id: looping
name: looping
defs:
  turn:
    input:
      count: !int
      last: !str
    nodes:
      place:
        kind: tool
        tool_id: place_order
        input:
          qtty: ${input.count}
          sym: X
    output:
      count: ${input.count + 1}
      last: ${place.output}
nodes:
  again:
    kind: loop
    call: turn
    input:
      count: 0
      last: ""
    while: count < 1
    max: 3
output: ${again.output}
"""


@pytest.mark.parametrize(
    ("source", "node_path"),
    [
        (_UNREACHED_BRANCH_FLOW, "place"),
        (_DEFS_CHILD_FLOW, "a/place"),
        (_MAP_BODY_FLOW, "each/place"),
        (_LOOP_BODY_FLOW, "again/place"),
    ],
    ids=["unreached-branch", "defs-child", "map-body", "loop-body"],
)
def test_tool_nodes_that_may_never_run_are_checked(tmp_path: Path, source: str, node_path: str):
    path = tmp_path / "run.jsonl"

    with pytest.raises(ValueError, match=re.escape(
        f"TOOL node {node_path!r} input 'qtty': tool 'place_order' takes no such input"
    )):
        run_flow(load_flow(source), {}, log=path, tools={"place_order": place_order})
    assert not path.exists()


def test_an_unknown_tool_id_still_fails_the_node():
    flow = load_flow(_one_tool_flow("      qty: 5"))

    result = run_flow(flow, {}, tools={"other": place_order})

    assert result.status == "failed"
    assert result.failure.node_id == "place"
    assert "unknown tool id" in result.failure.message


def place_int(qty: int) -> str:
    _calls.append({"qty": qty})
    return f"placed {qty}"


def place_float(qty: float) -> str:
    _calls.append({"qty": qty})
    return f"placed {qty}"


def place_str(qty: str) -> str:
    _calls.append({"qty": qty})
    return f"placed {qty}"


class Order(BaseModel):
    qty: int


class StrictOrder(BaseModel):
    model_config = ConfigDict(extra="forbid")

    qty: int


def place_model(order: Order) -> str:
    return f"placed {order.qty}"


def place_strict_model(order: StrictOrder) -> str:
    return f"placed {order.qty}"


class _JsonSchemaTool(BaseTool):
    """A tool whose argument schema is a JSON Schema dict."""

    name: str = "place_order"
    description: str = "Place one order."
    args_schema: dict = {
        "type": "object",
        "properties": {"qty": {"type": "integer"}},
        "required": ["qty"],
        "additionalProperties": False,
    }

    def _run(self, qty: int) -> str:
        return f"placed {qty}"


def _typed_flow(binding: str, *, flow_inputs: str = "", extra_nodes: str = "") -> str:
    """Return a flow whose TOOL node `place` binds `qty` as `binding`."""
    return f"""
id: placing
name: placing
{flow_inputs}
nodes:
{extra_nodes}
  place:
    kind: tool
    tool_id: place_order
    input:
      qty: {binding}
output: ${{place.output}}
"""


def test_a_disjoint_input_type_fails_at_run_start(tmp_path: Path):
    path = tmp_path / "run.jsonl"
    flow = load_flow(_typed_flow("${input.s}", flow_inputs="input:\n  s: !str"))

    with pytest.raises(ValueError, match=re.escape(
        "TOOL node 'place' input 'qty': tool 'place_order' takes int, but the binding is str"
    )):
        run_flow(flow, {"s": "10"}, log=path, tools={"place_order": place_int})
    assert not path.exists()
    assert _calls == []


def test_a_string_literal_into_an_int_fails_at_run_start():
    flow = load_flow(_typed_flow('"10"'))

    with pytest.raises(ValueError, match=re.escape(
        "TOOL node 'place' input 'qty': tool 'place_order' takes int, "
        'but the binding is "10" (str)'
    )):
        run_flow(flow, {}, tools={"place_order": place_int})
    assert _calls == []


def test_a_long_string_literal_is_truncated_in_the_message():
    flow = load_flow(_typed_flow('"' + "x" * 60 + '"'))

    with pytest.raises(ValueError) as excinfo:
        run_flow(flow, {}, tools={"place_order": place_int})
    message = str(excinfo.value)
    assert "x" * 60 not in message
    assert message.endswith('..." (str)')


def test_an_int_literal_into_a_str_fails_at_run_start_and_shows_its_value():
    with pytest.raises(ValueError, match=r"input 'qty'.*takes str, but the binding is 5 \(int\)$"):
        run_flow(load_flow(_typed_flow("5")), {}, tools={"place_order": place_str})


def test_an_int_literal_into_an_int_runs():
    result = run_flow(load_flow(_typed_flow("10")), {}, tools={"place_order": place_int})

    assert result.status == "succeeded", result.failure
    assert result.output == "placed 10"


def test_an_optional_input_starts_and_the_tool_validates_none():
    flow = load_flow(_typed_flow("${input.n}", flow_inputs="input:\n  n: !Optional(int)"))

    missing = run_flow(flow, {"n": None}, tools={"place_order": place_int})
    present = run_flow(flow, {"n": 3}, tools={"place_order": place_int})

    assert missing.status == "failed"
    assert missing.failure.node_id == "place"
    assert present.status == "succeeded", present.failure
    assert present.output == "placed 3"


def test_an_int_into_a_float_runs():
    flow = load_flow(_typed_flow("${input.n}", flow_inputs="input:\n  n: !int"))

    result = run_flow(flow, {"n": 2}, tools={"place_order": place_float})

    assert result.status == "succeeded", result.failure


def test_a_typed_binding_over_an_unproven_source_is_checked_by_its_declared_type():
    code = """
  c:
    kind: code
    code: 'return 1'
"""
    flow = load_flow(_typed_flow("!int ${c.output}", extra_nodes=code))

    with pytest.raises(ValueError, match=r"input 'qty'.*takes str, but the binding is int$"):
        run_flow(flow, {}, tools={"place_order": place_str})


_RECORD_FLOW = """
id: placing
name: placing
typedefs:
  Order: {qty: int, note: str}
input:
  order: !Order
nodes:
  place:
    kind: tool
    tool_id: place_order
    input:
      order: ${input.order}
output: ${place.output}
"""


def test_a_record_with_an_extra_field_into_a_default_model_runs():
    result = run_flow(
        load_flow(_RECORD_FLOW), {"order": {"qty": 2, "note": "n"}},
        tools={"place_order": place_model},
    )

    assert result.status == "succeeded", result.failure
    assert result.output == "placed 2"


def test_a_record_with_an_extra_field_into_a_closed_model_fails_at_run_start():
    with pytest.raises(ValueError, match=re.escape(
        'takes {"qty":int}, but the binding is {"note":str,"qty":int}'
    )):
        run_flow(
            load_flow(_RECORD_FLOW), {"order": {"qty": 2, "note": "n"}},
            tools={"place_order": place_strict_model},
        )


def test_a_json_schema_tool_with_a_disjoint_binding_fails_at_run_start():
    flow = load_flow(_typed_flow("${input.s}", flow_inputs="input:\n  s: !str"))

    with pytest.raises(ValueError, match=r"input 'qty'.*takes int, but the binding is str$"):
        run_flow(flow, {"s": "x"}, tools={"place_order": _JsonSchemaTool()})


_ASK_FLOW = """
id: asking
name: asking
nodes:
  ask:
    kind: tool
    tool_id: ask_user
    input:
      question: Proceed?
      options: "yes, no"
output: ${ask.output}
"""


def test_ask_user_with_string_options_runs():
    def handler(**request):
        return f"answered {request['question']}"

    result = run_flow(load_flow(_ASK_FLOW), {}, tools={"ask_user": handler})

    assert result.status == "succeeded", result.failure
    assert result.output == "answered Proceed?"


def _declared_output(fn):
    """Return the return type the run reads from `fn` as a tool."""
    return tool_types(ToolSet({"place_order": fn}).resolve(["place_order"])[0]).output


def test_a_return_type_disjoint_from_the_declared_output_fails_at_run_start(tmp_path: Path):
    path = tmp_path / "run.jsonl"
    assert _declared_output(place_int) is not None

    with pytest.raises(ValueError, match=re.escape(
        "TOOL node 'place' output: declared int, but tool 'place_order' returns str"
    )):
        run_flow(load_flow(_INT_FLOW), {}, log=path, tools={"place_order": place_int})
    assert not path.exists()
    assert _calls == []


class Filled(TypedDict):
    type: Literal["filled"]
    qty: int


class Interrupted(TypedDict):
    type: Literal["interrupted"]
    by: str
    partial: Optional[int]


class Rejected(TypedDict):
    type: Literal["rejected"]
    reason: str


class Cancelled(TypedDict):
    type: Literal["cancelled"]


def place_variant(qty: int) -> Union[Filled, Interrupted]:
    return {"type": "filled", "qty": qty}


def place_refused(qty: int) -> Union[Rejected, Cancelled]:
    return {"type": "cancelled"}


def test_a_union_return_matching_the_declared_union_runs():
    assert _declared_output(place_variant) is not None

    result = run_flow(load_flow(_ORDERS_FLOW), {}, tools={"place_order": place_variant})

    assert result.status == "succeeded", result.failure
    assert result.output == "filled 5"


def test_a_union_return_with_no_shared_tag_fails_at_run_start():
    assert _declared_output(place_refused) is not None

    with pytest.raises(ValueError, match=(
        r"output: declared Union\(.*\), but tool 'place_order' returns "
        r'Union\(\{"type":Literal\("cancelled"\)\},\{"reason":str,"type":Literal\("rejected"\)\}\)$'
    )):
        run_flow(load_flow(_ORDERS_FLOW), {}, tools={"place_order": place_refused})


_RECORD_OUTPUT_FLOW = """
id: placing
name: placing
nodes:
  place:
    kind: tool
    tool_id: place_order
    input:
      qty: 5
    output: {qty: int}
output: ${place.output}
"""


def place_dict(qty: int) -> dict:
    return {"qty": qty}


def place_bad_dict(qty: int) -> dict:
    return {"qty": "bad"}


def test_a_wider_return_type_runs_and_the_gate_still_checks_each_value():
    assert _declared_output(place_dict) is not None
    flow = load_flow(_RECORD_OUTPUT_FLOW)

    ok = run_flow(flow, {}, tools={"place_order": place_dict})
    bad = run_flow(flow, {}, tools={"place_order": place_bad_dict})

    assert ok.status == "succeeded", ok.failure
    assert ok.output == {"qty": 5}
    assert bad.status == "failed"
    assert bad.failure.code == "contract.output"


def place_unannotated(qty: int):
    return "x"


class _QtyArgs(BaseModel):
    qty: int


class _NoFuncTool(BaseTool):
    """A tool that defines `_run` instead of wrapping a function."""

    name: str = "place_order"
    description: str = "Place one order."
    args_schema: type[BaseModel] = _QtyArgs

    def _run(self, qty: int) -> str:
        return "x"


def place_model_result(qty: int) -> Order:
    return Order(qty=qty)


@pytest.mark.parametrize(
    "tool",
    [place_unannotated, _NoFuncTool(), place_model_result],
    ids=["unannotated", "no-func", "model-return"],
)
def test_an_unread_return_type_skips_the_check_and_the_gate_still_applies(tool):
    assert _declared_output(tool) is None

    result = run_flow(load_flow(_INT_FLOW), {}, tools={"place_order": tool})

    assert result.status == "failed"
    assert result.failure.node_id == "place"
    assert result.failure.code == "contract.output"


def test_a_node_without_output_runs_any_return_type():
    assert _declared_output(place_int) is not None
    flow = load_flow(_INT_FLOW.replace("    output: int\n", ""))

    result = run_flow(flow, {}, tools={"place_order": place_int})

    assert result.status == "succeeded", result.failure
    assert result.output == "placed 5"


_TWO_STEP_FLOW = """
id: two_step
name: two_step
nodes:
  first:
    kind: tool
    tool_id: first_step
  place:
    kind: tool
    tool_id: place_order
    depends_on: [first]
    input:
      qty: 5
output: ${place.output}
"""


def interrupt_first() -> str:
    raise _HostStop()


def first_step() -> str:
    return "ready"


def test_resume_checks_tool_types_before_appending(tmp_path: Path):
    path = tmp_path / "run.jsonl"
    flow = load_flow(_TWO_STEP_FLOW)
    with pytest.raises(_HostStop):
        run_flow(flow, {}, log=path, tools={"first_step": interrupt_first, "place_order": place_int})
    lines = path.read_text().splitlines()

    with pytest.raises(ValueError, match=r"input 'qty'.*takes str, but the binding is 5 \(int\)$"):
        resume_flow(flow, log=path, tools={"first_step": first_step, "place_order": place_str})
    assert path.read_text().splitlines() == lines

    resumed = resume_flow(flow, log=path, tools={"first_step": first_step, "place_order": place_int})
    assert resumed.status == "succeeded", resumed.failure
    assert resumed.output == "placed 5"


def test_a_finished_run_resumes_without_the_tool_type_check(tmp_path: Path):
    path = tmp_path / "run.jsonl"
    flow = load_flow(_TWO_STEP_FLOW)
    finished = run_flow(flow, {}, log=path, tools={"first_step": first_step, "place_order": place_int})
    assert finished.status == "succeeded", finished.failure

    replayed = resume_flow(flow, log=path, tools={"first_step": first_step, "place_order": place_str})

    assert replayed.status == "succeeded", replayed.failure
    assert replayed.output == "placed 5"
