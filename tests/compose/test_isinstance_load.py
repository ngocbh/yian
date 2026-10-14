"""`isinstance(x, T)` type names resolve at load into each node's type table."""

import json

import pytest

from yian.compose import LoadError, load_flow
from yian.compose.helpers.errors import LoadDiagnostics
from yian.compose.helpers.isinstance_types import collect_isinstance_names
from yian.compile.model import END_ID, START_ID
from yian.run import resume_flow, run_flow
from yian.typesys.types import read_typedefs, type_for

_TYPEDEFS_RAW = {
    "Buy": {"action": "Literal[buy]", "qty": "int"},
    "Sell": {"action": "Literal[sell]", "price": "float"},
    "Order": "Union[Buy, Sell]",
    "Running": "Literal[running]",
}
_REGISTRY = read_typedefs(_TYPEDEFS_RAW)
BUY = type_for("Buy", _REGISTRY)
SELL = type_for("Sell", _REGISTRY)
ORDER = type_for("Order", _REGISTRY)
RUNNING = type_for("Running", _REGISTRY)

TYPEDEFS = """typedefs:
  Buy: {action: 'Literal[buy]', qty: int}
  Sell: {action: 'Literal[sell]', price: float}
  Order: Union[Buy, Sell]
  Running: Literal[running]
"""

A_BUY = {"action": "buy", "qty": 3}
A_SELL = {"action": "sell", "price": 1.5}

# One CODE node `d` that returns the `order` input typed as `Order`.
_HEAD = (
    "id: p\nname: p\n"
    + TYPEDEFS
    + """input:
  order: object
nodes:
  d:
    kind: code
    input:
      order: ${input.order}
    output: Order
    code: |
      return inputs["order"]
"""
)


def _flow(rest: str) -> str:
    return _HEAD + rest


def _line_of(text: str, marker: str) -> int:
    """The 1-based line of the first line containing `marker`."""
    for number, line in enumerate(text.splitlines(), start=1):
        if marker in line:
            return number
    raise AssertionError(f"{marker!r} not in text")


def _load_error(text: str, **kwargs) -> LoadError:
    with pytest.raises(LoadError) as caught:
        load_flow(text, **kwargs)
    return caught.value


def _error_line(error: LoadError) -> int:
    return error.diagnostic.frames[-1].range.start_line


# --- collecting names --- #


@pytest.mark.parametrize(
    ("source", "condition", "names"),
    [
        ("${isinstance(d.output, Buy)}", False, ["Buy"]),
        ("a ${isinstance(x, Buy) and isinstance(y, int)} b ${isinstance(z, Buy)}", False,
         ["Buy", "int"]),
        ("isinstance(x, Buy)", False, []),
        ("isinstance(x, Buy) or not isinstance(x, Sell)", True, ["Buy", "Sell"]),
        ("${isinstance(x, Order)}", True, ["Order"]),
        ("plain text", False, []),
        (3, False, []),
    ],
)
def test_collect_isinstance_names(source, condition, names):
    assert collect_isinstance_names(source, condition=condition) == names


# --- resolution and evaluation --- #


_INPUT_FLOW = _flow(
    """  check:
    kind: code
    input:
      ok: ${isinstance(d.output, Buy)}
      known: ${isinstance(input.order, object)}
    code: |
      return [inputs["ok"], inputs["known"]]
output: ${check.output}
"""
)


@pytest.mark.parametrize(("order", "expected"), [(A_BUY, [True, True]), (A_SELL, [False, True])])
def test_isinstance_in_a_node_input(order, expected):
    flow = load_flow(_INPUT_FLOW)
    assert flow.compiled.nodes["check"].types == {"Buy": BUY, "object": type_for("object", {})}
    assert flow.compiled.nodes["d"].types == {}
    result = run_flow(flow, {"order": order})
    assert result.status == "succeeded", result.failure
    assert result.output == expected


_CASE_FLOW = _flow(
    """  route:
    kind: case
    cases:
      - when: isinstance(d.output, Buy)
        then: b
      - when: isinstance(d.output, Order)
        then: o
    else: n
  b:
    kind: code
    code: |
      return "buy"
  o:
    kind: code
    code: |
      return "order"
  n:
    kind: code
    code: |
      return "none"
output: ${b.output | o.output | n.output}
"""
)


@pytest.mark.parametrize(("order", "expected"), [(A_BUY, "buy"), (A_SELL, "order")])
def test_isinstance_in_a_searched_case_is_first_true_wins(order, expected):
    flow = load_flow(_CASE_FLOW)
    assert flow.compiled.nodes["route"].types == {"Buy": BUY, "Order": ORDER}
    result = run_flow(flow, {"order": order})
    assert result.status == "succeeded", result.failure
    assert result.output == expected


_NODE_ASSERT_FLOW = (
    "id: p\nname: p\n"
    + TYPEDEFS
    + """input:
  order: object
nodes:
  step:
    kind: code
    input:
      order: ${input.order}
    code: |
      return inputs["order"]
    asserts:
      - isinstance(order, Order)
      - isinstance(output, Buy)
output: ${step.output}
"""
)


@pytest.mark.parametrize(("order", "status"), [(A_BUY, "succeeded"), (A_SELL, "failed")])
def test_isinstance_in_a_node_assert(order, status):
    flow = load_flow(_NODE_ASSERT_FLOW)
    assert flow.compiled.nodes["step"].types == {"Buy": BUY, "Order": ORDER}
    result = run_flow(flow, {"order": order})
    assert result.status == status, result.failure
    if status == "failed":
        assert result.failure.code == "assertion.failed"


_FLOW_ASSERT_FLOW = _flow(
    """output: ${d.output}
asserts:
  - isinstance(input.order, Order)
  - isinstance(d.output, Buy)
"""
)


def test_isinstance_in_flow_asserts_fills_start_and_end():
    flow = load_flow(_FLOW_ASSERT_FLOW)
    assert flow.compiled.nodes[START_ID].types == {"Order": ORDER}
    assert flow.compiled.nodes[END_ID].types == {"Buy": BUY}
    assert run_flow(flow, {"order": A_BUY}).status == "succeeded"
    failed = run_flow(flow, {"order": A_SELL})
    assert failed.status == "failed"
    assert failed.failure.code == "assertion.failed"


def test_a_failing_flow_input_assert_fails_at_start():
    failed = run_flow(load_flow(_FLOW_ASSERT_FLOW), {"order": {"action": "hold"}})
    assert failed.status == "failed"
    assert failed.failure.node_id == START_ID


_FLOW_OUTPUT_FLOW = _flow(
    """output:
  order: ${d.output}
  is_buy: ${isinstance(d.output, Buy)}
"""
)


@pytest.mark.parametrize(("order", "is_buy"), [(A_BUY, True), (A_SELL, False)])
def test_isinstance_in_a_flow_output(order, is_buy):
    flow = load_flow(_FLOW_OUTPUT_FLOW)
    assert flow.compiled.nodes[END_ID].types == {"Buy": BUY}
    result = run_flow(flow, {"order": order})
    assert result.status == "succeeded", result.failure
    assert result.output == {"order": order, "is_buy": is_buy}


_LOOP_FLOW = (
    "id: counter\nname: counter\n"
    + TYPEDEFS
    + """defs:
  bump:
    input:
      n: int
      status: str
    nodes:
      step:
        kind: code
        input:
          n: ${input.n}
          status: ${input.status}
        code: |
          n = inputs["n"] + 1
          return {"n": n, "status": "done" if n >= 4 else "running"}
        output:
          n: int
          status: str
    output: ${step.output}
nodes:
  loop:
    kind: loop
    call: bump
    input:
      n: 0
      status: running
    while: isinstance(status, Running)
    max: 10
output: ${loop.output}
"""
)


def test_isinstance_in_a_loop_while_runs_several_iterations():
    flow = load_flow(_LOOP_FLOW)
    assert flow.compiled.nodes["loop"].types == {"Running": RUNNING}
    result = run_flow(flow, {})
    assert result.status == "succeeded", result.failure
    assert result.output == {"n": 4, "status": "done"}


def test_a_run_that_used_isinstance_replays(tmp_path):
    log = tmp_path / "run.jsonl"
    first = run_flow(load_flow(_LOOP_FLOW), {}, log=log)
    assert first.status == "succeeded", first.failure
    replayed = resume_flow(load_flow(_LOOP_FLOW), log=log)
    assert replayed.status == "succeeded", replayed.failure
    assert replayed.output == first.output
    # Cut after the second iteration, so resume replays two iterations and runs two live.
    kept = _cut_log_after(log, "loop#1/__end__")
    assert "loop~1" in kept
    assert not any(node_id and node_id.startswith(("loop~2", "loop#2/")) for node_id in kept)
    resumed = resume_flow(load_flow(_LOOP_FLOW), log=log)
    assert resumed.status == "succeeded", resumed.failure
    assert resumed.output == first.output


def _cut_log_after(log, node_id: str) -> list[str]:
    """Keep the log up to `node_id`'s success record; return the kept records' node ids."""
    lines = log.read_text().splitlines(keepends=True)
    records = [json.loads(line) for line in lines]
    end = next(
        index for index, record in enumerate(records)
        if record["kind"] == "node_succeeded" and record["payload"]["node_id"] == node_id
    )
    log.write_text("".join(lines[: end + 1]))
    return [record["payload"].get("node_id") for record in records[: end + 1]]


_CHILD_FLOWS = (
    "id: p\nname: p\n"
    + TYPEDEFS
    + """input:
  order: object
defs:
  classify:
    input:
      order: object
    nodes:
      inner:
        kind: code
        input:
          is_sell: ${isinstance(input.order, Sell)}
        code: |
          return inputs["is_sell"]
    output: ${inner.output}
nodes:
  named:
    kind: call
    call: classify
    input:
      order: ${input.order}
  inline:
    kind: call
    input:
      order: ${input.order}
    nodes:
      outer:
        kind: call
        input:
          order: ${input.order}
        nodes:
          inner:
            kind: code
            input:
              is_buy: ${isinstance(input.order, Buy)}
            code: |
              return inputs["is_buy"]
        output: ${inner.output}
    output: ${outer.output}
output:
  sell: ${named.output}
  buy: ${inline.output}
"""
)


@pytest.mark.parametrize(
    ("order", "expected"),
    [(A_BUY, {"sell": False, "buy": True}), (A_SELL, {"sell": True, "buy": False})],
)
def test_isinstance_in_defs_and_nested_inline_children(order, expected):
    flow = load_flow(_CHILD_FLOWS)
    nodes = flow.compiled.nodes
    assert nodes["named"].child.nodes["inner"].types == {"Sell": SELL}
    inline_outer = nodes["inline"].child.nodes["outer"]
    assert inline_outer.child.nodes["inner"].types == {"Buy": BUY}
    assert nodes["inline"].types == {}
    result = run_flow(flow, {"order": order})
    assert result.status == "succeeded", result.failure
    assert result.output == expected


_USES_CHILD = """id: child
name: child
typedefs:
  Ticket: {kind: 'Literal[ticket]', id: int}
input:
  value: object
nodes:
  check:
    kind: code
    input:
      ok: ${isinstance(input.value, Ticket)}
    code: |
      return inputs["ok"]
output: ${check.output}
"""

_USES_PARENT = """id: parent
name: parent
input:
  value: object
uses:
  child: child
nodes:
  run:
    kind: call
    call: child
    input:
      value: ${input.value}
output: ${run.output}
"""


def _uses_resolver():
    loaded = {"child": load_flow(_USES_CHILD)}

    def resolve(flow_id, version=None):
        try:
            return loaded[flow_id]
        except KeyError as exc:
            raise LoadDiagnostics().error(f"unknown child {flow_id!r}") from exc

    return resolve


def test_a_uses_child_resolves_names_against_its_own_typedefs():
    flow = load_flow(_USES_PARENT, child_resolver=_uses_resolver())
    ticket = flow.compiled.nodes["run"].child.nodes["check"].types["Ticket"]
    assert ticket.fields["id"] == type_for("int", {})
    result = run_flow(flow, {"value": {"kind": "ticket", "id": 7}})
    assert result.status == "succeeded", result.failure
    assert result.output is True


def test_a_parent_cannot_use_a_childs_typedef():
    text = _USES_PARENT.replace(
        "      value: ${input.value}\n",
        "      value: ${input.value}\n  probe:\n    kind: code\n    input:\n"
        "      ok: ${isinstance(input.value, Ticket)}\n    code: |\n      return 1\n",
    )
    error = _load_error(text, child_resolver=_uses_resolver())
    assert "unknown type 'Ticket'" in str(error)
    assert _error_line(error) == _line_of(text, "isinstance(input.value, Ticket)")


_FLAG_DEF = """defs:
  flag:
    input:
      is_buy: bool
    nodes:
      e:
        kind: code
        input:
          is_buy: ${input.is_buy}
        code: |
          return inputs["is_buy"]
    output: ${e.output}
"""

_MAP_FLOW = (
    "id: p\nname: p\n"
    + TYPEDEFS
    + _FLAG_DEF
    + """input:
  orders: list[object]
nodes:
  each:
    kind: map
    call: flag
    over: ${input.orders}
    input:
      is_buy: ${isinstance(item, Buy)}
output: ${each.output}
"""
)


def test_isinstance_in_a_map_per_item_input():
    flow = load_flow(_MAP_FLOW)
    assert flow.compiled.nodes["each"].types == {"Buy": BUY}
    result = run_flow(flow, {"orders": [A_BUY, A_SELL, A_BUY]})
    assert result.status == "succeeded", result.failure
    assert result.output == [True, False, True]


_CALL_ARG_FLOW = _flow(
    """  c:
    kind: code
    input:
      res: call(flag, is_buy=${isinstance(d.output, Buy)})
    code: |
      return inputs["res"]
"""
    + _FLAG_DEF
    + "output: ${c.output}\n"
)


@pytest.mark.parametrize(("order", "expected"), [(A_BUY, True), (A_SELL, False)])
def test_isinstance_in_a_call_directive_argument(order, expected):
    flow = load_flow(_CALL_ARG_FLOW)
    (synth,) = [node for nid, node in flow.compiled.nodes.items() if nid.startswith("__call")]
    assert synth.types == {"Buy": BUY}
    assert flow.compiled.nodes["c"].types == {}
    result = run_flow(flow, {"order": order})
    assert result.status == "succeeded", result.failure
    assert result.output is expected


_LOOP_SEED_FLOW = (
    "id: p\nname: p\n"
    + TYPEDEFS
    + """input:
  order: object
defs:
  bump:
    input:
      n: int
      buy: bool
    nodes:
      step:
        kind: code
        input:
          n: ${input.n}
          buy: ${input.buy}
        code: |
          return {"n": inputs["n"] + 1, "buy": inputs["buy"]}
        output:
          n: int
          buy: bool
    output: ${step.output}
nodes:
  loop:
    kind: loop
    call: bump
    input:
      n: 0
      buy: ${isinstance(input.order, Buy)}
    until: n >= 3 and isinstance(buy, bool)
    max: 10
output: ${loop.output}
"""
)


@pytest.mark.parametrize(("order", "buy"), [(A_BUY, True), (A_SELL, False)])
def test_isinstance_in_a_loop_seed_and_until(order, buy):
    flow = load_flow(_LOOP_SEED_FLOW)
    assert flow.compiled.nodes["loop"].types == {"Buy": BUY, "bool": type_for("bool", {})}
    result = run_flow(flow, {"order": order})
    assert result.status == "succeeded", result.failure
    assert result.output == {"n": 3, "buy": buy}


_ON_FLOW = _flow(
    """  route:
    kind: case
    on: ${isinstance(d.output, Buy)}
    cases:
      - when: true
        then: b
    else: o
  b:
    kind: code
    code: |
      return "buy"
  o:
    kind: code
    code: |
      return "other"
output: ${b.output | o.output}
"""
)


def test_isinstance_in_a_case_on():
    # The bool `on:` value matches `when: true`, so a buy routes to `b`; the run evaluates
    # the `on:` binding with the table.
    flow = load_flow(_ON_FLOW)
    assert flow.compiled.nodes["route"].types == {"Buy": BUY, "bool": type_for("bool", {})}
    result = run_flow(flow, {"order": A_BUY})
    assert result.status == "succeeded", result.failure
    assert result.output == "buy"
    flow.compiled.nodes["route"].types = {}
    failed = run_flow(flow, {"order": A_BUY})
    assert failed.status == "failed"
    assert failed.failure.node_id == "route"


def test_an_inline_body_cannot_take_a_whole_value_isinstance_argument():
    # Inline bodies infer each argument's type while the parent builds, and a whole-value
    # expression has no static type there; a named `defs:` body declares it instead.
    text = _flow(
        """  c:
    kind: call
    input:
      is_buy: ${isinstance(d.output, Buy)}
    nodes:
      e:
        kind: code
        input:
          is_buy: ${input.is_buy}
        code: |
          return inputs["is_buy"]
    output: ${e.output}
output: ${c.output}
"""
    )
    error = _load_error(text)
    assert "cannot infer type for inline binding 'c.input.is_buy'" in str(error)
    assert _error_line(error) == _line_of(text, "is_buy: ${isinstance")


# --- located load errors --- #


_UNKNOWN_SITES = {
    "node input": (_INPUT_FLOW, "d.output, Buy", "Buyy", "ok:"),
    "case when": (_CASE_FLOW, "d.output, Order", "Ordr", "Ordr"),
    "node assert": (_NODE_ASSERT_FLOW, "output, Buy", "Bu", "isinstance(output, Bu)"),
    "flow assert": (_FLOW_ASSERT_FLOW, "d.output, Buy", "Byu", "Byu"),
    "flow output": (_FLOW_OUTPUT_FLOW, "d.output, Buy", "Bye", "is_buy:"),
    "loop while": (_LOOP_FLOW, "status, Running", "Runing", "while:"),
    "defs child": (_CHILD_FLOWS, "input.order, Sell", "Sel", "is_sell:"),
    "inline child": (_CHILD_FLOWS, "input.order, Buy", "By", "is_buy:"),
    "map per-item input": (_MAP_FLOW, "item, Buy", "Buyy", "is_buy: ${isinstance(item"),
    "call argument": (_CALL_ARG_FLOW, "d.output, Buy", "Buyy", "res: call"),
    "loop seed": (_LOOP_SEED_FLOW, "input.order, Buy", "Buyy", "buy: ${isinstance"),
    "loop until": (_LOOP_SEED_FLOW, "buy, bool", "boo", "until:"),
    "case on": (_ON_FLOW, "d.output, Buy", "Buyy", "on: ${isinstance"),
}


@pytest.mark.parametrize("site", sorted(_UNKNOWN_SITES))
def test_an_unknown_type_name_is_a_located_load_error(site):
    flow, written, name, marker = _UNKNOWN_SITES[site]
    text = flow.replace(written, f"{written.split(', ')[0]}, {name}")
    assert text != flow
    error = _load_error(text)
    assert f"unknown type {name!r}" in str(error)
    assert "typedefs" in str(error)
    assert ":: " not in str(error)
    assert _error_line(error) == _line_of(text, marker)


def test_a_flow_assert_error_is_located_by_its_own_assert():
    text = _flow(
        """output: ${d.output}
asserts:
  - isinstance(d.output, Buy)
  - isinstance(d.output, Buy) or isinstance(d.output, Nope)
"""
    )
    error = _load_error(text)
    assert "unknown type 'Nope'" in str(error)
    assert _error_line(error) == _line_of(text, "Nope")


def test_list_is_not_an_isinstance_type_name():
    text = _INPUT_FLOW.replace("d.output, Buy", "d.output, list")
    assert "unknown type 'list'" in str(_load_error(text))


def test_isinstance_in_wait_until_is_a_located_load_error():
    text = """id: w
name: w
input:
  when: date
nodes:
  w:
    kind: wait
    until: ${isinstance(input.when, date)}
output: ${w.output}
"""
    error = _load_error(text)
    assert "isinstance is not supported in a wait node's until:" in str(error)
    assert _error_line(error) == _line_of(text, "until:")


def test_isinstance_in_map_over_is_a_located_load_error():
    text = (
        "id: m\nname: m\n"
        + TYPEDEFS
        + """input:
  items: list[int]
defs:
  one:
    input:
      x: int
    nodes:
      e:
        kind: code
        input:
          x: ${input.x}
        code: |
          return inputs["x"]
    output: ${e.output}
nodes:
  each:
    kind: map
    call: one
    over: ${isinstance(input.items, Buy)}
    input:
      x: ${item}
output: ${each.output}
"""
    )
    error = _load_error(text)
    assert "isinstance is not supported in a map node's over:" in str(error)
    assert _error_line(error) == _line_of(text, "over:")


def test_isinstance_in_a_call_nodes_asserts_is_a_located_load_error():
    text = _CHILD_FLOWS.replace(
        "    call: classify\n",
        "    call: classify\n    asserts:\n      - isinstance(output, bool)\n",
    )
    error = _load_error(text)
    assert (
        "isinstance in a call node's asserts is not supported; assert inside the child flow"
        in str(error)
    )
    assert _error_line(error) == _line_of(text, "isinstance(output, bool)")


def test_isinstance_in_a_loop_nodes_asserts_is_a_load_error():
    text = _LOOP_FLOW.replace(
        "    max: 10\n", "    max: 10\n    asserts:\n      - isinstance(output, Buy)\n"
    )
    assert "isinstance in a loop node's asserts is not supported" in str(_load_error(text))


def test_isinstance_in_an_agent_prompt_is_a_located_load_error():
    text = (
        "id: a\nname: a\n"
        + TYPEDEFS
        + """input:
  order: object
nodes:
  ask:
    kind: agent
    input:
      order: ${input.order}
    prompt: "Is it a buy? ${isinstance(order, Buy)}"
    output: str
output: ${ask.output}
"""
    )
    error = _load_error(text)
    assert "isinstance is not a prompt function; compute it in an input binding" in str(error)
    assert _error_line(error) == _line_of(text, "prompt:")


def test_isinstance_in_a_human_input_question_is_a_located_load_error():
    text = (
        "id: h\nname: h\n"
        + TYPEDEFS
        + """input:
  order: object
nodes:
  ask:
    kind: human_input
    input:
      order: ${input.order}
    questions: '${isinstance(order, Buy)}'
output: ${ask.output}
"""
    )
    error = _load_error(text)
    assert "exactly one `${name}`" in str(error)
    assert _error_line(error) == _line_of(text, "questions:")
