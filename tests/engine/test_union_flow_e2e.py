"""Unions of records end to end: load, route, narrow and replay through `run_flow`."""

import json

import pytest

from yian.compose.loader import load_flow
from yian.run import resume_flow, run_flow

_FLOW = """
id: unions
name: unions
typedefs:
  Buy: {action: 'Literal(buy)', qty: int}
  Sell: {action: 'Literal(sell)', qty: int, price: float}
  Hold: {action: 'Literal(hold)', note: str}
  Decision: Union(Buy, Sell, Hold)
  Text: {text: str}
  Num: {num: int}
  Note: Union(Text, Num)
input:
  decision: !object
  note: !object
defs:
  sell_flow:
    input:
      order: !Sell
    nodes:
      fill:
        kind: code
        input:
          qty: ${input.order.qty}
          price: ${input.order.price}
        code: |
          return inputs["qty"] * inputs["price"]
        output: float
    output: ${fill.output}
nodes:
  d:
    kind: code
    input:
      decision: ${input.decision}
    code: |
      return inputs["decision"]
    output: Decision
  route:
    kind: case
    on: ${d.output.action}
    cases:
      - when: buy
        then: place_buy
      - when: sell
        then: call(sell_flow, order=${d.output})
    else: log_hold
  place_buy:
    kind: code
    input:
      qty: ${d.output.qty}
    code: |
      return "buy " + str(inputs["qty"])
    output: str
  log_hold:
    kind: code
    input:
      note: ${d.output.note}
    code: |
      return "hold: " + inputs["note"]
    output: str
  n:
    kind: code
    input:
      note: ${input.note}
    code: |
      return inputs["note"]
    output: Note
  kind_of:
    kind: case
    cases:
      - when: isinstance(n.output, Text)
        then: shout
    else: double
  shout:
    kind: code
    input:
      text: ${n.output.text}
    code: |
      return inputs["text"].upper()
    output: str
  double:
    kind: code
    depends_on: [n]
    code: |
      return "number"
    output: str
output:
  action: ${d.output.action}
  trade: ${place_buy.output | __call_0.output | log_hold.output}
  note: ${shout.output | double.output}
"""


def _flow():
    return load_flow(_FLOW)


BUY = {"action": "buy", "qty": 3}
SELL = {"action": "sell", "qty": 2, "price": 1.5}
HOLD = {"action": "hold", "note": "wait"}


@pytest.mark.parametrize("decision, action, trade", [
    (BUY, "buy", "buy 3"),
    (SELL, "sell", 3.0),
    (HOLD, "hold", "hold: wait"),
])
def test_each_on_branch_reads_its_members_fields(decision, action, trade):
    result = run_flow(_flow(), {"decision": decision, "note": {"num": 1}})
    assert result.status == "succeeded", result.failure
    assert result.output == {"action": action, "trade": trade, "note": "number"}


@pytest.mark.parametrize("note, expected", [({"text": "hi"}, "HI"), ({"num": 4}, "number")])
def test_an_isinstance_case_routes_an_untagged_union(note, expected):
    result = run_flow(_flow(), {"decision": BUY, "note": note})
    assert result.status == "succeeded", result.failure
    assert result.output["note"] == expected


@pytest.mark.parametrize("decision, message", [
    ({"action": "wait"}, "'wait' is not a tag of the union (expected one of ['buy', 'hold', 'sell'])"),
    ({"qty": 2}, "missing tag field 'action' (expected one of ['buy', 'hold', 'sell'])"),
    ({"action": "sell", "qty": 2}, "record missing required fields: ['price']"),
])
def test_a_code_node_returning_an_invalid_tagged_member_fails_the_node(decision, message):
    result = run_flow(_flow(), {"decision": decision, "note": {"num": 1}})
    assert result.status == "failed"
    assert result.failure.node_id == "d"
    assert result.failure.code == "contract.output"
    assert result.failure.message == message


def test_a_code_node_returning_no_untagged_member_fails_the_node():
    result = run_flow(_flow(), {"decision": BUY, "note": {"zz": 1}})
    assert result.status == "failed"
    assert result.failure.node_id == "n"
    assert result.failure.code == "contract.output"
    assert result.failure.message.startswith("value matches no member of Union(")
    assert "record missing required fields: ['num']" in result.failure.message
    assert "record missing required fields: ['text']" in result.failure.message


_NESTED_FLOW = """
id: nested
name: nested
typedefs:
  Buy: {action: 'Literal(buy)', qty: int}
  Sell: {action: 'Literal(sell)', qty: int, price: float}
  Decision: Union(Buy, Sell)
  Text: {text: str}
  Num: {num: int}
  Note: Union(Text, Num)
input:
  value: !object
nodes:
  d:
    kind: code
    input:
      value: ${input.value}
    code: |
      return inputs["value"]
    output:
      decision: Decision
      notes: list(Note)
output: ${d.output}
"""


@pytest.mark.parametrize("value, message, field", [
    (
        {"decision": {"action": "bid", "qty": 1}, "notes": []},
        "'bid' is not a tag of the union (expected one of ['buy', 'sell'])",
        "decision",
    ),
    (
        {"decision": {"action": "sell", "qty": "x", "price": 1.0}, "notes": []},
        "'x' does not match declared type int",
        "decision",
    ),
    (
        {"decision": BUY, "notes": [{"text": "a"}, {"zz": 1}]},
        "value matches no member of Union(",
        "notes",
    ),
])
def test_a_nested_union_failure_points_at_the_output_field(value, message, field):
    flow = load_flow(_NESTED_FLOW)
    result = run_flow(flow, {"value": value})
    assert result.status == "failed"
    assert result.failure.code == "contract.output"
    assert result.failure.message.startswith(message)
    (origin,) = result.failure.trace
    assert origin.key == f"/s:root/s:node/s:d/s:output/s:contract/s:path/s:{field}"
    (frame,) = flow.diagnose(result.failure).frames
    authored = frame.text.splitlines()[frame.range.start_line - 1]
    assert authored.strip().startswith(f"{field}:")


def _cut_after_d(log):
    """Keep the log up to `d`'s success record; return the lines and that record's index."""
    lines = log.read_text().splitlines(keepends=True)
    end = next(
        index for index, line in enumerate(lines)
        if json.loads(line)["kind"] == "node_succeeded"
        and json.loads(line)["payload"]["node_id"] == "d"
    )
    log.write_text("".join(lines[: end + 1]))
    return lines[: end + 1], end


def test_replaying_a_finished_run_reproduces_the_union_branch(tmp_path):
    log = tmp_path / "run.jsonl"
    first = run_flow(_flow(), {"decision": SELL, "note": {"text": "hi"}}, log=log)
    assert first.status == "succeeded", first.failure
    replayed = resume_flow(_flow(), log=log)
    assert replayed.status == "succeeded", replayed.failure
    assert replayed.output == first.output == {"action": "sell", "trade": 3.0, "note": "HI"}


def test_resume_after_the_union_value_continues_down_its_branch(tmp_path):
    log = tmp_path / "run.jsonl"
    first = run_flow(_flow(), {"decision": SELL, "note": {"num": 1}}, log=log)
    assert first.status == "succeeded", first.failure
    _cut_after_d(log)
    resumed = resume_flow(_flow(), log=log)
    assert resumed.status == "succeeded", resumed.failure
    assert resumed.output == first.output


@pytest.mark.parametrize("forged, message", [
    ({"action": "wait", "qty": 2}, "'wait' is not a tag of the union"),
    ({"action": "sell", "qty": 2}, r"record missing required fields: \['price'\]"),
])
def test_resume_rejects_a_recorded_value_that_is_not_a_member(tmp_path, forged, message):
    log = tmp_path / "run.jsonl"
    first = run_flow(_flow(), {"decision": SELL, "note": {"num": 1}}, log=log)
    assert first.status == "succeeded", first.failure
    lines, end = _cut_after_d(log)
    record = json.loads(lines[end])
    record["payload"]["value"]["value"] = forged
    lines[end] = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
    log.write_text("".join(lines))
    with pytest.raises(ValueError, match=message):
        resume_flow(_flow(), log=log)
