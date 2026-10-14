"""AGENT structured output for a union of records: generation, error paths and a run."""

import json

import pytest
from pydantic import ValidationError

import yian.llm_clients as llm_clients_mod
from yian.compose.loader import load_flow
from yian.nodes.agent.structured import _structured_validation_path, generate_structured
from yian.run import run_flow
from yian.typesys.types import read_typedefs, type_for


NATIVE = {"provider": "openai", "model": "gpt-4o"}
FALLBACK = {"provider": "vllm", "model": "no-structured-sentinel"}

REGISTRY = read_typedefs({
    "Buy": {"action": "Literal(buy)", "qty": "int"},
    "Sell": {"action": "Literal(sell)", "price": "float"},
    "Hold": {"action": "Literal(hold)"},
    "Text": {"text": "str"},
    "Call": {"name": "str", "input": "object"},
    "Order": {"trade": "Union(Buy, Sell)", "note": "str"},
    "Ticket": {"kind": "Literal(ticket)", "trade": "Optional(Union(Buy, Sell))"},
    "Skip": {"kind": "Literal(skip)"},
    "Limit": {"px": "float"},
    "Bid": {"action": "Literal(bid)", "limit": "Limit"},
})
TRADE = type_for("Union(Buy, Sell)", REGISTRY)


class _Replies:
    """A fake model that answers each call with the next payload, natively or as text."""

    def __init__(self, *payloads):
        self.payloads = list(payloads)
        self.calls = 0

    def _next(self):
        payload = self.payloads[min(self.calls, len(self.payloads) - 1)]
        self.calls += 1
        return payload

    def with_structured_output(self, schema):
        replies = self

        class _Bound:
            def invoke(self, _messages):
                return schema.model_validate(replies._next())

        return _Bound()

    def invoke(self, _messages):
        return json.dumps(self._next())


def _exhausted(typ, payload, *, config=NATIVE, max_retries=0):
    """Return the model and the generated-schema error raised once retries run out."""
    model = _Replies(payload)
    with pytest.raises(ValidationError) as caught:
        generate_structured(model, [], typ, max_retries=max_retries, llm_config=config)
    return model, caught.value


def test_native_union_reply_returns_the_member_dict():
    model = _Replies({"value": {"action": "buy", "qty": 2}})
    out = generate_structured(model, [], TRADE, llm_config=NATIVE)
    assert out == {"action": "buy", "qty": 2}


def test_fallback_union_reply_parses_json():
    model = _Replies({"value": {"action": "sell", "price": 1.5}})
    out = generate_structured(model, [], TRADE, llm_config=FALLBACK)
    assert out == {"action": "sell", "price": 1.5}


@pytest.mark.parametrize("config", [NATIVE, FALLBACK])
def test_unknown_field_is_rejected_at_its_path(config):
    _, error = _exhausted(TRADE, {"value": {"action": "buy", "qty": 2, "zzz": 1}}, config=config)
    assert _structured_validation_path(error, TRADE) == ("zzz",)


@pytest.mark.parametrize("config", [NATIVE, FALLBACK])
def test_sell_reply_with_qty_retries_then_reports_the_sell_members_first_error(config):
    model, error = _exhausted(
        TRADE, {"value": {"action": "sell", "qty": 2}}, config=config, max_retries=2,
    )
    assert model.calls == 3
    # The `buy` member is dropped by its tag; `sell` reports `price` missing before `qty`.
    assert _structured_validation_path(error, TRADE) == ("price",)


def test_unknown_field_in_a_record_nested_in_a_member_is_rejected_at_its_path():
    typ = type_for("Union(Bid, Sell)", REGISTRY)
    _, error = _exhausted(typ, {"value": {"action": "bid", "limit": {"px": 1.0, "zz": 1}}})
    assert _structured_validation_path(error, typ) == ("limit", "zz")


def test_sell_reply_with_extra_qty_reports_the_unknown_field():
    _, error = _exhausted(TRADE, {"value": {"action": "sell", "price": 1.0, "qty": 2}})
    assert _structured_validation_path(error, TRADE) == ("qty",)


def test_matching_member_field_error_keeps_the_member_field_path():
    _, error = _exhausted(TRADE, {"value": {"action": "buy", "qty": "many"}})
    assert _structured_validation_path(error, TRADE) == ("qty",)


@pytest.mark.parametrize("payload", [{"qty": 2}, {"action": "hold"}])
def test_missing_or_unknown_tag_reports_the_tag_field(payload):
    _, error = _exhausted(TRADE, {"value": payload})
    assert _structured_validation_path(error, TRADE) == ("action",)


def test_nested_list_of_union_error_maps_to_the_item_field():
    typ = type_for("list(Union(Buy, Sell))", REGISTRY)
    _, error = _exhausted(typ, {"items": [{"action": "sell", "price": "high"}]})
    assert _structured_validation_path(error, typ) == (0, "price")


def test_record_field_of_union_type_error_maps_through_the_member():
    typ = type_for("Order", REGISTRY)
    _, error = _exhausted(typ, {"trade": {"action": "buy", "qty": "x"}, "note": "n"})
    assert _structured_validation_path(error, typ) == ("trade", "qty")


def test_union_inside_a_union_member_maps_through_both_members():
    typ = type_for("Union(Ticket, Skip)", REGISTRY)
    payload = {"value": {"kind": "ticket", "trade": {"action": "sell", "price": "high"}}}
    _, error = _exhausted(typ, payload)
    assert _structured_validation_path(error, typ) == ("trade", "price")

    _, error = _exhausted(typ, {"value": {"kind": "ticket", "trade": {"action": "hold"}}})
    assert _structured_validation_path(error, typ) == ("trade", "action")


def test_untagged_union_with_no_match_maps_to_the_union_position():
    typ = type_for("Union(Text, Call)", REGISTRY)
    _, error = _exhausted(typ, {"value": {"zz": 1}})
    assert _structured_validation_path(error, typ) == ()

    listed = type_for("list(Union(Text, Call))", REGISTRY)
    _, error = _exhausted(listed, {"items": [{"text": "a"}, {"name": 3}]})
    assert _structured_validation_path(error, listed) == (1,)


def test_untagged_union_reply_returns_the_matching_member():
    typ = type_for("Union(Text, Call)", REGISTRY)
    model = _Replies({"value": {"name": "lookup", "input": {"q": 1}}})
    assert generate_structured(model, [], typ, llm_config=NATIVE) == {
        "name": "lookup", "input": {"q": 1},
    }


_AGENT_FLOW = """
id: decide
name: decide
typedefs:
  Buy: {action: 'Literal(buy)', qty: int}
  Sell: {action: 'Literal(sell)', qty: int, price: float}
  Hold: {action: 'Literal(hold)', note: str}
  Decision: Union(Buy, Sell, Hold)
nodes:
  a:
    kind: agent
    prompt: decide a trade
    output: Decision
  route:
    kind: case
    on: ${a.output.action}
    cases:
      - when: buy
        then: place_buy
      - when: sell
        then: place_sell
    else: log_hold
  place_buy:
    kind: code
    input:
      qty: ${a.output.qty}
    code: |
      return "buy " + str(inputs["qty"])
    output: str
  place_sell:
    kind: code
    input:
      qty: ${a.output.qty}
      price: ${a.output.price}
    code: |
      return inputs["qty"] * inputs["price"]
    output: float
  log_hold:
    kind: code
    input:
      note: ${a.output.note}
    code: |
      return "hold: " + inputs["note"]
    output: str
output:
  decision: ${a.output}
  trade: ${place_buy.output | place_sell.output | log_hold.output}
"""


@pytest.mark.parametrize("decision, trade", [
    ({"action": "buy", "qty": 3}, "buy 3"),
    ({"action": "sell", "qty": 2, "price": 1.5}, 3.0),
    ({"action": "hold", "note": "wait"}, "hold: wait"),
])
def test_agent_union_output_routes_a_case_and_its_branch_reads_member_fields(
    monkeypatch, decision, trade,
):
    model = _Replies({"value": decision})
    monkeypatch.setattr(llm_clients_mod, "model_from_config", lambda _config: model)

    result = run_flow(load_flow(_AGENT_FLOW), {})
    assert result.status == "succeeded", result.failure
    assert result.output == {"decision": decision, "trade": trade}
    assert model.calls == 1


def test_agent_union_output_with_an_unknown_tag_fails_the_agent(monkeypatch):
    model = _Replies({"value": {"action": "wait"}})
    monkeypatch.setattr(llm_clients_mod, "model_from_config", lambda _config: model)

    result = run_flow(load_flow(_AGENT_FLOW), {})
    assert result.status == "failed"
    assert result.failure.node_id == "a"
    assert result.failure.code == "node.execute"
    assert "literal_error" in result.failure.message
    # The unknown tag maps to the union's tag field; a typedef output locates it at the contract.
    (origin,) = result.failure.trace
    assert origin.key == "/s:root/s:node/s:a/s:output/s:contract/s:root"
