/**
 * `generateStructured` for a union of records: the member value it returns and the output path of
 * an exhausted reply's error; mirrors the `generate_structured` rows of
 * `tests/engine/test_structured_union.py`.
 *
 * Ported (native and prompt where Python parametrizes): `test_native_union_reply_returns_the_member_dict`,
 * `test_fallback_union_reply_parses_json`,
 * `test_sell_reply_with_qty_retries_then_reports_the_sell_members_first_error`,
 * `test_sell_reply_with_extra_qty_reports_the_unknown_field` (`qty` is a contract path through the
 * `buy` member), `test_matching_member_field_error_keeps_the_member_field_path`,
 * `test_missing_or_unknown_tag_reports_the_tag_field`,
 * `test_nested_list_of_union_error_maps_to_the_item_field`,
 * `test_record_field_of_union_type_error_maps_through_the_member`,
 * `test_union_inside_a_union_member_maps_through_both_members`,
 * `test_untagged_union_with_no_match_maps_to_the_union_position` and
 * `test_untagged_union_reply_returns_the_matching_member`. The path is `OutputInvalidError.path`
 * rather than a function of the exception.
 *
 * Changed: an unknown field's output path is the record that refuses it, not the field, because
 * the field is not part of the output contract (`test_unknown_field_is_rejected_at_its_path`:
 * `[]` for Python's `("zzz",)`; `test_unknown_field_in_a_record_nested_in_a_member_is_rejected_at_its_path`:
 * `["limit"]` for `("limit", "zz")`). The value check's own error, the `cause`, keeps the field in
 * its path and message.
 *
 * Later (T4-8), they load and run a flow:
 * `test_agent_union_output_routes_a_case_and_its_branch_reads_member_fields` and
 * `test_agent_union_output_with_an_unknown_tag_fails_the_agent`.
 */

import { afterEach, beforeEach, describe, expect, test } from "bun:test";

import { BUILTIN, forget, install } from "../../src/defaults.ts";
import { Message, Response, TextPart } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { generateStructured, OutputInvalidError } from "../../src/nodes/agent/structured.ts";
import { readTypedefs, typeFor } from "../../src/typesys/types.ts";
import { type Type, TypeCheckError } from "../../src/typesys/values.ts";

beforeEach(() => install({ ...BUILTIN, modelRetries: 0 }));
afterEach(forget);

const NATIVE = { provider: "openai", model: "gpt-5.5" };
const PROMPTED = { provider: "vllm", model: "no-structured-sentinel" };
const CONFIGS = [
  ["native", NATIVE],
  ["prompt", PROMPTED],
] as const;

const REGISTRY = readTypedefs({
  Buy: { action: "Literal(buy)", qty: "int" },
  Sell: { action: "Literal(sell)", price: "float" },
  Hold: { action: "Literal(hold)" },
  Text: { text: "str" },
  Call: { name: "str", input: "object" },
  Order: { trade: "Union(Buy, Sell)", note: "str" },
  Ticket: { kind: "Literal(ticket)", trade: "Optional(Union(Buy, Sell))" },
  Skip: { kind: "Literal(skip)" },
  Limit: { px: "float" },
  Bid: { action: "Literal(bid)", limit: "Limit" },
});
const TRADE = t("Union(Buy, Sell)");

function t(expression: string): Type {
  return typeFor(expression, REGISTRY);
}

/** A model answering `payload` (JSON text) to each of `calls` requests. */
function replies(payload: string, calls = 1, options: { provider: string; model: string } = NATIVE): ScriptedModel {
  const response = new Response({ message: new Message({ role: "assistant", parts: [new TextPart(payload)] }) });
  return new ScriptedModel(Array.from({ length: calls }, () => response), options);
}

/** The model and the `OutputInvalidError` generating `typ` throws once `retries` are spent. */
async function exhausted(
  typ: Type,
  payload: string,
  options: { config?: { provider: string; model: string }; retries?: number } = {},
): Promise<[ScriptedModel, OutputInvalidError]> {
  const retries = options.retries ?? 0;
  const model = replies(payload, retries + 1, options.config);
  try {
    await generateStructured(model, [], typ, { retries });
  } catch (error) {
    expect(error).toBeInstanceOf(OutputInvalidError);
    return [model, error as OutputInvalidError];
  }
  throw new Error("generateStructured returned");
}

test("test_native_union_reply_returns_the_member_dict", async () => {
  const model = replies('{"value": {"action": "buy", "qty": 2}}');
  expect(await generateStructured(model, [], TRADE)).toEqual({ action: "buy", qty: 2n });
});

test("test_fallback_union_reply_parses_json", async () => {
  const model = replies('{"value": {"action": "sell", "price": 1.5}}', 1, PROMPTED);
  expect(await generateStructured(model, [], TRADE)).toEqual({ action: "sell", price: 1.5 });
});

describe("test_unknown_field_is_rejected_at_its_path (Changed: the record's path)", () => {
  test.each(CONFIGS)("%s", async (_label, config) => {
    const [, error] = await exhausted(TRADE, '{"value": {"action": "buy", "qty": 2, "zzz": 1}}', { config });
    expect(error.path).toEqual([]);
    expect(error.message).toBe('structured output at zzz: record has unknown fields: ["zzz"]');
    expect((error.cause as TypeCheckError).path).toEqual(["zzz"]);
  });
});

describe("test_sell_reply_with_qty_retries_then_reports_the_sell_members_first_error", () => {
  test.each(CONFIGS)("%s", async (_label, config) => {
    const [model, error] = await exhausted(TRADE, '{"value": {"action": "sell", "qty": 2}}', {
      config,
      retries: 2,
    });
    expect(model.requests).toHaveLength(3);
    // The `buy` member is dropped by its tag; `sell` reports `price` missing before `qty`.
    expect(error.path).toEqual(["price"]);
  });
});

test("test_unknown_field_in_a_record_nested_in_a_member_is_rejected_at_its_path (Changed: the record's path)", async () => {
  const [, error] = await exhausted(t("Union(Bid, Sell)"), '{"value": {"action": "bid", "limit": {"px": 1.0, "zz": 1}}}');
  expect(error.path).toEqual(["limit"]);
  expect(error.cause).toBeInstanceOf(TypeCheckError);
  expect((error.cause as TypeCheckError).path).toEqual(["limit", "zz"]);
});

test("test_sell_reply_with_extra_qty_reports_the_unknown_field", async () => {
  // `qty` is a contract path through the `buy` member, so the field itself is reported.
  const [, error] = await exhausted(TRADE, '{"value": {"action": "sell", "price": 1.0, "qty": 2}}');
  expect(error.path).toEqual(["qty"]);
});

test("test_matching_member_field_error_keeps_the_member_field_path", async () => {
  const [, error] = await exhausted(TRADE, '{"value": {"action": "buy", "qty": "many"}}');
  expect(error.path).toEqual(["qty"]);
});

describe("test_missing_or_unknown_tag_reports_the_tag_field", () => {
  test.each(['{"qty": 2}', '{"action": "hold"}'])("%s", async (payload) => {
    const [, error] = await exhausted(TRADE, `{"value": ${payload}}`);
    expect(error.path).toEqual(["action"]);
  });
});

test("test_nested_list_of_union_error_maps_to_the_item_field", async () => {
  const [, error] = await exhausted(t("list(Union(Buy, Sell))"), '{"items": [{"action": "sell", "price": "high"}]}');
  expect(error.path).toEqual([0, "price"]);
});

test("test_record_field_of_union_type_error_maps_through_the_member", async () => {
  const [, error] = await exhausted(t("Order"), '{"trade": {"action": "buy", "qty": "x"}, "note": "n"}');
  expect(error.path).toEqual(["trade", "qty"]);
});

test("test_union_inside_a_union_member_maps_through_both_members", async () => {
  const typ = t("Union(Ticket, Skip)");
  let [, error] = await exhausted(typ, '{"value": {"kind": "ticket", "trade": {"action": "sell", "price": "high"}}}');
  expect(error.path).toEqual(["trade", "price"]);

  [, error] = await exhausted(typ, '{"value": {"kind": "ticket", "trade": {"action": "hold"}}}');
  expect(error.path).toEqual(["trade", "action"]);
});

test("test_untagged_union_with_no_match_maps_to_the_union_position", async () => {
  let [, error] = await exhausted(t("Union(Text, Call)"), '{"value": {"zz": 1}}');
  expect(error.path).toEqual([]);

  [, error] = await exhausted(t("list(Union(Text, Call))"), '{"items": [{"text": "a"}, {"name": 3}]}');
  expect(error.path).toEqual([1]);
});

test("test_untagged_union_reply_returns_the_matching_member", async () => {
  const model = replies('{"value": {"name": "lookup", "input": {"q": 1}}}');
  expect(await generateStructured(model, [], t("Union(Text, Call)"))).toEqual({ name: "lookup", input: { q: 1n } });
});
