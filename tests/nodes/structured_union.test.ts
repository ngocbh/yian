/**
 * `generateStructured` for a union of records: the member value it returns and the output path of
 * an exhausted reply's error. An unknown field's output path is the record that refuses it, since
 * the field is not part of the output contract; the `cause` keeps the field in its path.
 */

import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterAll, afterEach, beforeEach, describe, expect, test } from "bun:test";

import { loadFlow } from "../../src/compose/loader.ts";
import { BUILTIN, forget, install } from "../../src/defaults.ts";
import { Message, Response, TextPart } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { generateStructured, OutputInvalidError } from "../../src/nodes/agent/structured.ts";
import { runFlow } from "../../src/run.ts";
import { readTypedefs, typeFor } from "../../src/typesys/types.ts";
import { type Type, TypeCheckError } from "../../src/typesys/values.ts";
import { hostEnv } from "../helpers/mcp_helpers.ts";
import { serving } from "../helpers/run_state_fixtures.ts";

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

test("native union reply returns the member dict", async () => {
  const model = replies('{"value": {"action": "buy", "qty": 2}}');
  expect(await generateStructured(model, [], TRADE)).toEqual({ action: "buy", qty: 2n });
});

test("fallback union reply parses json", async () => {
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

describe("sell reply with qty retries then reports the sell members first error", () => {
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

test("sell reply with extra qty reports the unknown field", async () => {
  // `qty` is a contract path through the `buy` member, so the field itself is reported.
  const [, error] = await exhausted(TRADE, '{"value": {"action": "sell", "price": 1.0, "qty": 2}}');
  expect(error.path).toEqual(["qty"]);
});

test("matching member field error keeps the member field path", async () => {
  const [, error] = await exhausted(TRADE, '{"value": {"action": "buy", "qty": "many"}}');
  expect(error.path).toEqual(["qty"]);
});

describe("missing or unknown tag reports the tag field", () => {
  test.each(['{"qty": 2}', '{"action": "hold"}'])("%s", async (payload) => {
    const [, error] = await exhausted(TRADE, `{"value": ${payload}}`);
    expect(error.path).toEqual(["action"]);
  });
});

test("nested list of union error maps to the item field", async () => {
  const [, error] = await exhausted(t("list(Union(Buy, Sell))"), '{"items": [{"action": "sell", "price": "high"}]}');
  expect(error.path).toEqual([0, "price"]);
});

test("record field of union type error maps through the member", async () => {
  const [, error] = await exhausted(t("Order"), '{"trade": {"action": "buy", "qty": "x"}, "note": "n"}');
  expect(error.path).toEqual(["trade", "qty"]);
});

test("union inside a union member maps through both members", async () => {
  const typ = t("Union(Ticket, Skip)");
  let [, error] = await exhausted(typ, '{"value": {"kind": "ticket", "trade": {"action": "sell", "price": "high"}}}');
  expect(error.path).toEqual(["trade", "price"]);

  [, error] = await exhausted(typ, '{"value": {"kind": "ticket", "trade": {"action": "hold"}}}');
  expect(error.path).toEqual(["trade", "action"]);
});

test("untagged union with no match maps to the union position", async () => {
  let [, error] = await exhausted(t("Union(Text, Call)"), '{"value": {"zz": 1}}');
  expect(error.path).toEqual([]);

  [, error] = await exhausted(t("list(Union(Text, Call))"), '{"items": [{"text": "a"}, {"name": 3}]}');
  expect(error.path).toEqual([1]);
});

test("untagged union reply returns the matching member", async () => {
  const model = replies('{"value": {"name": "lookup", "input": {"q": 1}}}');
  expect(await generateStructured(model, [], t("Union(Text, Call)"))).toEqual({ name: "lookup", input: { q: 1n } });
});

// ---------- an AGENT with a union output, through the runner ----------

const AGENT_FLOW = `
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
    on: \${a.output.action}
    cases:
      - when: buy
        then: place_buy
      - when: sell
        then: place_sell
    else: log_hold
  place_buy:
    kind: code
    input:
      qty: \${a.output.qty}
    code: |
      return "buy " + str(inputs["qty"])
    output: str
  place_sell:
    kind: code
    input:
      qty: \${a.output.qty}
      price: \${a.output.price}
    code: |
      return inputs["qty"] * inputs["price"]
    output: float
  log_hold:
    kind: code
    input:
      note: \${a.output.note}
    code: |
      return "hold: " + inputs["note"]
    output: str
output:
  decision: \${a.output}
  trade: \${place_buy.output | place_sell.output | log_hold.output}
`;

const dir = mkdtempSync(join(tmpdir(), "yian-structured-union-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));

/** Run `AGENT_FLOW` with `a` answering `decision` to every request. */
async function runDecision(decision: string): Promise<[ScriptedModel, Awaited<ReturnType<typeof runFlow>>]> {
  const model = replies(`{"value": ${decision}}`, 10);
  const result = await runFlow(await loadFlow(AGENT_FLOW), {}, {
    llmConfig: NATIVE,
    models: serving(model),
    env: hostEnv(dir),
  });
  return [model, result];
}

describe("agent union output routes a case and its branch reads member fields", () => {
  test.each([
    ['{"action": "sell", "qty": 2, "price": 1.5}', { action: "sell", qty: 2n, price: 1.5 }, 3n],
    ['{"action": "hold", "note": "wait"}', { action: "hold", note: "wait" }, "hold: wait"],
  ])("%s", async (reply, decision, trade) => {
    const [model, result] = await runDecision(reply);
    expect(result.failure).toBeNull();
    expect(result.output).toEqual({ decision, trade });
    expect(model.requests).toHaveLength(1);
  });
});

test("agent union output with an unknown tag fails the agent", async () => {
  const [, result] = await runDecision('{"action": "wait"}');
  expect(result.status).toBe("failed");
  const failure = result.failure!;
  expect([failure.nodeId, failure.code]).toEqual(["a", "agent.output_invalid"]);
  expect(failure.message).toBe(
    'structured output at action: "wait" is not a tag of the union (expected one of ["buy","hold","sell"])',
  );
  // The unknown tag maps to the union's tag field; a typedef output locates it at the contract.
  expect(failure.trace.map((origin) => origin.key)).toEqual(["/s:root/s:node/s:a/s:output/s:contract/s:root"]);
});
