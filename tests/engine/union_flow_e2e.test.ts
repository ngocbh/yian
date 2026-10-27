/**
 * Unions of records end to end: load, route, narrow and replay through `runFlow`: a port of
 * `tests/engine/test_union_flow_e2e.py`.
 *
 * Ported: every row. The CODE bodies are Python's, which read the same in JS (decision 11), but
 * for `String(...)` and `toUpperCase()` in place of `str(...)` and `upper()`. A refused resume
 * rejects with the recovery error, where Python raises `ValueError`. Ints read as `bigint`s
 * (decision 4a).
 *
 * Changed: the sell branch's trade `2 * 1.5` is the whole number 3, which reads back from the CODE
 * body as the int `3n` and stays an int in its `float` slot (decisions 9A, 38), where Python gives
 * `3.0`. Messages show text values JSON-style and lists of names as JSON text (decisions 7, 43):
 * `"wait" is not a tag of the union (expected one of ["buy","hold","sell"])`,
 * `record missing required fields: ["price"]`.
 *
 * The harvested `run-core/engine_union_flow_e2e__*` cases run reduced flows (no `note` half, or
 * only the nested output), so each covers no whole test.
 */

import { afterAll, expect, test } from "bun:test";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { loadFlow } from "../../src/compose/loader.ts";
import { type RunResult, advance, runFlow } from "../../src/run.ts";
import { hostEnv } from "./mcp_helpers.ts";
import { rejected } from "./run_state_fixtures.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-union-flow-e2e-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));
const env = hostEnv(dir);

let logs = 0;
/** A fresh log path. */
const freshLog = () => join(dir, `run-${++logs}.jsonl`);

const FLOW = `
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
          qty: \${input.order.qty}
          price: \${input.order.price}
        code: |
          return inputs["qty"] * inputs["price"]
        output: float
    output: \${fill.output}
nodes:
  d:
    kind: code
    input:
      decision: \${input.decision}
    code: |
      return inputs["decision"]
    output: Decision
  route:
    kind: case
    on: \${d.output.action}
    cases:
      - when: buy
        then: place_buy
      - when: sell
        then: call(sell_flow, order=\${d.output})
    else: log_hold
  place_buy:
    kind: code
    input:
      qty: \${d.output.qty}
    code: |
      return "buy " + String(inputs["qty"])
    output: str
  log_hold:
    kind: code
    input:
      note: \${d.output.note}
    code: |
      return "hold: " + inputs["note"]
    output: str
  n:
    kind: code
    input:
      note: \${input.note}
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
      text: \${n.output.text}
    code: |
      return inputs["text"].toUpperCase()
    output: str
  double:
    kind: code
    depends_on: [n]
    code: |
      return "number"
    output: str
output:
  action: \${d.output.action}
  trade: \${place_buy.output | __call_0.output | log_hold.output}
  note: \${shout.output | double.output}
`;

const flow = () => loadFlow(FLOW, { env });

const BUY = { action: "buy", qty: 3 };
const SELL = { action: "sell", qty: 2, price: 1.5 };
const HOLD = { action: "hold", note: "wait" };

/** Run `FLOW` on `inputs` in the test's environment. */
const run = async (inputs: Record<string, unknown>, log: string | null = null) => runFlow(await flow(), inputs, { env, log });

/** Assert `result` succeeded and return its output. */
function succeeded(result: RunResult): unknown {
  expect(result.failure).toBeNull();
  expect(result.status).toBe("succeeded");
  return result.output;
}

test.each([
  [BUY, "buy", "buy 3"],
  [SELL, "sell", 3n],
  [HOLD, "hold", "hold: wait"],
] as const)("each on branch reads its members fields (%o)", async (decision, action, trade) => {
  const output = succeeded(await run({ decision, note: { num: 1 } }));

  expect(output).toEqual({ action, trade, note: "number" });
});

test.each([
  [{ text: "hi" }, "HI"],
  [{ num: 4 }, "number"],
] as const)("an isinstance case routes an untagged union (%o)", async (note, expected) => {
  const output = succeeded(await run({ decision: BUY, note })) as Record<string, unknown>;

  expect(output["note"]).toBe(expected);
});

test.each([
  [{ action: "wait" }, '"wait" is not a tag of the union (expected one of ["buy","hold","sell"])'],
  [{ qty: 2 }, `missing tag field 'action' (expected one of ["buy","hold","sell"])`],
  [{ action: "sell", qty: 2 }, 'record missing required fields: ["price"]'],
] as const)("a code node returning an invalid tagged member fails the node (%o)", async (decision, message) => {
  const result = await run({ decision, note: { num: 1 } });

  expect(result.status).toBe("failed");
  expect(result.failure!.nodeId).toBe("d");
  expect(result.failure!.code).toBe("contract.output");
  expect(result.failure!.message).toBe(message);
});

test("a code node returning no untagged member fails the node", async () => {
  const result = await run({ decision: BUY, note: { zz: 1 } });

  expect(result.status).toBe("failed");
  expect(result.failure!.nodeId).toBe("n");
  expect(result.failure!.code).toBe("contract.output");
  expect(result.failure!.message.startsWith("value matches no member of Union(")).toBe(true);
  expect(result.failure!.message).toContain('record missing required fields: ["num"]');
  expect(result.failure!.message).toContain('record missing required fields: ["text"]');
});

const NESTED_FLOW = `
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
      value: \${input.value}
    code: |
      return inputs["value"]
    output:
      decision: Decision
      notes: list(Note)
output: \${d.output}
`;

test.each([
  [{ decision: { action: "bid", qty: 1 }, notes: [] }, '"bid" is not a tag of the union (expected one of ["buy","sell"])', "decision"],
  [{ decision: { action: "sell", qty: "x", price: 1.0 }, notes: [] }, '"x" does not match declared type int', "decision"],
  [{ decision: BUY, notes: [{ text: "a" }, { zz: 1 }] }, "value matches no member of Union(", "notes"],
] as const)("a nested union failure points at the output field (%o)", async (value, message, field) => {
  const loaded = await loadFlow(NESTED_FLOW, { env });
  const result = await runFlow(loaded, { value }, { env });

  expect(result.status).toBe("failed");
  expect(result.failure!.code).toBe("contract.output");
  expect(result.failure!.message.startsWith(message)).toBe(true);
  expect(result.failure!.trace.map((origin) => origin.key)).toEqual([`/s:root/s:node/s:d/s:output/s:contract/s:path/s:${field}`]);
  const frames = loaded.diagnose(result.failure!).frames;
  expect(frames.length).toBe(1);
  const authored = frames[0]!.text.split("\n")[frames[0]!.range.startLine - 1]!;
  expect(authored.trim().startsWith(`${field}:`)).toBe(true);
});

/** Keep the log up to `d`'s success record; return the lines kept and that record's index. */
function cutAfterD(log: string): [string[], number] {
  const lines = readFileSync(log, "utf8").split("\n").filter((line) => line !== "");
  const end = lines.findIndex((line) => {
    const record = JSON.parse(line);
    return record.kind === "node_succeeded" && record.payload.node_id === "d";
  });
  const kept = lines.slice(0, end + 1);
  writeFileSync(log, kept.join("\n") + "\n");
  return [kept, end];
}

test("replaying a finished run reproduces the union branch", async () => {
  const log = freshLog();
  const first = await run({ decision: SELL, note: { text: "hi" } }, log);
  succeeded(first);

  const replayed = await advance({ log, env });

  expect(succeeded(replayed)).toEqual(first.output);
  expect(first.output).toEqual({ action: "sell", trade: 3n, note: "HI" });
});

test("resume after the union value continues down its branch", async () => {
  const log = freshLog();
  const first = await run({ decision: SELL, note: { num: 1 } }, log);
  succeeded(first);
  cutAfterD(log);

  const resumed = await advance({ log, env });

  expect(succeeded(resumed)).toEqual(first.output);
});

test.each([
  [{ action: "wait", qty: 2 }, '"wait" is not a tag of the union'],
  [{ action: "sell", qty: 2 }, 'record missing required fields: ["price"]'],
] as const)("resume rejects a recorded value that is not a member (%o)", async (forged, message) => {
  const log = freshLog();
  const first = await run({ decision: SELL, note: { num: 1 } }, log);
  succeeded(first);
  const [lines, end] = cutAfterD(log);
  // d's output is its commit, the record before its success.
  const record = JSON.parse(lines[end - 1]!);
  expect(record.kind).toBe("pool_commit");
  record.payload.value = forged;
  lines[end - 1] = JSON.stringify(record);
  writeFileSync(log, lines.join("\n") + "\n");

  const error = await rejected(advance({ log, env }));

  expect(error).toBeInstanceOf(RangeError);
  expect((error as Error).message).toContain(message);
});
