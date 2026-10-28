/** Ints with more than 4300 decimal digits as yian values. */

import { afterAll, expect, test } from "bun:test";
import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { loadFlow } from "../../src/compose/loader.ts";
import { coerceParam } from "../../src/nodes/binding.ts";
import { Parameter, ValueContract } from "../../src/nodes/contracts.ts";
import { runFlow } from "../../src/run.ts";
import { Type, ValueKind } from "../../src/typesys/index.ts";
import { parseJson } from "../../src/typesys/json.ts";
import { fromHost, type Value, valueRepr } from "../../src/typesys/values.ts";
import { hostEnv } from "../helpers/mcp_helpers.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-long-ints-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));
const env = hostEnv(dir);

const BIG = 10n ** 4999n + 12345n; // 5000 digits
const D = "1".repeat(5000);

/** An `int` parameter that reads text, as the input readers build it. */
const INT = new Parameter("n", {
  contract: new ValueContract(Type.scalar(ValueKind.INTEGER)),
  coerceStrings: true,
});

test("parse int text skips leading zeros", () => {
  expect(coerceParam(INT, "0".repeat(4999) + "1")).toBe(1n);
});

test("parse int text rejects a long int", () => {
  expect(coerceParam(INT, "-" + D)).toBe(-BigInt(D));
});

test.each<[string, string, bigint]>([
  ["4300-nines", "9".repeat(4300), BigInt("9".repeat(4300))],
  ["plus", "+12", 12n],
  ["underscores", " 1_000 ", 1000n],
  ["zero-underscores", "0_".repeat(3000) + "1", 1n],
])("parse int text accepts int grammar (%s)", (_id, text, expected) => {
  expect(coerceParam(INT, text)).toBe(expected);
});

test("parse int text skips leading zeros of any script", () => {
  const text = "\u0660".repeat(4400) + "1";
  expect(coerceParam(INT, text)).toBe(text);
});

test("parse int text raises int own error for invalid text", () => {
  expect(coerceParam(INT, "abc")).toBe("abc");
});

test("parse json int marks a long token", () => {
  expect(parseJson("[1, " + D + "]", "plain")).toEqual([1n, BigInt(D)]);
});

test.each<[string, Value]>([
  ["root", BIG],
  ["negative", -BIG],
  ["nested", { a: [1n, BIG] }],
])("reject nonfinite rejects a long int (%s)", (_id, value) => {
  expect(fromHost(value, "input 'n'")).toEqual(value);
});

test("reject nonfinite passes ints within the limit and bools", () => {
  expect(fromHost(10n ** 4300n - 1n, "input 'n'")).toBe(10n ** 4300n - 1n);
  expect(fromHost(true, "input 'n'")).toBe(true);
});

test("value repr shows a digit count for a long int", () => {
  expect(valueRepr(BIG)).toBe(String(BIG).slice(0, 199) + "…");
  expect(valueRepr([BIG])).toBe("[" + String(BIG).slice(0, 198) + "…");
});

/** Return a flow whose `c` CODE node binds `y` to the input `declared` declares and returns it. */
const inputFlow = (declared: string) => `
id: long-input
name: long_input
input:
  ${declared}
nodes:
  c:
    kind: code
    input:
      y: \${input.${declared.split(":")[0]}}
    code: |
      return inputs.y
output: \${c.output}
`;

let runs = 0;

test.each<[string, string, Record<string, unknown>, Value]>([
  ["int", "n: !int", { n: BIG }, BIG],
  ["int-text", "n: !int", { n: D }, BigInt(D)],
  ["list", "xs: !list(int)", { xs: [1n, BIG] }, [1n, BIG]],
])("a long int run input is rejected before the run starts (%s)", async (_id, declared, inputs, output) => {
  const log = join(dir, `run-${runs++}.jsonl`);

  const result = await runFlow(await loadFlow(inputFlow(declared), { env }), inputs, { env, log });

  expect(result.failure).toBeNull();
  expect(result.status).toBe("succeeded");
  expect(result.output).toEqual(output);
  expect(existsSync(log)).toBe(true);
});

test("a long int run input is rejected before the run starts (str)", async () => {
  const result = await runFlow(await loadFlow(inputFlow("n: !str"), { env }), { n: BIG }, { env });

  expect(result.status).toBe("failed");
  expect(result.failure!.code).toBe("contract.input");
  expect(result.failure!.message).toBe(`input 'n': ${String(BIG).slice(0, 199)}… does not match declared type str`);
});
