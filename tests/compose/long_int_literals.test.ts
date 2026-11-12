/** Int literals with more than 4300 digits in a flow file. */

import { afterAll, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { LoadError } from "../../src/compose/helpers/errors.ts";
import { runFlow } from "../../src/run.ts";
import { parseType } from "../../src/typesys/types.ts";
import { TypeCheckError, type Value } from "../../src/typesys/values.ts";
import { loadConformanceFlow } from "../helpers/declarations.ts";
import { hostEnv } from "../helpers/mcp_helpers.ts";
import { rejected, thrownBy } from "../helpers/run_state_fixtures.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-long-int-literals-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));
// `D + 1` gives a 5000-digit int, over the default YIAN_MAX_INT_DIGITS.
const env = hostEnv(dir, { YIAN_MAX_INT_DIGITS: "5000" });

const D = "1".repeat(5000);
const ONES = BigInt(D);

const HEAD = "id: f\nname: f\n";

/** A `use` node echoing `topic`, bound to `binding`. */
const node = (binding: string) =>
  "  use:\n    kind: code\n    code: ./fixtures/compose_codefns.ts:echo\n" + `    input:\n      topic: ${binding}\n`;

/** A flow whose `use` node binds `binding`, after an optional `input:` block. */
const flow = (binding = "hi", inputs = "") => HEAD + inputs + "nodes:\n" + node(binding) + "output: ${use.output}\n";

/** Load and run `text` with `inputs`, assert it succeeded, and return its output. */
async function output(text: string, inputs: Record<string, unknown> = {}): Promise<unknown> {
  const result = await runFlow(await loadConformanceFlow(text, { env }), inputs, { env });
  expect(result.failure).toBeNull();
  expect(result.status).toBe("succeeded");
  return result.output;
}

// --- YAML int literals ------------------------------------------------------ //

test.each<[string, string, Value]>([
  ["input-default", flow("${input.n}", `input:\n  n: !int ${D}\n`), ONES],
  ["bare-binding", flow(D), ONES],
  ["tagged-binding", flow(`!int ${D}`), ONES],
  ["value-entry", HEAD + `nodes:\n  v: !int ${D}\noutput: \${v.output}\n`, ONES],
  ["negative", flow(`-${D}`), -ONES],
  ["hex", flow("0x" + "f".repeat(4000)), BigInt("0x" + "f".repeat(4000))],
  ["octal", flow("0o" + "7".repeat(5000)), BigInt("0o" + "7".repeat(5000))],
])("a long int literal is a located load error (%s)", async (_id, text, value) => {
  expect(await output(text)).toBe(value as bigint);
});

test("a long int literal is a located load error (input-default-long-form)", async () => {
  const error = await rejected(
    loadConformanceFlow(flow("${input.n}", `input:\n  n:\n    type: int\n    default: ${D}\n`), { env }),
  );

  expect(error).toBeInstanceOf(LoadError);
  expect((error as LoadError).message).toBe(
    "flow input 'n': name the record in `typedefs:` and tag the input with it, e.g. `n: !N`",
  );
  expect((error as LoadError).diagnostic.frames.at(-1)!.range!.startLine).toBe(5);
});

test("leading zeros do not count", async () => {
  expect(await output(HEAD + "nodes:\n  v: !int " + "0".repeat(4999) + "1\noutput: ${v.output}\n")).toBe(1n);
});

// --- range ends ------------------------------------------------------------- //

test.each([
  ["int", `!int(0..${D})`, ONES, ONES + 1n],
  ["neg", `!int(..-${D})`, -ONES, -ONES + 1n],
])("a long range end is a load error (%s)", async (_id, declared, end, past) => {
  const text = flow("${input.n}", `input:\n  n: ${declared}\n`);

  expect(await output(text, { n: end })).toBe(end);
  const result = await runFlow(await loadConformanceFlow(text, { env }), { n: past }, { env });
  expect(result.status).toBe("failed");
  expect(result.failure!.code).toBe("contract.input");
});

test("a long range end is a load error (float)", async () => {
  const error = await rejected(loadConformanceFlow(flow("hi", `input:\n  n: !float(0..${D})\n`), { env }));

  expect(error).toBeInstanceOf(LoadError);
  expect((error as LoadError).message).toBe(
    `'n': \`!float(0..${D})\` is not a type (invalid bounds in 'float(0..${D})': float bounds require finite endpoints)`,
  );
});

test.each([
  ["decimal", D, `types use parentheses: \`int(0..${D})\``],
  [
    "hex",
    "0x" + "f".repeat(4000),
    `types use parentheses, not brackets (malformed type expression 'int[0, 0x${"f".repeat(4000)}]')`,
  ],
])("a long end in the bracket spelling names the limit (%s)", (_id, end, message) => {
  const caught = thrownBy(() => parseType(`int[0, ${end}]`));

  expect(caught).toBeInstanceOf(TypeCheckError);
  expect((caught as TypeCheckError).message).toBe(message);
});

// --- expression number tokens ----------------------------------------------- //

const PT = "typedefs:\n  Pt:\n    x: int\n";

// A row's output is `undefined` when its flow needs a model to run, so the row only loads it.
test.each<[string, string, Value | undefined]>([
  ["sum", flow("${" + D + " + 1}"), ONES + 1n],
  ["lone", flow("${" + D + "}"), ONES],
  [
    "case-when",
    HEAD +
      "nodes:\n  gate:\n    kind: case\n    cases:\n" +
      "      - when: ${" + D + " > 0}\n        then: use\n" +
      node("hi") +
      "output: ${use.output}\n",
    "hi",
  ],
  [
    "agent-prompt",
    HEAD + "nodes:\n  a:\n    kind: agent\n" + '    prompt: "${' + D + '}"\n    output: str\noutput: ${a.output}\n',
    undefined,
  ],
  ["constructor-argument", HEAD + PT + "nodes:\n" + node("${Pt(x=" + D + ")}") + "output: ${use.output}\n", { x: ONES }],
  ["flow-assert", flow() + 'asserts:\n  - "' + D + ' > 0"\n', "hi"],
  ["node-assert", HEAD + "nodes:\n" + node("hi") + "    asserts:\n      - " + D + " > 0\noutput: ${use.output}\n", "hi"],
])("a long int in an expression is a load error (%s)", async (_id, text, value) => {
  if (value === undefined) await loadConformanceFlow(text, { env });
  else expect(await output(text)).toEqual(value);
});
