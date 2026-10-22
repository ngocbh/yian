/**
 * Arithmetic or rendering a binding can't complete fails the run, not the caller: a port of
 * `tests/engine/test_expr_arithmetic_errors.py`.
 *
 * Ported: every row but the one covered below. Flow inputs are `bigint`s (decision 4a).
 *
 * Changed: `test_a_float_power_overflow_fails_the_run` and
 * `test_an_int_too_large_for_float_division_fails_the_run` succeed: `2.0 ** 5000` and
 * `10 ** 400 / 3` are float infinity, a value (decision 34). `test_an_int_to_float_overflow_fails_the_node_naming_the_binding`
 * succeeds the same way with `input.x * 1.0`; the row then checks that a failing binding is named
 * in the failure's trace with a division by zero (decision 35) on the same binding. These rows
 * check the binding is `Infinity`. `test_an_embedded_int_too_large_for_text_fails_the_run`
 * succeeds: a 5000-digit int renders whole (decisions 4a, 4b), which the row checks. `test_a_missing_format_key_fails_the_run` fails because `%` with text on the
 * left no longer formats (decision 7), where Python fails on the missing key.
 *
 * Covered by `run-core/engine_expr_arithmetic_errors__an_overflow_in_a_flow_assert` (TS expect,
 * decision 34): `test_an_overflow_in_a_flow_assert_fails_the_run`.
 */

import { afterAll, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { loadFlow } from "../../src/compose/loader.ts";
import type { Failure } from "../../src/diagnostics.ts";
import { type RunResult, runFlow } from "../../src/run.ts";
import { hostEnv } from "./mcp_helpers.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-expr-arithmetic-errors-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));
const env = hostEnv(dir);

/** Return a flow whose CODE node `n` binds `v: <value>` over flow inputs `decl` and outputs it as text. */
const flow = (decl: string, value: string) => `
id: arith
name: arith
input:
  ${decl}
nodes:
  n:
    kind: code
    input:
      v: ${value}
    output: str
    code: return String(inputs.v)
output: \${n.output}
`;

/** Load and run `text` in the test's environment. */
async function run(text: string, inputs: Record<string, unknown>): Promise<RunResult> {
  return runFlow(await loadFlow(text, { env }), inputs, { env });
}

/** Run the flow and assert it succeeded, its binding rendering as `output`. */
async function succeeded(decl: string, value: string, inputs: Record<string, unknown>, output: string): Promise<void> {
  const result = await run(flow(decl, value), inputs);
  expect(result.failure).toBeNull();
  expect(result.status).toBe("succeeded");
  expect(result.output).toBe(output);
}

/** Run the flow and return its failure, asserting it failed at `n`'s binding. */
async function failed(decl: string, value: string, inputs: Record<string, unknown>): Promise<Failure> {
  const result = await run(flow(decl, value), inputs);
  expect(result.status).toBe("failed");
  expect(result.failure!.code).toBe("binding.evaluate");
  expect(result.failure!.nodeId).toBe("n");
  return result.failure!;
}

test("a float power overflow fails the run", async () => {
  await succeeded("n: !int", "${2.0 ** input.n}", { n: 5000n }, "Infinity");
});

test("an int too large for float division fails the run", async () => {
  await succeeded("n: !int", "${input.n / 3}", { n: 10n ** 400n }, "Infinity");
});

test("a type error on a huge int fails the run", async () => {
  const failure = await failed("n: !int", '${10 ** input.n + "a"}', { n: 5000n });
  expect(failure.message).toContain("type error");
});

test("an embedded int too large for text fails the run", async () => {
  await succeeded("n: !int", "'x=${10 ** input.n}'", { n: 4999n }, "x=1" + "0".repeat(4999));
});

test("a missing format key fails the run", async () => {
  const failure = await failed("r: !object", '${"%(k)s" % input.r}', { r: { j: 1n } });
  expect(failure.message).toBe("text formatting with `%` was removed; use a template");
});

/** Return the flow binding `c`'s input `y` as `binding` over the int input `x`, outputting it as text. */
const yFlow = (binding: string) => `
id: arith
name: arith
input:
  x: !int
nodes:
  c:
    kind: code
    input:
      y: ${binding}
    output: str
    code: return String(inputs.y)
output: \${c.output}
`;

test("an int to float overflow fails the node naming the binding", async () => {
  const overflow = await run(yFlow("${input.x * 1.0}"), { x: 10n ** 400n });
  expect(overflow.failure).toBeNull();
  expect(overflow.status).toBe("succeeded");
  expect(overflow.output).toBe("Infinity");

  const result = await run(yFlow("${input.x * 1.0 / 0}"), { x: 10n ** 400n });
  expect(result.status).toBe("failed");
  expect(result.failure!.code).toBe("binding.evaluate");
  expect(result.failure!.nodeId).toBe("c");
  expect(result.failure!.message).toContain("division by zero");
  expect(result.failure!.trace[0]!.key).toContain("/s:input/s:y/");
});
