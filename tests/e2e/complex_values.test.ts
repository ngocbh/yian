/**
 * A number with no real value fails at the first boundary that sees it. In an expression,
 * `(0 - 8) ** 0.5` is an expression error ("** on -8 and 0.5 has no numeric result"); in a JS CODE
 * body, `(-8) ** 0.5` is NaN, which fails the output contract as "NaN is not a yian value".
 */

import { afterAll, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { loadFlow } from "../../src/compose/loader.ts";
import { type RunResult, runFlow, type RunFlowOptions } from "../../src/run.ts";
import { hostEnv } from "../helpers/mcp_helpers.ts";
import { startedIds } from "../helpers/run_state_fixtures.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-complex-values-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));

/** The expression error `(0 - 8) ** 0.5` raises. */
const NO_RESULT = "** on -8 and 0.5 has no numeric result";

/** Load `source` and run it in the test's environment. */
async function run(source: string, inputs: Record<string, unknown> = {}, options: RunFlowOptions = {}) {
  return runFlow(await loadFlow(source), inputs, { env: hostEnv(dir), ...options });
}

/** Assert `nodeId` failed `binding.evaluate` with exactly `message`. */
function expectBindingFails(result: RunResult, nodeId: string, message: string): void {
  expect(result.status).toBe("failed");
  expect(result.failure!.code).toBe("binding.evaluate");
  expect(result.failure!.nodeId).toBe(nodeId);
  expect(result.failure!.message).toBe(message);
}

test("a binding that evaluates to no real number fails its node", async () => {
  const result = await run(
    `
id: complex
name: complex
nodes:
  n:
    kind: code
    input:
      v: \${(0 - 8) ** 0.5}
    code: |
      return inputs["v"]
output: \${n.output}
`,
    {},
  );

  expectBindingFails(result, "n", NO_RESULT);
  expect(result.failure!.trace[0]!.key).toBe("/s:root/s:node/s:n/s:input/s:v/s:binding");
});

/** A flow whose `k` CALL node binds `binding` to a def input `y: declared`. */
const callFlow = (declared: string, binding: string) => `
id: complex
name: complex
input:
  x: !float
defs:
  body:
    input:
      y: !${declared}
    nodes:
      inner:
        kind: code
        input:
          y: \${input.y}
        code: |
          return inputs["y"]
    output: \${inner.output}
nodes:
  k:
    kind: call
    call: body
    input:
      y: ${binding}
output: \${k.output}
`;

for (const declared of ["float", "int"]) {
  test(`a call binding that evaluates to no real number fails the call: ${declared}`, async () => {
    const result = await run(callFlow(declared, "${(0 - 8) ** 0.5}"), { x: 1.0 });

    expectBindingFails(result, "k", NO_RESULT);
    expect(startedIds(result)).toEqual(["__start__", "k"]);
  });
}

/** A flow whose `c` CODE node returns `value` and whose `d` node reads it. */
const readerFlow = (outputLine: string, value: string) => `
id: complex
name: complex
nodes:
  c:
    kind: code
${outputLine}
    code: |
      return ${value}
  d:
    kind: code
    input:
      seen: \${c.output}
    code: |
      return inputs["seen"]
output: \${d.output}
`;

test("a NaN object output from a CODE power fails the node at the field", async () => {
  const result = await run(readerFlow("    output: !object", '{"a": [(-8) ** 0.5]}'));

  expect(result.status).toBe("failed");
  expect(result.failure!.code).toBe("contract.output");
  expect(result.failure!.nodeId).toBe("c");
  expect(result.failure!.message).toBe("output at a[0]: NaN is not a yian value");
  expect(result.events.some((event) => (event as { nodeId?: string }).nodeId === "d")).toBe(false);
});

test("a NaN undeclared output from a CODE power fails the node", async () => {
  const result = await run(`
id: complex
name: complex
nodes:
  n:
    kind: code
    code: |
      return (-8) ** 0.5
output: \${n.output}
`);

  expect(result.status).toBe("failed");
  expect(result.failure!.code).toBe("contract.output");
  expect(result.failure!.nodeId).toBe("n");
  expect(result.failure!.message).toBe("output: NaN is not a yian value");
});

test("a flow output binding with no real number fails the END binding", async () => {
  const result = await run(`
id: complex
name: complex
nodes:
  c:
    kind: code
    code: |
      return 1
output: \${(0 - 8) ** 0.5}
`);

  expectBindingFails(result, "__end__", NO_RESULT);
});

test("a MAP over item with no real number fails the map", async () => {
  const result = await run(`
id: complex
name: complex
nodes:
  m:
    kind: map
    over: "\${[1.0, (0 - 8) ** 0.5]}"
    input:
      y: !float \${item}
    nodes:
      work:
        kind: code
        input:
          y: \${input.y}
        code: |
          return inputs["y"]
    output: \${work.output}
output: \${m.output}
`);

  expectBindingFails(result, "m", NO_RESULT);
  expect(result.failure!.trace[0]!.key).toBe("/s:root/s:node/s:m/s:reserved/s:over/s:binding");
});

test("no real number inside a template fails the binding", async () => {
  const result = await run(`
id: complex
name: complex
nodes:
  c:
    kind: code
    input:
      y: v=\${(0 - 8) ** 0.5}
    code: |
      return inputs["y"]
output: \${c.output}
`);

  expectBindingFails(result, "c", NO_RESULT);
});
