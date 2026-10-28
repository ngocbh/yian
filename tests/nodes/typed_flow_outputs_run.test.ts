/**
 * Typed flow outputs checked when a run reaches the flow's END.
 *
 * An output written `name: !Type value` declares the value's type. END's binder checks the
 * resolved value against it (code `binding.type`, located on the output), and END commits through
 * the declared contract. A whole number in a `float` slot stays an int (decision 38).
 */

import { afterAll, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { loadFlow } from "../../src/compose/loader.ts";
import { type RunResult, runFlow } from "../../src/run.ts";
import { hostEnv } from "../helpers/mcp_helpers.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-typed-flow-outputs-run-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));
const env = hostEnv(dir);

/** Load `text` and run it in the test's environment. */
async function run(text: string, inputs: Record<string, unknown> = {}): Promise<RunResult> {
  return runFlow(await loadFlow(text, { env }), inputs, { env });
}

/** Assert `result` succeeded and return its output. */
function succeeded(result: RunResult): unknown {
  expect(result.failure).toBeNull();
  expect(result.status).toBe("succeeded");
  return result.output;
}

/** Return a flow whose `n` node returns `value` into `output:` written as `output`. */
const valueFlow = (value: string, output: string) => `
id: typed_out
name: typed_out
nodes:
  n:
    kind: code
    code: |
      return ${value}
output: ${output}
`;

test("mismatched typed output fails end on the output line", async () => {
  const text = valueFlow("'x'", "!int ${n.output}");
  const loaded = await loadFlow(text, { env });

  const result = await runFlow(loaded, {}, { env });

  expect(result.status).toBe("failed");
  expect(result.failure!.code).toBe("binding.type");
  expect(result.failure!.nodeId).toBe("__end__");
  const diagnostic = loaded.diagnose(result.failure!);
  expect(diagnostic.frames.length).toBeGreaterThan(0);
  const line = text.split("\n").indexOf("output: !int ${n.output}") + 1;
  expect(diagnostic.frames.at(-1)!.range.coveredLines(text)).toEqual([line]);
});

test("matching typed output returns the value", async () => {
  expect(succeeded(await run(valueFlow("5", "!int ${n.output}")))).toBe(5n);
});

test("float output widens an int value", async () => {
  expect(succeeded(await run(valueFlow("30", "!float ${n.output}")))).toBe(30n);
});

/** Return a flow whose record output types `a: float` and writes `b` as `b`. */
const recordFlow = (b: string) => `
id: typed_record
name: typed_record
nodes:
  n:
    kind: code
    code: |
      return 30
  m:
    kind: code
    code: |
      return "m"
output:
  a: !float \${n.output}
  b: ${b}
`;

test("fully typed record output widens an int in a float field", async () => {
  expect(succeeded(await run(recordFlow("!str ${m.output}")))).toEqual({ a: 30n, b: "m" });
});

test("partly typed record output widens its typed float field", async () => {
  expect(succeeded(await run(recordFlow("${m.output}")))).toEqual({ a: 30n, b: "m" });
});

test("partly typed record output keeps its untyped field as given", async () => {
  const text = recordFlow("${m.output}").replace('return "m"', "return 7");

  expect(succeeded(await run(text))).toEqual({ a: 30n, b: 7n });
});

test("record node output widens like a whole float output", async () => {
  const text = `
id: recfloat
name: recfloat
typedefs:
  Score:
    value: float
input:
  n: !int
nodes:
  top:
    kind: code
    input: {n: "\${input.n}"}
    output: float
    code: |
      return inputs["n"]
  rec:
    kind: code
    input: {n: "\${input.n}"}
    output: Score
    code: |
      return {"value": inputs["n"]}
output:
  top: \${top.output}
  rec: \${rec.output.value}
`;

  expect(succeeded(await run(text, { n: 1 }))).toEqual({ top: 1n, rec: 1n });
});

test("partly typed record output rejects a mistyped typed field", async () => {
  const text = recordFlow("${m.output}").replace("return 30", 'return "x"');

  const result = await run(text);

  expect(result.status).toBe("failed");
  expect(result.failure!.code).toBe("binding.type");
  expect(result.failure!.nodeId).toBe("__end__");
});

/** Return a flow that CALLs a child whose `n` returns `childValue`. */
const callFlow = (childValue: string, childOutput: string) => `
id: typed_parent
name: typed_parent
defs:
  child:
    nodes:
      n:
        kind: code
        code: |
          return ${childValue}
    output: ${childOutput}
nodes:
  use:
    kind: call
    call: child
output: \${use.output}
`;

test("float child output widens through the call", async () => {
  expect(succeeded(await run(callFlow("30", "!float ${n.output}")))).toBe(30n);
});
