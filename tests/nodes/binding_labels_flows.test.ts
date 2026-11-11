/**
 * Runtime binding errors name a desugared param the way load errors name it.
 *
 * A CASE `on:` reads as `node '<id>' on:`, a value entry's one input reads as `value entry '<id>'`,
 * and a flow output reads as `flow output` (a record field as `flow output '<name>'`); template
 * paths add `at <path>`. An inline call's argument reads as `<site> call(<callee>) argument '<p>'`.
 * An ordinary node input keeps `input '<p>'`.
 *
 * Most rows fail the value check every labeled binding makes with a value nested too deep: the runs
 * see `YIAN_MAX_VALUE_NESTING=3`, `x` is an `!Any` input nested three lists deep, and the binding
 * `${[[v] for v in input.x]}` nests it one level deeper while staying untyped at load.
 */

import { afterAll, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { loadFlow } from "../../src/compose/loader.ts";
import type { FailureCode } from "../../src/diagnostics.ts";
import { type RunResult, runFlow } from "../../src/run.ts";
import { loadErrorAsync } from "../helpers/declarations.ts";
import { hostEnv } from "../helpers/mcp_helpers.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-binding-labels-flows-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));

/** The test's environment, with a value nesting limit a short binding can pass. */
const env = hostEnv(dir, { YIAN_MAX_VALUE_NESTING: "3" });

/** A flow input at the nesting limit; `${[[v] for v in input.x]}` nests it one level deeper. */
const DEEP = [[[1n]]];

/** The message a binding of `${[[v] for v in input.x]}` fails with, after its label. */
const DEEPER = " at [0][0][0]: value nests deeper than 3 levels, the limit for a yian value (each record and list is a level)";

/** Load `text` and run it in the test's environment. */
async function run(text: string, inputs: Record<string, unknown> = {}): Promise<RunResult> {
  return runFlow(await loadFlow(text, { env }), inputs, { env });
}

/** Assert `result` failed with `code` at `nodeId` with exactly `message`. */
function expectFails(result: RunResult, code: FailureCode, nodeId: string, message: string): void {
  expect(result.status).toBe("failed");
  expect(result.failure!.code).toBe(code);
  expect(result.failure!.nodeId).toBe(nodeId);
  expect(result.failure!.message).toBe(message);
}

test("a case on failure names the on field", async () => {
  const result = await run(
    `
id: f
name: f
input:
  x: !Any
nodes:
  g:
    kind: case
    on: \${[[v] for v in input.x]}
    cases:
      - when: 1
        then: a
    else: a
  a: !str hi
output: \${g.output}
`,
    { x: DEEP },
  );

  expectFails(result, "binding.evaluate", "g", `node 'g' on${DEEPER}`);
  expect(result.failure!.trace.at(-1)!.key).toBe("/s:root/s:node/s:g/s:field/s:on");
});

test("a case on failure in a called def names the on field", async () => {
  const result = await run(
    `
id: f
name: f
input:
  x: !Any
defs:
  route:
    input:
      x: !Any
    nodes:
      g:
        kind: case
        on: \${[[v] for v in input.x]}
        cases:
          - when: 1
            then: a
        else: a
      a: !str hi
    output: \${g.output}
nodes:
  invoke:
    kind: call
    call: route
    input:
      x: \${input.x}
output: \${invoke.output}
`,
    { x: DEEP },
  );

  expectFails(result, "binding.evaluate", "invoke/g", `node 'g' on${DEEPER}`);
});

test("a value entry failure names the entry", async () => {
  const result = await run(
    `
id: f
name: f
input:
  x: !Any
nodes:
  v: !float \${[[v] for v in input.x]}
output: \${v.output}
`,
    { x: DEEP },
  );

  expectFails(result, "binding.evaluate", "v", `value entry 'v'${DEEPER}`);
  expect(result.failure!.trace.at(-1)!.key).toBe("/s:root/s:node/s:v/s:input/s:v/s:binding");
});

test("a value entry type failure names the entry", async () => {
  const result = await run(`
id: f
name: f
nodes:
  p:
    kind: code
    code: |
      return 5
  x: !str \${p.output}
output: \${x.output}
`);

  expectFails(result, "binding.type", "x", "value entry 'x': 5 does not match declared type str");
});

test("a routed value entry failure points at the entry", async () => {
  const result = await run(
    `
id: f
name: f
input:
  x: !Any
  flag: !bool
nodes:
  g:
    kind: case
    on: \${input.flag}
    cases:
      - when: true
        then: a
    else: a
  a: !float \${[[v] for v in input.x]}
output: \${g.output}
`,
    { x: DEEP, flag: true },
  );

  expectFails(result, "binding.evaluate", "a", `value entry 'a'${DEEPER}`);
  expect(result.failure!.trace.at(-1)!.key).toBe("/s:root/s:node/s:a/s:input/s:a/s:binding");
});

test("a code node input failure keeps the input name", async () => {
  const result = await run(
    `
id: f
name: f
input:
  x: !Any
nodes:
  n:
    kind: code
    input:
      y: \${[[v] for v in input.x]}
    code: |
      return inputs["y"]
output: \${n.output}
`,
    { x: DEEP },
  );

  expectFails(result, "binding.evaluate", "n", `input 'y'${DEEPER}`);
});

/** Return a flow with a trivial CODE node `c` and the given `output:` text. */
const codeAnd = (output: string, inputs = "  x: !Any\n") => `
id: f
name: f
input:
${inputs}nodes:
  c:
    kind: code
    code: |
      return 1
output: ${output}
`;

test("a single flow output failure names the flow output", async () => {
  const result = await run(codeAnd("${[[v] for v in input.x]}"), { x: DEEP });

  expectFails(result, "binding.evaluate", "__end__", `flow output${DEEPER}`);
  expect(result.failure!.trace.at(-1)!.key).toBe("/s:root/s:output/i:0/s:binding");
});

test("a record flow output failure names the field", async () => {
  const result = await run(codeAnd("\n  r: ${[[v] for v in input.x]}"), { x: DEEP });

  expectFails(result, "binding.evaluate", "__end__", `flow output 'r'${DEEPER}`);
});

test("an unknown type in a record flow output names the field", async () => {
  const error = await loadErrorAsync(loadFlow(codeAnd("\n  r: ${isinstance(input.s, Nope)}", "  s: !str\n"), { env }));

  expect(error.message).toContain("flow output 'r': unknown type 'Nope'");
});

test("an inline call argument failure names the argument", async () => {
  const text = `
id: f
name: f
input:
  x: !Any
defs:
  child:
    input:
      y: !float
    nodes:
      c:
        kind: code
        code: "return inputs['y']"
        input:
          y: \${input.y}
        output: !float
    output: \${c.output}
nodes:
  n:
    kind: code
    code: "return inputs['a']"
    input:
      a: call(child, y=\${[[v] for v in input.x]})
    output: !float
output: \${n.output}
`;
  const result = await run(text, { x: DEEP });

  expect(result.status).toBe("failed");
  expect(result.failure!.nodeId).toBe("__call_0");
  expect(result.failure!.message).toBe(`node 'n' input 'a' call(child) argument 'y'${DEEPER}`);
});

const CHILD_DEFS = `
defs:
  child:
    input:
      y: !float
    nodes:
      c:
        kind: code
        code: "return inputs['y']"
        input:
          y: \${input.y}
        output: !float
    output: \${c.output}
  check:
    input:
      y: !float
    nodes:
      c:
        kind: code
        code: "return true"
        input:
          y: \${input.y}
        output: !bool
    output: \${c.output}
  lister:
    input:
      n: !float
    nodes:
      c:
        kind: code
        code: "return [inputs['n']]"
        input:
          n: \${input.n}
        output: !list(float)
    output: \${c.output}
`;

const BAD_ARG = "${[[v] for v in input.x]}";

/** Return a flow with the `child`, `check`, and `lister` defs and the given body. */
const callFlow = (body: string, inputs = "  x: !Any\n") =>
  "id: f\nname: f\n" + (inputs ? `input:\n${inputs}` : "") + CHILD_DEFS + body;

/** Assert `result` failed evaluating a call argument and return its last trace key. */
function argumentFailureKey(result: RunResult): string {
  expect(result.status).toBe("failed");
  expect(result.failure!.code).toBe("binding.evaluate");
  expect(result.failure!.trace.length).toBeGreaterThan(0);
  return result.failure!.trace.at(-1)!.key;
}

test("a node input call argument failure points at the input", async () => {
  const text = callFlow(`
nodes:
  n:
    kind: code
    code: "return inputs['a']"
    input:
      a: call(child, y=${BAD_ARG})
    output: !float
output: \${n.output}
`);
  const result = await run(text, { x: DEEP });

  expect(argumentFailureKey(result)).toBe("/s:root/s:node/s:n/s:input/s:a/s:binding");
});

test("a value entry call argument failure points at the entry", async () => {
  const text = callFlow(`
nodes:
  v: !float call(child, y=${BAD_ARG})
output: \${v.output}
`);
  const result = await run(text, { x: DEEP });

  expect(argumentFailureKey(result)).toBe("/s:root/s:node/s:v/s:input/s:v/s:binding");
});

test("a single flow output call argument failure points at the output", async () => {
  const text = callFlow(`
nodes:
  n: {kind: code, code: "return 1"}
output: call(child, y=${BAD_ARG})
`);
  const result = await run(text, { x: DEEP });

  expect(argumentFailureKey(result)).toBe("/s:root/s:output/i:0/s:binding");
});

test("a record flow output call argument failure points at the field", async () => {
  const text = callFlow(`
nodes:
  n: {kind: code, code: "return 1"}
output:
  q: \${n.output}
  r: call(child, y=${BAD_ARG})
`);
  const result = await run(text, { x: DEEP });

  expect(argumentFailureKey(result)).toBe("/s:root/s:output/i:1/s:binding");
});

test("a nested call argument failure points at the enclosing binding", async () => {
  const text = callFlow(`
nodes:
  n:
    kind: code
    code: "return inputs['a']"
    input:
      a: call(child, y=call(child, y=${BAD_ARG}))
    output: !float
output: \${n.output}
`);
  const result = await run(text, { x: DEEP });

  expect(argumentFailureKey(result)).toBe("/s:root/s:node/s:n/s:input/s:a/s:binding");
});

/** Return a flow whose CASE `g` routes on `input.flag` with the given branches. */
const caseCallFlow = (branches: string, readers = "") =>
  callFlow(
    `
nodes:
  g:
    kind: case
    on: \${input.flag}
${branches}
  b: !float 1.0
${readers}output: \${g.output}
`,
    "  x: !Any\n  flag: !bool\n",
  );

test("a case else call argument failure points at the else field", async () => {
  const text = caseCallFlow(`    cases:
      - when: true
        then: b
    else: call(child, y=${BAD_ARG})`);
  const result = await run(text, { x: DEEP, flag: false });

  expect(argumentFailureKey(result).endsWith("/s:node/s:g/s:field/s:else")).toBe(true);
});

test("a case then call argument failure points at the then field", async () => {
  const text = caseCallFlow(`    cases:
      - when: true
        then: call(child, y=${BAD_ARG})
    else: b`);
  const result = await run(text, { x: DEEP, flag: true });

  expect(argumentFailureKey(result).endsWith("/s:node/s:g/s:case/i:0/s:then")).toBe(true);
});

test("a read case else call argument failure points at the else field", async () => {
  const readers = `  r:
    kind: code
    code: "return inputs['v']"
    input:
      v: \${g.output}
    output: !float
`;
  const text = caseCallFlow(
    `    cases:
      - when: true
        then: b
    else: call(child, y=${BAD_ARG})`,
    readers,
  );
  const result = await run(text, { x: DEEP, flag: false });

  expect(argumentFailureKey(result).endsWith("/s:node/s:g/s:field/s:else")).toBe(true);
});

test("a flow assert call argument failure points at the assert", async () => {
  const text = callFlow(`
nodes:
  n: {kind: code, code: "return 1"}
output: \${n.output}
asserts:
  - call(check, y=${BAD_ARG})
`);
  const result = await run(text, { x: DEEP });

  expect(argumentFailureKey(result)).toBe("/s:root/s:assert/i:0");
});

test("a map over call argument failure points at the over field", async () => {
  const text = callFlow(`
nodes:
  m:
    kind: map
    call: child
    over: call(lister, n=${BAD_ARG})
    input:
      y: \${item}
output: \${m.output}
`);
  const result = await run(text, { x: DEEP });

  expect(argumentFailureKey(result)).toBe("/s:root/s:node/s:m/s:reserved/s:over/s:binding");
});

test("a child rejection of a call argument has the authored outer frame", async () => {
  const text = callFlow(
    `
nodes:
  m:
    kind: code
    code: return "x"
  n:
    kind: code
    code: "return inputs['a']"
    input:
      a: call(child, y=\${m.output})
    output: !float
output: \${n.output}
`,
    "",
  );
  const result = await run(text);

  expect(result.status).toBe("failed");
  expect(result.failure!.code).toBe("contract.input");
  expect(result.failure!.nodeId).toBe("__call_0/__start__");
  expect(result.failure!.trace[0]!.key).toBe("/s:root/s:node/s:n/s:input/s:a/s:binding");
  expect(result.failure!.trace.at(-1)!.key).toBe("/s:def/s:child/s:input/s:y/s:contract/s:type/s:root");
});
