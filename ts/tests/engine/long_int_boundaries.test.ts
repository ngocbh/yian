/**
 * A port of `tests/engine/test_long_int_boundaries.py`: an int over 4300 digits at each binding
 * and output boundary a run crosses. Each run is resumed from its log, which must end the same.
 *
 * Ported: every row but the one already ported below.
 *
 * Changed: ints are `bigint`s with no digit limit (decisions 4a, 4b, 23), so a 5000-digit int
 * crosses every boundary whole where Python fails at the first one with its digit-limit message.
 * The binding, node output, flow output, value entry, MAP `over` and LOOP seed rows succeed with
 * the whole value; the CODE nodes that returned `1` return their input instead (and the MAP body
 * binds `y`), so the result shows the value that crossed, and a resumed run must give the same
 * output. A long int bound or returned where `str` is declared is a type mismatch
 * (`binding.type`, `contract.output`), whose message shows the int's digits cut at the
 * shown-value cap; so is the partial-record field typed `str`, and the `str` default of
 * `test_a_long_int_default_is_rejected_before_its_contract`, whose `BindingError` keeps the code
 * `binding.evaluate`. The other two partial-record fields accept the int, and the constructor
 * mismatch shows the digits cut at the cap where Python shows `<int with 5000 digits>`. CODE
 * bodies are JS (`10n ** 4999n`). `test_a_long_int_in_a_partial_record_output_names_the_limit` is
 * ported whole here: `run_state_outcomes.test.ts` accepts a long int in a declared and an opaque
 * field, but has no mistyped-field row.
 *
 * Already ported: `test_a_long_int_default_is_rejected` in `flow_inputs.test.ts` (`applyDefaults
 * accepts a long int default`).
 */

import { afterAll, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { CompiledFlow, Edge, END_ID, START_ID } from "../../src/compile/model.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import type { FailureCode } from "../../src/diagnostics.ts";
import type { Node } from "../../src/nodes/base.ts";
import { applyDefaults, BindingError } from "../../src/nodes/binding.ts";
import { Parameter, PartialRecordContract, ValueContract } from "../../src/nodes/contracts.ts";
import { EndNode } from "../../src/nodes/end/node.ts";
import { StartNode } from "../../src/nodes/start/node.ts";
import { type RunResult, advance, runFlow } from "../../src/run.ts";
import { commitPrepared, NodeDataError, prepareNodeStarted } from "../../src/runtime/state.ts";
import { Type, TypeCheckError, type Value, ValueKind } from "../../src/typesys/index.ts";
import { hostEnv } from "./mcp_helpers.ts";
import {
  commitStartSuccess,
  FuncNode,
  initializedAndStarted,
  prepareOutput,
  thrownBy,
} from "./run_state_fixtures.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-long-int-boundaries-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));
const env = hostEnv(dir);

const BIG = 10n ** 4999n + 12345n; // 5000 digits
const POWER = 10n ** 4999n; // `10 ** input.n` with `n = 4999`
// A 5000-digit int as a message shows it: its first 199 characters and an ellipsis.
const SHOWN = (value: bigint) => String(value).slice(0, 199) + "…";

/** Check a message shows no long run of the int's digits. */
function assertClean(message: string): void {
  expect(message).not.toMatch(/[0-9]{600}/);
}

let runs = 0;

/** Run the flow `text` with `n = 4999`, check its log resumes to the same end, and return the result. */
async function run(text: string): Promise<RunResult> {
  const flow = await loadFlow(text, { env });
  const log = join(dir, `run-${runs++}.jsonl`);
  const result = await runFlow(flow, { n: 4999n }, { env, log });
  const resumed = await advance({ log, env });
  expect(resumed.status).toBe(result.status);
  if (result.failure !== null) expect(resumed.failure!.message).toBe(result.failure.message);
  else expect(resumed.output).toEqual(result.output);
  return result;
}

/** Assert the run succeeded with `output`. */
function assertSucceeds(result: RunResult, output: Value): void {
  expect(result.failure).toBeNull();
  expect(result.status).toBe("succeeded");
  expect(result.output).toEqual(output);
}

/** Assert the run failed `nodeId` with `code` and exactly `message`, without the int's digits. */
function assertFails(result: RunResult, code: FailureCode, nodeId: string, message: string): void {
  expect(result.status).toBe("failed");
  expect(result.failure!.code).toBe(code);
  expect(result.failure!.nodeId).toBe(nodeId);
  expect(result.failure!.message).toBe(message);
  assertClean(result.failure!.message);
}

/** Return a flow whose `c` CODE node binds its input `x` to `binding` and returns it. */
const codeInputFlow = (binding: string) => `
id: long
name: long
input:
  n: !int
nodes:
  c:
    kind: code
    input:
      x: ${binding}
    code: |
      return inputs.x
output: \${c.output}
`;

test.each<[string, string, Value]>([
  ["whole", "${10 ** input.n}", POWER],
  ["list-item", "${[1, 10 ** input.n]}", [1n, POWER]],
])("a long int binding fails its node (%s)", async (_id, binding, output) => {
  assertSucceeds(await run(codeInputFlow(binding)), output);
});

test.each([
  ["str-input", "!str ${10 ** input.n}", "binding 'x'"],
  ["nested-typed", "{a: !str '${10 ** input.n}'}", "binding 'x.a'"],
])("a long int binding fails its node (%s)", async (_id, binding, where) => {
  const result = await run(codeInputFlow(binding));

  assertFails(result, "binding.type", "c", `${where}: ${SHOWN(POWER)} does not match declared type str`);
});

test("a long int default is rejected before its contract", () => {
  const parameter = new Parameter("x", {
    contract: new ValueContract(Type.scalar(ValueKind.STRING)),
    hasDefault: true,
    default: BIG,
  });

  const caught = thrownBy(() => applyDefaults([parameter], {}));

  expect(caught).toBeInstanceOf(BindingError);
  expect((caught as BindingError).failureCode).toBe("binding.evaluate");
  expect((caught as BindingError).message).toBe(`${SHOWN(BIG)} does not match declared type str`);
});

/** Return a flow whose `c` CODE node returns `value`. */
const codeOutputFlow = (value: string, outputLine = "") => `
id: long
name: long
input:
  n: !int
nodes:
  c:
    kind: code
${outputLine}
    code: |
      return ${value}
output: \${c.output}
`;

test.each<[string, string, string, Value]>([
  ["untyped", "10n ** 4999n", "", POWER],
  ["int", "10n ** 4999n", "    output: int", POWER],
  ["record-untyped", '{"v": [10n ** 4999n]}', "", { v: [POWER] }],
  ["record-typed", '{"v": [10n ** 4999n]}', "    output: {v: list(int)}", { v: [POWER] }],
])("a long int node output fails the node (%s)", async (_id, value, outputLine, output) => {
  assertSucceeds(await run(codeOutputFlow(value, outputLine)), output);
});

test("a long int node output fails the node (str)", async () => {
  const result = await run(codeOutputFlow("10n ** 4999n", "    output: str"));

  assertFails(result, "contract.output", "c", `${SHOWN(POWER)} does not match declared type str`);
});

/** Return `START -> work -> END` where `work` declares a partial-record output, `work` in flight. */
function partialRecordWorking() {
  const work = new FuncNode("work", (inputs) => inputs, {
    outputContract: new PartialRecordContract([
      ["items", new ValueContract(Type.scalar(ValueKind.LIST_INTEGER))],
      ["label", new ValueContract(Type.scalar(ValueKind.STRING))],
      ["note", null],
    ]),
  });
  const flow = CompiledFlow.fromParts(
    new Map<string, Node>([
      [START_ID, new StartNode(START_ID)],
      ["work", work],
      [END_ID, EndNode.value(END_ID, { outputName: null })],
    ]),
    [new Edge("start-work", START_ID, "work"), new Edge("work-end", "work", END_ID)],
  );
  const { state, log } = initializedAndStarted(flow);
  commitStartSuccess(state, log);
  commitPrepared(log, state, prepareNodeStarted(state, "work"));
  return { state, log };
}

test.each<[string, Value]>([
  ["declared-field", { items: [1n, BIG], label: "a", note: "x" }],
  ["opaque-field", { items: [1n], label: "a", note: { deep: BIG } }],
])("a long int in a partial record output names the limit (%s)", (_id, output) => {
  const { state, log } = partialRecordWorking();

  commitPrepared(log, state, prepareOutput(log, state, "work", output));

  expect(state.pool.get("work")).toEqual(output);
});

test("a long int in a partial record output names the limit (mistyped-field)", () => {
  const { state, log } = partialRecordWorking();

  const caught = thrownBy(() => prepareOutput(log, state, "work", { items: [1n], label: BIG, note: "x" }));

  expect(caught).toBeInstanceOf(NodeDataError);
  const original = (caught as NodeDataError).original as TypeCheckError;
  expect(original).toBeInstanceOf(TypeCheckError);
  expect(original.message).toBe(`${SHOWN(BIG)} does not match declared type str`);
  expect(original.path).toEqual(["label"]);
  assertClean(original.message);
});

test("a long int flow output fails the end binding", async () => {
  const result = await run(`
id: long
name: long
input:
  n: !int
nodes:
  c:
    kind: code
    code: |
      return 1
output: \${10 ** input.n}
`);

  assertSucceeds(result, POWER);
});

test("a long int value entry fails the entry", async () => {
  const result = await run(`
id: long
name: long
input:
  n: !int
nodes:
  v: !int \${10 ** input.n}
output: \${v.output}
`);

  assertSucceeds(result, POWER);
});

test("a long int map over fails the map", async () => {
  const result = await run(`
id: long
name: long
input:
  n: !int
nodes:
  m:
    kind: map
    over: \${[10 ** input.n]}
    input:
      y: !int \${item}
    nodes:
      work:
        kind: code
        input:
          y: \${input.y}
        code: |
          return inputs.y
    output: \${work.output}
output: \${m.output}
`);

  assertSucceeds(result, [POWER]);
});

test("a long int loop seed fails the loop", async () => {
  const result = await run(`
id: long
name: long
input:
  n: !int
defs:
  bump:
    input:
      n: !int
      exited: !bool
    nodes: {}
    output:
      n: \${input.n}
      exited: \${input.exited}
nodes:
  loop:
    kind: loop
    call: bump
    input:
      n: \${10 ** input.n}
      exited: false
    times: 1
output: \${loop.output}
`);

  assertSucceeds(result, { n: POWER, exited: false });
});

test.each([
  ["compare", "${10 ** input.n > 0}", true],
  ["isinstance-str", "${isinstance(10 ** input.n, str)}", false],
  ["isinstance-int", "${isinstance(10 ** input.n, int)}", true],
])("a long int inside one expression is allowed (%s)", async (_id, expression, expected) => {
  assertSucceeds(await run(codeInputFlow(expression)), expected);
});

test("a constructor mismatch on a long int is a located type error", async () => {
  const result = await run(`
id: long
name: long
typedefs:
  Pt:
    x: str
input:
  n: !int
nodes:
  c:
    kind: code
    input:
      p: \${Pt(x=10 ** input.n)}
    code: |
      return 1
output: \${c.output}
`);

  assertFails(result, "binding.type", "c", `Pt(...): ${SHOWN(POWER)} does not match declared type str`);
  expect(result.failure!.trace[0]!.key).toBe("/s:root/s:node/s:c/s:input/s:p/s:binding");
});
