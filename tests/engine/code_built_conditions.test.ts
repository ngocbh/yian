/**
 * A CASE `when` or LOOP predicate set in code that doesn't parse is a setup error: `run` rejects
 * before the run starts, and an evaluation error still fails in `forward`.
 */

import { afterEach, beforeEach, expect, test } from "bun:test";
import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { ExpressionError } from "../../src/expr/index.ts";
import { Case, CaseNode } from "../../src/nodes/case/node.ts";
import { LoopNode } from "../../src/nodes/loop/node.ts";
import { runFlow, type RunFlowOptions } from "../../src/run.ts";

const FIXTURES = fileURLToPath(new URL("../conformance/fixtures", import.meta.url));

/** Return the error `promise` rejects with; fail when it resolves. */
async function rejection(promise: Promise<unknown>): Promise<unknown> {
  try {
    await promise;
  } catch (error) {
    return error;
  }
  throw new Error("expected a rejection");
}

test("case when evaluation error still fails in forward", async () => {
  // A `when` that parses but can't evaluate is unchanged: it rejects from `forward`.
  const node = new CaseNode("gate", [new Case("pos", { when: "x > 0" })]);

  const error = await rejection(node.run({ x: { a: 1n } }));

  expect(error).toBeInstanceOf(ExpressionError);
  expect((error as Error).message).not.toContain("is not a valid boolean expression");
  expect((error as Error).stack).toContain("forward");
});

test("unparseable case when raises its parse error from setup", async () => {
  // `run` names the node, case, and field instead of failing inside `forward`.
  const node = new CaseNode("gate", [new Case("ok", { when: "x > 0" }), new Case("pos", { when: "${x} > 0" })]);

  const error = await rejection(node.run({ x: 1n }));

  expect(error).toBeInstanceOf(ExpressionError);
  const message = (error as Error).message;
  expect(message).toStartWith("node 'gate' case 'pos' `when:` '${x} > 0' is not a valid boolean expression: ");
  expect(message).toContain("write `x > 0`");
});

for (const kind of ["while", "until"] as const) {
  test(`unparseable loop predicate raises before the first body (${kind})`, async () => {
    // Both predicate kinds are checked at setup, including `until` at iteration 0.
    const node = new LoopNode("loop", {
      flowId: "f",
      child: {},
      predicateKind: kind,
      predicate: "not ${exited}",
      maxIters: 10,
    });

    const error = await rejection(node.run({ exited: false }));

    expect(error).toBeInstanceOf(ExpressionError);
    const message = (error as Error).message;
    expect(message).toStartWith(`node 'loop' \`${kind}:\` 'not \${exited}' is not a valid boolean expression: `);
    expect(message).toContain("write `not exited`");
  });
}

test("times loop has no predicate to check", () => {
  // A `times` loop carries no predicate; setup accepts it.
  const node = new LoopNode("loop", {
    flowId: "f",
    child: {},
    predicateKind: "times",
    predicate: null,
    times: 2,
    maxIters: 2,
  });

  expect(() => node.validateConditions()).not.toThrow();
});

test("validate conditions names the given path", () => {
  // The pre-run check names a node by its static path.
  const node = new CaseNode("gate", [new Case("pos", { when: "${x} > 0" })]);

  expect(() => node.validateConditions({ path: "loop/gate" })).toThrow(ExpressionError);
  expect(() => node.validateConditions({ path: "loop/gate" })).toThrow(/^node 'loop\/gate' case 'pos' `when:`/);
});

test("loop validate conditions names the given path", () => {
  const node = new LoopNode("loop", {
    flowId: "f",
    child: {},
    predicateKind: "while",
    predicate: "not ${exited}",
    maxIters: 10,
  });

  expect(() => node.validateConditions({ path: "outer/loop" })).toThrow(ExpressionError);
  expect(() => node.validateConditions({ path: "outer/loop" })).toThrow(/^node 'outer\/loop' `while:`/);
});

// ---- conditions set in code fail before the run starts ----

const GATED = `
id: gated
name: gated
input:
  v: !int
nodes:
  a:
    kind: code
    code: ./compose_codefns.ts:double
    input:
      n: \${input.v}
    output: int
  gate:
    kind: case
    cases:
      - when: "a.output > 0"
        then: pos
    else: neg
  pos:
    kind: code
    code: ./compose_codefns.ts:double
    input:
      n: \${a.output}
    output: int
  neg:
    kind: code
    code: ./compose_codefns.ts:double
    input:
      n: \${a.output}
    output: int
output: \${pos.output}
`;

const COUNTER = `
id: counter
name: counter
defs:
  bump:
    input:
      n: !int
      exited: !bool
    nodes:
      step:
        kind: code
        code: ./compose_codefns.ts:loop_bump
        input:
          n: \${input.n}
          exited: \${input.exited}
        output:
          n: int
          exited: bool
    output: \${step.output}
nodes:
  loop:
    kind: loop
    call: bump
    input:
      n: 0
      exited: false
    while: not exited
    max: 10
output: \${loop.output}
`;

let dir: string;

beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), "yian-conditions-"));
});

afterEach(() => {
  rmSync(dir, { recursive: true, force: true });
});

/** `runFlow` with only the test's credential store in its environment, so no host variable or login decides a row. */
function run(flow: ComposedFlow, inputs: Readonly<Record<string, unknown>>, options: RunFlowOptions = {}) {
  const env = new Map([["YIAN_AUTH_FILE", join(dir, "auth.json")]]);
  return runFlow(flow, inputs, { env: (name) => env.get(name), ...options });
}

test("a case when set in code fails before the run starts", async () => {
  // A downstream CASE's bad `when` rejects before any node runs or any record is written.
  const flow = await loadFlow(GATED, { baseDir: FIXTURES });
  const gate = flow.compiled.nodes.get("gate") as CaseNode;
  const [first, ...rest] = gate.cases;
  gate.cases = [new Case(first!.handle, { when: "${__r0} > 0", origin: first!.origin }), ...rest];
  const path = join(dir, "run.jsonl");

  const error = await rejection(run(flow, { v: 1 }, { log: path }));

  expect(error).toBeInstanceOf(ExpressionError);
  const message = (error as Error).message;
  expect(message).toStartWith("node 'gate' case 'pos' `when:` '${__r0} > 0' is not a valid boolean expression: ");
  expect(message).toContain("write `__r0 > 0`");
  expect(existsSync(path)).toBe(false);
});

test("a loop predicate set in code fails before the run starts", async () => {
  const flow = await loadFlow(COUNTER, { baseDir: FIXTURES });
  (flow.compiled.nodes.get("loop") as LoopNode).predicate = "not ${exited}";
  const path = join(dir, "run.jsonl");

  const error = await rejection(run(flow, {}, { log: path }));

  expect(error).toBeInstanceOf(ExpressionError);
  expect((error as Error).message).toStartWith(
    "node 'loop' `while:` 'not ${exited}' is not a valid boolean expression: ",
  );
  expect(existsSync(path)).toBe(false);
});
