/**
 * CASE's else route has its own handle: a branch can't take it, a route to an unknown handle is
 * rejected, the handle without an edge skips every branch, and a node named `default` runs only on
 * its own branch.
 */

import { afterAll, expect, test } from "bun:test";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { loadFlow } from "../../src/compose/loader.ts";
import { NodeStarted } from "../../src/events.ts";
import { ELSE_HANDLE } from "../../src/nodes/base.ts";
import { Case, CaseNode } from "../../src/nodes/case/node.ts";
import { advance, type RunResult, runFlow } from "../../src/run.ts";
import { NodeState } from "../../src/runtime/common.ts";
import { commitPrepared, NodeDataError, prepareNodeRouted } from "../../src/runtime/state.ts";
import { loadErrorAsync } from "../helpers/declarations.ts";
import { hostEnv } from "../helpers/mcp_helpers.ts";
import { callRedirectFlow, rejected, routeInFlight } from "../helpers/run_state_fixtures.ts";

const runDir = mkdtempSync(join(tmpdir(), "yian-case-else-runs-"));
afterAll(() => rmSync(runDir, { recursive: true, force: true }));

const SEARCHED_FLOW = `
id: f
name: f
input:
  n: !int
nodes:
  gate:
    kind: case
    cases:
      - when: "input.n > 5"
        then: default
    else: small
  default: {kind: code, code: "return 'big'", output: !str}
  small: {kind: code, code: "return 'small'", output: !str}
output: \${default.output | small.output}
`;

const NO_ELSE_FLOW = `
id: f
name: f
input:
  n: !int
nodes:
  gate:
    kind: case
    cases:
      - when: "input.n > 5"
        then: default
  default: {kind: code, code: "return 'big'", output: !str}
output: \${default.output :- "none"}
`;

const ON_FLOW = `
id: f
name: f
input:
  s: !str
nodes:
  gate:
    kind: case
    on: \${input.s}
    cases:
      - when: a
        then: default
    else: other
  default: {kind: code, code: "return 'a'", output: !str}
  other: {kind: code, code: "return 'other'", output: !str}
output: \${default.output | other.output}
`;

/** Load `text` and run it on `inputs` in `hostEnv`. */
async function run(text: string, inputs: Record<string, unknown>) {
  return runFlow(await loadFlow(text), inputs, { env: hostEnv(runDir) });
}

/** The ids of every node that started in a run. */
function started(result: RunResult): Set<string> {
  return new Set(result.events.filter((event) => event instanceof NodeStarted).map((event) => event.nodeId));
}

test.each([
  [1, "small", "default", "small"],
  [9, "default", "small", "big"],
])("a then target named default runs only on its own branch: n=%p", async (n, ran, skipped, expected) => {
  const result = await run(SEARCHED_FLOW, { n });

  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(result.output).toBe(expected);
  expect(started(result).has(ran)).toBe(true);
  expect(started(result).has(skipped)).toBe(false);
});

test("a then target named default does not run on no match without else", async () => {
  const result = await run(NO_ELSE_FLOW, { n: 1 });

  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(result.output).toBe("none");
  expect(started(result).has("default")).toBe(false);
});

test.each([
  ["a", "default", "other", "a"],
  ["z", "other", "default", "other"],
])("an on case target named default runs only on its own branch: s=%p", async (s, ran, skipped, expected) => {
  const result = await run(ON_FLOW, { s });

  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(result.output).toBe(expected);
  expect(started(result).has(ran)).toBe(true);
  expect(started(result).has(skipped)).toBe(false);
});

test("the else handle is no possible node id", async () => {
  expect(ELSE_HANDLE).toBe("#else");
  const flow = `
id: f
name: f
nodes:
  "#else": {kind: code, code: "return 1", output: !int}
output: \${#else.output}
`;
  const error = await loadErrorAsync(loadFlow(flow));
  expect(error.message).toBe(
    "node id '#else' uses a reserved separator ('#' and '/' are reserved for runtime graph expansion; '.' is reserved for reference paths)",
  );
});

test("a case may not take the else handle as a branch", () => {
  expect(() => new Case(ELSE_HANDLE, { when: "True" })).toThrow(RangeError);
  expect(() => new Case(ELSE_HANDLE, { when: "True" })).toThrow(
    "CASE handle '#else' is reserved for the else route; name the branch by its target node id",
  );

  const held = Object.assign(Object.create(Case.prototype) as Case, { handle: ELSE_HANDLE, when: "x", origin: null });
  const node = new CaseNode("g", [held]);
  expect(() => node.validateExecutionSetup()).toThrow(RangeError);
  expect(() => node.validateExecutionSetup()).toThrow(
    "node 'g' case handle '#else' is reserved for the else route; name the branch by its target node id",
  );
});

test("a route to an unknown handle named default is rejected", () => {
  const { state } = routeInFlight(callRedirectFlow());

  let caught: unknown;
  try {
    prepareNodeRouted(state, "router", "default");
  } catch (error) {
    caught = error;
  }

  expect(caught).toBeInstanceOf(NodeDataError);
  const original = (caught as NodeDataError).original;
  expect((original as Error).constructor).toBe(RangeError);
  expect((original as Error).message).toBe("NodeRouted handle 'default' matches no outgoing edge of 'router'");
});

test("the else handle without an edge skips every branch", () => {
  const { state, log } = routeInFlight(callRedirectFlow());

  commitPrepared(log, state, prepareNodeRouted(state, "router", ELSE_HANDLE));

  expect(state.edgeState.get("router-child-end")).toBe(NodeState.SKIPPED);
  expect(state.ready).toEqual([]);
});

const ELSE_LESS_FLOW = `
id: f
name: f
input:
  n: !int
nodes:
  gate:
    kind: case
    cases:
      - when: "input.n > 5"
        then: big
  big: {kind: code, code: "return 'big'", output: !str}
output: \${big.output :- "none"}
`;

test("a log routed to the old default handle does not resume", async () => {
  const dir = mkdtempSync(join(tmpdir(), "yian-case-else-"));
  const env = hostEnv(dir);
  try {
    const path = join(dir, "run.jsonl");
    const first = await runFlow(await loadFlow(ELSE_LESS_FLOW), { n: 1 }, { log: path, env });
    expect(first.status).toBe("succeeded");
    expect(first.output).toBe("none");

    const lines = readFileSync(path, "utf8").split("\n").slice(0, -1);
    const records = lines.map((line) => JSON.parse(line) as { kind: string; payload: { handle?: string } });
    const routed = records.findIndex((record) => record.kind === "node_routed");
    expect(records[routed]!.payload.handle).toBe(ELSE_HANDLE);
    records[routed]!.payload.handle = "default";
    lines[routed] = JSON.stringify(records[routed]);
    writeFileSync(path, `${lines.join("\n")}\n`);

    const caught = await rejected(advance({ log: path, env }));
    expect((caught as Error).constructor).toBe(RangeError);
    expect((caught as Error).message).toBe("NodeRouted handle 'default' matches no outgoing edge of 'gate'");
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
