/**
 * TS-only: the reducer's producer index and the `derived` node trait.
 *
 * `RunState.producers` maps each pool name to the seq of the accepted `pool_commit` that last
 * wrote it, in every run: run inputs and system values map to their run-start commits, any other
 * name to its node's commit once that node's `node_succeeded` names it. It is pruned with the
 * pool and rebuilt by replay. A derived node (START, END, LOOP, CALL, MAP: its value only reads
 * earlier values through its binding) gets an in-memory input route in a non-learnable run too,
 * whose source leaves name the commits its binding read; nothing more is written to the log.
 */

import { afterAll, expect, spyOn, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { NodeKind } from "../../src/nodes/base.ts";
import { CallNode } from "../../src/nodes/call/node.ts";
import { EndNode } from "../../src/nodes/end/node.ts";
import { LoopNode } from "../../src/nodes/loop/node.ts";
import { MapNode } from "../../src/nodes/map/node.ts";
import { StartNode } from "../../src/nodes/start/node.ts";
import { runFlow } from "../../src/run.ts";
import { NO_ENV } from "./run_state_fixtures.ts";

const tmp = mkdtempSync(join(tmpdir(), "yian-producer-index-"));
afterAll(() => rmSync(tmp, { recursive: true, force: true }));

let logs = 0;
/** A fresh log path under the test's temp folder. */
const freshLog = () => join(tmp, `log-${++logs}.jsonl`);

/** A loop whose body CALLs another def, so child ENDs commit under their spawners. */
const CALL_IN_LOOP = `
id: count
name: count
defs:
  add1:
    input: {x: !int}
    nodes:
      inc: {kind: code, input: {x: "\${input.x}"}, output: {y: int}, code: "return {y: inputs.x + 1}"}
    output: {y: "\${inc.output.y}"}
  body:
    input: {x: !int}
    nodes:
      step: {kind: call, call: add1, input: {x: "\${input.x}"}}
    output: {x: "\${step.output.y}"}
nodes:
  count: {kind: loop, call: body, input: {x: 0}, times: 2}
output: \${count.output.x}
`;

/** A MAP over the run's input list. */
const DOUBLES = `
id: doubles
name: doubles
input: {xs: !list(int)}
defs:
  double:
    input: {x: !int}
    nodes:
      times2: {kind: code, input: {x: "\${input.x}"}, output: {y: int}, code: "return {y: inputs.x * 2}"}
    output: \${times2.output.y}
nodes:
  each: {kind: map, over: "\${input.xs}", call: double, input: {x: "\${item}"}}
output: \${each.output}
`;

/** One node of every kind; loaded, never run. */
const EVERY_KIND = `
id: kinds
name: kinds
input: {xs: !list(int)}
defs:
  one:
    input: {x: !int}
    nodes:
      c: {kind: code, input: {x: "\${input.x}"}, output: int, code: "return inputs.x"}
    output: {x: "\${c.output}"}
nodes:
  ask: {kind: human_input, prompt: "q?", output: str}
  w: {kind: wait, until: "2030-01-01T00:00:00Z"}
  route:
    kind: case
    cases:
      - when: \${ask.output == "a"}
        then: yes
    else: no
  yes: {kind: agent, prompt: hi}
  no: {kind: tool, tool_id: echo, input: {n: "x"}, output: str}
  code: {kind: code, code: "return 1", output: int}
  one_call: {kind: call, call: one, input: {x: 1}}
  each: {kind: map, over: "\${input.xs}", call: one, input: {x: "\${item}"}}
  again: {kind: loop, call: one, input: {x: 0}, times: 2}
output: \${again.output}
`;

const load = (text: string) => loadFlow(text, { sourceLabel: "flow.yaml" });

/** Run `flow` serially into `log`, learnable or not. */
const run = (flow: ComposedFlow, log: string, inputs: Record<string, unknown> = {}, learnable = false) =>
  runFlow(flow, inputs, { env: NO_ENV, log, runId: "r1", numWorkers: 0, learnable });

const DERIVED_KINDS = new Set<string>([NodeKind.START, NodeKind.END, NodeKind.LOOP, NodeKind.CALL, NodeKind.MAP]);

test("derived is true for START, END, LOOP, CALL and MAP and false for every other kind", async () => {
  const flow = await load(EVERY_KIND);
  const kinds = new Set<string>();
  for (const node of flow.compiled.nodes.values()) {
    kinds.add(node.kind);
    expect([node.kind, node.derived]).toEqual([node.kind, DERIVED_KINDS.has(node.kind)]);
  }
  expect([...kinds].sort()).toEqual(Object.values(NodeKind).sort());
});

test("a derived kind's forward is synchronous: it returns no promise", async () => {
  const spies = [StartNode, EndNode, LoopNode, CallNode, MapNode].map((kind) => spyOn(kind.prototype, "forward"));
  try {
    await run(await load(CALL_IN_LOOP), freshLog());
    await run(await load(DOUBLES), freshLog(), { xs: [1, 2] });
    for (const spy of spies) {
      expect(spy.mock.calls.length).toBeGreaterThan(0);
      for (const result of spy.mock.results) expect(result.value).not.toBeInstanceOf(Promise);
    }
  } finally {
    for (const spy of spies) spy.mockRestore();
  }
});
